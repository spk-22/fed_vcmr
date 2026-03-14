import faiss
import numpy as np
import sqlite3
from typing import List, Tuple
import torch
from src.config import INDEX_PATH, DIMENSION, DB_PATH
from src.transformer_model import TransformerInference

class SearchIndex:
    def __init__(self, dimension=DIMENSION):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension) # Inner Product on normalized vectors = Cosine
        self.chunk_ids = []

    def load_index(self):
        """Load the FAISS index and chunk_ids mapping."""
        self.index = faiss.read_index(str(INDEX_PATH))
        # We need to reload the chunk_ids from the DB in the same order
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id FROM chunks")
        self.chunk_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

    def build_from_cache(self):
        """Load chunk vectors from SQLite and .npy files and build FAISS index."""
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id, cache_path FROM chunks")
        rows = cursor.fetchall()
        
        embeddings = []
        for chunk_id, cache_path in rows:
            features = np.load(cache_path) # (8, 512)
            # Weighted average as per Technical Reference: 
            # weights = [0.5, 0.75, 1.0, 1.25, 1.25, 1.0, 0.75, 0.5]
            weights = np.array([0.5, 0.75, 1.0, 1.25, 1.25, 1.0, 0.75, 0.5])
            chunk_emb = np.sum(features.astype(np.float32) * weights[:, np.newaxis], axis=0) / np.sum(weights)
            # Normalize
            chunk_emb /= (np.linalg.norm(chunk_emb) + 1e-10)
            
            embeddings.append(chunk_emb)
            self.chunk_ids.append(chunk_id)
        
        if embeddings:
            self.index.add(np.stack(embeddings).astype(np.float32))
            faiss.write_index(self.index, str(INDEX_PATH))
        
        conn.close()
        print(f"FAISS index built with {len(self.chunk_ids)} chunks.")

    def coarse_search(self, query_emb: np.ndarray, top_k=100) -> List[Tuple[str, float]]:
        """Search the index and return chunk_ids and scores."""
        scores, indices = self.index.search(query_emb.astype(np.float32), top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append((self.chunk_ids[idx], float(scores[0][i])))
        return results

def maxsim_rerank(query_embs: np.ndarray, chunk_scores: List[Tuple[str, float]], top_k=5) -> List[Tuple[str, float]]:
    """
    MaxSim reranking.
    query_embs: (N_query, 512) - usually 3 expanded phrasings
    chunk_scores: List of (chunk_id, coarse_score)
    """
    reranked = []
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    for chunk_id, coarse_score in chunk_scores:
        cursor.execute("SELECT cache_path FROM chunks WHERE chunk_id=?", (chunk_id,))
        cache_path = cursor.fetchone()[0]
        frames = np.load(cache_path).astype(np.float32) # (8, 512)
        
        # MaxSim: (1/8) sum_i max_k cosine(q_k, v_i)
        # frames: (8, 512), query_embs: (N_q, 512)
        # Cosine similarity matrix: (N_q, 8)
        sim_matrix = np.matmul(query_embs, frames.T) # (N_q, 8)
        
        # For each frame, find its BEST matching query phrasing
        max_sim_per_frame = np.max(sim_matrix, axis=0) # (8,)
        maxsim_score = np.mean(max_sim_per_frame)
        
        reranked.append((chunk_id, maxsim_score))
    
    conn.close()
    # Sort by maxsim score descending
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]

class TransformerReranker:
    def __init__(self):
        self.inference = TransformerInference()

    def rerank(self, query_emb: np.ndarray, query_embs_multi: np.ndarray, chunk_scores: List[Tuple[str, float]], top_k=5) -> List[dict]:
        """
        Hybrid Rerank: 
        1. Use MaxSim (query_embs_multi) to pick the best segments.
        2. Use Transformer (query_emb) to refine boundaries of those segments.
        """
        if not self.inference.is_loaded:
            print("Warning: Transformer not loaded. Falling back to coarse segments.")
            
        # 1. Use MaxSim to pick Top-K from the coarse candidates
        maxsim_results = maxsim_rerank(query_embs_multi, chunk_scores, top_k=top_k)
        
        reranked = []
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Prepare query tensor (1, 1, 512) for Transformer adjustment
        q_tensor = torch.from_numpy(query_emb).float().to(self.inference.device)
        if q_tensor.dim() == 2:
            q_tensor = q_tensor.unsqueeze(0) # (1, 1, 512)
        
        for chunk_id, ms_score in maxsim_results:
            cursor.execute("SELECT cache_path, t_start, t_end FROM chunks WHERE chunk_id=?", (chunk_id,))
            cache_path, t_orig_start, t_orig_end = cursor.fetchone()
            
            # Load frames (8, 512)
            frames = np.load(cache_path).astype(np.float32)
            v_tensor = torch.from_numpy(frames).float().unsqueeze(0).to(self.inference.device) # (1, 8, 512)
            
            # Predict boundary adjustments [start_perc, end_perc]
            with torch.no_grad():
                boundaries = self.inference.model(v_tensor, q_tensor).cpu().numpy()[0]
            
            # Map percentiles back to temporal duration
            duration = t_orig_end - t_orig_start
            t_pred_start = t_orig_start + (boundaries[0] * duration)
            t_pred_end = t_orig_start + (boundaries[1] * duration)
            
            # Ensure start < end
            if t_pred_start >= t_pred_end:
                t_pred_end = t_pred_start + 0.1
                
            reranked.append({
                "chunk_id": chunk_id,
                "score": float(ms_score),
                "t_start": float(t_pred_start),
                "t_end": float(t_pred_end)
            })
            
        conn.close()
        return reranked
