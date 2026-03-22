import faiss
import numpy as np
import torch
import sqlite3
from typing import List, Tuple
from src.config import INDEX_PATH, DIMENSION, DB_PATH, DEVICE
from src.model import apply_projection
from src.adapter import apply_adapter

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
            features = np.load(cache_path).astype(np.float32) # (8, 512)

            # Apply projection if verified
            with torch.no_grad():
                tensor = torch.tensor(features, device=DEVICE)
                features = apply_projection(tensor) # returns tensor

                # Apply adapter (after projection)
                features = apply_adapter(features, client_id="client_1")

                features = features.cpu().numpy()

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
    MaxSim reranking using PyTorch for GPU acceleration if available.
    query_embs: (N_query, 512) - usually 3 expanded phrasings
    chunk_scores: List of (chunk_id, coarse_score)
    """
    reranked = []
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Convert query embeddings to tensor on device
    # query_embs might already be projected from query.py
    query_tensor = torch.tensor(query_embs, device=DEVICE, dtype=torch.float32)

    for chunk_id, coarse_score in chunk_scores:
        cursor.execute("SELECT cache_path FROM chunks WHERE chunk_id=?", (chunk_id,))
        row = cursor.fetchone()
        if not row:
            continue

        cache_path = row[0]
        # Load frames (CPU) -> Tensor (Device)
        frames_np = np.load(cache_path).astype(np.float32) # (8, 512)
        frames_tensor = torch.tensor(frames_np, device=DEVICE, dtype=torch.float32)

        # Apply Projection
        with torch.no_grad():
            frames_tensor = apply_projection(frames_tensor)

            # Apply Adapter
            frames_tensor = apply_adapter(frames_tensor, client_id="client_1")

        # MaxSim: (1/8) sum_i max_k cosine(q_k, v_i)
        # sim_matrix: (N_q, 8)
        sim_matrix = torch.matmul(query_tensor, frames_tensor.T)

        # For each frame, find its BEST matching query phrasing
        max_sim_per_frame, _ = torch.max(sim_matrix, dim=0) # (8,)
        maxsim_score = torch.mean(max_sim_per_frame).item() # scalar

        reranked.append((chunk_id, maxsim_score))
    
    conn.close()
    # Sort by maxsim score descending
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]
