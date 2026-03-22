# scripts/infer_vcmr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sqlite3
import open_clip
import cv2
import os
import sys
import argparse
import faiss
from typing import List, Tuple
from tqdm import tqdm

sys.path.insert(0, '.')

# ── Model Classes ────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)

class CrossModalTransformer(nn.Module):
    def __init__(self, visual_dim=512, query_dim=512, hidden_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.query_proj  = nn.Linear(query_dim,  hidden_dim)
        self.pos_embed   = nn.Parameter(torch.zeros(1, 8, hidden_dim))
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim*2,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.regressor   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()
        )
    def forward(self, visual_features, query_features):
        v = self.visual_proj(visual_features)
        q = self.query_proj(query_features).unsqueeze(1)
        v = v + self.pos_embed
        tokens = torch.cat([q, v], dim=1)
        transformed = self.transformer(tokens)
        q_out = transformed[:, 0, :]
        v_out = transformed[:, 1:, :]
        conf = torch.sigmoid(torch.sum(v_out * q_out.unsqueeze(1), dim=-1))
        return self.regressor(q_out), conf

# ── Setup Models ─────────────────────────────────────────────────
def setup_pipeline(args, device='cuda'):
    print(f'Loading models on {device}...')
    backbone, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
    backbone = backbone.to(device).eval()
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')

    ph_ckpt = torch.load(args.proj_heads, map_location=device)
    text_head = ProjectionHead().to(device)
    vision_head = ProjectionHead().to(device)
    text_head.load_state_dict(ph_ckpt['text_head'])
    vision_head.load_state_dict(ph_ckpt['vision_head'])
    text_head.eval()
    vision_head.eval()

    tg_ckpt = torch.load(args.transformer_ckpt, map_location=device)
    transformer = CrossModalTransformer().to(device)
    state_dict = tg_ckpt['model'] if isinstance(tg_ckpt, dict) and 'model' in tg_ckpt else tg_ckpt
    transformer.load_state_dict(state_dict)
    transformer.eval()

    print('Loading FAISS index...')
    index = faiss.read_index(args.faiss_index)
    conn = sqlite3.connect('fedvcmr.db')
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_id, video_id FROM chunks ORDER BY chunk_id")
    chunk_data = cursor.fetchall()
    chunk_ids = [row[0] for row in chunk_data]
    vid_map = {row[0]: row[1] for row in chunk_data}
    conn.close()

    cache = np.memmap('cache/frame_features.bin', dtype='float16', mode='r', shape=(len(chunk_ids), 8, 512))

    return backbone, tokenizer, text_head, vision_head, transformer, index, chunk_ids, vid_map, cache

# ── Inference Stages ─────────────────────────────────────────────
def run_inference(query, backbone, tokenizer, text_head, vision_head, transformer, index, chunk_ids, vid_map, cache, device='cuda', top_k_faiss=100, top_k_rerank=5):
    timing = {}
    
    # Stage 1: CLIP Encode Query
    t1 = time.time()
    with torch.no_grad():
        tokens = tokenizer([query]).to(device)
        t_raw = F.normalize(backbone.encode_text(tokens), dim=-1)
        q_emb = text_head(t_raw)
        q_emb_np = q_emb.cpu().numpy()
    timing['Stage 1 (CLIP)'] = (time.time() - t1) * 1000

    # Stage 2: FAISS Coarse Retrieval
    t2 = time.time()
    scores, indices = index.search(q_emb_np.astype(np.float32), top_k_faiss)
    candidate_cids = [chunk_ids[idx] for idx in indices[0] if idx != -1]
    timing['Stage 2 (FAISS)'] = (time.time() - t2) * 1000

    # Stage 3: MaxSim Reranking
    t3 = time.time()
    reranked = []
    chunk_idx_map = {cid: i for i, cid in enumerate(chunk_ids)}
    for cid in candidate_cids:
        c_idx = chunk_idx_map[cid]
        frames = cache[c_idx].astype(np.float32)
        sim_matrix = np.matmul(q_emb_np, frames.T)
        maxsim_score = np.mean(np.max(sim_matrix, axis=0))
        reranked.append((cid, maxsim_score))
    
    reranked.sort(key=lambda x: x[1], reverse=True)
    best_cid, _ = reranked[0]
    best_vid = vid_map[best_cid]
    timing['Stage 3 (MaxSim)'] = (time.time() - t3) * 1000

    # Stage 4: Transformer Grounding & Confidence
    t4 = time.time()
    c_idx = chunk_idx_map[best_cid]
    frames_t = torch.tensor(cache[c_idx].astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred, conf = transformer(frames_t, q_emb)
        pred_s, pred_e = pred[0, 0].item(), pred[0, 1].item()
        conf_scores = conf[0].cpu().numpy()
        
    timing['Stage 4 (Trans)'] = (time.time() - t4) * 1000
        
    return best_vid, pred_s, pred_e, np.mean(conf_scores), timing

# ── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, default='a person is cooking in the kitchen')
    parser.add_argument('--faiss_index', type=str, default='faiss_index.bin')
    parser.add_argument('--proj_heads', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--transformer_ckpt', type=str, default='checkpoints/temporal_grounding_best.pt')
    args = parser.parse_args()

    import time
    import psutil
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024 # MB
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    backbone, tokenizer, text_head, vision_head, transformer, index, chunk_ids, vid_map, cache = setup_pipeline(args, DEVICE)
    
    mem_after_load = process.memory_info().rss / 1024 / 1024 # MB
    
    print(f'\nRunning end-to-end inference for: "{args.query}"')
    
    # Warm-up + Multi-pass for profiling
    n_iters = 5
    all_timings = []
    
    for i in range(n_iters):
        start_time = time.time()
        vid_id, s_norm, e_norm, confidence, timing = run_inference(args.query, backbone, tokenizer, text_head, vision_head, transformer, index, chunk_ids, vid_map, cache, DEVICE)
        full_latency = (time.time() - start_time) * 1000
        
        if i > 0: # Skip first pass (warm-up)
            all_timings.append((full_latency, timing))
            
    mem_after_infer = process.memory_info().rss / 1024 / 1024 # MB
    gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if DEVICE == 'cuda' else 0

    # Calculate averages from iterations 2-5
    avg_full = sum(t[0] for t in all_timings) / len(all_timings)
    avg_stages = {}
    for stage in all_timings[0][1].keys():
        avg_stages[stage] = sum(t[1][stage] for t in all_timings) / len(all_timings)

    print(f'\n{"="*50}')
    print(f'INFERENCE RESULT (Steady State - Avg of {n_iters-1} runs)')
    print(f'{"="*50}')
    print(f'Video ID:   {vid_id}')
    print(f'Timestamp:  [{s_norm:.2%} — {e_norm:.2%}] of the segment')
    print(f'Confidence: {confidence:.1%}')
    print(f'Avg Latency:    {avg_full:.1f}ms')
    print(f'System RAM:     {mem_after_infer:.1f}MB (Delta: {mem_after_infer - mem_before:.1f}MB)')
    print(f'Peak GPU RAM:   {gpu_mem:.1f}MB')
    print(f'\nLatency Breakdown:')
    for stage, ms in avg_stages.items():
        print(f'  - {stage:<15}: {ms:>8.1f}ms')
    print(f'{"="*50}\n')
