# federated_learning/audit_dap_hw.py
"""
DAPHW Personalization Audit.
Compares Global FL weights vs. Local Fine-tuned weights on non-IID shards.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sqlite3
import numpy as np
import os
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Model Definition ────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.linear.weight)
    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)

def info_nce_loss(v, t, temp=0.07):
    sims = (v @ t.T) / temp
    labels = torch.arange(v.size(0)).to(v.device)
    return (F.cross_entropy(sims, labels) + F.cross_entropy(sims.T, labels)) / 2

# ── Evaluator ───────────────────────────────────────────────────
def evaluate_recall(text_head, vision_head, vids, text_features, frame_cache, idx_map, vid_chunks):
    text_head.eval()
    vision_head.eval()
    
    # 1. Compute all embeddings for the shard
    t_embs = []
    v_embs = []
    for vid in vids:
        # Text
        t_raw = text_features.get(vid, torch.zeros(512)).to(DEVICE)
        t_embs.append(text_head(t_raw.unsqueeze(0)))
        # Vision
        chunks = vid_chunks.get(vid, [])
        if chunks:
            frames = frame_cache[idx_map[chunks[0]]].astype('float32')
            frames /= (np.linalg.norm(frames, axis=1, keepdims=True) + 1e-8)
            v_feat = torch.from_numpy(frames.mean(0)).to(DEVICE)
        else:
            v_feat = torch.zeros(512, device=DEVICE)
        v_embs.append(vision_head(v_feat.unsqueeze(0)))
    
    T = torch.cat(t_embs) # (N, 512)
    V = torch.cat(v_embs) # (N, 512)
    
    # 2. Compute Sim matrix
    sims = T @ V.T # (N, N)
    preds = torch.argmax(sims, dim=1)
    correct = (preds == torch.arange(len(vids)).to(DEVICE)).sum().item()
    return (correct / len(vids)) * 100

# ── Main Audit ──────────────────────────────────────────────────
def main():
    print(f"Starting DAPHW Audit on {DEVICE}...")
    
    # 1. Load Data
    text_features = torch.load('cache/fl_text_features.pt', map_location='cpu')
    conn = sqlite3.connect('fedvcmr.db')
    n_total = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
    all_chunks = conn.execute('SELECT chunk_id, video_id FROM chunks ORDER BY chunk_id').fetchall()
    frame_cache = np.memmap('cache/frame_features.bin', dtype='float16', mode='r', shape=(n_total, 8, 512))
    idx_map = {cid: i for i, (cid, _) in enumerate(all_chunks)}
    vid_chunks = {}
    for cid, vid in all_chunks:
        vid_chunks.setdefault(vid, []).append(cid)
    
    # Focus on Client 0 (Gaming)
    client_id = 0
    vids = [r[0] for r in conn.execute('SELECT video_id FROM fl_shards WHERE client_id = ?', (client_id,)).fetchall()]
    conn.close()
    print(f"Target: Client {client_id} (Gaming) with {len(vids)} videos.")

    # 2. Load Global Weights
    t_head = ProjectionHead().to(DEVICE)
    v_head = ProjectionHead().to(DEVICE)
    checkpoint = torch.load('checkpoints/fl_global_model.pt', map_location=DEVICE)
    t_head.load_state_dict(checkpoint['text_head'])
    v_head.load_state_dict(checkpoint['vision_head'])
    
    r1_global = evaluate_recall(t_head, v_head, vids, text_features, frame_cache, idx_map, vid_chunks)
    print(f"\n[GLOBAL MODEL] R@1 on Shard {client_id}: {r1_global:.2f}%")

    # 3. Local Fine-Tuning (Deep Personalization)
    print("\nSimulating Deep Local Personalization (15 Epochs)...")
    opt = torch.optim.AdamW(list(t_head.parameters()) + list(v_head.parameters()), lr=2e-5)
    t_head.train(); v_head.train()
    
    for epoch in range(15):
        total_loss = 0
        indices = np.random.permutation(len(vids))
        for i in range(0, len(vids), 64):
            batch_indices = indices[i:i+64]
            batch_vids = [vids[idx] for idx in batch_indices]
            
            t_raw = torch.stack([text_features.get(v, torch.zeros(512)) for v in batch_vids]).to(DEVICE)
            v_list = []
            for v in batch_vids:
                chunks = vid_chunks.get(v, [])
                frames = frame_cache[idx_map[chunks[0]]].astype('float32') if chunks else np.zeros((8, 512))
                frames /= (np.linalg.norm(frames, axis=1, keepdims=True) + 1e-8)
                v_list.append(torch.from_numpy(frames.mean(0)).to(DEVICE))
            
            t_emb = t_head(t_raw)
            v_emb = v_head(torch.stack(v_list))
            loss = info_nce_loss(v_emb, t_emb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1} / 15 Loss: {total_loss/(len(vids)/64):.4f}")

    r1_local = evaluate_recall(t_head, v_head, vids, text_features, frame_cache, idx_map, vid_chunks)
    print(f"\n[LOCAL MODEL] R@1 on Shard {client_id}: {r1_local:.2f}%")
    print(f"Personalization Gain: {r1_local - r1_global:+.2f}%")

    # 4. Save results
    with open('federated_learning/audit_results.txt', 'w') as f:
        f.write(f"Global R@1: {r1_global:.2f}%\n")
        f.write(f"Local R@1: {r1_local:.2f}%\n")
        f.write(f"Gain: {r1_local - r1_global:+.2f}%\n")

if __name__ == "__main__":
    main()
