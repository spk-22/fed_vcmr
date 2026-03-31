# federated_learning/audit_dap_hw.py
"""
DAPHW Personalization Audit.
Compares Global FL weights vs. Local Fine-tuned weights on non-IID shards across all clients.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sqlite3
import numpy as np
import os
from tqdm import tqdm
import copy

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
    
    t_embs = []
    v_embs = []
    for vid in vids:
        # Text
        t_raw = text_features.get(vid, torch.zeros(512)).to(DEVICE)
        t_embs.append(text_head(t_raw.unsqueeze(0)))
        
        # Vision (Max-Pool as established in optimized strategy)
        chunks = vid_chunks.get(vid, [])
        if chunks:
            frames = frame_cache[idx_map[chunks[0]]].astype('float32')
            frames /= (np.linalg.norm(frames, axis=1, keepdims=True) + 1e-8)
            v_feat = torch.from_numpy(frames).max(dim=0)[0].to(DEVICE)
        else:
            v_feat = torch.zeros(512, device=DEVICE)
        v_embs.append(vision_head(v_feat.unsqueeze(0)))
    
    T = torch.cat(t_embs)
    V = torch.cat(v_embs)
    
    sims = T @ V.T
    preds = torch.argmax(sims, dim=1)
    correct = (preds == torch.arange(len(vids)).to(DEVICE)).sum().item()
    return (correct / len(vids)) * 100

# ── Per-Client Personalization ──────────────────────────────────
def run_client_audit(client_id, seed, text_features, frame_cache, idx_map, vid_chunks, conn):
    vids = [r[0] for r in conn.execute('SELECT video_id FROM fl_shards WHERE client_id = ?', (client_id,)).fetchall()]
    
    # Train/Test Split
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(vids))
    split = int(0.8 * len(vids))
    train_vids = [vids[i] for i in shuffled[:split]]
    test_vids  = [vids[i] for i in shuffled[split:]]

    # Load Global Weights
    t_head = ProjectionHead().to(DEVICE)
    v_head = ProjectionHead().to(DEVICE)
    checkpoint = torch.load('checkpoints/fl_global_model.pt', map_location=DEVICE, weights_only=True)
    t_head.load_state_dict(checkpoint['text_head'])
    v_head.load_state_dict(checkpoint['vision_head'])

    # Global Baseline
    r1_global = evaluate_recall(t_head, v_head, test_vids, text_features, frame_cache, idx_map, vid_chunks)

    # Local Fine-Tuning: Vision-Centric Quick-Adapt (3 Epochs, lr=5e-5)
    v_head_global = copy.deepcopy(v_head)
    for p in t_head.parameters(): p.requires_grad = False
    
    opt = torch.optim.AdamW(v_head.parameters(), lr=5e-5, weight_decay=0.01)
    t_head.eval(); v_head.train()
    mu = 0.01

    for _ in range(3):
        indices = np.random.permutation(len(train_vids))
        for i in range(0, len(train_vids), 64):
            batch_indices = indices[i:i+64]
            batch_vids = [train_vids[idx] for idx in batch_indices]
            
            t_raw = torch.stack([text_features.get(v, torch.zeros(512)) for v in batch_vids]).to(DEVICE)
            v_list = []
            for v in batch_vids:
                chunks = vid_chunks.get(v, [])
                if chunks:
                    frames = frame_cache[idx_map[chunks[0]]].astype('float32')
                    frames /= (np.linalg.norm(frames, axis=1, keepdims=True) + 1e-8)
                    v_feat = torch.from_numpy(frames).max(dim=0)[0].to(DEVICE)
                else:
                    v_feat = torch.zeros(512, device=DEVICE)
                v_list.append(v_feat)
            
            with torch.no_grad(): t_emb = t_head(t_raw)
            v_emb = v_head(torch.stack(v_list))
            
            loss = info_nce_loss(v_emb, t_emb)
            prox_loss = sum(((p - p_g)**2).sum() for p, p_g in zip(v_head.parameters(), v_head_global.parameters()))
            total_batch_loss = loss + mu * prox_loss
            
            opt.zero_grad(); total_batch_loss.backward(); opt.step()

    # Local Evaluation
    r1_local = evaluate_recall(t_head, v_head, test_vids, text_features, frame_cache, idx_map, vid_chunks)
    return r1_global, r1_local

# ── Main Audit ──────────────────────────────────────────────────
def main():
    print(f"Starting Multi-Client DAPHW Audit on {DEVICE}...")
    SEEDS = [42, 7, 123]
    CLIENTS = {0: "Gaming", 1: "Sports", 2: "Cooking", 3: "News"}
    
    text_features = torch.load('cache/fl_text_features.pt', map_location='cpu', weights_only=True)
    conn = sqlite3.connect('fedvcmr.db')
    n_total = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
    all_chunks = conn.execute('SELECT chunk_id, video_id FROM chunks ORDER BY chunk_id').fetchall()
    frame_cache = np.memmap('cache/frame_features.bin', dtype='float16', mode='r', shape=(n_total, 8, 512))
    idx_map = {cid: i for i, (cid, _) in enumerate(all_chunks)}
    vid_chunks = {}
    for cid, vid in all_chunks:
        vid_chunks.setdefault(vid, []).append(cid)
    
    results = {}
    for cid, cname in CLIENTS.items():
        print(f"Auditing Client {cid} ({cname})...")
        client_seeds = []
        for seed in SEEDS:
            g, l = run_client_audit(cid, seed, text_features, frame_cache, idx_map, vid_chunks, conn)
            client_seeds.append(l - g)
        results[cname] = (np.mean(client_seeds), np.std(client_seeds))
        print(f"  Gain: {results[cname][0]:.2f}% ± {results[cname][1]:.2f}%")

    # Save and Print Summary
    with open('federated_learning/audit_results.txt', 'w') as f:
        f.write("DAPHW Cross-Client Personalization Audit (3 Seeds)\n")
        f.write("-" * 50 + "\n")
        for cname, (mean, std) in results.items():
            line = f"{cname:10} | Mean Gain: {mean:+.2f}% | Std: {std:.2f}%\n"
            f.write(line)
            print(line, end="")
        
        avg_gain = np.mean([m for m, s in results.values()])
        f.write("-" * 50 + "\n")
        f.write(f"OVERALL MEAN GAIN: {avg_gain:+.2f}%\n")
        print(f"\nOverall Mean Gain: {avg_gain:+.2f}%")

    conn.close()

if __name__ == "__main__":
    main()
