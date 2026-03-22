"""
Charades-STA VCMR Evaluation Script for M17
Evaluates the full pipeline: CLIP -> FAISS -> MaxSim -> Transformer
on the Charades-STA test set.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sqlite3
import open_clip
import os
import sys
import argparse
import faiss
from tqdm import tqdm

sys.path.insert(0, '.')

# ── Model Classes ────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
    def forward(self, x): return F.normalize(self.linear(x), dim=-1)

class CrossModalTransformer(nn.Module):
    def __init__(self, visual_dim=512, query_dim=512, hidden_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.query_proj  = nn.Linear(query_dim,  hidden_dim)
        self.pos_embed   = nn.Parameter(torch.zeros(1, 8, hidden_dim))
        encoder_layer    = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.regressor   = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2), nn.Sigmoid())
    def forward(self, v, q):
        v = self.visual_proj(v) + self.pos_embed
        q = self.query_proj(q).unsqueeze(1)
        tokens = torch.cat([q, v], dim=1)
        transformed = self.transformer(tokens)
        return self.regressor(transformed[:, 0, :])

# ── Metrics ──────────────────────────────────────────────────────
def calculate_iou(pred, target):
    p_s, p_e = pred
    t_s, t_e = target
    inter_s = max(p_s, t_s)
    inter_e = min(p_e, t_e)
    inter = max(0, inter_e - inter_s)
    union = (p_e - p_s) + (t_e - t_s) - inter
    return inter / (union + 1e-8)

# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/temporal_grounding_best.pt')
    parser.add_argument('--proj_heads', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--top_k_faiss', type=int, default=50)
    parser.add_argument('--no_proj', action='store_true')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load Models
    print(f'Loading models (No Projection: {args.no_proj})...')
    backbone, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
    backbone = backbone.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')
    
    if not args.no_proj:
        ph_ckpt = torch.load(args.proj_heads, map_location=DEVICE)
        text_head = ProjectionHead().to(DEVICE).eval()
        text_head.load_state_dict(ph_ckpt['text_head'])
        vision_head = ProjectionHead().to(DEVICE).eval()
        vision_head.load_state_dict(ph_ckpt['vision_head'])

    tg_ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    transformer = CrossModalTransformer().to(DEVICE).eval()
    state_dict = tg_ckpt['model'] if isinstance(tg_ckpt, dict) and 'model' in tg_ckpt else tg_ckpt
    transformer.load_state_dict(state_dict)

    # 2. Load Data from DB
    conn = sqlite3.connect('fedvcmr.db')
    segments = conn.execute('SELECT video_id, sentence, start_time, end_time, duration, feature_path FROM charades_sta_segments').fetchall()
    conn.close()
    
    if not segments:
        print("Error: No Charades-STA segments found in DB.")
        return

    # 3. Build Video-level FAISS Index
    print('Building FAISS index for Charades-STA subset...')
    unique_vids = sorted(list(set([s[0] for s in segments])))
    
    vid_feats = []
    indexed_vids = []
    w = np.array([0.5, 0.75, 1.0, 1.25, 1.25, 1.0, 0.75, 0.5])
    
    for vid in tqdm(unique_vids, desc="Indexing Videos"):
        feat_path = f"cache/charades_frames/{vid}.npy"
        if not os.path.exists(feat_path):
            continue
        frames = np.load(feat_path).astype(np.float32)
        v_e = np.sum(frames * w[:, None], axis=0) / np.sum(w)
        v_e /= (np.linalg.norm(v_e) + 1e-10)
        
        if not args.no_proj:
            v_e_t = vision_head(torch.tensor(v_e).float().unsqueeze(0).to(DEVICE)).cpu().detach().numpy()[0]
        else:
            v_e_t = v_e
            
        vid_feats.append(v_e_t)
        indexed_vids.append(vid)
    
    print(f"  Indexed {len(indexed_vids)} videos")
    faiss_idx = faiss.IndexFlatIP(512)
    faiss_idx.add(np.stack(vid_feats).astype(np.float32))

    # 4. Deduplicate — one sample per video for fair retrieval eval
    seen = set()
    deduped = []
    for s in segments:
        vid = s[0]
        if vid not in seen:
            seen.add(vid)
            deduped.append(s)
    print(f'After dedup: {len(deduped)} unique videos (was {len(segments)} segments)')
    segments = deduped

    # 5. Evaluation Loop
    print(f'Starting VCMR evaluation on {len(segments)} samples (pool={len(indexed_vids)})...')
    hits_05 = 0
    hits_07 = 0
    retrieval_hits = 0
    total_evaluated = 0
    
    frame_cache = {vid: np.load(f"cache/charades_frames/{vid}.npy").astype(np.float32) for vid in indexed_vids}

    for vid, sentence, s_time, e_time, duration, _ in tqdm(segments, desc="Evaluating"):
        if vid not in frame_cache:
            continue
        total_evaluated += 1

        with torch.no_grad():
            tok = tokenizer([sentence]).to(DEVICE)
            q_emb = F.normalize(backbone.encode_text(tok).float(), dim=-1)
            
            if not args.no_proj:
                q_emb = text_head(q_emb)
            
            q_emb_np = q_emb.cpu().numpy()

        scores, indices = faiss_idx.search(q_emb_np.astype(np.float32), args.top_k_faiss)
        candidate_vids = [indexed_vids[idx] for idx in indices[0] if idx != -1]

        reranked = []
        for cvid in candidate_vids:
            frames = frame_cache[cvid]
            sim_matrix = np.matmul(q_emb_np, frames.T)
            score = np.mean(np.max(sim_matrix, axis=0))
            reranked.append((cvid, score))
        reranked.sort(key=lambda x: x[1], reverse=True)
        best_vid = reranked[0][0]

        if best_vid == vid:
            retrieval_hits += 1
            frames_t = torch.tensor(frame_cache[best_vid]).float().unsqueeze(0).to(DEVICE)
            target_norm = [s_time / duration, e_time / duration]
            
            with torch.no_grad():
                pred_norm = transformer(frames_t, q_emb)[0].cpu().numpy()
            
            iou = calculate_iou(pred_norm, target_norm)
            if iou >= 0.5: hits_05 += 1
            if iou >= 0.7: hits_07 += 1

    r1_iou05 = hits_05 / total_evaluated if total_evaluated > 0 else 0
    r1_iou07 = hits_07 / total_evaluated if total_evaluated > 0 else 0
    vid_r1 = retrieval_hits / total_evaluated if total_evaluated > 0 else 0
    
    print(f'\n{"="*45}')
    print(f'CHARADES-STA VCMR RESULTS (Subset={len(indexed_vids)})')
    print(f'{"="*45}')
    print(f'Total Evaluated:       {total_evaluated}')
    print(f'Video R@1 (Retrieval): {vid_r1*100:.2f}%')
    print(f'R@1 IoU@0.5:           {r1_iou05*100:.2f}%')
    print(f'R@1 IoU@0.7:           {r1_iou07*100:.2f}%')
    print(f'{"="*45}\n')

if __name__ == "__main__":
    main()
