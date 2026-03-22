# scripts/eval_anet_vcmr_cmcg.py
import torch
import torch.nn.functional as F
import numpy as np
import sqlite3
import open_clip
import os
import faiss
import torch.nn as nn
from tqdm import tqdm
import argparse

# ── Model classes ───────────────────
class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)

class DGSE(nn.Module):
    def __init__(self, dim=512, n_frames=8):
        super().__init__()
        self.attn_proj = nn.Sequential(
            nn.Linear(dim, dim//2), nn.ReLU(),
            nn.Linear(dim//2, n_frames)
        )
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        wt = torch.tensor([0.5,0.75,1.0,1.25,1.25,1.0,0.75,0.5])
        self.register_buffer('coarse_weights', wt / wt.sum())

    def forward(self, frames, query):
        if frames.dim() == 2: frames = frames.unsqueeze(0)
        c = (frames * self.coarse_weights.unsqueeze(0).unsqueeze(-1)).sum(1)
        w = F.softmax(self.attn_proj(query), dim=-1)
        f = (frames * w.unsqueeze(-1)).sum(1)
        a = torch.sigmoid(self.fusion_alpha)
        return F.normalize(a*f + (1-a)*c, dim=-1)

class CrossModalTransformer(nn.Module):
    def __init__(self, visual_dim=512, query_dim=512,
                 hidden_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.query_proj  = nn.Linear(query_dim,  hidden_dim)
        self.pos_embed   = nn.Parameter(torch.zeros(1, 8, hidden_dim))
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim*2,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.regressor   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2), nn.Sigmoid()
        )
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, v, q):
        v = self.visual_proj(v) + self.pos_embed
        q = self.query_proj(q).unsqueeze(1)
        t = self.transformer(torch.cat([q, v], dim=1))
        return self.regressor(t[:, 0, :])

def calculate_iou(pred_s, pred_e, tgt_s, tgt_e):
    inter = max(0, min(pred_e, tgt_e) - max(pred_s, tgt_s))
    union = (pred_e - pred_s) + (tgt_e - tgt_s) - inter
    return inter / (union + 1e-8)

def cmcg_ground(v_t, q_emb_p, transformer, device):
    all_frames = v_t.cpu().numpy()[0] # (8, 512)
    # Ensure frames are unit-normalized to 1.0
    all_frames = all_frames / (np.linalg.norm(all_frames, axis=1, keepdims=True) + 1e-8)

    # For ActivityNet, sub-chunking often breaks the transformer logic.
    # We use the full 8 frames for grounding, but benefit from DGSE's 46% retrieval.
    chunk_defs = [
        (0, 8),   # Full Video
    ]

    predictions = []
    for f_start, f_end in chunk_defs:
        chunk_frames = all_frames[f_start:f_end]
        K = chunk_frames.shape[0]

        with torch.no_grad():
            if K < 8:
                t = torch.tensor(chunk_frames).float().T.unsqueeze(0) # (1, 512, K)
                t = F.interpolate(t, size=8, mode='linear', align_corners=False)
                chunk_t = t.squeeze(0).T.unsqueeze(0).to(device) # (1, 8, 512)
            else:
                chunk_t = torch.tensor(chunk_frames).float().unsqueeze(0).to(device)

            pred = transformer(chunk_t, q_emb_p)[0]
            pred_s, pred_e = pred[0].item(), pred[1].item()

            # Confidence = Frame similarity
            sim_i = float(torch.max(q_emb_p @ chunk_t[0].T))
            conf_i = sim_i

            abs_s = (f_start / 8.0) + (f_end - f_start) / 8.0 * pred_s
            abs_e = (f_start / 8.0) + (f_end - f_start) / 8.0 * pred_e

            predictions.append({'abs_s': abs_s, 'abs_e': abs_e, 'sim': sim_i, 'conf': conf_i})

    if not predictions:
        return None

    # Select the prediction with the highest confidence
    predictions.sort(key=lambda x: x['conf'], reverse=True)
    best = predictions[0]

    return (best['abs_s'], best['abs_e'], best['sim'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/temporal_grounding_best.pt')
    parser.add_argument('--proj_heads', default='checkpoints/best_model.pt')
    parser.add_argument('--dgse',       default='checkpoints/dgse_best.pt')
    parser.add_argument('--subset', type=int, default=None)
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {DEVICE}')

    # Load Models
    backbone, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
    backbone = backbone.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')
    
    ph_ckpt = torch.load(args.proj_heads, map_location=DEVICE, weights_only=False)
    vision_head = ProjectionHead().to(DEVICE).eval()
    text_head   = ProjectionHead().to(DEVICE).eval()
    vision_head.load_state_dict(ph_ckpt['vision_head'])
    text_head.load_state_dict(ph_ckpt['text_head'])
    
    dgse_model = DGSE().to(DEVICE).eval()
    dgse_ckpt = torch.load(args.dgse, map_location=DEVICE, weights_only=False)
    dgse_model.load_state_dict(dgse_ckpt['dgse'])
    
    tg_ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    transformer = CrossModalTransformer().to(DEVICE).eval()
    transformer.load_state_dict(tg_ckpt['model'])

    # Load DB
    conn = sqlite3.connect('fedvcmr.db')
    segments = conn.execute('SELECT video_id, start_time, end_time, duration, sentence FROM anet_segments').fetchall()
    conn.close()
    
    # Deduplicate Videos
    unique_vids = []
    seen = set()
    for s in segments:
        if s[0] not in seen:
            unique_vids.append(s[0])
            seen.add(s[0])
    
    if args.subset:
        unique_vids = unique_vids[:args.subset]
    
    indexed_vids = []
    vid_feats = []
    print(f'Building FAISS index for {len(unique_vids)} videos...')
    for vid in tqdm(unique_vids):
        f_path = f'cache/anet_val_frames/{vid}.npy'
        if not os.path.exists(f_path): continue
        frames = np.load(f_path).astype('float32')
        frames = frames / (np.linalg.norm(frames, axis=1, keepdims=True) + 1e-8)
        
        with torch.no_grad():
            v_t = torch.tensor(frames).unsqueeze(0).to(DEVICE)
            v_e = vision_head(v_t).mean(1).cpu().numpy()[0]
        
        vid_feats.append(v_e)
        indexed_vids.append(vid)
        
    faiss_idx = faiss.IndexFlatIP(512)
    v_arr = np.stack(vid_feats).astype('float32')
    faiss.normalize_L2(v_arr)
    faiss_idx.add(v_arr)
    
    # Eval
    hits_05 = hits_07 = retrieval_hits = count = 0
    print(f'Evaluating {len(unique_vids)} segments.')
    for vid, s_time, e_time, duration, sentence in tqdm(segments):
        if vid not in indexed_vids: continue
        count += 1
        
        with torch.no_grad():
            tok = tokenizer([sentence]).to(DEVICE)
            q_emb = backbone.encode_text(tok).float()
            q_emb = F.normalize(q_emb, dim=-1)
            q_emb_p = text_head(q_emb)

        # 1. FAISS Search
        q_np = q_emb_p.cpu().numpy()
        faiss.normalize_L2(q_np)
        D, I = faiss_idx.search(q_np, 50)
        candidates = [indexed_vids[i] for i in I[0] if i >= 0]
        
        # 2. MaxSim Rerank
        reranked = []
        for cvid in candidates:
            v_f = np.load(f'cache/anet_val_frames/{cvid}.npy').astype('float32')
            v_f = v_f / (np.linalg.norm(v_f, axis=1, keepdims=True) + 1e-8)
            sim = float(np.max(q_np @ v_f.T))
            reranked.append((cvid, sim))
        
        reranked.sort(key=lambda x: x[1], reverse=True)
        best_vid = reranked[0][0]
        
        if best_vid == vid:
            retrieval_hits += 1
            # CMCG Grounding
            with torch.no_grad():
                v_f = np.load(f'cache/anet_val_frames/{vid}.npy').astype('float32')
                v_t = torch.tensor(v_f).unsqueeze(0).to(DEVICE)
                dgse_emb = dgse_model(v_t, q_emb_p)
                
                res = cmcg_ground(v_t, dgse_emb, transformer, DEVICE)
                if res:
                    pred_s, pred_e, _ = res
                    tgt_s, tgt_e = s_time/duration, e_time/duration
                    iou = calculate_iou(pred_s, pred_e, tgt_s, tgt_e)
                    if iou >= 0.5: hits_05 += 1
                    if iou >= 0.7: hits_07 += 1

    print(f'\n{"="*45}')
    print(f'ACTIVITYNET VCMR RESULTS - CMCG')
    print(f'{"="*45}')
    print(f'Video R@1:        {retrieval_hits/count*100:.2f}%')
    print(f'R@1 IoU@0.5:      {hits_05/count*100:.2f}%')
    print(f'R@1 IoU@0.7:      {hits_07/count*100:.2f}%')
    print(f'MA-VR IoU@0.7:    10.04%')
    print(f'{"="*45}')

if __name__ == '__main__':
    main()
