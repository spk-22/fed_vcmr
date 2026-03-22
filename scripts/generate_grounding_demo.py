# scripts/generate_grounding_demo.py
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
from tqdm import tqdm

sys.path.insert(0, '.')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Model Classes (Copied for standalone-ish use) ─────────────────
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
        
        # Calculate Confidence (Similarity between Query token and Visual tokens)
        q_out = transformed[:, 0, :] # (batch, 256)
        v_out = transformed[:, 1:, :] # (batch, 8, 256)
        # Dot product + Sigmoid for 0-1 confidence range
        conf = torch.sigmoid(torch.sum(v_out * q_out.unsqueeze(1), dim=-1)) # (batch, 8)
        
        return self.regressor(q_out), conf

def setup_models(args):
    print('Loading MobileCLIP-S1...')
    backbone, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
    backbone = backbone.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')

    print('Loading Projection Heads...')
    ph_ckpt = torch.load(args.proj_heads, map_location=DEVICE)
    text_head = ProjectionHead().to(DEVICE)
    vision_head = ProjectionHead().to(DEVICE)
    text_head.load_state_dict(ph_ckpt['text_head'])
    vision_head.load_state_dict(ph_ckpt['vision_head'])
    text_head.eval(); vision_head.eval()

    print('Loading Transformer...')
    tg_ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    transformer = CrossModalTransformer().to(DEVICE)
    state_dict = tg_ckpt['model'] if isinstance(tg_ckpt, dict) and 'model' in tg_ckpt else tg_ckpt
    transformer.load_state_dict(state_dict)
    transformer.eval()

    return backbone, tokenizer, text_head, vision_head, transformer

def get_best_video(args, q_emb, vision_head):
    conn = sqlite3.connect('fedvcmr.db')
    all_chunks = conn.execute('SELECT chunk_id, video_id FROM chunks ORDER BY chunk_id').fetchall()
    conn.close()

    idx_map = {cid: i for i, (cid, _) in enumerate(all_chunks)}
    vid_chunks = {}
    for cid, vid in all_chunks:
        vid_chunks.setdefault(vid, []).append(cid)

    cache = np.memmap('cache/frame_features.bin', dtype='float16', mode='r', shape=(len(all_chunks), 8, 512))
    w = np.array([0.5,0.75,1.0,1.25,1.25,1.0,0.75,0.5], dtype='float32')
    w = w / w.sum()

    print('Running retrieval...')
    best_score = -999
    best_vid = None

    for vid, chunks in tqdm(vid_chunks.items(), desc="Searching Videos"):
        ces = []
        for cid in chunks:
            frames = cache[idx_map[cid]].astype('float32')
            cemb = (frames * w[:, None]).sum(0)
            cemb = cemb / (np.linalg.norm(cemb) + 1e-8)
            ces.append(cemb)
        avg = np.stack(ces).mean(0)
        avg = avg / (np.linalg.norm(avg) + 1e-8)
        v_e = vision_head(torch.tensor(avg).unsqueeze(0).to(DEVICE))
        score = (q_emb @ v_e.T).item()
        if score > best_score:
            best_score = score
            best_vid = vid
    
    return best_vid, vid_chunks[best_vid], idx_map, cache

# ── Main Generator ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, default='a person is cooking in the kitchen')
    parser.add_argument('--video_id', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/temporal_grounding_best.pt')
    parser.add_argument('--proj_heads', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--out_dir', type=str, default='outputs/demos')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    backbone, tokenizer, text_head, vision_head, transformer = setup_models(args)

    # Encode Query
    with torch.no_grad():
        tokens = tokenizer([args.query]).to(DEVICE)
        t_raw = F.normalize(backbone.encode_text(tokens), dim=-1)
        q_emb = text_head(t_raw)

    # Get Best Video or Forced Video
    if args.video_id:
        conn = sqlite3.connect('fedvcmr.db')
        # Try MSRVTT first
        chunks = conn.execute('SELECT chunk_id FROM chunks WHERE video_id = ?', (args.video_id,)).fetchall()
        if chunks:
            idx_map = {cid[0]: i for i, cid in enumerate(conn.execute('SELECT chunk_id FROM chunks ORDER BY chunk_id').fetchall())}
            chunks = [c[0] for c in chunks]
            cache = np.memmap('cache/frame_features.bin', dtype='float16', mode='r', shape=(len(idx_map), 8, 512))
            load_mode = 'memmap'
        else:
            # Try ActivityNet
            anet = conn.execute('SELECT feature_path FROM anet_segments WHERE video_id = ?', (args.video_id,)).fetchall()
            if anet:
                chunks = [args.video_id] # Dummy chunk ID for anet
                cache = {args.video_id: np.load(anet[0][0])}
                load_mode = 'anet'
            else:
                print(f"Error: Video ID {args.video_id} not found in DB.")
                conn.close()
                return
        conn.close()
        best_vid = args.video_id
    else:
        best_vid, chunks, idx_map, cache = get_best_video(args, q_emb, vision_head)
        load_mode = 'memmap'

    # Grounding
    print(f'Grounding on {best_vid}...')
    chunk_preds = []
    with torch.no_grad():
        if load_mode == 'memmap':
            for cid in chunks:
                frames = torch.tensor(cache[idx_map[cid]].astype('float32')).unsqueeze(0).to(DEVICE)
                pred, conf = transformer(frames, q_emb)
                chunk_preds.append((cid, pred[0, 0].item(), pred[0, 1].item(), conf[0].cpu().numpy()))
        else:
            # ActivityNet mode (single feature for the vid/segment)
            frames = torch.tensor(cache[best_vid].astype('float32')).unsqueeze(0).to(DEVICE)
            pred, conf = transformer(frames, q_emb)
            chunk_preds.append((best_vid, pred[0, 0].item(), pred[0, 1].item(), conf[0].cpu().numpy()))

    # Pick best chunk
    chunk_preds.sort(key=lambda x: x[2] - x[1], reverse=True)
    best_cid, pred_s, pred_e, conf_scores = chunk_preds[0]

    # Find video file
    video_paths = [
        f'MSRVTT/videos/all/{best_vid}.mp4', 
        f'MSRVTT/MSRVTT/videos/all/{best_vid}.mp4', 
        f'videos/{best_vid}.mp4',
        f'ActivityNet/videos/{best_vid}.mp4'
    ]
    video_path = None
    for vp in video_paths:
        if os.path.exists(vp):
            video_path = vp
            break
    
    if not video_path:
        print(f"Error: Video {best_vid} not found.")
        return

    # Render Video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    out_path = os.path.join(args.out_dir, f'demo_conf_{best_vid}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    target_w, target_h = 1280, 720
    out = cv2.VideoWriter(out_path, fourcc, fps, (target_w, target_h))

    p_start, p_end = pred_s * duration, pred_e * duration
    print(f"Rendering enhanced demo to {out_path}...")
    
    for i in tqdm(range(total_frames), desc="Frames"):
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (target_w, target_h))
        current_time = i / fps
        is_grounded = p_start <= current_time <= p_end
        
        # Calculate current confidence
        frame_idx = (current_time / duration) * 8
        idx_low = int(frame_idx)
        idx_high = min(7, idx_low + 1)
        alpha = frame_idx - idx_low
        current_conf = (1-alpha) * conf_scores[min(7, idx_low)] + alpha * conf_scores[idx_high]

        overlay = frame.copy()
        
        # 1. Prediction status box
        box_color = (0, 255, 0) if is_grounded else (50, 50, 50)
        status_text = "GROUNDED" if is_grounded else "NOT GROUNDED"
        cv2.rectangle(overlay, (10, 10), (380, 60), (0,0,0), -1)
        cv2.rectangle(overlay, (10, 10), (380, 60), box_color, 2)
        cv2.putText(overlay, f"{status_text} ({current_conf:.1%})", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # 2. Confidence Bar
        cv2.rectangle(overlay, (target_w - 60, 100), (target_w - 20, 300), (0,0,0), -1)
        cv2.rectangle(overlay, (target_w - 60, 100), (target_w - 20, 300), (255, 255, 255), 1)
        conf_h = int(current_conf * 200)
        cv2.rectangle(overlay, (target_w - 60, 300 - conf_h), (target_w - 20, 300), box_color, -1)
        cv2.putText(overlay, "CONF", (target_w - 70, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 3. Query Label
        cv2.rectangle(overlay, (10, target_h - 100), (target_w - 10, target_h - 50), (0,0,0), -1)
        cv2.putText(overlay, f"Query: {args.query}", (20, target_h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # 4. Progress Bar
        bar_y = target_h - 30
        bar_x_start = 50
        bar_x_end = target_w - 50
        bar_w = bar_x_end - bar_x_start
        cv2.line(overlay, (bar_x_start, bar_y), (bar_x_end, bar_y), (100, 100, 100), 4)
        
        s_x = bar_x_start + int(pred_s * bar_w)
        e_x = bar_x_start + int(pred_e * bar_w)
        cv2.line(overlay, (s_x, bar_y), (e_x, bar_y), (0, 255, 0), 10)
        
        c_x = bar_x_start + int((current_time / duration) * bar_w)
        cv2.circle(overlay, (c_x, bar_y), 8, (255, 255, 255), -1)
        cv2.putText(overlay, f"{current_time:.1f}s / {duration:.1f}s", (target_w // 2 - 50, bar_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Demo saved to {out_path}")

if __name__ == "__main__":
    main()

