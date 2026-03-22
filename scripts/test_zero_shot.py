# scripts/test_zero_shot.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open_clip
import cv2
import os
import sys
import argparse
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, '.')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# ── Setup ────────────────────────────────────────────────────────
def setup_models(args):
    print('Loading MobileCLIP-S1...')
    model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
    model = model.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')

    print('Loading Projection Heads...')
    ph_ckpt = torch.load(args.proj_heads, map_location=DEVICE)
    text_head = ProjectionHead().to(DEVICE)
    text_head.load_state_dict(ph_ckpt['text_head'])
    text_head.eval()

    print('Loading Transformer...')
    tg_ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    transformer = CrossModalTransformer().to(DEVICE)
    state_dict = tg_ckpt['model'] if isinstance(tg_ckpt, dict) and 'model' in tg_ckpt else tg_ckpt
    transformer.load_state_dict(state_dict)
    transformer.eval()

    return model, preprocess, tokenizer, text_head, transformer

def extract_features(video_path, model, preprocess):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, 8, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(preprocess(Image.fromarray(frame_rgb)))
    cap.release()
    
    if len(frames) < 8:
        raise ValueError("Could not extract 8 frames from video")
        
    frames_t = torch.stack(frames).to(DEVICE)
    with torch.no_grad():
        features = model.encode_image(frames_t)
        features /= features.norm(dim=-1, keepdim=True)
    return features.unsqueeze(0) # (1, 8, 512)

# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/temporal_grounding_best.pt')
    parser.add_argument('--proj_heads', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--out_dir', type=str, default='outputs/demos')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, preprocess, tokenizer, text_head, transformer = setup_models(args)

    # 1. Extract Visual Features
    print(f'Extracting live features from: {args.video}...')
    visual_features = extract_features(args.video, model, preprocess)

    # 2. Encode Query
    print(f'Encoding query: "{args.query}"...')
    with torch.no_grad():
        tokens = tokenizer([args.query]).to(DEVICE)
        t_raw = F.normalize(model.encode_text(tokens), dim=-1)
        q_emb = text_head(t_raw)

    # 3. Grounding
    print('Performing temporal grounding...')
    with torch.no_grad():
        pred, conf = transformer(visual_features, q_emb)
        pred_s, pred_e = pred[0, 0].item(), pred[0, 1].item()
        conf_scores = conf[0].cpu().numpy()

    # 4. Render Demo
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    p_start, p_end = pred_s * duration, pred_e * duration
    print(f"Predicted moment: {p_start:.1f}s - {p_end:.1f}s")

    out_path = os.path.join(args.out_dir, 'zero_shot_demo.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    target_w, target_h = 1280, 720
    out = cv2.VideoWriter(out_path, fourcc, fps, (target_w, target_h))

    for i in tqdm(range(total_frames), desc="Rendering"):
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (target_w, target_h))
        current_time = i / fps
        is_grounded = p_start <= current_time <= p_end
        
        # Confidence interpolation
        frame_idx = (current_time / duration) * 8
        idx_low = int(frame_idx)
        idx_high = min(7, idx_low + 1)
        alpha = frame_idx - idx_low
        current_conf = (1-alpha) * conf_scores[min(7, idx_low)] + alpha * conf_scores[idx_high]

        overlay = frame.copy()
        
        # UI
        box_color = (0, 255, 0) if is_grounded else (50, 50, 50)
        status_text = "GROUNDED" if is_grounded else "NOT GROUNDED"
        cv2.rectangle(overlay, (10, 10), (450, 60), (0,0,0), -1)
        cv2.rectangle(overlay, (10, 10), (450, 60), box_color, 2)
        cv2.putText(overlay, f"ZERO-SHOT: {status_text} ({current_conf:.1%})", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Confidence Bar
        cv2.rectangle(overlay, (target_w - 60, 100), (target_w - 20, 300), (0,0,0), -1)
        conf_h = int(current_conf * 200)
        cv2.rectangle(overlay, (target_w - 60, 300 - conf_h), (target_w - 20, 300), box_color, -1)
        cv2.putText(overlay, "CONF", (target_w - 70, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Query
        cv2.rectangle(overlay, (10, target_h - 100), (target_w - 10, target_h - 50), (0,0,0), -1)
        cv2.putText(overlay, f"Query: {args.query}", (20, target_h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Progress
        bar_y = target_h - 30
        bar_x_start, bar_x_end = 50, target_w - 50
        bar_w = bar_x_end - bar_x_start
        cv2.line(overlay, (bar_x_start, bar_y), (bar_x_end, bar_y), (100, 100, 100), 4)
        s_x = bar_x_start + int(pred_s * bar_w)
        e_x = bar_x_start + int(pred_e * bar_w)
        cv2.line(overlay, (s_x, bar_y), (e_x, bar_y), (0, 255, 0), 10)
        c_x = bar_x_start + int((current_time / duration) * bar_w)
        cv2.circle(overlay, (c_x, bar_y), 8, (255, 255, 255), -1)

        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Zero-shot demo saved: {out_path}")

if __name__ == "__main__":
    main()
