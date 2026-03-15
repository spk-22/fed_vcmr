# scripts/test_visual_grounding.py
# Give it a query → it finds the video → predicts timestamp → shows you the frames
import torch
import torch.nn.functional as F
import numpy as np
import sqlite3
import open_clip
import cv2
import os
import sys
import argparse
sys.path.insert(0, '.')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Args ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--query',      type=str,
                    default='a person is cooking in the kitchen')
parser.add_argument('--video_id',   type=str, default=None,
                    help='Force a specific video_id to test grounding')
parser.add_argument('--checkpoint', type=str,
                    default='checkpoints/temporal_grounding_best.pt')
parser.add_argument('--proj_heads', type=str,
                    default='checkpoints/best_model.pt')
parser.add_argument('--out_dir',    type=str,
                    default='outputs/visual_test')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── Models ────────────────────────────────────────────────────────
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.linear.weight)
    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)

class CrossModalTransformer(nn.Module):
    def __init__(self, visual_dim=512, query_dim=512,
                 hidden_dim=256, n_heads=4, n_layers=2):
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
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, visual_features, query_features):
        v           = self.visual_proj(visual_features)
        q           = self.query_proj(query_features).unsqueeze(1)
        v           = v + self.pos_embed
        tokens      = torch.cat([q, v], dim=1)
        transformed = self.transformer(tokens)
        return self.regressor(transformed[:, 0, :])

# ── Load backbone ─────────────────────────────────────────────────
print('Loading MobileCLIP-S1...')
backbone, _, _ = open_clip.create_model_and_transforms(
    'MobileCLIP-S1', pretrained='datacompdr'
)
backbone  = backbone.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')
for p in backbone.parameters():
    p.requires_grad = False

# ── Load projection heads ─────────────────────────────────────────
ph_ckpt     = torch.load(args.proj_heads, map_location=DEVICE)
text_head   = ProjectionHead().to(DEVICE)
vision_head = ProjectionHead().to(DEVICE)
text_head.load_state_dict(ph_ckpt['text_head'])
vision_head.load_state_dict(ph_ckpt['vision_head'])
text_head.eval()
vision_head.eval()
print('Projection heads loaded.')

# ── Load transformer ──────────────────────────────────────────────
tg_ckpt = torch.load(args.checkpoint, map_location=DEVICE)
transformer = CrossModalTransformer().to(DEVICE)

if isinstance(tg_ckpt, dict) and 'model' in tg_ckpt:
    transformer.load_state_dict(tg_ckpt['model'])
    epoch = tg_ckpt.get('epoch', 'unknown')
else:
    transformer.load_state_dict(tg_ckpt)
    epoch = 'unknown'

transformer.eval()
print(f'Transformer loaded (epoch {epoch}).')


# ── DB + cache ────────────────────────────────────────────────────
conn = sqlite3.connect('fedvcmr.db')
all_chunks = conn.execute(
    'SELECT chunk_id, video_id FROM chunks ORDER BY chunk_id'
).fetchall()
all_caps = conn.execute(
    'SELECT video_id, caption FROM captions'
).fetchall()
conn.close()

idx_map    = {cid: i for i, (cid, _) in enumerate(all_chunks)}
vid_chunks = {}
for cid, vid in all_chunks:
    vid_chunks.setdefault(vid, []).append(cid)

caps_map = {}
for vid, cap in all_caps:
    caps_map.setdefault(vid, []).append(cap)

cache = np.memmap('cache/frame_features.bin',
                  dtype='float16', mode='r',
                  shape=(len(all_chunks), 8, 512))

w = np.array([0.5,0.75,1.0,1.25,1.25,1.0,0.75,0.5], dtype='float32')
w = w / w.sum()

# ── Encode query ──────────────────────────────────────────────────
print(f'\nQuery: "{args.query}"')
with torch.no_grad():
    tokens  = tokenizer([args.query]).to(DEVICE)
    t_raw   = F.normalize(backbone.encode_text(tokens), dim=-1)
    q_emb   = text_head(t_raw)                              # (1, 512)

# ── Retrieval — find best matching video ──────────────────────────
if args.video_id:
    best_vid = args.video_id
    print(f'Using forced video_id: {best_vid}')
else:
    print('Running retrieval...')
    best_score = -999
    best_vid   = None

    for vid, chunks in vid_chunks.items():
        ces = []
        for cid in chunks:
            frames = cache[idx_map[cid]].astype('float32')
            cemb   = (frames * w[:, None]).sum(0)
            cemb   = cemb / (np.linalg.norm(cemb) + 1e-8)
            ces.append(cemb)
        avg  = np.stack(ces).mean(0)
        avg  = avg / (np.linalg.norm(avg) + 1e-8)
        v_e  = vision_head(
            torch.tensor(avg).unsqueeze(0).to(DEVICE)
        )
        score = (q_emb @ v_e.T).item()
        if score > best_score:
            best_score = score
            best_vid   = vid

    print(f'Best match: {best_vid}  (score={best_score:.4f})')

# ── Temporal grounding ────────────────────────────────────────────
print(f'\nRunning temporal grounding on {best_vid}...')

chunks      = vid_chunks.get(best_vid, [])
chunk_preds = []

with torch.no_grad():
    for cid in chunks:
        frames    = torch.tensor(
            cache[idx_map[cid]].astype('float32')
        ).unsqueeze(0).to(DEVICE)                           # (1, 8, 512)
        pred      = transformer(frames, q_emb)              # (1, 2)
        pred_s    = pred[0, 0].item()
        pred_e    = max(pred[0, 0].item() + 0.01,
                        pred[0, 1].item())
        chunk_preds.append((cid, pred_s, pred_e))

# Pick chunk with most confident prediction (widest span = most specific)
chunk_preds.sort(key=lambda x: x[2] - x[1], reverse=True)
best_cid, pred_s, pred_e = chunk_preds[0]

print(f'Predicted segment (normalized): [{pred_s:.3f}, {pred_e:.3f}]')

# ── Load actual video and extract frames ──────────────────────────
# Try common video locations
video_paths = [
    f'MSRVTT/videos/all/{best_vid}.mp4',
    f'MSRVTT/MSRVTT/videos/all/{best_vid}.mp4',
    f'videos/{best_vid}.mp4',
]
video_path = None
for vp in video_paths:
    if os.path.exists(vp):
        video_path = vp
        break

if video_path is None:
    print(f'\nVideo file not found. Checked:')
    for vp in video_paths:
        print(f'  {vp}')
    print('\nShowing feature-level visualization instead.')

    # Feature visualization — show frame weights as heatmap
    frames_arr = cache[idx_map[best_cid]].astype('float32')  # (8, 512)
    with torch.no_grad():
        frames_t = torch.tensor(frames_arr).unsqueeze(0).to(DEVICE)
        # Use transformer's internal attention or logits
        # Here we just show which frames the model "thinks" are start/end
        # But a better way is to show the normalized pred_s, pred_e coverage.
        
        # We'll just print the bar chart as the user suggested
        print(f'\nPredicted moment: {pred_s:.2f} to {pred_e:.2f}')
        bar_chars = 40
        for i in range(8):
            center = (i + 0.5) / 8.0
            in_window = pred_s <= center <= pred_e
            bar = '█' * (bar_chars if in_window else 2)
            mark = ' [PREDICTED]' if in_window else ''
            print(f'  Frame {i+1}: {bar:<40} {mark}')

else:
    # ── Extract actual video frames ───────────────────────────────
    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps

    pred_start_sec = pred_s * duration
    pred_end_sec   = pred_e * duration

    print(f'Video duration: {duration:.1f}s')
    print(f'Predicted time: {pred_start_sec:.1f}s — {pred_end_sec:.1f}s')

    # Extract 6 frames: 3 from predicted window, 3 outside
    def get_frame(cap, t_sec):
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
        ret, frame = cap.read()
        return frame if ret else None

    # Sample frames across predicted window
    pred_times    = np.linspace(pred_start_sec, pred_end_sec, 4)
    outside_times = [max(0, pred_start_sec - 2),
                     min(duration - 0.1, pred_end_sec + 2)]
    all_times     = list(outside_times[:1]) + list(pred_times) + list(outside_times[1:])

    frames_out = []
    for t in all_times:
        f = get_frame(cap, t)
        if f is not None:
            f = cv2.resize(f, (320, 180))
            # Label frame
            in_window = pred_start_sec <= t <= pred_end_sec
            color     = (0, 255, 0) if in_window else (0, 0, 255)
            label     = f't={t:.1f}s [PREDICTED]' if in_window else f't={t:.1f}s'
            cv2.rectangle(f, (0, 0), (320, 30), color, -1)
            cv2.putText(f, label, (5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)
            frames_out.append(f)
    cap.release()

    if frames_out:
        # Stack into grid: top row = outside window, bottom = predicted
        row1 = np.hstack(frames_out[:2] + [np.zeros((180, 320, 3),
                                             dtype=np.uint8)])
        row2 = np.hstack(frames_out[2:5])
        row3_f = frames_out[5] if len(frames_out) > 5 else \
                 np.zeros((180, 320, 3), dtype=np.uint8)
        row3 = np.hstack([row3_f,
                          np.zeros((180, 640, 3), dtype=np.uint8)])

        grid = np.vstack([row1, row2, row3])

        # Add title
        title = np.zeros((60, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(title,
                    f'Query: {args.query[:70]}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1)
        cv2.putText(title,
                    f'Video: {best_vid} | '
                    f'Predicted: {pred_start_sec:.1f}s-{pred_end_sec:.1f}s '
                    f'/ {duration:.1f}s total',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)
        out = np.vstack([title, grid])

        out_path = os.path.join(args.out_dir,
                                f'{best_vid}_{args.query[:30]}.jpg')
        out_path = out_path.replace(' ', '_')
        cv2.imwrite(out_path, out)
        print(f'\nSaved visual result to: {out_path}')

# ── Summary ───────────────────────────────────────────────────────
print(f'\n{"="*55}')
print(f'VISUAL GROUNDING TEST SUMMARY')
print(f'{"="*55}')
print(f'Query:           {args.query}')
print(f'Retrieved video: {best_vid}')
print(f'Predicted:       [{pred_s:.3f}, {pred_e:.3f}] (normalized)')
if video_path:
    print(f'                 [{pred_start_sec:.1f}s, {pred_end_sec:.1f}s] '
          f'of {duration:.1f}s')
print(f'Known captions:')
for cap in caps_map.get(best_vid, [])[:3]:
    print(f'  - {cap}')
print(f'{"="*55}')
