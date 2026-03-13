# scripts/eval_m10.py
# Auto-detects checkpoint architecture: identity_linear_512 or mlp_residual_512_256
import argparse
import torch
import faiss
import numpy as np
import sqlite3
import open_clip
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',
                    default='checkpoints/proj_heads_best.pt')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

# ── Projection Heads ─────────────────────────────────────────────
class IdentityLinearHead(nn.Module):
    """Single linear layer, identity-init. arch='identity_linear_512'"""
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)

class MLPResidualHead(nn.Module):
    """2-layer MLP + residual + dropout. arch='mlp_residual_512_256'"""
    def __init__(self, in_dim=512, out_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(in_dim, out_dim),
        )
        self.residual = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, x):
        return F.normalize(self.net(x) + self.residual(x), dim=-1)

# ── Load Checkpoint ──────────────────────────────────────────────
print(f'Loading checkpoint: {args.checkpoint}')
ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)

arch = ckpt.get('arch', 'unknown')
print(f'Architecture: {arch}')

if arch == 'identity_linear_512':
    out_dim = 512
    vision_head = IdentityLinearHead(512).to(DEVICE)
    text_head   = IdentityLinearHead(512).to(DEVICE)
elif arch == 'mlp_residual_512_256':
    out_dim = 256
    vision_head = MLPResidualHead(512, 256).to(DEVICE)
    text_head   = MLPResidualHead(512, 256).to(DEVICE)
else:
    # Fallback: try to detect from keys
    keys = list(ckpt['vision_head'].keys())
    if 'linear.weight' in keys and len(keys) == 1:
        out_dim = 512
        vision_head = IdentityLinearHead(512).to(DEVICE)
        text_head   = IdentityLinearHead(512).to(DEVICE)
    else:
        out_dim = 256
        vision_head = MLPResidualHead(512, 256).to(DEVICE)
        text_head   = MLPResidualHead(512, 256).to(DEVICE)

vision_head.load_state_dict(ckpt['vision_head'])
text_head.load_state_dict(ckpt['text_head'])
vision_head.eval()
text_head.eval()

loss_val = ckpt.get('loss', 'N/A')
val_r1   = ckpt.get('val_r1', 'N/A')
print(f'Training loss: {loss_val}')
if val_r1 != 'N/A':
    print(f'Validation R@1: {val_r1:.1f}%')
if 'temperature' in ckpt:
    print(f'Temperature: {ckpt["temperature"]}')

# ── Load Backbone ────────────────────────────────────────────────
print('Loading MobileCLIP-S1...')
model, _, _ = open_clip.create_model_and_transforms(
    'MobileCLIP-S1', pretrained='datacompdr'
)
model = model.to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad = False
tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')

# ── Load 1K-A Test Split ─────────────────────────────────────────
with open('MSRVTT/MSRVTT/structured-symlinks/test_list_miech.txt') as f:
    test_ids = set(l.strip() for l in f)
print(f'Test split: {len(test_ids)} videos')

conn = sqlite3.connect('fedvcmr.db')
all_sentences = conn.execute(
    'SELECT video_id, caption FROM captions'
).fetchall()
all_chunks = conn.execute(
    'SELECT chunk_id, video_id FROM chunks ORDER BY chunk_id'
).fetchall()
conn.close()

test_caps = {}
for vid, cap in all_sentences:
    if vid in test_ids and vid not in test_caps:
        test_caps[vid] = cap
print(f'Test captions: {len(test_caps)}')

# ── Load Feature Cache ───────────────────────────────────────────
n_total = len(all_chunks)
cache   = np.memmap('cache/frame_features.bin',
                    dtype='float16', mode='r',
                    shape=(n_total, 8, 512))

idx_map    = {cid: i for i, (cid, _) in enumerate(all_chunks)}
vid_chunks = {}
for cid, vid in all_chunks:
    vid_chunks.setdefault(vid, []).append(cid)

w = np.array([0.5, 0.75, 1.0, 1.25, 1.25, 1.0, 0.75, 0.5],
             dtype='float32')
w = w / w.sum()

# ── Build Test Video Embeddings ──────────────────────────────────
print('Building video embeddings...')
test_vids = list(test_caps.keys())
test_embs = []

with torch.no_grad():
    for vid in tqdm(test_vids):
        chunks = vid_chunks.get(vid, [])
        if not chunks:
            test_embs.append(np.zeros(out_dim, dtype='float32'))
            continue
        chunk_embs = []
        for cid in chunks:
            frames = cache[idx_map[cid]].astype('float32')  # (8, 512)
            cemb = (frames * w[:, None]).sum(0)  # (512,)
            cemb = cemb / (np.linalg.norm(cemb) + 1e-8)
            chunk_embs.append(cemb)
        avg = np.stack(chunk_embs).mean(0)  # (512,)
        avg = avg / (np.linalg.norm(avg) + 1e-8)
        emb = vision_head(
            torch.tensor(avg).unsqueeze(0).to(DEVICE)
        ).cpu().numpy()[0]
        test_embs.append(emb)

test_embs = np.stack(test_embs).astype('float32')
faiss.normalize_L2(test_embs)

# ── Encode Queries ───────────────────────────────────────────────
print('Encoding queries...')
queries    = [test_caps[vid] for vid in test_vids]
query_embs = []
BATCH      = 128

with torch.no_grad():
    for i in tqdm(range(0, len(queries), BATCH)):
        tokens = tokenizer(queries[i:i+BATCH]).to(DEVICE)
        embs   = model.encode_text(tokens)
        embs   = F.normalize(embs, dim=-1)
        embs   = text_head(embs).cpu().numpy()
        query_embs.append(embs)

query_embs = np.concatenate(query_embs).astype('float32')
faiss.normalize_L2(query_embs)

# ── Retrieve + Rank ──────────────────────────────────────────────
print('Running retrieval...')
index = faiss.IndexFlatIP(out_dim)
index.add(test_embs)
D, I = index.search(query_embs, 10)

ranks = []
for i, vid in enumerate(test_vids):
    retrieved = [test_vids[j] for j in I[i]]
    rank = retrieved.index(vid) + 1 if vid in retrieved else 11
    ranks.append(rank)

# ── Metrics ──────────────────────────────────────────────────────
r1  = sum(r == 1 for r in ranks) / len(ranks) * 100
r5  = sum(r <= 5 for r in ranks) / len(ranks) * 100
r10 = sum(r <= 10 for r in ranks) / len(ranks) * 100
mdr = sorted(ranks)[len(ranks) // 2]

BASELINE = 25.80
print(f'\n--- M10 Evaluation Results ---')
print(f'Zero-shot baseline : R@1 = {BASELINE}%')
print(f'After training     : R@1 = {r1:.2f}%')
print(f'Improvement        : {r1 - BASELINE:+.2f}%')
print(f'R@5  : {r5:.2f}%')
print(f'R@10 : {r10:.2f}%')
print(f'MdR  : {mdr}')

if r1 > BASELINE + 5:
    print('\nM10 GATE PASSED')
elif r1 > BASELINE + 2:
    print('\nM10 GATE MARGINAL -- consider more epochs')
else:
    print('\nM10 GATE FAILED -- check checkpoint and normalization')
