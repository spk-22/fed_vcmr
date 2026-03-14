# scripts/eval_m11.py
import torch
import faiss
import numpy as np
import sqlite3
import open_clip
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import sys
sys.path.insert(0, '.')
from src.models.dgse import DGSE

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',  default='checkpoints/best_model.pt')
parser.add_argument('--candidates',  type=int, default=50)
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ── Projection Head — must match v4 exactly ──────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.linear.weight)
    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)

# ── Load checkpoint ───────────────────────────────────────────────
print(f'Loading: {args.checkpoint}')
ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
vision_head = ProjectionHead().to(DEVICE)
text_head   = ProjectionHead().to(DEVICE)
vision_head.load_state_dict(ckpt['vision_head'])
text_head.load_state_dict(ckpt['text_head'])
vision_head.eval()
text_head.eval()
print(f'Arch: {ckpt.get("arch", "unknown")}')

# Load DGSE if checkpoint contains it
dgse = DGSE(dim=512, n_frames=8).to(DEVICE)
if 'dgse' in ckpt:
    dgse.load_state_dict(ckpt['dgse'])
    print('DGSE weights loaded.')
else:
    print('No DGSE weights in checkpoint — using random init.')
dgse.eval()

# ── Backbone ──────────────────────────────────────────────────────
model, _, _ = open_clip.create_model_and_transforms(
    'MobileCLIP-S1', pretrained='datacompdr'
)
model     = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')
for p in model.parameters():
    p.requires_grad = False

# ── DB — ORDER BY rowid to match frame_features.bin ──────────────
# CRITICAL: frame_features.bin was built with ORDER BY rowid
# Using ORDER BY chunk_id gives wrong frame→chunk mapping
conn = sqlite3.connect('fedvcmr.db')
all_chunks = conn.execute(
    'SELECT chunk_id, video_id FROM chunks ORDER BY chunk_id'  # Testing chunk_id
).fetchall()
all_caps = conn.execute(
    'SELECT video_id, caption FROM captions'
).fetchall()
conn.close()

# Test split
with open('MSRVTT/MSRVTT/structured-symlinks/test_list_miech.txt') as f:
    test_ids = set(l.strip() for l in f)

test_caps = {}
for vid, cap in all_caps:
    if vid in test_ids and vid not in test_caps:
        test_caps[vid] = cap
print(f'Test videos: {len(test_caps)}')

# ── Frame cache ───────────────────────────────────────────────────
n_total = len(all_chunks)
cache   = np.memmap('cache/frame_features.bin',
                    dtype='float16', mode='r',
                    shape=(n_total, 8, 512))

idx_map    = {cid: i for i, (cid, _) in enumerate(all_chunks)}
vid_chunks = {}
for cid, vid in all_chunks:
    vid_chunks.setdefault(vid, []).append(cid)

w = np.array([0.5,0.75,1.0,1.25,1.25,1.0,0.75,0.5], dtype='float32')
w = w / w.sum()

# ── Build video embeddings ────────────────────────────────────────
print('Building video embeddings...')
test_vids = list(test_caps.keys())
test_embs = []

with torch.no_grad():
    for vid in tqdm(test_vids):
        chunks = vid_chunks.get(vid, [])
        if not chunks:
            test_embs.append(np.zeros(512, dtype='float32'))
            continue
        chunk_embs = []
        for cid in chunks:
            frames = cache[idx_map[cid]].astype('float32')
            cemb   = (frames * w[:, None]).sum(0)
            norm   = np.linalg.norm(cemb) + 1e-8
            chunk_embs.append(cemb / norm)
        avg  = np.stack(chunk_embs).mean(0)
        avg  = avg / (np.linalg.norm(avg) + 1e-8)
        emb  = vision_head(
            torch.tensor(avg).unsqueeze(0).to(DEVICE)
        ).cpu().numpy()[0]
        test_embs.append(emb)

test_embs = np.stack(test_embs).astype('float32')
faiss.normalize_L2(test_embs)

index = faiss.IndexFlatIP(512)
index.add(test_embs)

# ── Encode queries ────────────────────────────────────────────────
print('Encoding queries...')
queries    = [test_caps[vid] for vid in test_vids]
query_embs = []
BATCH      = 128

with torch.no_grad():
    for i in range(0, len(queries), BATCH):
        tokens = tokenizer(queries[i:i+BATCH]).to(DEVICE)
        embs   = F.normalize(model.encode_text(tokens), dim=-1)
        embs   = text_head(embs)
        query_embs.append(embs.cpu())

query_embs = torch.cat(query_embs, dim=0)

# ── Coarse retrieval (M10 baseline) ──────────────────────────────
print('Coarse retrieval...')
q_np = query_embs.numpy().astype('float32')
faiss.normalize_L2(q_np)
D, I = index.search(q_np, 10)

ranks_coarse = []
for i, vid in enumerate(test_vids):
    retrieved = [test_vids[j] for j in I[i]]
    rank = retrieved.index(vid) + 1 if vid in retrieved else 11
    ranks_coarse.append(rank)

r1_c  = sum(r == 1 for r in ranks_coarse) / len(ranks_coarse) * 100
r5_c  = sum(r <= 5 for r in ranks_coarse) / len(ranks_coarse) * 100
r10_c = sum(r <= 10 for r in ranks_coarse) / len(ranks_coarse) * 100

# ── DGSE reranking (M11) ──────────────────────────────────────────
print('DGSE reranking...')

ranks_dgse = []
with torch.no_grad():
    for i, vid in enumerate(tqdm(test_vids)):
        q_emb  = query_embs[i].unsqueeze(0).to(DEVICE)
        q_np_i = q_emb.cpu().numpy().astype('float32')
        faiss.normalize_L2(q_np_i)
        _, I_i = index.search(q_np_i, args.candidates)

        scores = []
        for idx in I_i[0]:
            if idx < 0:
                scores.append((-999, idx))
                continue
            cand_vid = test_vids[idx]
            chunks   = vid_chunks.get(cand_vid, [])
            if not chunks:
                scores.append((-999, idx))
                continue
            cid    = chunks[0]
            frames = torch.tensor(
                cache[idx_map[cid]].astype('float32')
            ).unsqueeze(0).to(DEVICE)
            dgse_emb = dgse(frames, q_emb)
            score    = (dgse_emb @ q_emb.T).item()
            scores.append((score, idx))

        reranked       = sorted(scores, key=lambda x: x[0], reverse=True)
        retrieved_vids = [test_vids[idx] for _, idx in reranked if idx >= 0]
        rank = retrieved_vids.index(vid) + 1 \
               if vid in retrieved_vids else args.candidates + 1
        ranks_dgse.append(rank)

r1_d  = sum(r == 1 for r in ranks_dgse) / len(ranks_dgse) * 100
r5_d  = sum(r <= 5 for r in ranks_dgse) / len(ranks_dgse) * 100
r10_d = sum(r <= 10 for r in ranks_dgse) / len(ranks_dgse) * 100

# ── Results ───────────────────────────────────────────────────────
print(f'\n{"="*52}')
print(f'M11 Evaluation Results')
print(f'{"="*52}')
print(f'                R@1      R@5      R@10')
print(f'Zero-shot:     25.80%')
print(f'M10 coarse:    {r1_c:.2f}%   {r5_c:.2f}%   {r10_c:.2f}%')
print(f'M11 DGSE:      {r1_d:.2f}%   {r5_d:.2f}%   {r10_d:.2f}%')
print(f'DGSE delta:    {r1_d - r1_c:+.2f}%')
print(f'{"="*52}')

if abs(r1_c - 31.66) < 2.0:
    print('M10 coarse matches expected 31.66%')
    if r1_d > r1_c:
        print(f'DGSE improvement: {r1_d - r1_c:+.2f}%')
    else:
        print('DGSE did not improve R@1 over coarse.')
else:
    print(f'WARNING: M10 coarse = {r1_c:.2f}%, expected ~31.66%')
    print('Check checkpoint path and ORDER BY chunk_id')
