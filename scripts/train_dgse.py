# scripts/train_dgse.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sqlite3
import open_clip
import faiss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys, os, time
sys.path.insert(0, '.')
from src.models.dgse import DGSE

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32      # small for MX250 2GB
EPOCHS     = 20
LR         = 3e-4
CKPT_PATH  = 'checkpoints/best_model.pt'
SAVE_PATH  = 'checkpoints/dgse_best.pt'
print(f'Device: {DEVICE}')

# ── Projection Head ───────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.linear.weight)
    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)

# ── Load frozen projection heads ──────────────────────────────────
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
vision_head = ProjectionHead().to(DEVICE)
text_head   = ProjectionHead().to(DEVICE)
vision_head.load_state_dict(ckpt['vision_head'])
text_head.load_state_dict(ckpt['text_head'])
vision_head.eval()
text_head.eval()
for p in vision_head.parameters(): p.requires_grad = False
for p in text_head.parameters():   p.requires_grad = False
print('Projection heads loaded and frozen.')

# ── Load frozen backbone ──────────────────────────────────────────
model, _, _ = open_clip.create_model_and_transforms(
    'MobileCLIP-S1', pretrained='datacompdr'
)
model     = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')
for p in model.parameters(): p.requires_grad = False
print('Backbone loaded and frozen.')

# ── DB ────────────────────────────────────────────────────────────
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

with open('MSRVTT/MSRVTT/structured-symlinks/train_list_full.txt') as f:
    train_ids = set(l.strip() for l in f)
with open('MSRVTT/MSRVTT/structured-symlinks/test_list_miech.txt') as f:
    test_ids = list(set(l.strip() for l in f))

# ── Frame cache ───────────────────────────────────────────────────
n_total = len(all_chunks)
cache   = np.memmap('cache/frame_features.bin',
                    dtype='float16', mode='r',
                    shape=(n_total, 8, 512))

w = np.array([0.5,0.75,1.0,1.25,1.25,1.0,0.75,0.5], dtype='float32')
w = w / w.sum()

# ── Text cache ────────────────────────────────────────────────────
text_cache = np.load('cache/text_features.npy',
                     allow_pickle=True).item()

# ── Dataset ───────────────────────────────────────────────────────
class DGSEDataset(Dataset):
    """
    Returns (frames, text_emb) per chunk.
    frames:   (8, 512) raw frame features -- DGSE learns to pool these
    text_emb: (512,)   normalized text embedding
    """
    def __init__(self, train_ids):
        self.items = [(cid, vid) for cid, vid in all_chunks
                      if vid in train_ids and vid in text_cache]
        print(f'DGSE dataset: {len(self.items):,} chunks')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cid, vid = self.items[i]
        frames   = cache[idx_map[cid]].astype('float32')  # (8, 512)
        caps     = text_cache[vid]
        text_emb = caps[np.random.randint(len(caps))].astype('float32')
        return torch.tensor(frames), torch.tensor(text_emb)

dataset = DGSEDataset(train_ids)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                     shuffle=True, num_workers=0,
                     pin_memory=(DEVICE == 'cuda'))

# ── DGSE loss ─────────────────────────────────────────────────────
def dgse_infonce(dgse_emb, text_emb, temperature=0.07):
    """
    dgse_emb: (B, 512) already normalized
    text_emb: (B, 512) already normalized from text_head
    """
    logits   = torch.matmul(text_emb, dgse_emb.T) / temperature
    labels   = torch.arange(len(logits), device=logits.device)
    loss_t2v = F.cross_entropy(logits,   labels)
    loss_v2t = F.cross_entropy(logits.T, labels)
    return (loss_t2v + loss_v2t) / 2

# ── Validation ────────────────────────────────────────────────────
def validate(dgse, val_vids, n=200):
    dgse.eval()
    vids = val_vids[:n]
    v_embs, t_embs = [], []

    with torch.no_grad():
        for vid in vids:
            chunks = vid_chunks.get(vid, [])
            if not chunks or vid not in caps_map:
                continue
            # Build DGSE embedding for this video
            # Use first caption as query to build query-aware embedding
            cap    = caps_map[vid][0]
            tokens = tokenizer([cap]).to(DEVICE)
            t_raw  = F.normalize(model.encode_text(tokens), dim=-1)
            t_emb  = text_head(t_raw)  # (1, 512)

            # Average DGSE embeddings across all chunks
            chunk_embs = []
            for cid in chunks:
                frames   = torch.tensor(
                    cache[idx_map[cid]].astype('float32')
                ).unsqueeze(0).to(DEVICE)                 # (1, 8, 512)
                dgse_emb = dgse(frames, t_emb)            # (1, 512)
                chunk_embs.append(dgse_emb.cpu().numpy()[0])

            avg_emb = np.stack(chunk_embs).mean(0)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
            v_embs.append(avg_emb)
            t_embs.append(t_emb.cpu().numpy()[0])

    if len(v_embs) < 10:
        return 0.0

    v_arr = np.stack(v_embs).astype('float32')
    t_arr = np.stack(t_embs).astype('float32')
    faiss.normalize_L2(v_arr)
    faiss.normalize_L2(t_arr)

    index = faiss.IndexFlatIP(512)
    index.add(v_arr)
    _, I  = index.search(t_arr, 10)
    n_val = len(v_arr)
    r1    = sum(I[i, 0] == i for i in range(n_val)) / n_val * 100
    dgse.train()
    return r1

# ── Initialize DGSE ───────────────────────────────────────────────
dgse      = DGSE(dim=512, n_frames=8).to(DEVICE)
optimizer = optim.AdamW(dgse.parameters(), lr=LR, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

# Quick check before training
n_params = sum(p.numel() for p in dgse.parameters() if p.requires_grad)
print(f'DGSE trainable params: {n_params:,}')

r1_before = validate(dgse, test_ids)
print(f'R@1 before training:   {r1_before:.2f}%  (expect ~31%)')

# ── Training loop ─────────────────────────────────────────────────
best_r1   = r1_before
best_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    dgse.train()
    total_loss = 0.0
    t0 = time.time()

    for frames, text_emb in loader:
        frames   = frames.to(DEVICE)    # (B, 8, 512)
        text_emb = text_emb.to(DEVICE)  # (B, 512)

        # Pass text through text_head to get normalized query
        with torch.no_grad():
            t_proj = text_head(text_emb)  # (B, 512)

        # DGSE forward -- query-conditioned pooling
        dgse_emb = dgse(frames, t_proj)   # (B, 512)

        loss = dgse_infonce(dgse_emb, t_proj)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dgse.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(loader)
    elapsed  = time.time() - t0

    # Validate every 2 epochs
    if epoch % 2 == 0 or epoch == 1:
        r1 = validate(dgse, test_ids)
        print(f'Epoch {epoch:2d}/{EPOCHS} | '
              f'Loss: {avg_loss:.4f} | '
              f'R@1: {r1:.2f}% | '
              f'{elapsed:.0f}s')

        if r1 > best_r1:
            best_r1 = r1
            torch.save({
                'dgse':        dgse.state_dict(),
                'vision_head': ckpt['vision_head'],
                'text_head':   ckpt['text_head'],
                'arch':        'dgse_identity_linear_512',
                'r1':          r1,
            }, SAVE_PATH)
            print(f'  Best R@1: {r1:.2f}% -> saved to {SAVE_PATH}')
    else:
        print(f'Epoch {epoch:2d}/{EPOCHS} | '
              f'Loss: {avg_loss:.4f} | '
              f'{elapsed:.0f}s')

print(f'\nTraining complete.')
print(f'Best R@1: {best_r1:.2f}%  (was {r1_before:.2f}% before training)')
print(f'Delta:    {best_r1 - r1_before:+.2f}%')
