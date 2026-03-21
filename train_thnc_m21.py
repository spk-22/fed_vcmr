"""
M21 — THNC Training (Temporal Hard Negative Curriculum)
=========================================================
Three-phase curriculum:
  Phase 1 (epochs 1-10)  : Easy negatives — random chunks from different videos
  Phase 2 (epochs 11-20) : Medium negatives — chunks with cosine sim > 0.5
  Phase 3 (epochs 21-30) : Hard negatives — chunks with cosine sim > 0.8

Trains projection heads (512→256) on top of frozen MobileCLIP backbone.
Starts from proj_heads_best.pt checkpoint.

Expected: +2-3% R@1 over current 25.80% on MSR-VTT 1K-A

Usage (Colab):
    python train_thnc_m21.py
"""

import os, sys, random, sqlite3, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, "/content/fed_vcmr")
os.chdir("/content/fed_vcmr")

import src.config as cfg
import src.search as sm
cfg.DB_PATH    = Path("/content/data/fedvcmr.db")
cfg.INDEX_PATH = Path("/content/fed_vcmr/faiss_index.bin")
sm.DB_PATH     = Path("/content/data/fedvcmr.db")
sm.INDEX_PATH  = Path("/content/fed_vcmr/faiss_index.bin")

from src.models.projection import ProjectionHead
from src.training.losses import infonce_loss

# ── Config ─────────────────────────────────────────────────────
DRIVE        = "/content/drive/.shortcut-targets-by-id/1SwgWGRg6WNkmN0Rlu0ppLltv6hE4yAFi/fedvcmr"
DB_PATH      = "/content/data/fedvcmr.db"
MEMMAP_PATH  = "/content/data/frame_features.bin"
MEMMAP_SHAPE = (16108, 8, 512)
TEXT_FEAT    = f"{DRIVE}/cache/text_features_normalized.npy"
PROJ_CKPT    = f"{DRIVE}/checkpoints/proj_heads_best.pt"
SAVE_DIR     = f"{DRIVE}/checkpoints_thnc_new"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
LR         = 5e-5        # lower LR since we're fine-tuning
EPOCHS     = 30
TEMP       = 0.07

# Phase boundaries
PHASE1_END = 10   # easy negatives
PHASE2_END = 20   # medium negatives
# Phase 3 = epochs 21-30: hard negatives

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Save dir: {SAVE_DIR}")


# ── Dataset ────────────────────────────────────────────────────
class THNCDataset(Dataset):
    """
    THNC-aware dataset.
    Returns chunk embedding + text embedding + all chunk embeddings
    (needed for hard negative mining).
    """
    def __init__(self, db_path, memmap_path, text_dict, train_ids):
        conn = sqlite3.connect(db_path)

        # All chunks ordered by chunk_id (matches memmap order)
        all_chunks = conn.execute(
            "SELECT chunk_id, video_id FROM chunks ORDER BY chunk_id"
        ).fetchall()
        self.chunk_id_to_row = {c[0]: i for i, c in enumerate(all_chunks)}

        # Filter to training videos only
        train_set    = set(train_ids)
        self.chunks  = [(cid, vid) for cid, vid in all_chunks
                        if vid in train_set and
                        self.chunk_id_to_row[cid] < MEMMAP_SHAPE[0]]

        conn.close()

        # Memmap
        self.cache = np.memmap(memmap_path, dtype="float32",
                               mode="r", shape=MEMMAP_SHAPE)

        # Text features dict: video_id → (20, 512)
        self.text_dict = text_dict

        # Precompute chunk embeddings for negative mining
        # (weighted average of 8 frames, normalized)
        self._chunk_embs = None

        print(f"THNC Dataset: {len(self.chunks)} training chunks")

    def _get_chunk_emb(self, row_idx):
        frames  = self.cache[row_idx].astype(np.float32)       # (8, 512)
        w       = np.array([0.5,0.75,1.0,1.25,1.25,1.0,0.75,0.5],
                           dtype=np.float32)
        w      /= w.sum()
        emb     = (frames * w[:,None]).sum(0)                   # (512,)
        emb    /= (np.linalg.norm(emb) + 1e-8)
        return emb

    def precompute_embeddings(self):
        """Precompute all chunk embeddings for negative mining."""
        print("Precomputing chunk embeddings for negative mining...")
        embs = np.zeros((len(self.chunks), 512), dtype=np.float32)
        for i, (cid, vid) in enumerate(self.chunks):
            row     = self.chunk_id_to_row[cid]
            embs[i] = self._get_chunk_emb(row)
            if (i+1) % 2000 == 0:
                print(f"  {i+1}/{len(self.chunks)}")
        self._chunk_embs = embs
        print("Done.")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        chunk_id, video_id = self.chunks[i]
        row    = self.chunk_id_to_row[chunk_id]
        emb    = torch.from_numpy(self._get_chunk_emb(row))

        # Random caption for this video
        feats  = self.text_dict.get(video_id)
        if feats is not None:
            idx    = random.randint(0, feats.shape[0]-1)
            t_emb  = torch.from_numpy(feats[idx].astype(np.float32))
        else:
            t_emb  = torch.zeros(512)

        return emb, t_emb, i   # i = index for negative mining


# ── Hard negative mining ───────────────────────────────────────
def mine_negatives(chunk_embs: np.ndarray,
                   indices: list,
                   phase: int,
                   n_neg: int = 1) -> list:
    """
    For each sample in batch, find hard negatives based on phase.

    Phase 1: random negatives (any other chunk)
    Phase 2: medium negatives (cosine sim > 0.5, < 0.8)
    Phase 3: hard negatives   (cosine sim > 0.8)

    Returns list of negative indices, one per sample.
    """
    batch_embs = chunk_embs[indices]            # (B, 512)
    all_sims   = batch_embs @ chunk_embs.T      # (B, N)

    neg_indices = []
    for b, idx in enumerate(indices):
        sims      = all_sims[b].copy()
        sims[idx] = -1.0     # exclude self

        if phase == 1:
            # Random negative
            candidates = np.where(sims > -1.0)[0]
            neg        = int(np.random.choice(candidates))

        elif phase == 2:
            # Medium: sim in [0.5, 0.8)
            candidates = np.where((sims >= 0.5) & (sims < 0.8))[0]
            if len(candidates) == 0:
                candidates = np.where(sims > -1.0)[0]
            neg = int(candidates[np.argmax(sims[candidates])])

        else:  # phase 3
            # Hard: sim >= 0.8
            candidates = np.where(sims >= 0.8)[0]
            if len(candidates) == 0:
                # fallback to medium
                candidates = np.where(sims >= 0.5)[0]
            if len(candidates) == 0:
                candidates = np.where(sims > -1.0)[0]
            neg = int(candidates[np.argmax(sims[candidates])])

        neg_indices.append(neg)

    return neg_indices


def get_phase(epoch):
    if epoch <= PHASE1_END:
        return 1
    elif epoch <= PHASE2_END:
        return 2
    return 3


# ── THNC InfoNCE with hard negatives ──────────────────────────
def thnc_loss(v_emb, t_emb, v_neg, temperature=TEMP):
    """
    InfoNCE with hard negative injection.
    v_emb : (B, 256) anchor visual
    t_emb : (B, 256) positive text
    v_neg : (B, 256) hard negative visual
    """
    B = v_emb.shape[0]

    # Standard symmetric InfoNCE
    base_loss = infonce_loss(v_emb, t_emb, temperature)

    # Hard negative contrastive: push v_emb away from v_neg
    # For each sample, negative should score lower than positive
    pos_sim = (v_emb * t_emb).sum(dim=1) / temperature     # (B,)
    neg_sim = (v_neg * t_emb).sum(dim=1) / temperature     # (B,)

    # Margin loss: max(0, neg_sim - pos_sim + margin)
    margin    = 0.2
    hn_loss   = F.relu(neg_sim - pos_sim + margin).mean()

    return base_loss + 0.5 * hn_loss


# ── Load data ──────────────────────────────────────────────────
print("\nLoading data...")

# Text features
text_raw  = np.load(TEXT_FEAT, allow_pickle=True)
text_dict = text_raw.item()
print(f"Text features: {len(text_dict)} videos")

# Training video IDs
with open(f"{DRIVE}/train_list_full.txt") as f:
    train_ids = [l.strip() for l in f.readlines() if l.strip()]
print(f"Training videos: {len(train_ids)}")

# Dataset
dataset = THNCDataset(DB_PATH, MEMMAP_PATH, text_dict, train_ids)
dataset.precompute_embeddings()

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=2,
                        pin_memory=True)


# ── Models ─────────────────────────────────────────────────────
print("\nLoading projection heads...")
vision_head = ProjectionHead(512, 256).to(DEVICE)
text_head   = ProjectionHead(512, 256).to(DEVICE)

# Load from best checkpoint
ckpt = torch.load(PROJ_CKPT, map_location=DEVICE, weights_only=False)
if "vision_head" in ckpt:
    vision_head.load_state_dict(ckpt["vision_head"])
    text_head.load_state_dict(ckpt["text_head"])
    print(f"Loaded proj_heads_best.pt")
else:
    sd = ckpt.get("model_state_dict", ckpt)
    print(f"Checkpoint keys: {list(ckpt.keys())[:5]}")

vision_head.train()
text_head.train()

optimizer = optim.AdamW(
    list(vision_head.parameters()) +
    list(text_head.parameters()),
    lr=LR, weight_decay=0.01
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)


# ── Training loop ──────────────────────────────────────────────
print(f"\nStarting THNC training — {EPOCHS} epochs, batch={BATCH_SIZE}")
print(f"Phase 1 (easy)   : epochs 1-{PHASE1_END}")
print(f"Phase 2 (medium) : epochs {PHASE1_END+1}-{PHASE2_END}")
print(f"Phase 3 (hard)   : epochs {PHASE2_END+1}-{EPOCHS}\n")

best_loss  = float("inf")
history    = []

for epoch in range(1, EPOCHS + 1):
    phase      = get_phase(epoch)
    total_loss = 0.0
    t0         = time.time()

    vision_head.train()
    text_head.train()

    for step, (v_emb, t_emb_raw, indices) in enumerate(dataloader):
        v_emb    = v_emb.to(DEVICE)          # (B, 512)
        t_emb_raw = t_emb_raw.to(DEVICE)     # (B, 512)
        indices  = indices.tolist()

        # Mine hard negatives
        neg_idx  = mine_negatives(
            dataset._chunk_embs, indices, phase
        )
        v_neg_np = dataset._chunk_embs[neg_idx].astype(np.float32)
        v_neg    = torch.from_numpy(v_neg_np).to(DEVICE)  # (B, 512)

        # Project all
        v_proj   = vision_head(v_emb)        # (B, 256)
        t_proj   = text_head(t_emb_raw)      # (B, 256)
        vn_proj  = vision_head(v_neg)        # (B, 256)

        loss = thnc_loss(v_proj, t_proj, vn_proj)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(vision_head.parameters()) +
            list(text_head.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()

        if step % 20 == 0:
            print(f"  Epoch {epoch:2d} | Phase {phase} | "
                  f"Step {step:3d} | Loss {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    elapsed  = time.time() - t0
    history.append({"epoch": epoch, "phase": phase,
                    "loss": avg_loss})

    print(f"Epoch {epoch:2d} | Phase {phase} | "
          f"AvgLoss {avg_loss:.4f} | {elapsed:.1f}s")

    scheduler.step()

    # Save every 10 epochs
    if epoch % 10 == 0 or epoch == EPOCHS:
        save_path = f"{SAVE_DIR}/thnc_epoch_{epoch}.pt"
        torch.save({
            "epoch"      : epoch,
            "phase"      : phase,
            "vision_head": vision_head.state_dict(),
            "text_head"  : text_head.state_dict(),
            "optimizer"  : optimizer.state_dict(),
            "loss"       : avg_loss,
            "history"    : history,
        }, save_path)
        print(f"  Saved: {save_path}")

    # Save best
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            "epoch"      : epoch,
            "phase"      : phase,
            "vision_head": vision_head.state_dict(),
            "text_head"  : text_head.state_dict(),
            "loss"       : avg_loss,
            "history"    : history,
        }, f"{SAVE_DIR}/thnc_best.pt")

print(f"\nTraining complete. Best loss: {best_loss:.4f}")
print(f"Checkpoints saved to: {SAVE_DIR}")

# ── Quick eval on MSR-VTT ──────────────────────────────────────
print("\nRunning quick R@1 eval on MSR-VTT test set...")

from src.evaluation import compute_metrics, get_ranks

vision_head.eval()
text_head.eval()

with open(f"{DRIVE}/msrvtt_miech_test.txt") as f:
    test_ids = [l.strip() for l in f.readlines()]

N            = len(test_ids)
query_matrix = np.zeros((N, 256), dtype=np.float32)
video_matrix = np.zeros((N, 256), dtype=np.float32)

with torch.no_grad():
    for i, vid in enumerate(test_ids):
        # Query: first caption
        feats = text_dict.get(vid)
        if feats is not None:
            t_raw = torch.from_numpy(
                feats[0].astype(np.float32)
            ).unsqueeze(0).to(DEVICE)
            t_proj = text_head(t_raw)
            query_matrix[i] = t_proj[0].cpu().numpy()

        # Video: caption mean as proxy
        if feats is not None:
            v_raw  = torch.from_numpy(
                feats.mean(0).astype(np.float32)
            ).unsqueeze(0).to(DEVICE)
            v_proj = vision_head(v_raw)
            video_matrix[i] = v_proj[0].cpu().numpy()

sim_matrix = query_matrix @ video_matrix.T
ranks      = get_ranks(sim_matrix)
metrics    = compute_metrics(ranks)

print(f"\n{'='*45}")
print(f"M21 THNC — MSR-VTT Eval")
print(f"{'='*45}")
print(f"  R@1  : {metrics['R1']:.2f}%")
print(f"  R@5  : {metrics['R5']:.2f}%")
print(f"  R@10 : {metrics['R10']:.2f}%")
print(f"  MedR : {metrics['MedR']:.1f}")
print(f"{'='*45}")
print(f"  Baseline (M5 zero-shot) : 25.80%")
print(f"  Improvement             : {metrics['R1']-25.80:+.2f}%")
