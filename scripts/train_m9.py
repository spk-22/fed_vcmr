import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.projection import ProjectionHead
from src.training.losses import infonce_loss
from src.training.dataset import MSRVTTDataset
import sys
import os
import numpy as np
import random

def train():
    # Config
    EPOCHS      = 3  # Initial 3-epoch run as per user request
    BATCH_SIZE  = 128
    LR          = 1e-4
    DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_PATH   = 'checkpoints/proj_heads_epoch_{}.pt'

    os.makedirs('checkpoints', exist_ok=True)

    print("Loading precomputed text features...")
    text_cache_path = 'cache/text_features.npy'
    if not os.path.exists(text_cache_path):
        print(f"Error: {text_cache_path} not found. Run precompute_text.py first.")
        return
    text_cache = np.load(text_cache_path, allow_pickle=True).item()

    vision_head  = ProjectionHead(512, 256).to(DEVICE)
    text_head    = ProjectionHead(512, 256).to(DEVICE)

    optimizer = optim.AdamW(
        list(vision_head.parameters()) +
        list(text_head.parameters()),
        lr=LR, weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30 
    )

    # Data
    print("Initializing dataset...")
    dataset    = MSRVTTDataset('fedvcmr.db', 'cache/frame_features.bin')
    # num_workers=0 for Windows stability
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0,
                            pin_memory=True)

    print(f"Starting optimized training for {EPOCHS} epochs...")
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        vision_head.train()
        text_head.train()
        total_loss = 0.0

        for step, (chunk_emb, captions, frames, video_ids) in enumerate(dataloader):
            chunk_emb = chunk_emb.to(DEVICE)

            # Load precomputed text features
            text_feat = torch.stack([
                torch.from_numpy(text_cache[vid][random.randint(0, 19)].astype('float32'))
                for vid in video_ids
            ]).to(DEVICE) # (B, 512)

            # Project both modalities
            v_emb = vision_head(chunk_emb)    # (B, 256)
            t_emb = text_head(text_feat)      # (B, 256)

            loss = infonce_loss(v_emb, t_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(vision_head.parameters()) +
                list(text_head.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} DONE | Avg Loss: {avg_loss:.4f}")
        scheduler.step()

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'vision_head': vision_head.state_dict(),
            'text_head':   text_head.state_dict(),
            'optimizer':   optimizer.state_dict(),
        }, SAVE_PATH.format(epoch))
        print(f"Checkpoint saved: epoch {epoch}")

if __name__ == "__main__":
    train()
