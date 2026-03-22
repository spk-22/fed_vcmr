import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# Ensure src can be imported
sys.path.append(os.getcwd())

import src.config
from src.config import DEVICE, CACHE_ROOT
from src.dataset import FedVCMRDataset
from src.model import get_projection_head, ProjectionHead
from src.backbone import MobileCLIPWrapper

def train_projection_head(epochs=5, batch_size=32, limit=None, lr=1e-4):
    print("=== Training Projection Head ===")

    # 1. Setup
    src.config.USE_PROJECTION = True

    # 2. Model
    model = get_projection_head()
    # Ensure usage of shared instance, do not create manually
    assert model is not None, "get_projection_head() returned None despite USE_PROJECTION=True"

    model = model.to(DEVICE)
    model.train()
    print(f"Model loaded on {DEVICE}.")

    # 3. Data
    backbone = MobileCLIPWrapper(device="cpu") # CPU is fine for caching text
    train_dataset = FedVCMRDataset(backbone=backbone, split='train', limit=limit)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 4. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0

        for batch_idx, (video_emb, text_emb) in enumerate(train_loader):
            # Move to DEVICE
            video_emb = video_emb.to(DEVICE).float()
            text_emb = text_emb.to(DEVICE).float()

            optimizer.zero_grad()

            # Forward pass
            v_proj = model(video_emb)
            t_proj = model(text_emb)

            # Similarity matrix (B, B)
            # Add temperature scaling (clip standard) for sharper gradients
            temperature = 0.07
            sim = torch.matmul(v_proj, t_proj.T) / temperature

            # Labels and Loss
            labels = torch.arange(video_emb.size(0), device=DEVICE)
            loss_v = criterion(sim, labels)
            loss_t = criterion(sim.T, labels)
            loss = (loss_v + loss_t) / 2

            # Critical: Check for stability
            if torch.isnan(loss):
                print("NaN detected — stopping training")
                break

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

            # Print loss every few iterations
            if batch_idx % 10 == 0:
                 print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / count if count > 0 else 0
        print(f"Epoch [{epoch+1}/{epochs}] Complete. Avg Loss: {avg_loss:.4f}")

    # 5. Save
    save_path = CACHE_ROOT / "projection_head.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_projection_head(epochs=50, batch_size=32, limit=None, lr=1e-3)
