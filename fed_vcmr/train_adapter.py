import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Ensure src can be imported
sys.path.append(os.getcwd())

import src.config
from src.config import DEVICE, CACHE_ROOT
from src.dataset import FedVCMRDataset
from src.model import get_projection_head
from src.adapter import get_adapter_manager
from src.backbone import MobileCLIPWrapper

def train_adapter(client_id="client_1", epochs=5, batch_size=32, limit=None, lr=1e-4):
    print(f"=== Training Adapter for {client_id} ===")

    # 1. Setup Config
    src.config.USE_PROJECTION = True
    src.config.USE_ADAPTER = True

    # 2. Load Frozen ProjectionHead
    projection_head = get_projection_head()
    if projection_head is None:
        raise RuntimeError("ProjectionHead not found! Ensure USE_PROJECTION is True and weights exist.")

    # Freeze ProjectionHead
    projection_head.eval()
    for p in projection_head.parameters():
        p.requires_grad = False
    print("ProjectionHead loaded and frozen.")

    # 3. Load/Create Adapter
    manager = get_adapter_manager()
    adapter = manager.get_adapter(client_id)
    adapter.train()
    print(f"Adapter for {client_id} initialized and set to train mode.")

    # 4. Data
    # Initialize backbone on CPU for caching text embeddings
    backbone = MobileCLIPWrapper(device="cpu")
    train_dataset = FedVCMRDataset(backbone=backbone, split='train', limit=limit)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 5. Optimizer (Adapter params only)
    optimizer = optim.Adam(adapter.parameters(), lr=lr)
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

            # Forward pass: Projection (Frozen)
            with torch.no_grad():
                v_proj = projection_head(video_emb)
                t_proj = projection_head(text_emb)

            # Forward pass: Adapter (Trainable)
            v_final = adapter(v_proj)
            t_final = adapter(t_proj)

            # Raw similarity for monitoring
            sim_raw = torch.matmul(v_final, t_final.T)

            # Similarity matrix (B, B) with temperature
            temperature = 0.05
            sim = sim_raw / temperature

            # Labels
            labels = torch.arange(video_emb.size(0), device=DEVICE)

            # Symmetric Loss
            loss_v = criterion(sim, labels)
            loss_t = criterion(sim.T, labels)
            contrastive_loss = (loss_v + loss_t) / 2

            # 1. Regularization: Prevent collapse (push std towards 1.0)
            std_v = v_final.std(dim=0).mean()
            std_t = t_final.std(dim=0).mean()
            reg_loss = 0.2 * (torch.relu(1.0 - std_v) + torch.relu(1.0 - std_t))

            # 2. Centering: Push mean similarity -> 0
            mean_sim = sim_raw.mean()
            center_loss = 0.05 * (mean_sim ** 2)

            loss = contrastive_loss + reg_loss + center_loss

            # Stability Check
            if torch.isnan(loss):
                print("NaN detected — stopping training")
                break

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

            if batch_idx % 10 == 0:
                 print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}], Loss: {loss.item():.4f}, Sim: {mean_sim.item():.4f}, StdV: {std_v.item():.4f}, StdT: {std_t.item():.4f}")

        avg_loss = total_loss / count if count > 0 else 0
        print(f"Epoch [{epoch+1}/{epochs}] Complete. Avg Loss: {avg_loss:.4f}")

    # 6. Save Adapter Weights
    save_path = CACHE_ROOT / f"adapter_{client_id}.pth"
    torch.save(adapter.state_dict(), save_path)
    print(f"Adapter model saved to {save_path}")

if __name__ == "__main__":
    train_adapter(client_id="client_1", epochs=5, batch_size=32)
