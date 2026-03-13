import torch
from torch.utils.data import DataLoader
from src.training.dataset import MSRVTTDataset
from src.training.losses import infonce_loss
from src.models.projection import ProjectionHead
from src.backbone import MobileCLIPWrapper
import math
import os

def sanity_check():
    print("--- Milestone M8: Sanity Check ---")
    
    # 1. Setup paths
    db_path = "fedvcmr.db"
    cache_path = "cache/frame_features.bin"
    batch_size = 128
    
    # 2. Init Dataset and Loader
    dataset = MSRVTTDataset(db_path, cache_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Init Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    backbone = MobileCLIPWrapper(device=device)
    vision_head = ProjectionHead(512, 256).to(device)
    text_head = ProjectionHead(512, 256).to(device)
    
    # 4. Get one batch
    chunk_embs, captions, _ = next(iter(loader))
    chunk_embs = chunk_embs.to(device) # (B, 512)
    
    # 5. Encode text
    with torch.no_grad():
        text_features_np = backbone.encode_text(captions)
        text_features = torch.from_numpy(text_features_np).to(device) # (B, 512)
    
    # 6. Pass through projection heads
    v_proj = vision_head(chunk_embs)
    t_proj = text_head(text_features)
    
    # 7. Compute Loss
    loss = infonce_loss(v_proj, t_proj)
    
    expected_loss = math.log(batch_size)
    print(f"Batch Size: {batch_size}")
    print(f"Calculated Loss: {loss.item():.4f}")
    print(f"Expected Initial Loss (log(B)): {expected_loss:.4f}")
    
    # Verification
    if abs(loss.item() - expected_loss) < 0.5:
        print("\nGATE PASSED: Loss starts near theoretical random value.")
    else:
        print("\nGATE WARNING: Loss differs significantly from theoretical value.")
    
    # 8. Dummy backward pass to verify gradients
    loss.backward()
    print("Backward pass successful. Gradients present.")

if __name__ == "__main__":
    sanity_check()
