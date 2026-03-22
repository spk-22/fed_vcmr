# scripts/fix_anet_normalization.py
import torch
import torch.nn.functional as F
import os

def fix_normalization():
    path = 'cache/anet_data_consolidated.pt'
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return

    print(f"Loading {path}...")
    data = torch.load(path, weights_only=False)
    print(f"Processing {len(data)} samples...")

    for i in range(len(data)):
        # 1. Normalize frames (8, 512)
        frames = data[i]['features'].float()
        frames = F.normalize(frames, dim=-1)
        data[i]['features'] = frames.half() # Back to half for consistency

        # 2. Normalize query embed (512,)
        q_emb = data[i]['query_embed'].float()
        if q_emb.dim() == 1:
            q_emb = F.normalize(q_emb, dim=-1)
        else:
            q_emb = F.normalize(q_emb, dim=-1)
        data[i]['query_embed'] = q_emb.half()

    print(f"Saving fixed data to {path}...")
    torch.save(data, path)
    print("Optimization Complete: All features are now unit-normalized.")

if __name__ == "__main__":
    fix_normalization()
