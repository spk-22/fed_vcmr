# scripts/fix_anet_features.py
# Normalize all features in anet_data_consolidated.pt
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os

IN_PATH  = 'cache/anet_data_consolidated.pt'
OUT_PATH = 'cache/anet_data_consolidated_fixed.pt'

if not os.path.exists(IN_PATH):
    print(f"Error: {IN_PATH} not found.")
    exit(1)

print('Loading...')
data = torch.load(IN_PATH, weights_only=False)
print(f'Samples: {len(data):,}')

# Check before
sample_norm = data[0]['features'][0].norm().item()
query_norm  = data[0]['query_embed'].norm().item()
print(f'Before — feature norm: {sample_norm:.2f}  query norm: {query_norm:.4f}')

fixed = []
for item in tqdm(data):
    # Normalize each frame vector to unit sphere
    frames = item['features'].float()           # (8, 512)
    frames = F.normalize(frames, dim=-1)        # each frame → unit norm

    # Normalize query embedding
    q = item['query_embed'].float()
    q = F.normalize(q, dim=-1)

    # Clamp target just in case
    target = torch.clamp(item['target'].float(), 0.0, 1.0)

    fixed.append({
        'video_id':    item['video_id'],
        'features':    frames,
        'query_embed': q,
        'target':      target,
        'sentence':    item.get('sentence', '')
    })

# Verify after
sample_norm_fixed = fixed[0]['features'][0].norm().item()
query_norm_fixed  = fixed[0]['query_embed'].norm().item()
print(f'After  — feature norm: {sample_norm_fixed:.4f}  query norm: {query_norm_fixed:.4f}')
assert abs(sample_norm_fixed - 1.0) < 0.01, 'Frame norm still wrong'
assert abs(query_norm_fixed  - 1.0) < 0.01, 'Query norm still wrong'

print(f'Saving to {OUT_PATH}...')
torch.save(fixed, OUT_PATH)
print('Done.')
