# scripts/fix_anet_val_features.py
import numpy as np
import os
from tqdm import tqdm

VAL_DIR = 'cache/anet_val_frames'
if not os.path.exists(VAL_DIR):
    print(f"Error: {VAL_DIR} not found.")
    exit(1)

files   = [f for f in os.listdir(VAL_DIR) if f.endswith('.npy')]
print(f'Val files: {len(files)}')

for fname in tqdm(files):
    path   = os.path.join(VAL_DIR, fname)
    frames = np.load(path).astype('float32')    # (8, 512)

    # Check if already normalized
    norms = np.linalg.norm(frames, axis=1)
    if np.allclose(norms, 1.0, atol=0.05):
        continue                                 # already fine

    # Normalize each frame
    frames = frames / (norms[:, None] + 1e-8)
    np.save(path, frames.astype('float16'))

print('Val features fixed.')

# Verify
if files:
    sample = np.load(os.path.join(VAL_DIR, files[0])).astype('float32')
    norms  = np.linalg.norm(sample, axis=1)
    print(f'Norms after fix: {norms}  (all should be ~1.0)')
