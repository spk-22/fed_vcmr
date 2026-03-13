import numpy as np
import torch
from torch.utils.data import Dataset
import sqlite3
import random
import os
from pathlib import Path

class MSRVTTDataset(Dataset):
    def __init__(self, db_path, cache_path, split_file='MSRVTT/MSRVTT/structured-symlinks/train_list_full.txt'):
        # 1. Connect and Load split-specific video IDs
        if not os.path.isabs(split_file):
            split_file = os.path.join(os.getcwd(), split_file)
            
        with open(split_file, 'r') as f:
            valid_ids = set(line.strip() for line in f if line.strip())

        conn = sqlite3.connect(db_path)
        
        # 2. Get chunks for valid videos
        # We need chunk_id to map to mmap index, and video_id for caption lookup
        all_chunks = conn.execute(
            'SELECT chunk_id, video_id FROM chunks ORDER BY chunk_id'
        ).fetchall()
        
        # Filter by training split
        self.chunks = [c for c in all_chunks if c[1] in valid_ids]

        # 3. Caption lookup: video_id -> [caption1, caption2, ...]
        rows = conn.execute(
            'SELECT video_id, caption FROM captions'
        ).fetchall()
        self.captions = {}
        for vid, cap in rows:
            if vid in valid_ids:
                self.captions.setdefault(vid, []).append(cap)
        conn.close()

        # 4. Load mmap cache
        # Note: shape must match consolidate_cache.py (N, 8, 512)
        # N=32216 is the known size for the full 10k set
        n_total = 32216 
        self.cache = np.memmap(cache_path,
                               dtype='float16',
                               mode='r',
                               shape=(n_total, 8, 512))

        # We need the global index for each chunk in the mmap
        # Since mmap was built with ORDER BY chunk_id, we map them here
        self.chunk_id_to_global_idx = {c[0]: i for i, c in enumerate(all_chunks)}
        
        print(f"Dataset initialized with {len(self.chunks)} training chunks.")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        chunk_id, video_id = self.chunks[i]
        global_idx = self.chunk_id_to_global_idx[chunk_id]

        # Randomly sample one caption per video (MSR-VTT has 20 per vid)
        caps = self.captions.get(video_id, [''])
        caption = random.choice(caps)

        # Load frame features from mmap (shape: 8, 512)
        frames = self.cache[global_idx].astype('float32')

        # FIX: Normalize each frame to unit sphere BEFORE averaging
        frame_norms = np.linalg.norm(frames, axis=1, keepdims=True) + 1e-8
        frames = frames / frame_norms

        # Feature Pooling (Center-biased weighted average)
        w = np.array([0.5, 0.75, 1.0, 1.25, 1.25, 1.0, 0.75, 0.5],
                     dtype='float32')
        w = w / w.sum()
        chunk_emb = (frames * w[:, None]).sum(0)  # (512,)

        # L2-normalize the pooled result
        chunk_emb = chunk_emb / (np.linalg.norm(chunk_emb) + 1e-8)
        chunk_emb = torch.from_numpy(chunk_emb)

        return chunk_emb, caption, frames, video_id

import os # Required for os.path check in __init__
