import torch
from torch.utils.data import Dataset
import numpy as np
import sqlite3
from typing import List, Tuple
from src.config import DB_PATH, DEVICE
from src.backbone import MobileCLIPWrapper

class FedVCMRDataset(Dataset):
    """
    Dataset for training the ProjectionHead.
    Loads cached video chunks and corresponding captions.
    Aggregates video frames into a single vector (consistency with search.py).
    """
    def __init__(self, db_path=DB_PATH, backbone=None, split='train', limit=None):
        self.chunk_data = [] # List of (cache_path, caption)

        print(f"Initializing Dataset (Limit: {limit})...")

        # 1. Load Metadata
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get chunks: (chunk_id, cache_path, video_id)
        cursor.execute("SELECT chunk_id, cache_path, video_id FROM chunks")
        chunks = cursor.fetchall()

        # Get captions: (video_id, caption)
        cursor.execute("SELECT video_id, caption FROM captions")
        all_captions = cursor.fetchall()
        conn.close()

        # Index captions by video_id
        vid_to_caps = {}
        for vid, cap in all_captions:
            if vid not in vid_to_caps:
                vid_to_caps[vid] = []
            vid_to_caps[vid].append(cap)

        # Join chunks with captions
        # We create a sample for every (chunk, caption) pair
        # This effectively uses all available text supervision for every chunk
        count = 0
        for chunk_id, cache_path, video_id in chunks:
            if video_id in vid_to_caps:
                caps = vid_to_caps[video_id]
                for cap in caps:
                    self.chunk_data.append({
                        'cache_path': cache_path,
                        'caption': cap
                    })
                    count += 1
                    if limit and count >= limit:
                        break
            if limit and count >= limit:
                break

        print(f"Loaded {len(self.chunk_data)} samples.")

        # 2. Pre-compute Text Embeddings
        # We do this once to avoid running the backbone in the training loop
        if backbone is None:
            print("Backbone not provided, initializing generic MobileCLIPWrapper...")
            backbone = MobileCLIPWrapper()

        print("Pre-computing text embeddings...")
        unique_captions = list(set(d['caption'] for d in self.chunk_data))
        text_cache = {}

        batch_size = 256
        total_batches = (len(unique_captions) + batch_size - 1) // batch_size

        # Simple batching logic
        for i in range(0, len(unique_captions), batch_size):
            batch_texts = unique_captions[i:i+batch_size]
            # Backbone returns (B, 512) numpy
            embs = backbone.encode_text(batch_texts)
            for txt, emb in zip(batch_texts, embs):
                text_cache[txt] = emb

        # Store embeddings in self.chunk_data to save lookup time or kept in generic cache?
        # Storing in list might duplicate data in RAM?
        # Actually storing a reference to the numpy array is cheap.
        for item in self.chunk_data:
            item['text_emb'] = text_cache[item['caption']]

        del text_cache # Free dict overhead
        print("Text embeddings cached.")

        # 3. Setup Aggregation Weights
        # Exact weights from technical reference / search.py
        self.weights = np.array([0.5, 0.75, 1.0, 1.25, 1.25, 1.0, 0.75, 0.5])

    def __len__(self):
        return len(self.chunk_data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.chunk_data[idx]

        # A. Video: Load -> Weighted Avg -> Normalize
        # Load (8, 512) or similar
        features = np.load(item['cache_path']).astype(np.float32)

        # Handle frame count mismatch by slicing weights
        n_frames = features.shape[0]
        w = self.weights[:n_frames]
        w_sum = np.sum(w)

        # Weighted Average
        if w_sum > 0:
            video_emb = np.sum(features * w[:, np.newaxis], axis=0) / w_sum
        else:
            video_emb = np.mean(features, axis=0)

        # Normalize (Input to ProjectionHead should be normalized if it expects generic CLIP embeddings)
        video_emb /= (np.linalg.norm(video_emb) + 1e-10)

        # B. Text: Retrieve cached embedding (Already normalized by backbone)
        text_emb = item['text_emb']

        # Return tensors (CPU) - Training loop handles device move
        return torch.from_numpy(video_emb), torch.from_numpy(text_emb)

def get_dataloader(batch_size=32, limit=None):
    from torch.utils.data import DataLoader
    ds = FedVCMRDataset(limit=limit)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    # Smoke test
    dl = get_dataloader(batch_size=4, limit=10)
    for v, t in dl:
        print(f"Video Batch: {v.shape}, Text Batch: {t.shape}")
        break

