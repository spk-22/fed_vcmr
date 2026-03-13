import torch
import open_clip
import sqlite3
import numpy as np
from tqdm import tqdm
import os

import torch.nn.functional as F

def precompute():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading MobileCLIP-S1 on {DEVICE}...")
    model, _, _ = open_clip.create_model_and_transforms(
        'MobileCLIP-S1', pretrained='datacompdr'
    )
    model = model.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')

    # Load all captions from DB
    db_path = 'fedvcmr.db'
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found")
        return

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        'SELECT video_id, caption FROM captions'
    ).fetchall()
    conn.close()

    # Group captions by video_id
    video_text = {}
    for vid, cap in rows:
        video_text.setdefault(vid, []).append(cap)

    print(f"Precomputing text features for {len(video_text)} videos...")
    result = {}
    video_ids = list(video_text.keys())

    # MX250 (2GB VRAM): model ~300MB, leaves ~1.5GB for inference
    # 5 videos × 20 captions = 100 captions/batch → ~500MB, safe margin
    batch_size_vids = 5

    os.makedirs('cache', exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(0, len(video_ids), batch_size_vids)):
            batch_vids = video_ids[i:i+batch_size_vids]
            all_caps = []
            vid_cap_counts = []

            for vid in batch_vids:
                caps = video_text[vid]
                all_caps.extend(caps)
                vid_cap_counts.append(len(caps))

            tokens = tokenizer(all_caps).to(DEVICE)
            embs = model.encode_text(tokens)  # (total_captions, 512)
            embs = F.normalize(embs, dim=-1)  # ← L2 normalize before saving
            embs_cpu = embs.cpu().numpy().astype('float16')

            # Free GPU memory between batches
            del tokens, embs
            torch.cuda.empty_cache()

            start = 0
            for vid, count in zip(batch_vids, vid_cap_counts):
                result[vid] = embs_cpu[start:start+count]
                start += count

    save_path = 'cache/text_features.npy'
    np.save(save_path, result, allow_pickle=True)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Done. Saved to {save_path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    precompute()
