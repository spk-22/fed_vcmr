import torch
import numpy as np
import os
import sqlite3
from tqdm import tqdm
import open_clip

# Config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DB_PATH = 'fedvcmr.db'
CACHE_DIR = 'cache/anet_frames'
OUT_PATH = 'cache/anet_data_consolidated.pt'

def main():
    if not os.path.exists(DB_PATH):
        print("Error: DB not found")
        return
        
    print(f"Loading MobileCLIP-S1 for text encoding on {DEVICE}...")
    model, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
    model = model.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('''
        SELECT video_id, segment_idx, start_time, end_time, duration, sentence, feature_path 
        FROM anet_segments
    ''').fetchall()
    conn.close()
    
    print(f"Consolidating {len(rows)} segments...")
    
    all_data = []
    features_cache = {}
    
    # Pre-tokenize all sentences for efficiency
    sentences = [row[5] for row in rows]
    # In batches to avoid OOM or huge tensors
    text_features = []
    batch_size = 128
    
    print("Encoding text features...")
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_sent = sentences[i : i + batch_size]
        text = tokenizer(batch_sent).to(DEVICE)
        with torch.no_grad():
            feat = model.encode_text(text)
            feat = F.normalize(feat, dim=-1)
            text_features.append(feat.cpu())
    
    text_features = torch.cat(text_features, dim=0) # (Total, 512)

    print("Packaging video features and labels...")
    for i, row in enumerate(tqdm(rows)):
        vid, idx, s, e, dur, sent, feat_path = row
        
        if feat_path not in features_cache:
            if os.path.exists(feat_path):
                features_cache[feat_path] = np.load(feat_path) # (8, 512)
            else:
                continue
        
        feat = features_cache[feat_path]
        
        # Normalize timestamps to [0, 1] and clamp for precision issues
        s_norm = max(0.0, min(1.0, s / dur))
        e_norm = max(0.0, min(1.0, e / dur))
        
        # Ensure s < e for valid segments
        if s_norm >= e_norm:
            # Shift slightly or skip if invalid
            e_norm = min(1.0, s_norm + 0.01)
            if s_norm >= 1.0: # Edge case
                s_norm = 0.99
                e_norm = 1.0

        
        all_data.append({
            'video_id': vid,
            'features': torch.from_numpy(feat).float(),
            'query_embed': text_features[i].float(),
            'target': torch.tensor([s_norm, e_norm]).float(),
            'sentence': sent
        })

    print(f"Saving {len(all_data)} segments to {OUT_PATH}...")
    torch.save(all_data, OUT_PATH)
    print("Done. This file is ready for Colab.")

import torch.nn.functional as F

if __name__ == "__main__":
    main()
