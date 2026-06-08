import json
import os
import struct
import numpy as np
from pathlib import Path
import hashlib

# Config
CACHE_DIR = 'cache/anet_benchmark_features'
VAL_JSON = 'ActivityNet/val_1.json'
VIDEO_DIR = 'ActivityNet/videos'
OUTPUT_DIR = 'phone_payload_anet'
TARGET_VIDS = 733
DIM = 512
FRAMES = 8

def get_file_hash(path):
    # Simple hash of first 1MB for "file_hash" field
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(VAL_JSON, 'r') as f:
        val_data = json.load(f)
    
    existing_vids = set([f[:-4] for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])
    all_ids = sorted(val_data.keys())
    
    selected_vids = []
    for vid in all_ids:
        if vid in existing_vids:
            selected_vids.append(vid)
            if len(selected_vids) >= TARGET_VIDS:
                break

    print(f"Packaging {len(selected_vids)} videos...")

    user_index = []
    features_index = {}
    blob_bytes = bytearray()
    
    current_offset = 0
    
    for vid in selected_vids:
        feat_path = os.path.join(CACHE_DIR, f"{vid}.npy")
        video_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
        
        if not os.path.exists(feat_path):
            print(f"Warning: {vid} missing features. Skipping.")
            continue
            
        # 1. Load and append to blob
        features = np.load(feat_path) # (8, 512)
        if features.shape != (FRAMES, DIM):
            print(f"Warning: {vid} has wrong shape {features.shape}. Skipping.")
            continue
            
        features_f32 = features.astype(np.float32)
        blob_bytes += struct.pack(f"{FRAMES * DIM}f", *features_f32.flatten())
        
        # 2. Features Index (video_id -> byte offset)
        features_index[vid] = current_offset
        
        # 3. User Index (metadata)
        duration = val_data[vid]['duration']
        user_index.append({
            "video_id": vid,
            "duration": round(duration, 3),
            "num_frames": FRAMES,
            "file_hash": get_file_hash(video_path),
            "ingested_at": 0 # Not relevant for benchmark but expected by app
        })
        
        current_offset += (FRAMES * DIM * 4)

    # Write files
    with open(os.path.join(OUTPUT_DIR, "user_features_blob.bin"), "wb") as f:
        f.write(blob_bytes)
    
    with open(os.path.join(OUTPUT_DIR, "user_features_index.json"), "w") as f:
        json.dump(features_index, f, indent=2)
        
    with open(os.path.join(OUTPUT_DIR, "user_index.json"), "w") as f:
        json.dump(user_index, f, indent=2)

    print(f"Packaging complete in {OUTPUT_DIR}")
    print(f"  user_features_blob.bin: {len(blob_bytes) / (1024*1024):.2f} MB")
    print(f"  Total videos: {len(user_index)}")

if __name__ == "__main__":
    main()
