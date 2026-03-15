import torch
import cv2
import json
import sqlite3
import numpy as np
import open_clip
import os
from tqdm import tqdm

# Config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ANET_JSON = 'ActivityNet/train.json'
VIDEO_DIR = 'ActivityNet/videos'
CACHE_DIR = 'cache/anet_frames'
DB_PATH = 'fedvcmr.db'

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_video_frames(video_path, n_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return None
    
    indices = np.linspace(0, total_frames - 1, n_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Try to grab last successful frame or zero
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            continue
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to 224x224 (std CLIP size)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    return np.stack(frames)

def setup_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS anet_segments (
            video_id TEXT,
            segment_idx INTEGER,
            start_time REAL,
            end_time REAL,
            duration REAL,
            sentence TEXT,
            feature_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def main():
    setup_db()
    
    print(f"Loading MobileCLIP-S1 on {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'MobileCLIP-S1', pretrained='datacompdr'
    )
    model = model.to(DEVICE).eval()
    
    with open(ANET_JSON, 'r') as f:
        data = json.load(f)
    
    downloaded_vids = [f[:-4] for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    print(f"Found {len(downloaded_vids)} downloaded videos.")
    
    conn = sqlite3.connect(DB_PATH)
    
    for vid in tqdm(downloaded_vids):
        # Check if already processed
        exists = conn.execute('SELECT 1 FROM anet_segments WHERE video_id=?', (vid,)).fetchone()
        if exists:
            continue
            
        video_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
        frames = get_video_frames(video_path, n_frames=8)
        if frames is None:
            continue
            
        # Process through CLIP
        # preprocess expects PIL or tensor. Let's use torch directly.
        # frames shape: (8, 224, 224, 3)
        input_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        # Normalize (approximate for CLIP)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        input_tensor = (input_tensor - mean) / std
        
        with torch.no_grad():
            features = model.encode_image(input_tensor.to(DEVICE))
            features = features.cpu().numpy().astype('float16') # (8, 512)
            
        feat_path = os.path.join(CACHE_DIR, f"{vid}.npy")
        np.save(feat_path, features)
        
        # Save segments to DB
        duration = data[vid]['duration']
        for i, (ts, sent) in enumerate(zip(data[vid]['timestamps'], data[vid]['sentences'])):
            conn.execute('''
                INSERT INTO anet_segments 
                (video_id, segment_idx, start_time, end_time, duration, sentence, feature_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (vid, i, ts[0], ts[1], duration, sent, feat_path))
        
        conn.commit()
        
    conn.close()
    print("Extraction complete.")

if __name__ == "__main__":
    main()
