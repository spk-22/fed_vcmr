import torch
import cv2
import sqlite3
import numpy as np
import open_clip
import os
from tqdm import tqdm

# Config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VIDEO_DIR = 'ActivityNet/videos_val'
CACHE_DIR = 'cache/anet_val_frames'
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
            if frames: frames.append(frames[-1])
            else: frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    return np.stack(frames)

from PIL import Image

def main():
    print(f"Loading MobileCLIP-S1 on {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
    model = model.to(DEVICE).eval()
    
    conn = sqlite3.connect(DB_PATH)
    vids = [row[0] for row in conn.execute('SELECT DISTINCT video_id FROM anet_val_segments').fetchall()]
    
    print(f"Processing {len(vids)} validation videos...")
    
    for vid in tqdm(vids):
        video_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
        if not os.path.exists(video_path):
            continue
            
        feat_path = os.path.join(CACHE_DIR, f"{vid}.npy")
        if os.path.exists(feat_path):
            continue
            
        # Get 8 frames using cv2 but convert to PIL for official preprocess
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue
        
        indices = np.linspace(0, total - 1, 8).astype(int)
        batch_tensors = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                batch_tensors.append(batch_tensors[-1] if batch_tensors else torch.zeros(3, 224, 224))
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            batch_tensors.append(preprocess(pil_img))
        cap.release()
        
        input_tensor = torch.stack(batch_tensors).to(DEVICE) # (8, 3, 224, 224)
        
        with torch.no_grad():
            features = model.encode_image(input_tensor).float()
            features = features.cpu().numpy()
            
        np.save(feat_path, features)
        conn.execute('UPDATE anet_val_segments SET feature_path=? WHERE video_id=?', (feat_path, vid))
        conn.commit()
        
    conn.close()
    print("Validation feature extraction complete.")

if __name__ == "__main__":
    main()
