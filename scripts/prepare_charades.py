"""
Charades-STA Preparation Script for M17
- Parses Charades_v1_test.csv to generate sentence-temporal annotations
- Downloads only the test set videos (~138 videos)
- Extracts features using official open_clip preprocessing
- Populates fedvcmr.db with charades_sta_segments table
"""
import csv
import os
import sys
import sqlite3
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import open_clip
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '.')

DB_PATH    = 'fedvcmr.db'
CSV_PATH   = 'Charades/Charades_annotations/Charades/Charades_v1_test.csv'
VIDEO_DIR  = 'Charades/Charades_v1_480'
CACHE_DIR  = 'cache/charades_frames'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Step 1: Parse CSV and build segments ──────────────────────────
def parse_charades_csv(csv_path):
    """Parse Charades test CSV into sentence-temporal segments.
    
    Charades-STA uses video descriptions as queries and action
    temporal boundaries as ground-truth moments.
    """
    segments = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['id']
            duration = float(row['length'])
            descriptions = row['descriptions'].split(';')
            
            # Use the first description as the query sentence 
            sentence = descriptions[0].strip() if descriptions else ""
            if not sentence:
                continue
                
            # Parse action temporal annotations: "c077 12.10 18.00;c079 11.80 17.30"
            actions_raw = row['actions']
            if not actions_raw:
                continue
                
            action_list = actions_raw.split(';')
            for act in action_list:
                parts = act.strip().split()
                if len(parts) >= 3:
                    try:
                        start = float(parts[1])
                        end = float(parts[2])
                        if end > start and end <= duration:
                            segments.append({
                                'video_id': vid,
                                'sentence': sentence,
                                'start_time': start,
                                'end_time': end,
                                'duration': duration
                            })
                    except ValueError:
                        continue
    return segments

import zipfile

# ── Step 2: Extract videos from zip ────────────────────────────────
def extract_videos_from_zip(zip_path, video_dir, vids):
    """Extract specific video files from the Charades zip archive."""
    vids_set = {f"{vid}.mp4" for vid in vids}
    success = 0
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Some zips have a nested folder like 'Charades_v1_480/'
        # Let's find the correct paths inside the zip
        namelist = z.namelist()
        for name in tqdm(namelist, desc="Checking ZIP"):
            basename = os.path.basename(name)
            if basename in vids_set:
                out_path = os.path.join(video_dir, basename)
                if not os.path.exists(out_path):
                    with open(out_path, "wb") as f:
                        f.write(z.read(name))
                success += 1
    return success

# ── Step 3: Extract features ────────────────────────────────────
def extract_features(video_dir, cache_dir, model, preprocess, vids):
    """Extract 8-frame features using official open_clip preprocessing."""
    os.makedirs(cache_dir, exist_ok=True)
    
    for vid in tqdm(vids, desc="Extracting Features"):
        feat_path = os.path.join(cache_dir, f"{vid}.npy")
        if os.path.exists(feat_path):
            continue
            
        video_path = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(video_path):
            continue
            
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
        
        input_tensor = torch.stack(batch_tensors).to(DEVICE)
        with torch.no_grad():
            features = model.encode_image(input_tensor).float()
            features = features.cpu().numpy()
        np.save(feat_path, features)

# ── Step 4: Populate DB ─────────────────────────────────────────
def populate_db(segments, db_path):
    """Create charades_sta_segments table and insert data."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute('''CREATE TABLE IF NOT EXISTS charades_sta_segments (
        video_id TEXT, sentence TEXT, start_time REAL, end_time REAL,
        duration REAL, feature_path TEXT
    )''')
    conn.execute('DELETE FROM charades_sta_segments')
    
    for seg in segments:
        feat_path = f"cache/charades_frames/{seg['video_id']}.npy"
        conn.execute(
            'INSERT INTO charades_sta_segments VALUES (?,?,?,?,?,?)',
            (seg['video_id'], seg['sentence'], seg['start_time'],
             seg['end_time'], seg['duration'], feat_path)
        )
    conn.commit()
    conn.close()

# ── Main ────────────────────────────────────────────────────────
def main():
    # Step 1: Parse all test videos (no subset limit)
    print("Parsing Charades test CSV...")
    segments = parse_charades_csv(CSV_PATH)
    unique_vids = sorted(list(set(s['video_id'] for s in segments)))
    print(f"  Found {len(segments)} segments across {len(unique_vids)} unique videos")
    
    # Step 2: Videos already extracted — verify
    available = [v for v in unique_vids if os.path.exists(os.path.join(VIDEO_DIR, f"{v}.mp4"))]
    print(f"\n  Videos found on disk: {len(available)} / {len(unique_vids)}")
    
    # Step 3: Extract Features
    print("\nLoading MobileCLIP-S1...")
    model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
    model = model.to(DEVICE).eval()
    
    downloaded_vids = available
    extract_features(VIDEO_DIR, CACHE_DIR, model, preprocess, downloaded_vids)
    
    # Step 4: Populate DB
    print("\nPopulating database...")
    # Filter segments to only include downloaded videos
    valid_segments = [s for s in segments if os.path.exists(os.path.join(CACHE_DIR, f"{s['video_id']}.npy"))]
    populate_db(valid_segments, DB_PATH)
    print(f"  Stored {len(valid_segments)} segments in charades_sta_segments table")
    
    print("\nCharades-STA preparation complete!")

if __name__ == "__main__":
    main()
