import json
import os
import random
import subprocess
import sqlite3
from tqdm import tqdm
import concurrent.futures
import sys

# Config
VAL_JSON = 'ActivityNet/val_1.json'
OUT_DIR = 'ActivityNet/videos_val'
MAX_VAL_VIDS = 600
CONCURRENT_DOWNLOADS = 4

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def download_video(video_id):
    out_path = os.path.join(OUT_DIR, f"{video_id}.mp4")
    if os.path.exists(out_path):
        return True, "Already exists"
    
    url = f"https://www.youtube.com/watch?v={video_id[2:]}"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "best[height<=360]",
        "--merge-output-format", "mp4",
        "-o", out_path,
        url
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True, "Downloaded"
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

def main():
    with open(VAL_JSON, 'r') as f:
        data = json.load(f)
    
    # Filter for videos with non-empty captions
    all_vids = [vid for vid, vdata in data.items() if vdata.get('sentences')]
    random.seed(42) # For repeatability
    random.shuffle(all_vids)
    
    to_download = all_vids[:MAX_VAL_VIDS]
    
    print(f"Targeting {len(to_download)} validation videos from val_1.json...")
    
    downloaded = 0
    failed = 0
    
    # Initialize DB table first with timeout
    conn = sqlite3.connect('fedvcmr.db', timeout=30.0)
    conn.execute('DROP TABLE IF EXISTS anet_val_segments')
    conn.execute('''
        CREATE TABLE anet_val_segments (
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

    downloaded = 0
    failed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_DOWNLOADS) as executor:
        futures = {executor.submit(download_video, vid): vid for vid in to_download}
        pbar = tqdm(total=len(to_download))
        for future in concurrent.futures.as_completed(futures):
            vid = futures[future]
            success, msg = future.result()
            if success:
                downloaded += 1
                # Insert metadata for successfully downloaded video
                conn = sqlite3.connect('fedvcmr.db', timeout=30.0)
                vdata = data[vid]
                for i, (times, sentence) in enumerate(zip(vdata['timestamps'], vdata['sentences'])):
                    conn.execute('''
                        INSERT INTO anet_val_segments 
                        (video_id, segment_idx, start_time, end_time, duration, sentence, feature_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (vid, i, times[0], times[1], vdata['duration'], sentence, f"cache/anet_val_frames/{vid}_{i}.npy"))
                conn.commit()
                conn.close()
            else:
                failed += 1
            pbar.update(1)
            pbar.set_description(f"New:{downloaded} F:{failed}")
        pbar.close()

    print(f"\nFinal Status: {downloaded} downloads, {failed} failed.")
    print("Validation metadata stored in DB (anet_val_segments).")

if __name__ == "__main__":
    main()
