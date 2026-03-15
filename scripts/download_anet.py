import json
import os
import subprocess
from tqdm import tqdm
import concurrent.futures

# Config
ANET_JSON = 'ActivityNet/train.json'
OUT_DIR = 'ActivityNet/videos'
MAX_VIDS = 3300
CONCURRENT_DOWNLOADS = 4

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

import sys

def download_video(video_id):
    out_path = os.path.join(OUT_DIR, f"{video_id}.mp4")
    if os.path.exists(out_path):
        return True, "Already exists"
    
    # ActivityNet URLs are usually https://www.youtube.com/watch?v=VIDEO_ID
    url = f"https://www.youtube.com/watch?v={video_id[2:]}" # v_XXXXXX -> XXXXXX
    
    # yt-dlp command
    # -f 'bestvideo[height<=360]+bestaudio/best[height<=360]' to save space
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "best[height<=360]",
        "--merge-output-format", "mp4",
        "-o", out_path,
        url
    ]
    
    try:
        # Using subprocess.run to avoid being blocked by interactive prompts
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True, "Downloaded"
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

import random

def main():
    with open(ANET_JSON, 'r') as f:
        data = json.load(f)
    
    all_vids = list(data.keys())
    existing_vids = set([f[:-4] for f in os.listdir(OUT_DIR) if f.endswith('.mp4')])
    current_count = len(existing_vids)
    
    print(f"Current successful downloads: {current_count}")
    if current_count >= MAX_VIDS:
        print("Target reached.")
        return

    to_try = [vid for vid in all_vids if vid not in existing_vids]
    # Randomize to avoid clusters of dead links
    random.shuffle(to_try)
    
    needed = MAX_VIDS - current_count
    print(f"Need {needed} more videos. Searching in randomized list of {len(to_try)}...")
    
    downloaded = 0
    failed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_DOWNLOADS) as executor:
        batch_size = 50
        try_idx = 0
        
        pbar = tqdm(total=needed)
        while current_count + downloaded < MAX_VIDS and try_idx < len(to_try):
            batch = to_try[try_idx : try_idx + batch_size]
            try_idx += batch_size
            
            futures = {executor.submit(download_video, vid): vid for vid in batch}
            first_fail_msg = None
            
            for future in concurrent.futures.as_completed(futures):
                success, msg = future.result()
                if success:
                    if msg == "Downloaded":
                        downloaded += 1
                        pbar.update(1)
                else:
                    failed += 1
                    if first_fail_msg is None:
                        first_fail_msg = msg
                
                pbar.set_description(f"New:{downloaded} F:{failed} Total:{current_count + downloaded}")
                if current_count + downloaded >= MAX_VIDS:
                    break
            
            # Print one error sample per batch if failures occur
            if first_fail_msg:
                # Truncate and clean for ASCII-safe printing
                clean_msg = str(first_fail_msg).split('\n')[0][:100]
                print(f"\nSample failure: {clean_msg}")
                
        pbar.close()

    print(f"\nFinal Status: {downloaded} new downloads, {failed} failed in this session.")
    print(f"Total videos in {OUT_DIR}: {len([f for f in os.listdir(OUT_DIR) if f.endswith('.mp4')])}")


if __name__ == "__main__":
    main()
