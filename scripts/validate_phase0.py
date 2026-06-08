import json
import os

# Config
VAL_JSON = 'ActivityNet/val_1.json'
VIDEO_DIR = 'ActivityNet/videos'
TARGET_VIDS = 733

def validate_phase0():
    if not os.path.exists(VAL_JSON):
        print(f"Error: {VAL_JSON} not found.")
        return

    with open(VAL_JSON, 'r') as f:
        data = json.load(f)
    
    # ActivityNet/videos contains v_XXXXX.mp4
    existing_vids = set([f[:-4] for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])
    
    # Sort all IDs from val_1.json to keep it deterministic as per plan
    all_ids = sorted(data.keys())
    
    valid_vids = []
    total_duration = 0
    
    for vid in all_ids:
        if vid in existing_vids:
            valid_vids.append(vid)
            total_duration += data[vid]['duration']
            if len(valid_vids) >= TARGET_VIDS:
                break
    
    print(f"Total videos found from val_1.json: {len(valid_vids)}")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")
    
    if len(valid_vids) >= TARGET_VIDS:
        print(f"Phase 0 Validated: Found {len(valid_vids)} videos covering {total_duration/3600:.2f} hours.")
    else:
        print(f"Phase 0 FAILED: Only found {len(valid_vids)} videos.")

if __name__ == "__main__":
    validate_phase0()
