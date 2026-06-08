import json
import os

PAYLOAD_DIR = 'phone_payload_anet'
BLOB_PATH = os.path.join(PAYLOAD_DIR, 'user_features_blob.bin')
FEAT_IDX_PATH = os.path.join(PAYLOAD_DIR, 'user_features_index.json')
USER_IDX_PATH = os.path.join(PAYLOAD_DIR, 'user_index.json')

def verify():
    print("Verifying PC-side Payload...")
    
    with open(USER_IDX_PATH, 'r') as f:
        user_index = json.load(f)
    with open(FEAT_IDX_PATH, 'r') as f:
        feat_index = json.load(f)
    
    blob_size = os.path.getsize(BLOB_PATH)
    
    # Check counts
    if len(user_index) != len(feat_index):
        print(f"Error: Count mismatch! User Index: {len(user_index)}, Feat Index: {len(feat_index)}")
        return
    
    # Check consistency
    for entry in user_index:
        vid = entry['video_id']
        if vid not in feat_index:
            print(f"Error: {vid} missing from features index.")
            return
        
        # Check num_frames
        if entry['num_frames'] != 8:
            print(f"Error: {vid} num_frames is {entry['num_frames']}, expected 8.")
            return

    # Check blob size math
    # Expected size = videos * 8 * 512 * 4
    expected_size = len(user_index) * 8 * 512 * 4
    if blob_size != expected_size:
        print(f"Error: Blob size mismatch! Found {blob_size}, expected {expected_size}.")
        return

    print("--- Verification Successful ---")
    print(f"Total Videos: {len(user_index)}")
    print(f"Blob Size: {blob_size / (1024*1024):.2f} MB")
    print(f"All Rule 2 & 3 checks passed.")

if __name__ == "__main__":
    verify()
