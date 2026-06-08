import cv2
import os
import glob

vids_dir = "pc_verification/user_videos/user_videos"
out_dir = "pc_verification/ground_truth"
os.makedirs(out_dir, exist_ok=True)

vids = glob.glob(os.path.join(vids_dir, "*.mp4"))
print(f"Processing {len(vids)} videos...")

for vpath in vids:
    vid_id = os.path.basename(vpath).replace(".mp4", "")
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"Failed to open {vid_id}")
        continue
    
    # Extract midpoint frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    if ret:
        out_path = os.path.join(out_dir, f"{vid_id}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"  ✓ {vid_id}")
    cap.release()

print("Done.")
