import torch
import cv2
import json
import numpy as np
import open_clip
import os
from tqdm import tqdm
from PIL import Image

# Config
DEVICE = 'cpu'
VIDEO_DIR = 'ActivityNet/videos'
CACHE_DIR = 'cache/anet_benchmark_features'
VAL_JSON = 'ActivityNet/val_1.json'
TARGET_VIDS = 733
FRAMES = 8
IMAGE_SIZE = 256 # Correct size for MobileCLIP-S1

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def main():
    print(f"Loading MobileCLIP-S1 on {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
    model = model.to(DEVICE).eval()
    
    # Adjust preprocess if necessary (though datacompdr usually handles resize)
    # The official MobileCLIP-S1 uses 256x256.
    
    with open(VAL_JSON, 'r') as f:
        data = json.load(f)
    
    existing_vids = set([f[:-4] for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])
    all_ids = sorted(data.keys())
    
    selected_vids = []
    for vid in all_ids:
        if vid in existing_vids:
            selected_vids.append(vid)
            if len(selected_vids) >= TARGET_VIDS:
                break
    
    print(f"Extracting features for {len(selected_vids)} videos...")
    
    for vid in tqdm(selected_vids):
        video_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
        feat_path = os.path.join(CACHE_DIR, f"{vid}.npy")
        
        if os.path.exists(feat_path):
            continue
            
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            print(f"Warning: {vid} has 0 frames.")
            continue
        
        indices = np.linspace(0, total - 1, FRAMES).astype(int)
        batch_tensors = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                batch_tensors.append(batch_tensors[-1] if batch_tensors else torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            # Resize manually to ensure 256x256 as per rule 2
            pil_img = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
            batch_tensors.append(preprocess(pil_img))
        cap.release()
        
        input_tensor = torch.stack(batch_tensors).to(DEVICE)
        
        with torch.no_grad():
            features = model.encode_image(input_tensor).float()
            # Normalize frame-wise as the app will normalize results
            # but we want raw features to be accurate
            features = features.cpu().numpy()
            
        np.save(feat_path, features)

    print(f"Extraction complete. Features saved in {CACHE_DIR}")

if __name__ == "__main__":
    main()
