import torch
import cv2
import json
import numpy as np
import open_clip
import os
import struct

def get_pc_video_features(video_path, model, preprocess, n_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return None
    
    # Matches original VideoIngestor sliding window behavior (first 4s window)
    # Actually VideoIngestor does: tStart=0, tEnd=4
    # durationSec = total_frames / fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / fps
    
    t_start, t_end = 0.0, min(4.0, duration_sec)
    span = t_end - t_start
    
    indices = []
    for i in range(n_frames):
        frac = i / (float(n_frames) - 1)
        time_s = t_start + frac * span
        indices.append(int(time_s * fps))
        
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames - 1))
        ret, frame = cap.read()
        if not ret:
            if frames: frames.append(frames[-1])
            else: frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    
    # Preprocess
    from PIL import Image
    processed_frames = [preprocess(Image.fromarray(f)) for f in frames]
    input_tensor = torch.stack(processed_frames).to('cpu')
    
    with torch.no_grad():
        features = model.encode_image(input_tensor)
        features /= features.norm(dim=-1, keepdim=True)
    return features.numpy()

def get_phone_features(video_id, chunk_idx=0):
    with open('phone_payload/index.json', 'r') as f:
        index = json.load(f)
    
    chunk_id = f"{video_id}_c{chunk_idx}"
    offset = -1
    for entry in index:
        if entry.get('chunk_id') == chunk_id:
            # We need to find the offset in the binary blob
            # The index.json doesn't have offset, but features_index.json might
            pass
            
    # Let's check features_index.json
    if os.path.exists('phone_payload/features_index.json'):
        with open('phone_payload/features_index.json', 'r') as f:
            f_idx = json.load(f)
            if chunk_id in f_idx:
                offset = f_idx[chunk_id]
                
    if offset == -1:
        # Fallback: assume sequential? No, let's look for features_index.json
        return None

    with open('phone_payload/features_blob.bin', 'rb') as f:
        f.seek(offset)
        # 8 frames * 512 dims * 4 bytes
        data = f.read(8 * 512 * 4)
        features = np.frombuffer(data, dtype='<f4').reshape(8, 512)
    return features

def main():
    print("Loading MobileCLIP-S1...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'MobileCLIP-S1', pretrained='datacompdr'
    )
    model = model.to('cpu').eval()
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')

    # 1. Text Comparison
    query = "person drinking water"
    with torch.no_grad():
        tokens = tokenizer([query])
        pc_text_emb = model.encode_text(tokens)
        pc_text_emb /= pc_text_emb.norm(dim=-1, keepdim=True)
        pc_text_emb = pc_text_emb.numpy().flatten()
    
    print(f"\n--- Text Analysis: '{query}' ---")
    print(f"PC Embedding norm: {np.linalg.norm(pc_text_emb):.4f}")
    # Dump for manual check or just print first few values
    print(f"PC Embedding (first 5): {pc_text_emb[:5]}")
    
    # Check if there is a way to get phone-side text embedding.
    # Usually we can't easily unless it's in a log or a specific file.
    # But we can compare if the PC script's output matches what the phone expects.
    
    # 2. Vision Comparison (GFPDD)
    video_id = "GFPDD"
    video_path = f"Charades/Charades_v1_480/{video_id}.mp4"
    
    print(f"\n--- Vision Analysis: {video_id} ---")
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found")
        return

    pc_vision_features = get_pc_video_features(video_path, model, preprocess)
    phone_vision_features = get_phone_features(video_id)
    
    if phone_vision_features is not None:
        print("Phone features found.")
        for i in range(8):
            sim = np.dot(pc_vision_features[i], phone_vision_features[i])
            print(f"Frame {i} cosine similarity: {sim:.4f}")
            if sim < 0.90:
                print(f"  WARNING: High drift on frame {i}!")
    else:
        print("Phone features NOT found in blob.")

if __name__ == "__main__":
    main()
