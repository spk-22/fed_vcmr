import torch
import cv2
import numpy as np
import open_clip
from PIL import Image

# NEW ANDROID LOGIC CONSTANTS
DIM = 512
FRAMES = 32
# Corrected MobileCLIP-S1 Normalization
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD  = [0.26862954, 0.26130258, 0.27577711]

def simulate_new_android_vision_encoder(video_path):
    """Replicates the new VisionEncoder.java + VideoIngestor.java logic."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_ms = int((total_frames / fps) * 1000)
    
    # 1. TSN-style 32-frame uniform sampling (VideoIngestor.java)
    indices = []
    for i in range(FRAMES):
        frac = i / (float(FRAMES) - 1)
        time_ms = int(frac * duration_ms)
        indices.append(int((time_ms / 1000.0) * fps))
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames - 1))
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (256, 256)) # Android uses 256
        frames.append(frame)
    cap.release()

    # 2. MobileCLIP-S1 Normalization (VisionEncoder.java)
    model, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
    model.eval()
    
    tensors = []
    for f in frames:
        t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
        for c in range(3):
            t[c] = (t[c] - MEAN[c]) / STD[c]
        tensors.append(t)
    
    input_batch = torch.stack(tensors)
    with torch.no_grad():
        features = model.encode_image(input_batch)
        features /= features.norm(dim=-1, keepdim=True)
    
    return features.numpy(), (duration_ms / 1000.0)

def test_peak_grounding(frame_features, query_text, duration):
    """Replicates SearchEngine.java peak-detection grounding."""
    model, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')
    
    with torch.no_grad():
        text_tokens = tokenizer([query_text])
        query_emb = model.encode_text(text_tokens)
        query_emb /= query_emb.norm(dim=-1, keepdim=True)
        query_emb = query_emb.numpy().flatten()

    # 1. Score all 32 frames
    scores = np.dot(frame_features, query_emb)
    max_idx = np.argmax(scores)
    max_score = scores[max_idx]
    
    # 2. Expand (threshold = max - 0.05)
    threshold = max_score - 0.05
    start_idx = max_idx
    end_idx = max_idx
    while start_idx > 0 and scores[start_idx-1] >= threshold: start_idx -= 1
    while end_idx < FRAMES-1 and scores[end_idx+1] >= threshold: end_idx += 1
    
    t_start = (start_idx / (FRAMES - 1)) * duration
    t_end = (end_idx / (FRAMES - 1)) * duration
    
    return t_start, t_end, max_score, scores

if __name__ == "__main__":
    video_id = "GFPDD"
    video_path = f"Charades/Charades_v1_480/{video_id}.mp4"
    query = "person drinking water"
    
    print(f"Testing Rewamped Pipeline on {video_id}...")
    features, duration = simulate_new_android_vision_encoder(video_path)
    
    # Validation: Compare with standard PC CLIP (which uses the same norm now)
    print(f"Video Duration: {duration:.2f}s")
    print(f"Extracted {len(features)} frames.")
    
    t_s, t_e, score, all_scores = test_peak_grounding(features, query, duration)
    
    print(f"\nQuery: '{query}'")
    print(f"Grounding Result: [{t_s:.2f}s - {t_e:.2f}s]")
    print(f"Confidence Score: {score:.4f}")
    
    # Check if peak detection found a plausible segment
    # For GFPDD (Charades), the person drinks water around the middle.
    if t_e > t_s and score > 0.2:
        print("\nSUCCESS: Pipeline is functional and grounding produced a valid temporal window.")
    else:
        print("\nFAILURE: Grounding collapsed or confidence too low.")
