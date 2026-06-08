import os
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
import glob
import numpy as np

def verify():
    print("🚀 Initializing PC-Side Visual Auditor (using open_clip)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Matching src/config.py
    model_name = "MobileCLIP-S1"
    pretrained = "datacompdr"
    
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Try: pip install open-clip-torch")
        return

    model = model.to(device)
    model.eval()

    image_paths = glob.glob("pc_verification/**/*.jpg", recursive=True)
    if not image_paths:
        print("❌ No evidence frames found in pc_verification/")
        return

    results = []
    print(f"🧐 Auditing {len(image_paths)} frames on {device}...\n")
    print(f"{'Query Excerpt':<30} | {'Video ID':<15} | {'PC CLIP Score':<10}")
    print("-" * 65)

    for img_path in image_paths:
        # Extract query and video_id from filename
        base = os.path.basename(img_path).replace("evidence_", "").replace(".jpg", "")
        # The filename was evidence_query_excerpt_video_id.jpg
        # We need to find where the video_id starts (it has 'phone_')
        if "_phone_" in base:
            parts = base.split("_phone_")
            query_excerpt = parts[0].replace("_", " ")
            video_id = "phone_" + parts[1]
        else:
            query_excerpt = base
            video_id = "unknown"

        # 1. Process Image
        try:
            image_obj = Image.open(img_path).convert("RGB")
            image = preprocess(image_obj).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
        
        # 2. Process Text
        text = tokenizer([query_excerpt]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            # Normalize
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Cosine Similarity
            score = (image_features @ text_features.T).item()
        
        results.append((query_excerpt, video_id, score))
        status = "✅ PASS" if score > 0.25 else "⚠️ HUB?" if score > 0.15 else "❌ FAIL"
        print(f"{query_excerpt[:30]:<30} | {video_id:<15} | {score:.4f} {status}")

    if not results:
        print("No valid results to analyze.")
        return

    avg_score = sum(r[2] for r in results) / len(results)
    print("\n" + "="*65)
    print(f"FINAL AUDIT SCORE: {avg_score:.4f}")
    if avg_score > 0.24:
        print("RESULT: SYSTEM STABLE. Ranking is semantic and bias is suppressed.")
    elif avg_score > 0.18:
        print("RESULT: SYSTEM MARGINAL. Some hubness detected in ranking.")
    else:
        print("RESULT: SYSTEM NOISY. Significant ranking drift or normalization error.")

if __name__ == "__main__":
    verify()
