# federated_learning/precompute_text.py
"""
Precomputes CLIP text features for all sharded training videos.
Reduces VRAM usage during multi-client simulation.
"""
import torch, sqlite3, open_clip, os
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    print(f"Device: {DEVICE}")
    model, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
    model = model.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')

    conn = sqlite3.connect('fedvcmr.db')
    # Get all training videos (those in fl_shards)
    vids = [r[0] for r in conn.execute('SELECT video_id FROM fl_shards').fetchall()]
    # Get one caption per video
    caps = {r[0]: r[1] for r in conn.execute('SELECT video_id, caption FROM captions WHERE video_id IN (SELECT video_id FROM fl_shards) GROUP BY video_id').fetchall()}
    conn.close()

    print(f"Precomputing features for {len(vids)} videos...")
    features = {}
    BATCH_SIZE = 128
    vid_list = list(caps.keys())

    with torch.no_grad():
        for i in tqdm(range(0, len(vid_list), BATCH_SIZE)):
            batch_vids = vid_list[i : i + BATCH_SIZE]
            batch_caps = [caps[v] for v in batch_vids]
            tokens = tokenizer(batch_caps).to(DEVICE)
            raw = torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1)
            for j, vid in enumerate(batch_vids):
                features[vid] = raw[j].cpu()

    torch.save(features, 'cache/fl_text_features.pt')
    print(f"Saved {len(features)} text features to cache/fl_text_features.pt")

if __name__ == "__main__":
    main()
