# scripts/eval_maxsim_baseline.py
"""
MaxSim Retrieval: Scores videos by the maximum frame-to-query similarity.
This avoids 'Talking Head' dilution seen in average pooling.
"""
import torch, torch.nn.functional as F, numpy as np
import sqlite3, open_clip, faiss
import torch.nn as nn
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)

def evaluate():
    print(f'Device: {DEVICE}')
    backbone, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
    backbone = backbone.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')

    ckpt = torch.load('checkpoints/best_model.pt', map_location=DEVICE, weights_only=False)
    text_head = ProjectionHead().to(DEVICE).eval()
    vision_head = ProjectionHead().to(DEVICE).eval()
    text_head.load_state_dict(ckpt['text_head'])
    vision_head.load_state_dict(ckpt['vision_head'])

    # Data
    conn = sqlite3.connect('fedvcmr.db')
    test_ids = set(l.strip() for l in open('MSRVTT/MSRVTT/structured-symlinks/test_list_miech.txt'))
    all_caps = conn.execute('SELECT video_id, caption FROM captions').fetchall()
    all_chunks = conn.execute('SELECT chunk_id, video_id FROM chunks ORDER BY chunk_id').fetchall()
    conn.close()

    test_caps = {vid: cap for vid, cap in all_caps if vid in test_ids}
    test_vids = list(test_caps.keys())

    # Build Index Mapping
    idx_map = {cid: i for i, (cid, _) in enumerate(all_chunks)}
    vid_chunks = {}
    for cid, vid in all_chunks:
        vid_chunks.setdefault(vid, []).append(cid)
    
    cache = np.memmap('cache/frame_features.bin', dtype='float16', mode='r', shape=(len(all_chunks), 8, 512))

    print(f"Loading Frame Features for {len(test_vids)} videos...")
    video_frames = {} 
    with torch.no_grad():
        for vid in tqdm(test_vids):
            chunks = vid_chunks.get(vid, [])
            if not chunks: continue
            
            # Load frames and project them
            chunk_frames = cache[idx_map[chunks[0]]].astype('float32') # (8, 512)
            chunk_frames /= (np.linalg.norm(chunk_frames, axis=1, keepdims=True) + 1e-8)
            
            f_torch = torch.from_numpy(chunk_frames).to(DEVICE)
            f_proj  = vision_head(f_torch) # (8, 512)
            video_frames[vid] = f_proj

    # 4. Evaluation Loop
    print("\nEvaluating with MaxSim Strategy...")
    ranks = []
    
    for vid, query in tqdm(list(test_caps.items())):
        tokens = tokenizer([query]).to(DEVICE)
        with torch.no_grad():
            q_raw = F.normalize(backbone.encode_text(tokens).float(), dim=-1)
            q_emb = text_head(q_raw) # (1, 512)
        
        scores = []
        for c_vid in test_vids:
            if c_vid not in video_frames:
                scores.append(-1.0)
                continue
            
            f_proj = video_frames[c_vid] # (8, 512)
            # MaxSim: max of (f_proj @ q_emb.T)
            # Both are L2-normalized, so dot product is cosine similarity
            sims = torch.matmul(f_proj, q_emb.T) # (8, 1)
            max_sim = float(torch.max(sims))
            scores.append(max_sim)
        
        # Sort and Rank
        best_indices = np.argsort(scores)[::-1]
        retrieved_vids = [test_vids[i] for i in best_indices[:100]]
        ranks.append(retrieved_vids.index(vid) + 1 if vid in retrieved_vids else 101)

    r1 = sum(r == 1 for r in ranks) / len(ranks) * 100
    r5 = sum(r <= 5 for r in ranks) / len(ranks) * 100
    print(f"\nMaxSim Baseline Final Result: **{r1:.2f}%** R@1  |  **{r5:.2f}%** R@5")

if __name__ == '__main__':
    evaluate()
