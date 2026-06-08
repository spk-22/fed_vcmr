# scripts/eval_aqe_dgse.py
"""
AQE-DGSE: Aggressive Query Expansion + DGSE (Dual-Granularity Segment Embedding).
Final attempt to push R@1 by de-crowding the text embedding space.
"""
import torch, torch.nn.functional as F, numpy as np
import sqlite3, open_clip, faiss, math
import torch.nn as nn
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)

# ── Aggressive Query Expansion (AQE) ────────────────────────────────
EXPANSIONS = {
    'someone ':  ['a person ', 'a human ', 'an individual ', 'a guy '],
    'a person ': ['someone ', 'a human ', 'an individual '],
    'people ':   ['a group ', 'a crowd ', 'individuals '],
    'a man ':    ['a guy ', 'a gentleman ', 'a male '],
    'a woman ':  ['a lady ', 'a female '],
    'a boy ':    ['a young man ', 'a kid '],
    'a girl ':   ['a young woman ', 'a kid '],
}

def expand_query(query: str, tokenizer, backbone, text_head):
    q = query.lower().strip()
    queries = [q]
    for p, rs in EXPANSIONS.items():
        if q.startswith(p):
            queries.extend([q.replace(p, r, 1) for r in rs])
            break # only replace first matching prefix
    
    tokens = tokenizer(queries).to(DEVICE)
    with torch.no_grad():
        raw = F.normalize(backbone.encode_text(tokens).float(), dim=-1)
        embs = text_head(raw)
        # Average embeddings
        q_emb = embs.mean(0, keepdim=True)
    return F.normalize(q_emb, dim=-1)

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

    # Build Mapping
    idx_map = {cid: i for i, (cid, _) in enumerate(all_chunks)}
    vid_chunks = {}
    for cid, vid in all_chunks:
        vid_chunks.setdefault(vid, []).append(cid)
    
    cache = np.memmap('cache/frame_features.bin', dtype='float16', mode='r', shape=(len(all_chunks), 8, 512))
    w = np.array([0.5,0.75,1.0,1.25,1.25,1.0,0.75,0.5], dtype='float32')
    w /= w.sum()

    print("Building M11 Visual Index (DGSE Baseline)...")
    test_embs = []
    with torch.no_grad():
        for vid in tqdm(test_vids):
            chunks = vid_chunks.get(vid, [])
            if not chunks:
                test_embs.append(np.zeros(512, dtype='float32'))
                continue
            ces = []
            for cid in chunks:
                frames = cache[idx_map[cid]].astype('float32')
                frames /= (np.linalg.norm(frames, axis=1, keepdims=True) + 1e-8)
                cemb = (frames * w[:, None]).sum(0)
                cemb /= np.linalg.norm(cemb) + 1e-8
                ces.append(cemb)
            # Use M11 Pooling (Mean of chunks + re-norm)
            avg = np.stack(ces).mean(0)
            avg /= np.linalg.norm(avg) + 1e-8
            emb = vision_head(torch.tensor(avg).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
            test_embs.append(emb)

    test_embs = np.stack(test_embs).astype('float32')
    faiss.normalize_L2(test_embs)
    index = faiss.IndexFlatIP(512)
    index.add(test_embs)

    print("\nEvaluating with AQE...")
    ranks = []
    for vid, query in tqdm(list(test_caps.items())):
        q_emb = expand_query(query, tokenizer, backbone, text_head)
        q_np = q_emb.cpu().numpy().astype('float32')
        
        _, I = index.search(q_np, 100)
        results = [test_vids[i] for i in I[0] if i >= 0]
        ranks.append(results.index(vid) + 1 if vid in results else 101)

    r1 = sum(r == 1 for r in ranks) / len(ranks) * 100
    print(f"\nAQE-DGSE Final Result: **{r1:.2f}%** R@1")

if __name__ == '__main__':
    evaluate()
