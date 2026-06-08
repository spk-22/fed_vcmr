# scripts/eval_mcr_vqrf.py
"""
MCR-VQRF: Mean-Centered Retrieval + VQRF v2.
Suppresses 'Black Hole' embeddings by centering the visual index.
"""
import torch, torch.nn.functional as F, numpy as np
import sqlite3, open_clip, faiss, math
import torch.nn as nn
from tqdm import tqdm
from collections import Counter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Constants ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.28
ALPHA_MAX            = 0.30
TOP_K_CANDIDATES     = 5
SOFTMAX_TEMP         = 10.0

# ── Model ─────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)

# ── Classifier ────────────────────────────────────────────────────
SPECIFIC_OBJECTS = {
    'paper','guitar','basketball','soccer','football','baseball','volleyball',
    'minecraft','roblox','fortnite','piano','violin','drums','microphone',
    'potato','pork','chicken','steak','pizza','sushi','noodle','noodles',
    'rubber','horse','dog','cat','car','bike','motorcycle','skateboard',
    'cartoon','commercial','trailer','tutorial','demonstration','review',
    'barbeque','barbecue','barbequeing','barbecuing','origami','craft',
    'cooking','swimming','climbing','surfing','skateboarding','wrestling',
    'fire','water','snow','mountain','ocean','forest','beach','stage',
    'interview','talk show','comedy','documentary','anime','animation',
    'phone','laptop','computer','keyboard','camera'
}

def classify_query(query: str) -> str:
    q = query.lower().strip()
    stops = {'a','an','the','is','are','was','were','be','been','being'}
    words = [w.strip('.,!?:') for w in q.split()]
    content_words = [w for w in words if w not in stops]
    n = len(content_words)
    if any(w in SPECIFIC_OBJECTS for w in words):
        return 'descriptive' if n >= 4 else 'neutral'
    if n <= 2: return 'ambiguous'
    if n >= 6: return 'descriptive'
    return 'neutral'

# ── Text expansion (Path A) ───────────────────────────────────────
EXPANSIONS = {
    'someone ':  ['a person ', 'a human '],
    'a person ': ['someone ', 'a human '],
    'people ':   ['a group ', 'individuals '],
    'a man ':    ['a guy ', 'a person '],
}

def embed_with_expansion(query: str, tokenizer, backbone, text_head) -> np.ndarray:
    queries = [query]
    for p, rs in EXPANSIONS.items():
        if query.lower().startswith(p):
            queries.extend([query.lower().replace(p, r, 1) for r in rs])
    tokens = tokenizer(queries).to(DEVICE)
    with torch.no_grad():
        raw = F.normalize(backbone.encode_text(tokens).float(), dim=-1)
        embs = text_head(raw).mean(dim=0, keepdim=True)
    return F.normalize(embs, dim=-1).cpu().numpy().astype('float32')

# ── Evaluator ─────────────────────────────────────────────────────
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

    # Load Data
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

    print("Building Global Index...")
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
            avg = np.stack(ces).mean(0)
            avg /= np.linalg.norm(avg) + 1e-8
            emb = vision_head(torch.tensor(avg).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
            test_embs.append(emb)

    original_test_embs = np.stack(test_embs).astype('float32')
    faiss.normalize_L2(original_test_embs)

    # ── MEAN CENTERED RETRIEVAL (MCR) ──────────────────────────────
    print("\n--- Applying Mean-Centered Optimization ---")
    global_mean = original_test_embs.mean(axis=0, keepdims=True)
    mcr_test_embs = original_test_embs - global_mean
    faiss.normalize_L2(mcr_test_embs)
    
    index = faiss.IndexFlatIP(512)
    index.add(mcr_test_embs)

    # ── Run Evaluation ─────────────────────────────────────────────
    print("\n--- Overall Audit ---")
    ranks = []
    
    for vid, query in tqdm(list(test_caps.items())):
        # Path decision
        bucket = classify_query(query)
        
        if bucket == 'ambiguous':
            q_np = embed_with_expansion(query, tokenizer, backbone, text_head)
        else:
            tokens = tokenizer([query]).to(DEVICE)
            with torch.no_grad():
                q_raw = F.normalize(backbone.encode_text(tokens).float(), dim=-1)
                q_emb = text_head(q_raw)
            q_np = q_emb.cpu().numpy().astype('float32')

        # Apply MCR to query
        q_np = q_np - global_mean
        faiss.normalize_L2(q_np)
        
        _, I = index.search(q_np, 100)
        results = [test_vids[i] for i in I[0] if i >= 0]
        ranks.append(results.index(vid) + 1 if vid in results else 101)

    r1 = sum(r == 1 for r in ranks) / len(ranks) * 100
    print(f"\nMCR-VQRF Final Result: **{r1:.2f}%** R@1")

if __name__ == '__main__':
    evaluate()
