# scripts/eval_vqrf_v2.py
"""
VQRF v2 — three-path query routing
  ambiguous  → text expansion (synonym averaging)
  neutral    → confidence-gated weighted VQRF
  descriptive → pass-through (alpha ≈ 0)
"""

import torch, torch.nn.functional as F, numpy as np
import sqlite3, open_clip, faiss, math
import torch.nn as nn
from tqdm import tqdm
from collections import Counter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Constants ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.28   # only fuse when base recall is meaningful
ALPHA_MAX            = 0.30   # max fusion weight for neutral queries
TOP_K_CANDIDATES     = 5
SOFTMAX_TEMP         = 10.0   # sharpen centroid weights

# ── Model ─────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)

# ── Classifier (fixed) ────────────────────────────────────────────
# Object veto: if any recognizable content word is present, it's NOT ambiguous.
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
    'phone','laptop','computer','keyboard','camera','makeup','fashion',
    'news','weather','sports','workout','gym','yoga','dance','dancing',
    'singing','painting','drawing','drawing','carving','woodworking'
}

def classify_query(query: str) -> str:
    q = query.lower().strip()
    # Strip stopwords for measurement
    stops = {'a','an','the','is','are','was','were','be','been','being',
             'in','on','at','to','of','and','or','with','for','by','as', 'shown', 'shows'}
    words = [w.strip('.,!?:') for w in q.split()]
    content_words = [w for w in words if w not in stops]
    n = len(content_words)

    # Object veto: specific noun/action present → never truly ambiguous
    if any(w in SPECIFIC_OBJECTS for w in words):
        if n >= 4:
            return 'descriptive'
        return 'neutral'

    # Pure length heuristic on content words
    if n <= 2:   return 'ambiguous'
    if n >= 6:   return 'descriptive'
    return 'neutral'


# ── Text expansion (Path A — ambiguous) ───────────────────────────
EXPANSIONS = {
    'someone ':  ['a person ', 'a human ', 'an individual '],
    'a person ': ['someone ', 'a human '],
    'people ':   ['a group ', 'individuals '],
    'a man ':    ['a guy ', 'a person '],
    'a woman ':  ['a lady ', 'a person '],
    'men ':      ['guys ', 'people '],
    'women ':    ['ladies ', 'people '],
    'a boy ':    ['a young man ', 'a child '],
    'a girl ':   ['a young woman ', 'a child '],
    'kids ':     ['children ', 'young people '],
}

def generate_expansions(query: str) -> list[str]:
    q = query.lower().strip()
    exps = []
    for prefix, replacements in EXPANSIONS.items():
        if q.startswith(prefix):
            for rep in replacements:
                exps.append(q.replace(prefix, rep, 1))
    return exps

def embed_with_expansion(query: str, tokenizer, backbone, text_head) -> np.ndarray:
    queries = [query] + generate_expansions(query)
    tokens  = tokenizer(queries).to(DEVICE)
    with torch.no_grad():
        raw  = F.normalize(backbone.encode_text(tokens).float(), dim=-1)
        embs = text_head(raw)                # (N, 512)
    # Average embeddings to explore neighborhood
    mean = embs.mean(dim=0, keepdim=True)
    return F.normalize(mean, dim=-1).cpu().numpy().astype('float32')


# ── Confidence-gated weighted VQRF (Path B — neutral) ─────────────

def vqrf_refine(q_np: np.ndarray, index, test_embs: np.ndarray,
                alpha: float) -> np.ndarray:
    """
    Returns refined query embedding, or original if gating rejects.
    """
    D, I = index.search(q_np, TOP_K_CANDIDATES)
    mean_score = float(D[0].mean())

    if mean_score < CONFIDENCE_THRESHOLD:
        return q_np   # noisy retrieval — trust text, skip fusion

    valid_idx = [i for i in I[0] if i >= 0]
    if not valid_idx:
        return q_np

    v_top  = test_embs[valid_idx]                           # (K, 512)
    scores = torch.tensor(D[0][:len(valid_idx)], dtype=torch.float32)
    # Softmax sharpening for centroid
    w      = torch.softmax(scores * SOFTMAX_TEMP, dim=0).numpy()[:, None]
    v_cent = (w * v_top).sum(axis=0, keepdims=True)
    v_cent = v_cent / (np.linalg.norm(v_cent) + 1e-8)

    fused  = (1 - alpha) * q_np + alpha * v_cent
    return  fused / (np.linalg.norm(fused) + 1e-8)


def compute_alpha(query: str) -> float:
    """Decays with query length."""
    n = len(query.strip().split())
    return ALPHA_MAX * math.exp(-n / 7.0)


# ── Single-query baseline embedding ───────────────────────────────

def embed_query(query: str, tokenizer, backbone, text_head) -> np.ndarray:
    tokens = tokenizer([query]).to(DEVICE)
    with torch.no_grad():
        raw = F.normalize(backbone.encode_text(tokens).float(), dim=-1)
        emb = text_head(raw)
    q_np = emb.cpu().numpy().astype('float32')
    faiss.normalize_L2(q_np)
    return q_np


# ── Main eval ─────────────────────────────────────────────────────

def evaluate():
    print(f'Device: {DEVICE}')

    backbone, _, _ = open_clip.create_model_and_transforms(
        'MobileCLIP-S1', pretrained='datacompdr')
    backbone  = backbone.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')

    # Load frozen MSR-VTT model
    ckpt        = torch.load('checkpoints/best_model.pt',
                             map_location=DEVICE, weights_only=False)
    text_head   = ProjectionHead().to(DEVICE).eval()
    vision_head = ProjectionHead().to(DEVICE).eval()
    text_head.load_state_dict(ckpt['text_head'])
    vision_head.load_state_dict(ckpt['vision_head'])

    # ── Load data ─────────────────────────────────────────────────
    test_list_path = 'MSRVTT/MSRVTT/structured-symlinks/test_list_miech.txt'
    with open(test_list_path) as f:
        test_ids = set(l.strip() for l in f)

    conn       = sqlite3.connect('fedvcmr.db')
    all_caps   = conn.execute(
        'SELECT video_id, caption FROM captions').fetchall()
    all_chunks = conn.execute(
        'SELECT chunk_id, video_id FROM chunks ORDER BY chunk_id').fetchall()
    conn.close()

    test_caps = {}
    for vid, cap in all_caps:
        if vid in test_ids and vid not in test_caps:
            test_caps[vid] = cap

    # ── Audit distribution ────────────────────────────────────────
    dist = Counter(classify_query(c) for c in test_caps.values())
    print(f"\nQuery distribution (FIXED): {dict(dist)}")
    
    # Print sample ambiguous queries
    print("\nSAMPLE AMBIGUOUS QUERIES:")
    count = 0
    for vid, cap in test_caps.items():
        if classify_query(cap) == 'ambiguous':
            print(f"  - '{cap}'")
            count += 1
            if count >= 10: break

    # ── Build FAISS index ─────────────────────────────────────────
    idx_map    = {cid: i for i, (cid, _) in enumerate(all_chunks)}
    vid_chunks = {}
    for cid, vid in all_chunks:
        vid_chunks.setdefault(vid, []).append(cid)

    cache = np.memmap('cache/frame_features.bin', dtype='float16',
                      mode='r', shape=(len(all_chunks), 8, 512))
    w = np.array([0.5,0.75,1.0,1.25,1.25,1.0,0.75,0.5], dtype='float32')
    w = w / w.sum()

    test_vids = list(test_caps.keys())
    test_embs = []

    print(f'Building index for {len(test_vids)} videos...')
    with torch.no_grad():
        for vid in tqdm(test_vids):
            chunks = vid_chunks.get(vid, [])
            if not chunks:
                test_embs.append(np.zeros(512, dtype='float32'))
                continue
            ces = []
            for cid in chunks:
                frames = cache[idx_map[cid]].astype('float32')
                # Correct normalization from fix_anet_features logic
                fnorms = np.linalg.norm(frames, axis=1, keepdims=True) + 1e-8
                frames = frames / fnorms
                
                cemb   = (frames * w[:, None]).sum(0)
                cemb  /= np.linalg.norm(cemb) + 1e-8
                ces.append(cemb)
            avg  = np.stack(ces).mean(0)
            avg /= np.linalg.norm(avg) + 1e-8
            emb  = vision_head(
                torch.tensor(avg).unsqueeze(0).to(DEVICE)
            ).cpu().numpy()[0]
            test_embs.append(emb)

    test_embs = np.stack(test_embs).astype('float32')
    faiss.normalize_L2(test_embs)
    index = faiss.IndexFlatIP(512)
    index.add(test_embs)

    # ── Evaluate all three paths ───────────────────────────────────
    print('\n--- Evaluating VQRF v2 Strategy ---')
    all_ranks = {'baseline': [], 'vqrf_v2': []}

    buckets = {'ambiguous': {}, 'neutral': {}, 'descriptive': {}}
    for vid, cap in test_caps.items():
        buckets[classify_query(cap)][vid] = cap

    for bucket_name, queries in buckets.items():
        if not queries:
            continue
        b_ranks, v_ranks = [], []

        for vid, query in tqdm(queries.items(), desc=f'Evaluating {bucket_name}'):
            # 1. Baseline
            q_base = embed_query(query, tokenizer, backbone, text_head)
            _, base_I = index.search(q_base, 100)
            retrieved_base = [test_vids[j] for j in base_I[0] if j >= 0]
            b_ranks.append(retrieved_base.index(vid) + 1 if vid in retrieved_base else 101)

            # 2. VQRF v2 Path
            if bucket_name == 'ambiguous':
                refined = embed_with_expansion(query, tokenizer, backbone, text_head)
            elif bucket_name == 'neutral':
                alpha  = compute_alpha(query)
                refined = vqrf_refine(q_base, index, test_embs, alpha)
            else:
                refined = q_base # Pass-through

            _, ref_I = index.search(refined, 100)
            retrieved_ref = [test_vids[j] for j in ref_I[0] if j >= 0]
            v_ranks.append(retrieved_ref.index(vid) + 1 if vid in retrieved_ref else 101)

        n    = len(b_ranks)
        b_r1 = sum(r == 1 for r in b_ranks) / n * 100
        b_r5 = sum(r <= 5 for r in b_ranks) / n * 100
        v_r1 = sum(r == 1 for r in v_ranks) / n * 100
        v_r5 = sum(r <= 5 for r in v_ranks) / n * 100
        delta = v_r1 - b_r1

        print(f'\n{bucket_name.upper()} (n={n})')
        print(f'  Baseline  R@1: {b_r1:5.2f}%  R@5: {b_r5:5.2f}%')
        print(f'  VQRF v2   R@1: {v_r1:5.2f}%  R@5: {v_r5:5.2f}%  Delta={delta:+.2f}%')

        all_ranks['baseline'].extend(b_ranks)
        all_ranks['vqrf_v2'].extend(v_ranks)

    # ── Overall ───────────────────────────────────────────────────
    n    = len(all_ranks['baseline'])
    b_r1 = sum(r == 1 for r in all_ranks['baseline']) / n * 100
    v_r1 = sum(r == 1 for r in all_ranks['vqrf_v2'])  / n * 100
    print(f'\n{"="*45}')
    print(f'VQRF v2 FINAL SUMMARY (n={n})')
    print(f'{"="*45}')
    print(f'  Baseline R@1: {b_r1:.2f}%')
    print(f'  VQRF v2  R@1: {v_r1:.2f}%  Delta={v_r1-b_r1:+.2f}%')
    print(f'{"="*45}')

if __name__ == '__main__':
    evaluate()
