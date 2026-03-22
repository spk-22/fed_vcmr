# scripts/eval_m19_query_robustness.py
"""
M19: Query Robustness Analysis
Split test queries into Descriptive vs Ambiguous categories
and measure R@1/5/10 gap to quantify model robustness.
"""
import torch
import torch.nn.functional as F
import numpy as np
import sqlite3
import open_clip
import faiss
import torch.nn as nn
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ── Projection Head ───────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)

# ── Load models ───────────────────────────────────────────────────
backbone, _, _ = open_clip.create_model_and_transforms(
    'MobileCLIP-S1', pretrained='datacompdr'
)
backbone  = backbone.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')
for p in backbone.parameters():
    p.requires_grad = False

ckpt        = torch.load('checkpoints/best_model.pt', map_location=DEVICE)
text_head   = ProjectionHead().to(DEVICE).eval()
vision_head = ProjectionHead().to(DEVICE).eval()
text_head.load_state_dict(ckpt['text_head'])
vision_head.load_state_dict(ckpt['vision_head'])

# ── Query classifier ──────────────────────────────────────────────
def classify_query(query):
    """
    Classify a query as descriptive or ambiguous based on
    lexical signals: length, specificity, colour/location words.

    Returns: 'descriptive', 'ambiguous', or 'neutral'
    """
    q     = query.lower().strip()
    words = q.split()
    n     = len(words)

    # Scoring: positive = descriptive, negative = ambiguous
    score = 0

    # Length signal
    if n >= 10: score += 3
    elif n >= 7: score += 2
    elif n >= 5: score += 1
    elif n <= 4: score -= 2

    # Colour words
    colours = {'red','blue','green','black','white','yellow',
               'orange','purple','pink','brown','grey','gray'}
    if any(w in colours for w in words):
        score += 2

    # Location/setting words
    locations = {'court','kitchen','street','park','beach','field',
                 'room','stage','office','gym','pool','road',
                 'indoor','outdoor','inside','outside','water',
                 'grass','sand','snow','mountain','forest'}
    if any(w in locations for w in words):
        score += 2

    # Generic subjects (ambiguous signal)
    generic_subjects = {'someone','somebody','people','person',
                        'man','woman','child','kid','guy','girl',
                        'they','he','she'}
    specific_count = sum(1 for w in words if w not in generic_subjects)
    if specific_count >= 4:
        score += 2
    elif specific_count <= 2:
        score -= 1

    # Specific vs generic verbs
    generic_verbs = {'doing','playing','going','making','having',
                     'using','getting','taking','coming','looking'}
    specific_verbs = {'dribbling','cooking','swimming','climbing',
                      'dancing','singing','cutting','throwing',
                      'jumping','running','eating','drinking',
                      'riding','driving','typing','painting',
                      'laughing','crying','walking','sitting',
                      'standing','lifting','pushing','pulling'}
    if any(w in specific_verbs for w in words):
        score += 2
    if any(w in generic_verbs for w in words):
        score -= 1

    # Adjectives / descriptors
    descriptors = {'tall','short','large','small','big','little',
                   'young','old','fast','slow','bright','dark',
                   'long','wide','narrow','high','low','heavy'}
    if any(w in descriptors for w in words):
        score += 1

    # Classify
    if score >= 3:
        return 'descriptive'
    elif score <= 1:
        return 'ambiguous'
    else:
        return 'neutral'

# ── Load test data ────────────────────────────────────────────────
with open('MSRVTT/MSRVTT/structured-symlinks/test_list_miech.txt') as f:
    test_ids = set(l.strip() for l in f)

conn     = sqlite3.connect('fedvcmr.db')
all_caps = conn.execute(
    'SELECT video_id, caption FROM captions'
).fetchall()
all_chunks = conn.execute(
    'SELECT chunk_id, video_id FROM chunks ORDER BY chunk_id'
).fetchall()
conn.close()

# One caption per test video
test_caps = {}
for vid, cap in all_caps:
    if vid in test_ids and vid not in test_caps:
        test_caps[vid] = cap

print(f'Test videos: {len(test_caps)}')

# Classify all queries
descriptive = {}
ambiguous   = {}
neutral     = {}

for vid, cap in test_caps.items():
    cat = classify_query(cap)
    if cat == 'descriptive':
        descriptive[vid] = cap
    elif cat == 'ambiguous':
        ambiguous[vid] = cap
    else:
        neutral[vid] = cap

print(f'\nQuery classification:')
print(f'  Descriptive: {len(descriptive)}')
print(f'  Ambiguous:   {len(ambiguous)}')
print(f'  Neutral:     {len(neutral)}')
print()

# Show examples
print('Sample DESCRIPTIVE queries:')
for vid, cap in list(descriptive.items())[:5]:
    print(f'  [{len(cap.split()):2d} words] {cap}')

print('\nSample AMBIGUOUS queries:')
for vid, cap in list(ambiguous.items())[:5]:
    print(f'  [{len(cap.split()):2d} words] {cap}')

# ── Build video index ─────────────────────────────────────────────
idx_map    = {cid: i for i, (cid, _) in enumerate(all_chunks)}
vid_chunks = {}
for cid, vid in all_chunks:
    vid_chunks.setdefault(vid, []).append(cid)

n_total = len(all_chunks)
cache   = np.memmap('cache/frame_features.bin',
                    dtype='float16', mode='r',
                    shape=(n_total, 8, 512))

w = np.array([0.5,0.75,1.0,1.25,1.25,1.0,0.75,0.5], dtype='float32')
w = w / w.sum()

test_vids = list(test_caps.keys())
print(f'\nBuilding FAISS index for {len(test_vids)} videos...')
test_embs = []

with torch.no_grad():
    for vid in tqdm(test_vids, desc="Building Index"):
        chunks = vid_chunks.get(vid, [])
        if not chunks:
            test_embs.append(np.zeros(512, dtype='float32'))
            continue
        ces = []
        for cid in chunks:
            frames = cache[idx_map[cid]].astype('float32')
            cemb   = (frames * w[:, None]).sum(0)
            cemb   = cemb / (np.linalg.norm(cemb) + 1e-8)
            ces.append(cemb)
        avg  = np.stack(ces).mean(0)
        avg  = avg / (np.linalg.norm(avg) + 1e-8)
        emb  = vision_head(
            torch.tensor(avg).unsqueeze(0).to(DEVICE)
        ).cpu().numpy()[0]
        test_embs.append(emb)

test_embs = np.stack(test_embs).astype('float32')
faiss.normalize_L2(test_embs)
index = faiss.IndexFlatIP(512)
index.add(test_embs)

# ── Evaluation function ───────────────────────────────────────────
def evaluate_subset(query_dict, label):
    if not query_dict:
        return None

    vids    = list(query_dict.keys())
    queries = list(query_dict.values())
    q_embs  = []
    BATCH   = 128

    with torch.no_grad():
        for i in range(0, len(queries), BATCH):
            tokens = tokenizer(queries[i:i+BATCH]).to(DEVICE)
            embs   = F.normalize(backbone.encode_text(tokens).float(), dim=-1)
            embs   = text_head(embs).cpu()
            q_embs.append(embs)

    q_arr = torch.cat(q_embs).numpy().astype('float32')
    faiss.normalize_L2(q_arr)
    _, I  = index.search(q_arr, 10)

    ranks = []
    for i, vid in enumerate(vids):
        if vid not in test_vids:
            continue
        retrieved = [test_vids[j] for j in I[i] if j >= 0]
        rank = retrieved.index(vid) + 1 \
               if vid in retrieved else 11
        ranks.append(rank)

    n    = len(ranks)
    r1   = sum(r == 1 for r in ranks) / n * 100
    r5   = sum(r <= 5 for r in ranks) / n * 100
    r10  = sum(r <= 10 for r in ranks) / n * 100
    mdr  = sorted(ranks)[n // 2]

    # Word length stats
    lengths = [len(q.split()) for q in queries]
    avg_len = np.mean(lengths)

    print(f'\n{label} ({n} queries | avg {avg_len:.1f} words):')
    print(f'  R@1:  {r1:.2f}%')
    print(f'  R@5:  {r5:.2f}%')
    print(f'  R@10: {r10:.2f}%')
    print(f'  MdR:  {mdr}')

    return {'r1': r1, 'r5': r5, 'r10': r10, 'mdr': mdr,
            'n': n, 'avg_len': avg_len}

# ── Run evaluation ────────────────────────────────────────────────
print(f'\n{"="*55}')
print(f'M19 QUERY ROBUSTNESS ANALYSIS')
print(f'{"="*55}')

all_results  = evaluate_subset(test_caps,    'ALL QUERIES    ')
desc_results = evaluate_subset(descriptive,  'DESCRIPTIVE    ')
amb_results  = evaluate_subset(ambiguous,    'AMBIGUOUS      ')
neut_results = evaluate_subset(neutral,      'NEUTRAL        ')

# ── Summary table ─────────────────────────────────────────────────
print(f'\n{"="*55}')
print(f'SUMMARY TABLE')
print(f'{"="*55}')
print(f'{"Category":<16} {"N":>5} {"AvgLen":>7} '
      f'{"R@1":>7} {"R@5":>7} {"R@10":>7}')
print(f'{"-"*55}')

for label, res in [
    ('All',         all_results),
    ('Descriptive', desc_results),
    ('Ambiguous',   amb_results),
    ('Neutral',     neut_results),
]:
    if res:
        print(f'{label:<16} {res["n"]:>5} {res["avg_len"]:>7.1f} '
              f'{res["r1"]:>7.2f} {res["r5"]:>7.2f} '
              f'{res["r10"]:>7.2f}')

if desc_results and amb_results:
    gap = desc_results['r1'] - amb_results['r1']
    print(f'\nRobustness Gap (Descriptive - Ambiguous): {gap:+.2f}% R@1')
    print()
    if gap > 10:
        print('Model significantly benefits from detailed queries')
        print('-> Query expansion could improve real-world performance')
    elif gap > 5:
        print('Moderate gap -- model handles ambiguity reasonably')
    else:
        print('Small gap -- model is robust to query ambiguity')

print(f'{"="*55}')
