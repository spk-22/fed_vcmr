# scripts/demo_robustness.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sqlite3
import open_clip
import faiss
import os
import sys
from tqdm import tqdm

sys.path.insert(0, '.')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Model Classes ────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
    def forward(self, x): return F.normalize(self.linear(x), dim=-1)

# ── Query Classifier (copied from M19) ───────────────────────────
def classify_query(query):
    q = query.lower().strip()
    words = q.split()
    n = len(words)
    score = 0
    if n >= 10: score += 3
    elif n >= 7: score += 2
    elif n >= 5: score += 1
    elif n <= 4: score -= 2
    colours = {'red','blue','green','black','white','yellow','orange','purple','pink','brown','grey','gray'}
    if any(w in colours for w in words): score += 2
    locations = {'court','kitchen','street','park','beach','field','room','stage','office','gym','pool','road','indoor','outdoor','inside','outside','water','grass','sand','snow','mountain','forest'}
    if any(w in locations for w in words): score += 2
    generic_subjects = {'someone','somebody','people','person','man','woman','child','kid','guy','girl','they','he','she'}
    specific_count = sum(1 for w in words if w not in generic_subjects)
    if specific_count >= 4: score += 2
    elif specific_count <= 2: score -= 1
    specific_verbs = {'dribbling','cooking','swimming','climbing','dancing','singing','cutting','throwing','jumping','running','eating','drinking','riding','driving','typing','painting','laughing','crying','walking','sitting','standing','lifting','pushing','pulling'}
    if any(w in specific_verbs for w in words): score += 2
    if score >= 3: return 'DESCRIPTIVE'
    elif score <= 1: return 'AMBIGUOUS'
    else: return 'NEUTRAL'

# ── Setup ────────────────────────────────────────────────────────
print("Loading models and index...")
backbone, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
backbone = backbone.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')

ckpt = torch.load('checkpoints/best_model.pt', map_location=DEVICE)
text_head = ProjectionHead().to(DEVICE).eval()
text_head.load_state_dict(ckpt['text_head'])
vision_head = ProjectionHead().to(DEVICE).eval()
vision_head.load_state_dict(ckpt['vision_head'])

conn = sqlite3.connect('fedvcmr.db')
all_chunks = conn.execute('SELECT chunk_id, video_id FROM chunks ORDER BY chunk_id').fetchall()
# Load ONE caption per video for the whole test set to ensure we have labels
all_caps = {}
print("Loading captions...")
cursor = conn.execute('SELECT video_id, caption FROM captions')
for vid, cap in cursor:
    if vid not in all_caps:
        all_caps[vid] = cap
conn.close()

idx_map = {cid: i for i, (cid, _) in enumerate(all_chunks)}
vid_chunks = {}
for cid, vid in all_chunks:
    vid_chunks.setdefault(vid, []).append(cid)

n_total = len(all_chunks)
cache = np.memmap('cache/frame_features.bin', dtype='float16', mode='r', shape=(n_total, 8, 512))
w = np.array([0.5,0.75,1.0,1.25,1.25,1.0,0.75,0.5], dtype='float32')
w /= w.sum()

test_vids = sorted(list(vid_chunks.keys()))[:1000] # Use the 1k set from eval
test_embs = []
with torch.no_grad():
    for vid in tqdm(test_vids, desc="Building Index"):
        chunks = vid_chunks.get(vid, [])
        ces = []
        for cid in chunks:
            frames = cache[idx_map[cid]].astype('float32')
            cemb = (frames * w[:, None]).sum(0)
            cemb /= (np.linalg.norm(cemb) + 1e-8)
            ces.append(cemb)
        avg = np.stack(ces).mean(0)
        avg /= (np.linalg.norm(avg) + 1e-8)
        emb = vision_head(torch.tensor(avg).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
        test_embs.append(emb)

test_embs = np.stack(test_embs).astype('float32')
faiss.normalize_L2(test_embs)
index = faiss.IndexFlatIP(512)
index.add(test_embs)

print("\n" + "="*50)
print("FedVCMR ROBUSTNESS DEMO")
print("="*50)
print("Type a query to see results.")
print("Example Ambiguous: 'a person cooking'")
print("Example Descriptive: 'a woman in a white coat is cooking soup in a kitchen'")
print("Type 'exit' to quit.")

while True:
    query = input("\nYour Query: ").strip()
    if query.lower() in ('exit', 'quit'): break
    if not query: continue

    cat = classify_query(query)
    print(f"-> Classification: {cat}")

    with torch.no_grad():
        tok = tokenizer([query]).to(DEVICE)
        q_emb = F.normalize(backbone.encode_text(tok).float(), dim=-1)
        q_emb = text_head(q_emb).cpu().numpy().astype('float32')
    
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, 3)

    print("\nTop 3 Matches:")
    for i in range(3):
        vid_idx = indices[0][i]
        score = scores[0][i]
        vid_id = test_vids[vid_idx]
        caption = all_caps.get(vid_id, "No caption found")
        print(f"  {i+1}. [{vid_id}] (Score: {score:.3f})")
        print(f"     Caption: {caption}")

print("\nDemo ended.")
