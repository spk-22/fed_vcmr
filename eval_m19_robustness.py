"""
M19 — Query Robustness Analysis
=================================
Splits MSR-VTT 1K-A test queries into:
  - Descriptive : specific, detailed queries (>8 words OR contains specific nouns)
  - Ambiguous   : short, vague queries (<=8 words AND generic)

Runs retrieval on each category separately.
Reports R@1 gap between descriptive vs ambiguous.

Usage:
    python eval_m19_robustness.py
"""

import os, sys, sqlite3, time
import numpy as np
from pathlib import Path

sys.path.insert(0, "/content/fed_vcmr")
os.chdir("/content/fed_vcmr")

import src.config as cfg
import src.search as sm
cfg.DB_PATH    = Path("/content/data/fedvcmr.db")
cfg.INDEX_PATH = Path("/content/fed_vcmr/faiss_index.bin")
sm.DB_PATH     = Path("/content/data/fedvcmr.db")
sm.INDEX_PATH  = Path("/content/fed_vcmr/faiss_index.bin")

from src.backbone import MobileCLIPWrapper
from src.search import SearchIndex, maxsim_rerank
from src.query import QueryService

# ── Config ─────────────────────────────────────────────────────
DRIVE        = "/content/drive/.shortcut-targets-by-id/1SwgWGRg6WNkmN0Rlu0ppLltv6hE4yAFi/fedvcmr"
TEST_FILE    = f"{DRIVE}/msrvtt_miech_test.txt"
DB_PATH      = "/content/data/fedvcmr.db"
MEMMAP_PATH  = "/content/data/frame_features.bin"
MEMMAP_SHAPE = (16108, 8, 512)

# ── Specificity classifier ──────────────────────────────────────
SPECIFIC_NOUNS = {
    "guitar", "piano", "violin", "drum", "trumpet", "saxophone",
    "basketball", "football", "soccer", "tennis", "baseball", "golf",
    "dog", "cat", "horse", "elephant", "lion", "tiger", "bird",
    "car", "truck", "motorcycle", "bicycle", "boat", "airplane",
    "cooking", "swimming", "dancing", "singing", "running", "jumping",
    "wedding", "interview", "concert", "match", "race", "ceremony",
    "beach", "mountain", "forest", "street", "kitchen", "stage",
    "baby", "child", "woman", "man", "girl", "boy",
    "red", "blue", "green", "white", "black", "yellow",
}

def classify_query(query: str) -> str:
    """
    Returns 'descriptive' or 'ambiguous'.

    Descriptive if ANY of:
      - More than 8 words
      - Contains a specific noun from our list
      - Contains a colour word
      - Contains a number

    Ambiguous if:
      - 8 words or fewer AND no specific nouns
    """
    words  = query.lower().split()
    n_words = len(words)
    word_set = set(words)

    has_specific = bool(word_set & SPECIFIC_NOUNS)
    has_number   = any(w.isdigit() for w in words)

    if n_words > 8 or has_specific or has_number:
        return "descriptive"
    return "ambiguous"


# ── Load test data ──────────────────────────────────────────────
print("Loading test data...")
with open(TEST_FILE) as f:
    test_video_ids = [l.strip() for l in f.readlines()]

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get one caption per video (first caption = standard 1K-A eval protocol)
video_captions = {}
for vid in test_video_ids:
    row = cursor.execute(
        "SELECT caption FROM captions WHERE video_id=? LIMIT 1", (vid,)
    ).fetchone()
    if row:
        video_captions[vid] = row[0]

# Build chunk metadata
id2row   = {}
chunk_meta = {}
rows = cursor.execute(
    "SELECT chunk_id, video_id, t_start, t_end FROM chunks"
).fetchall()
for i, (chunk_id, video_id, t_start, t_end) in enumerate(rows):
    chunk_meta[chunk_id] = video_id
    if i < MEMMAP_SHAPE[0]:
        id2row[chunk_id] = i

conn.close()
print(f"  Test videos    : {len(video_captions)}")
print(f"  Chunks in meta : {len(chunk_meta)}")

# ── Load FAISS + backbone ───────────────────────────────────────
print("Loading retrieval components...")
import torch
backbone  = MobileCLIPWrapper(device="cuda")
query_svc = QueryService(backbone=backbone)

search_index = SearchIndex()
search_index.load_index()

memmap = np.memmap(MEMMAP_PATH, dtype="float32", mode="r",
                   shape=MEMMAP_SHAPE)
print("  Ready.")

# ── Retrieval function ──────────────────────────────────────────
def retrieve_video(query: str, top_k: int = 10) -> list:
    """Returns ranked list of video_ids for a query."""
    phrasings  = query_svc.expand_query(query)
    multi_emb  = query_svc.encode_queries(phrasings)
    single_emb = query_svc.encode_query(query)

    coarse = search_index.coarse_search(single_emb, top_k=100)
    if not coarse:
        return []

    # MaxSim using memmap
    scored = []
    for chunk_id, _ in coarse:
        row = id2row.get(chunk_id)
        if row is None:
            continue
        frames = memmap[row].astype(np.float32)
        sim    = multi_emb @ frames.T          # (3, 8)
        score  = float(np.mean(np.max(sim, axis=0)))
        scored.append((chunk_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate by video_id keeping best score
    seen     = {}
    for chunk_id, score in scored:
        vid = chunk_meta.get(chunk_id)
        if vid and vid not in seen:
            seen[vid] = score
        if len(seen) >= top_k:
            break

    return list(seen.keys())


# ── Run evaluation ──────────────────────────────────────────────
print("\nClassifying queries and running retrieval...")

descriptive_correct = []
ambiguous_correct   = []
descriptive_queries = []
ambiguous_queries   = []

for i, (video_id, caption) in enumerate(video_captions.items()):
    category = classify_query(caption)

    ranked = retrieve_video(caption, top_k=10)
    hit    = 1 if (ranked and ranked[0] == video_id) else 0

    if category == "descriptive":
        descriptive_correct.append(hit)
        descriptive_queries.append(caption)
    else:
        ambiguous_correct.append(hit)
        ambiguous_queries.append(caption)

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/1000 done...")


# ── Results ────────────────────────────────────────────────────
desc_r1 = np.mean(descriptive_correct) * 100 if descriptive_correct else 0
amb_r1  = np.mean(ambiguous_correct)   * 100 if ambiguous_correct   else 0
gap     = desc_r1 - amb_r1

print(f"\n{'='*55}")
print(f"M19 — Query Robustness Analysis")
print(f"{'='*55}")
print(f"  Total queries      : {len(video_captions)}")
print(f"  Descriptive        : {len(descriptive_correct)} queries")
print(f"  Ambiguous          : {len(ambiguous_correct)} queries")
print(f"{'─'*55}")
print(f"  R@1 Descriptive    : {desc_r1:.2f}%")
print(f"  R@1 Ambiguous      : {amb_r1:.2f}%")
print(f"  Gap (Desc - Amb)   : {gap:.2f}%")
print(f"{'='*55}")

print(f"\n--- Sample Descriptive Queries ---")
for q in descriptive_queries[:5]:
    print(f"  {q}")

print(f"\n--- Sample Ambiguous Queries ---")
for q in ambiguous_queries[:5]:
    print(f"  {q}")

print(f"\n--- Failure Mode Analysis ---")
print(f"  The {gap:.1f}% gap shows the model struggles with vague queries.")
print(f"  Descriptive queries with specific nouns/actions perform better")
print(f"  because MobileCLIP text encoder captures specific concepts well.")
print(f"  Ambiguous queries like 'someone doing something' lack discriminative")
print(f"  signal, causing retrieval to rely more on visual priors.")

# ── Save results ───────────────────────────────────────────────
results_text = f"""M19 — Query Robustness Analysis
================================
Total queries    : {len(video_captions)}
Descriptive      : {len(descriptive_correct)} queries
Ambiguous        : {len(ambiguous_correct)} queries

R@1 Descriptive  : {desc_r1:.2f}%
R@1 Ambiguous    : {amb_r1:.2f}%
Gap              : {gap:.2f}%
"""
with open("/content/fed_vcmr/results_m19_robustness.txt", "w") as f:
    f.write(results_text)
print("\nResults saved to results_m19_robustness.txt")
