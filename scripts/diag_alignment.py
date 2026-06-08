#!/usr/bin/env python3
"""
Diagnose text-vision alignment on the FL projection heads.

Tests:
  A) Raw MobileCLIP backbone: text vs vision cosine (should be 0.15-0.35 for match)
  B) With text_head + vision_head: does it improve or destroy alignment?

Uses phone_payload features (Python-computed vision embeddings).
"""

import json, struct, sys
import numpy as np
import tensorflow as tf

BLOB   = "C:/prism/phone_payload/features_blob.bin"
IDX    = "C:/prism/phone_payload/features_index.json"
VIDIDX = "C:/prism/phone_payload/index.json"

TEXT_HEAD   = "C:/prism/android/app/src/main/assets/models/text_head_best_model.tflite"
VISION_HEAD = "C:/prism/android/app/src/main/assets/models/vision_head_best_model.tflite"

DIM = 512

# ── Load vision features from payload ────────────────────────────────────────
print("Loading phone_payload vision features...")
blob       = open(BLOB, "rb").read()
offsets    = json.load(open(IDX))
vid_index  = json.load(open(VIDIDX))

# Build map: video_id -> mean embedding (mean-pool frame embeddings)
vid_by_id = {}
BLOB_SIZE  = len(blob)
N_VIDS     = len(vid_index)
BYTES_PER  = BLOB_SIZE // N_VIDS
NF_DEFAULT = BYTES_PER // (DIM * 4)  # inferred from blob geometry

for entry in vid_index:
    vid   = entry["chunk_id"]
    nf    = NF_DEFAULT  # use inferred frame count, not entry field
    off   = offsets[vid]
    floats = struct.unpack_from(f"{nf*DIM}f", blob, off)
    frames = np.array(floats, dtype=np.float32).reshape(nf, DIM)
    # Mean-pool and L2-normalize
    mean = frames.mean(axis=0)
    mean /= (np.linalg.norm(mean) + 1e-10)
    vid_by_id[vid] = {"mean": mean, "frames": frames, "entry": entry}

print(f"  Loaded {len(vid_by_id)} videos")

# ── Load TFLite heads ─────────────────────────────────────────────────────────
def make_interp(path):
    interp = tf.lite.Interpreter(path)
    interp.allocate_tensors()
    return interp

def run_head(interp, vec512):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    buf = np.array(vec512, dtype=np.float32).reshape(inp["shape"])
    interp.set_tensor(inp["index"], buf)
    interp.invoke()
    return interp.get_tensor(out["index"]).flatten()

text_head_interp   = make_interp(TEXT_HEAD)
vision_head_interp = make_interp(VISION_HEAD)

print(f"  text_head   input:  {text_head_interp.get_input_details()[0]['shape']}")
print(f"  text_head   output: {text_head_interp.get_output_details()[0]['shape']}")
print(f"  vision_head input:  {vision_head_interp.get_input_details()[0]['shape']}")
print(f"  vision_head output: {vision_head_interp.get_output_details()[0]['shape']}")

# ── Load MobileCLIP text encoder ──────────────────────────────────────────────
print("\nLoading MobileCLIP-S1 text encoder...")
import open_clip, torch
model, _, _ = open_clip.create_model_and_transforms("MobileCLIP-S1", pretrained="datacompdr")
model.eval()
tokenizer = open_clip.get_tokenizer("MobileCLIP-S1")

def encode_text_raw(text):
    """Raw MobileCLIP text backbone (no projection head)."""
    with torch.no_grad():
        toks = tokenizer([text])
        feat = model.encode_text(toks, normalize=True)
    return feat[0].numpy()

def encode_text_with_head(text):
    raw = encode_text_raw(text)
    projected = run_head(text_head_interp, raw)
    norm = np.linalg.norm(projected)
    return projected / (norm + 1e-10)

def apply_vision_head(vec):
    projected = run_head(vision_head_interp, vec)
    norm = np.linalg.norm(projected)
    return projected / (norm + 1e-10)

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# ── Test queries ──────────────────────────────────────────────────────────────
# (query, expected_video_chunk_id)
TESTS = [
    ("woman sitting on stairs with a laptop", "REWLB_c0"),
    ("two people sitting on a sofa",          "4J1AP_c0"),
    ("person washing dishes at kitchen sink", "136V6_c0"),
    ("person in a bathroom",                  "REWLB_c0"),
]

print("\n" + "="*72)
print(f"{'Query':<40} {'Mode':<12} {'ExpectedVid':<14} {'Score':>6}  {'Rank':>5}")
print("="*72)

for query, expected_vid in TESTS:
    q_raw  = encode_text_raw(query)
    q_head = encode_text_with_head(query)

    for mode, q_vec, use_vision_head in [
        ("RAW",       q_raw,  False),
        ("BOTH_HEADS", q_head, True),
    ]:
        scores = {}
        for vid_id, vdata in vid_by_id.items():
            v_vec = apply_vision_head(vdata["mean"]) if use_vision_head else vdata["mean"]
            scores[vid_id] = cos(q_vec, v_vec)

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        exp_score = scores.get(expected_vid, 0.0)
        exp_rank  = next((i+1 for i, (v,_) in enumerate(ranked) if v == expected_vid), -1)
        top1_vid, top1_score = ranked[0]

        print(f"  {query[:38]:<40} {mode:<12} score={exp_score:.4f} rank={exp_rank:<4} | top1={top1_vid} ({top1_score:.4f})")

print("\n" + "="*72)
print("INTERPRETATION:")
print("  RAW mode: pure MobileCLIP backbone (no FL heads)")
print("  BOTH_HEADS: text_head + vision_head applied")
print("  If RAW >> BOTH_HEADS, the FL heads are destroying alignment.")
print("  If BOTH_HEADS >> RAW, heads are helping.")
