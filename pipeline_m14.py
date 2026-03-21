"""
M14 — End-to-End Inference Pipeline
=====================================
Input  : query string
Output : video_id, start_sec, end_sec, confidence

Chain  : FAISS (coarse) → MaxSim (rerank) → DGSE (rerank) → Transformer (grounding)

Usage
-----
    # From another script
    from pipeline_m14 import VCMRPipeline
    pipe = VCMRPipeline()
    result = pipe.infer("a person is cooking food on a stove")
    print(result)

    # From the command line
    python pipeline_m14.py --query "someone playing guitar"
    python pipeline_m14.py --query "a dog running in a park" --top_k 3
"""

import time
import argparse
import sqlite3
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ── Project root on path so src.* imports resolve ─────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.backbone import MobileCLIPWrapper
from src.config import INDEX_PATH, DB_PATH, DEVICE
from src.query import QueryService
from src.search import SearchIndex, maxsim_rerank
from src.models.dgse import DGSE
from src.models.temporal_grounding import CrossModalTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR   = Path("checkpoints")
DGSE_CKPT        = CHECKPOINT_DIR / "dgse_best.pt"
TRANSFORMER_CKPT = CHECKPOINT_DIR / "temporal_grounding_best.pt"


# ── Return type ────────────────────────────────────────────────────────────────
@dataclass
class RetrievalResult:
    video_id   : str
    start_sec  : float
    end_sec    : float
    confidence : float          # cosine similarity from DGSE stage
    query      : str

    def __str__(self):
        return (
            f"video_id   : {self.video_id}\n"
            f"start_sec  : {self.start_sec:.2f}s\n"
            f"end_sec    : {self.end_sec:.2f}s\n"
            f"confidence : {self.confidence:.4f}\n"
            f"query      : {self.query}"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────
def _load_chunk_metadata(db_path: str) -> dict:
    """
    Returns a dict mapping chunk_id → (video_id, start_sec, end_sec, cache_path).
    Tries common column names; adjust the SELECT if your schema differs.
    """
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Detect actual column names
    cols = [c[1] for c in cursor.execute("PRAGMA table_info(chunks)").fetchall()]

    # Flexible column matching
    vid_col   = next((c for c in cols if "video" in c.lower()), None)
    start_col = next((c for c in cols if "start" in c.lower()), None)
    end_col   = next((c for c in cols if "end"   in c.lower()), None)
    path_col  = next((c for c in cols if "cache" in c.lower() or "path" in c.lower()), None)

    if not all([vid_col, start_col, end_col, path_col]):
        conn.close()
        raise ValueError(
            f"Could not find required columns in 'chunks' table.\n"
            f"Found columns: {cols}\n"
            f"Expected something like: video_id, start_sec, end_sec, cache_path"
        )

    rows = cursor.execute(
        f"SELECT chunk_id, {vid_col}, {start_col}, {end_col}, {path_col} FROM chunks"
    ).fetchall()
    conn.close()

    return {row[0]: (row[1], float(row[2]), float(row[3]), row[4]) for row in rows}


def _frames_for_chunk(cache_path: str, device: torch.device) -> torch.Tensor:
    """Load (8, 512) frame features from .npy file → (1, 8, 512) tensor."""
    feat = np.load(cache_path).astype(np.float32)   # (8, 512)
    return torch.from_numpy(feat).unsqueeze(0).to(device)  # (1, 8, 512)


# ── Main pipeline class ────────────────────────────────────────────────────────
class VCMRPipeline:
    """
    Loads all components once at init; call infer() for each query.
    All heavy models stay on device — subsequent queries are fast.
    """

    def __init__(
        self,
        faiss_candidates : int = 100,   # FAISS coarse retrieval pool
        maxsim_top_k     : int = 20,    # after MaxSim rerank
        dgse_top_k       : int = 5,     # after DGSE rerank
        device           : str = DEVICE,
    ):
        self.faiss_candidates = faiss_candidates
        self.maxsim_top_k     = maxsim_top_k
        self.dgse_top_k       = dgse_top_k
        self.device           = torch.device(device)

        print(f"[M14] Loading pipeline on {device} ...")
        t0 = time.time()

        # 1. Backbone (MobileCLIP-S1)
        self.backbone = MobileCLIPWrapper(device=device)
        self.query_svc = QueryService(backbone=self.backbone)

        # 2. FAISS index
        self.search_index = SearchIndex()
        self.search_index.load_index()

        # 3. SQLite metadata  { chunk_id → (video_id, start, end, cache_path) }
        self.chunk_meta = _load_chunk_metadata(str(DB_PATH))

        # 4. DGSE model
        self.dgse = DGSE(dim=512, n_frames=8).to(self.device)
        if DGSE_CKPT.exists():
            state = torch.load(DGSE_CKPT, map_location=self.device, weights_only=False)
            # Handle checkpoints saved as full dict or just state_dict
            sd = state.get("model_state_dict", state.get("state_dict", state))
            self.dgse.load_state_dict(sd, strict=False)
            print(f"  [✓] DGSE loaded from {DGSE_CKPT}")
        else:
            print(f"  [!] DGSE checkpoint not found at {DGSE_CKPT} — using random weights")
        self.dgse.eval()

        # 5. CrossModalTransformer
        self.transformer = CrossModalTransformer(
            visual_dim=512, query_dim=512, hidden_dim=256, n_heads=4, n_layers=2
        ).to(self.device)
        if TRANSFORMER_CKPT.exists():
            state = torch.load(TRANSFORMER_CKPT, map_location=self.device, weights_only=False)
            sd = state.get("model_state_dict", state.get("state_dict", state))
            self.transformer.load_state_dict(sd, strict=False)
            print(f"  [✓] Transformer loaded from {TRANSFORMER_CKPT}")
        else:
            print(f"  [!] Transformer checkpoint not found at {TRANSFORMER_CKPT} — using random weights")
        self.transformer.eval()

        elapsed = time.time() - t0
        print(f"[M14] Pipeline ready in {elapsed:.2f}s\n")

    # ── Single query inference ─────────────────────────────────────────────────
    @torch.no_grad()
    def infer(self, query: str, return_top_k: int = 1) -> List[RetrievalResult]:
        """
        Full pipeline: query → video_id, start_sec, end_sec, confidence.

        Parameters
        ----------
        query        : natural language query string
        return_top_k : how many ranked results to return (default 1)

        Returns
        -------
        List[RetrievalResult] sorted by confidence descending
        """
        timings = {}

        # ── Stage 1: Query encoding ──────────────────────────────────────────
        t = time.time()
        phrasings  = self.query_svc.expand_query(query)          # 3 phrasings
        multi_emb  = self.query_svc.encode_queries(phrasings)    # (3, 512) numpy
        single_emb = self.query_svc.encode_query(query)          # (1, 512) numpy
        timings["encode"] = time.time() - t

        # ── Stage 2: FAISS coarse retrieval ──────────────────────────────────
        t = time.time()
        coarse_hits = self.search_index.coarse_search(
            single_emb, top_k=self.faiss_candidates
        )                                                         # [(chunk_id, score)]
        timings["faiss"] = time.time() - t

        if not coarse_hits:
            return []

        # ── Stage 3: MaxSim rerank ────────────────────────────────────────────
        t = time.time()
        maxsim_hits = maxsim_rerank(
            multi_emb, coarse_hits, top_k=self.maxsim_top_k
        )                                                         # [(chunk_id, score)]
        timings["maxsim"] = time.time() - t

        # ── Stage 4: DGSE rerank ──────────────────────────────────────────────
        t = time.time()
        query_tensor = torch.from_numpy(single_emb).to(self.device)  # (1, 512)

        dgse_hits = []
        for chunk_id, _ in maxsim_hits:
            meta = self.chunk_meta.get(chunk_id)
            if meta is None:
                continue
            video_id, start_sec, end_sec, cache_path = meta
            try:
                frames = _frames_for_chunk(cache_path, self.device)  # (1, 8, 512)
            except Exception as e:
                print(f"  [!] Could not load frames for {chunk_id}: {e}")
                continue

            dgse_emb = self.dgse(frames, query_tensor)                # (1, 512)
            score    = (dgse_emb @ query_tensor.T).item()
            dgse_hits.append((chunk_id, video_id, start_sec, end_sec, score, frames))

        dgse_hits.sort(key=lambda x: x[4], reverse=True)
        dgse_top  = dgse_hits[:self.dgse_top_k]
        timings["dgse"] = time.time() - t

        if not dgse_top:
            return []

        # ── Stage 5: Transformer temporal grounding ───────────────────────────
        t = time.time()
        results = []
        for chunk_id, video_id, start_sec, end_sec, dgse_score, frames in dgse_top:
            # frames: (1, 8, 512), query_tensor: (1, 512)
            pred = self.transformer(frames, query_tensor)  # (1, 2) — [start_norm, end_norm]

            # Convert normalized [0,1] predictions to absolute seconds
            duration     = end_sec - start_sec
            pred_start_n = pred[0, 0].item()   # 0..1 relative to chunk
            pred_end_n   = pred[0, 1].item()

            abs_start = start_sec + pred_start_n * duration
            abs_end   = start_sec + pred_end_n   * duration

            # Clamp to chunk bounds and ensure start < end
            abs_start = max(start_sec, min(abs_start, end_sec))
            abs_end   = max(abs_start + 0.1, min(abs_end, end_sec))

            results.append(RetrievalResult(
                video_id   = video_id,
                start_sec  = round(abs_start, 2),
                end_sec    = round(abs_end,   2),
                confidence = round(dgse_score, 4),
                query      = query,
            ))

        timings["transformer"] = time.time() - t
        timings["total"]       = sum(timings.values())

        # ── Print stage timings ────────────────────────────────────────────────
        print(f"  encode      : {timings['encode']*1000:.1f}ms")
        print(f"  faiss       : {timings['faiss']*1000:.1f}ms")
        print(f"  maxsim      : {timings['maxsim']*1000:.1f}ms")
        print(f"  dgse        : {timings['dgse']*1000:.1f}ms")
        print(f"  transformer : {timings['transformer']*1000:.1f}ms")
        print(f"  ─────────────────────────────")
        print(f"  TOTAL       : {timings['total']*1000:.1f}ms  (target <100ms)")

        return results[:return_top_k]


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M14 end-to-end VCMR inference")
    parser.add_argument("--query",  type=str, required=True, help="Natural language query")
    parser.add_argument("--top_k",  type=int, default=1,     help="Number of results to return")
    parser.add_argument("--device", type=str, default=DEVICE, help="cpu / cuda")
    args = parser.parse_args()

    pipe    = VCMRPipeline(device=args.device)
    results = pipe.infer(args.query, return_top_k=args.top_k)

    print(f"\n{'='*50}")
    print(f"Query: {args.query}")
    print(f"{'='*50}")

    if not results:
        print("No results returned. Check FAISS index and metadata DB.")
    else:
        for i, r in enumerate(results, 1):
            print(f"\nResult #{i}")
            print(r)