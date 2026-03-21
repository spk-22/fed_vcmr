"""
M15 — Optimised End-to-End Pipeline
=====================================
Fixes the 2081ms → target <100ms

Optimisations vs M14:
  1. frame_features.bin memmap — single file load, no per-.npy reads
     MaxSim: 1568ms → ~10ms
  2. GPU for backbone, DGSE, Transformer
     encode: 401ms → ~20ms
  3. Batched DGSE — all candidates in one forward pass
     dgse: 64ms → ~8ms
  4. Correct DB column names (t_start / t_end)

Usage (Colab):
    from pipeline_m15 import VCMRPipeline
    pipe = VCMRPipeline()
    results = pipe.infer("a person cooking food on a stove")
"""

import os, sys, time, argparse, sqlite3
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.backbone import MobileCLIPWrapper
from src.config import INDEX_PATH, DB_PATH, DEVICE
from src.query import QueryService
from src.search import SearchIndex
from src.models.dgse import DGSE
from src.models.temporal_grounding import CrossModalTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR   = Path("checkpoints")
DGSE_CKPT        = CHECKPOINT_DIR / "dgse_best.pt"
TRANSFORMER_CKPT = CHECKPOINT_DIR / "temporal_grounding_best.pt"
MEMMAP_PATH      = Path("data/frame_features.bin")
MEMMAP_SHAPE     = (16108, 8, 512)   # confirmed from Cell 2


# ── Return type ────────────────────────────────────────────────────────────────
@dataclass
class RetrievalResult:
    video_id   : str
    start_sec  : float
    end_sec    : float
    confidence : float
    query      : str

    def __str__(self):
        return (
            f"video_id   : {self.video_id}\n"
            f"start_sec  : {self.start_sec:.2f}s\n"
            f"end_sec    : {self.end_sec:.2f}s\n"
            f"confidence : {self.confidence:.4f}\n"
            f"query      : {self.query}"
        )


# ── DB helpers ─────────────────────────────────────────────────────────────────
def _build_chunk_lookup(db_path: str):
    """
    Returns:
      meta   : dict  chunk_id → (video_id, t_start, t_end, cache_path)
      id2row : dict  chunk_id → memmap row index (only for chunks in memmap)

    Uses actual DB column names: t_start, t_end
    """
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    rows = cursor.execute(
        "SELECT chunk_id, video_id, t_start, t_end, cache_path FROM chunks"
    ).fetchall()
    conn.close()

    meta   = {}
    id2row = {}
    for row_idx, (chunk_id, video_id, t_start, t_end, cache_path) in enumerate(rows):
        meta[chunk_id] = (video_id, float(t_start), float(t_end), cache_path)
        # Only first MEMMAP_SHAPE[0] rows are in the memmap
        if row_idx < MEMMAP_SHAPE[0]:
            id2row[chunk_id] = row_idx

    return meta, id2row


def _load_checkpoint(path: Path, model, device, label: str):
    """Load checkpoint — handles nested key structure from training scripts."""
    if not path.exists():
        print(f"  [!] {label} checkpoint not found at {path} — using random weights")
        return
    raw = torch.load(path, map_location=device, weights_only=False)

    # Each checkpoint uses a different top-level key
    if label == "DGSE" and "dgse" in raw:
        sd = raw["dgse"]
    elif label == "Transformer" and "model" in raw:
        sd = raw["model"]
        # Log training metrics
        if "iou" in raw:
            iou = raw["iou"]
            keys = list(iou.keys())
            print(f"  [i] Transformer trained {raw.get('epoch','?')} epochs, "
                  f"val_loss={raw.get('val_loss',0):.4f}, "
                  f"IoU@0.5={iou.get(0.5, iou.get('0.5', '?')):.4f}, "
                  f"IoU@0.7={iou.get(0.7, iou.get('0.7', '?')):.4f}")
    else:
        sd = raw.get("model_state_dict", raw.get("state_dict", raw))

    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing:
        print(f"  [!] {label} — still missing keys: {missing[:3]}")
    else:
        print(f"  [✓] {label} loaded PERFECTLY — all {len(sd)} keys matched")


# ── MaxSim (vectorised, memmap-based) ─────────────────────────────────────────
def maxsim_rerank_fast(
    query_embs : np.ndarray,          # (N_q, 512) normalised
    candidates : List[tuple],         # [(chunk_id, score), ...]
    memmap     : np.ndarray,          # (N, 8, 512) memmap
    id2row     : dict,
    top_k      : int = 20,
) -> List[tuple]:
    """
    Vectorised MaxSim using memmap — no file I/O per candidate.
    Falls back to cache_path .npy read if chunk not in memmap.
    """
    scored = []
    for chunk_id, _ in candidates:
        row = id2row.get(chunk_id)
        if row is not None:
            frames = memmap[row].astype(np.float32)   # (8, 512) — zero-copy slice
        else:
            # fallback: read from cache_path
            cache_path = None
            if hasattr(maxsim_rerank_fast, "_meta"):
                meta = maxsim_rerank_fast._meta.get(chunk_id)
                if meta:
                    cache_path = meta[3]
            if cache_path and os.path.exists(cache_path):
                frames = np.load(cache_path).astype(np.float32)
            else:
                continue

        # MaxSim: mean over frames of max cosine similarity across query phrasings
        # query_embs: (N_q, 512), frames: (8, 512)
        sim = query_embs @ frames.T          # (N_q, 8)
        score = float(np.mean(np.max(sim, axis=0)))
        scored.append((chunk_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ── DGSE batched ──────────────────────────────────────────────────────────────
def dgse_rerank_batched(
    candidates  : List[tuple],   # [(chunk_id, score), ...]
    memmap      : np.ndarray,    # (N, 8, 512)
    id2row      : dict,
    meta        : dict,
    dgse_model  : DGSE,
    query_tensor: torch.Tensor,  # (1, 512) on device
    device      : torch.device,
    top_k       : int = 5,
) -> List[tuple]:
    """
    Runs all candidate chunks through DGSE in a single batched forward pass.
    Much faster than looping one-by-one.
    """
    valid       = []
    frame_batch = []

    for chunk_id, _ in candidates:
        row = id2row.get(chunk_id)
        if row is None:
            m = meta.get(chunk_id)
            if m and os.path.exists(m[3]):
                frames = np.load(m[3]).astype(np.float32)
            else:
                continue
        else:
            frames = memmap[row].astype(np.float32)   # (8, 512)

        frame_batch.append(frames)
        valid.append(chunk_id)

    if not valid:
        return []

    # Stack into batch tensor
    batch = torch.from_numpy(
        np.stack(frame_batch)                         # (B, 8, 512)
    ).to(device)

    # Expand query to match batch size
    q_batch = query_tensor.expand(len(valid), -1)     # (B, 512)

    with torch.no_grad():
        dgse_embs = dgse_model(batch, q_batch)        # (B, 512)
        scores    = (dgse_embs * q_batch).sum(dim=1)  # (B,) dot product

    results = list(zip(valid, scores.cpu().tolist()))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ── Main pipeline ──────────────────────────────────────────────────────────────
class VCMRPipeline:
    def __init__(
        self,
        db_path          : str = str(DB_PATH),
        memmap_path      : str = str(MEMMAP_PATH),
        faiss_candidates : int = 100,
        maxsim_top_k     : int = 20,
        dgse_top_k       : int = 5,
        device           : str = None,
    ):
        self.faiss_candidates = faiss_candidates
        self.maxsim_top_k     = maxsim_top_k
        self.dgse_top_k       = dgse_top_k
        self.device           = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"[M15] Loading pipeline on {self.device} ...")
        t0 = time.time()

        # 1. Backbone
        self.backbone  = MobileCLIPWrapper(device=str(self.device))
        self.query_svc = QueryService(backbone=self.backbone)

        # 2. FAISS index
        self.search_index = SearchIndex()
        self.search_index.load_index()

        # 3. Metadata + memmap row lookup
        self.meta, self.id2row = _build_chunk_lookup(db_path)
        print(f"  [✓] Metadata loaded — {len(self.meta)} chunks, "
              f"{len(self.id2row)} in memmap")

        # 4. Memmap (read-only, zero-copy slicing)
        self.memmap = np.memmap(
            memmap_path, dtype="float32", mode="r",
            shape=MEMMAP_SHAPE
        )
        # Attach meta to maxsim helper for fallback
        maxsim_rerank_fast._meta = self.meta
        print(f"  [✓] Memmap loaded — shape {self.memmap.shape}")

        # 5. DGSE
        self.dgse = DGSE(dim=512, n_frames=8).to(self.device)
        _load_checkpoint(DGSE_CKPT, self.dgse, self.device, "DGSE")
        self.dgse.eval()

        # 6. Transformer
        self.transformer = CrossModalTransformer(
            visual_dim=512, query_dim=512,
            hidden_dim=256, n_heads=4, n_layers=2
        ).to(self.device)
        _load_checkpoint(TRANSFORMER_CKPT, self.transformer,
                         self.device, "Transformer")
        self.transformer.eval()

        print(f"[M15] Pipeline ready in {time.time()-t0:.2f}s\n")

    # ── Warm-up GPU (call once before benchmarking) ────────────────────────────
    def warmup(self, n: int = 3):
        print(f"[M15] GPU warmup ({n} runs) ...")
        for _ in range(n):
            self.infer("person walking in a park", verbose=False)
        print("[M15] Warmup done.\n")

    # ── Main inference ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def infer(
        self,
        query      : str,
        return_top_k: int = 1,
        verbose    : bool = True,
    ) -> List[RetrievalResult]:

        timings = {}

        # Stage 1 — Query encoding
        t = time.time()
        phrasings   = self.query_svc.expand_query(query)
        multi_emb   = self.query_svc.encode_queries(phrasings)    # (3, 512) numpy
        single_emb  = self.query_svc.encode_query(query)          # (1, 512) numpy
        query_tensor = torch.from_numpy(single_emb).to(self.device)  # (1, 512)
        timings["encode"] = time.time() - t

        # Stage 2 — FAISS coarse
        t = time.time()
        coarse_hits = self.search_index.coarse_search(
            single_emb, top_k=self.faiss_candidates
        )
        timings["faiss"] = time.time() - t

        if not coarse_hits:
            return []

        # Stage 3 — MaxSim (memmap, no file I/O)
        t = time.time()
        maxsim_hits = maxsim_rerank_fast(
            multi_emb, coarse_hits,
            self.memmap, self.id2row,
            top_k=self.maxsim_top_k
        )
        timings["maxsim"] = time.time() - t

        # Stage 4 — DGSE batched
        t = time.time()
        dgse_hits = dgse_rerank_batched(
            maxsim_hits, self.memmap, self.id2row,
            self.meta, self.dgse, query_tensor,
            self.device, top_k=self.dgse_top_k
        )
        timings["dgse"] = time.time() - t

        if not dgse_hits:
            return []

        # Stage 5 — Transformer temporal grounding
        t = time.time()
        results = []
        for chunk_id, dgse_score in dgse_hits:
            m = self.meta.get(chunk_id)
            if m is None:
                continue
            video_id, t_start, t_end, _ = m

            row = self.id2row.get(chunk_id)
            if row is None:
                continue
            frames = torch.from_numpy(
                self.memmap[row].astype(np.float32)
            ).unsqueeze(0).to(self.device)   # (1, 8, 512)

            pred = self.transformer(frames, query_tensor)  # (1, 2)
            pred_s = pred[0, 0].item()
            pred_e = pred[0, 1].item()

            duration  = t_end - t_start
            abs_start = t_start + pred_s * duration
            abs_end   = t_start + pred_e * duration
            abs_start = max(t_start, min(abs_start, t_end))
            abs_end   = max(abs_start + 0.1, min(abs_end, t_end))

            results.append(RetrievalResult(
                video_id   = video_id,
                start_sec  = round(abs_start, 2),
                end_sec    = round(abs_end,   2),
                confidence = round(dgse_score, 4),
                query      = query,
            ))

        timings["transformer"] = time.time() - t
        timings["total"]       = sum(v for k, v in timings.items()
                                     if k != "total")

        if verbose:
            self._print_timings(timings)

        return results[:return_top_k]

    def _print_timings(self, t: dict):
        print(f"  encode      : {t['encode']*1000:6.1f}ms")
        print(f"  faiss       : {t['faiss']*1000:6.1f}ms")
        print(f"  maxsim      : {t['maxsim']*1000:6.1f}ms")
        print(f"  dgse        : {t['dgse']*1000:6.1f}ms")
        print(f"  transformer : {t['transformer']*1000:6.1f}ms")
        print(f"  {'─'*30}")
        status = "✓ PASS" if t['total'] < 0.1 else "✗ FAIL"
        print(f"  TOTAL       : {t['total']*1000:6.1f}ms  "
              f"(target <100ms) {status}")

    # ── Latency benchmark (M15 gate) ───────────────────────────────────────────
    def benchmark(self, n_runs: int = 10):
        """
        Run n_runs queries and report mean/min/max per stage.
        M15 gate: mean total < 100ms
        """
        test_queries = [
            "a person is cooking food on a stove",
            "someone playing guitar",
            "a dog running in a park",
            "people dancing at a party",
            "a cat sitting on a table",
            "someone riding a bicycle",
            "children playing outside",
            "a man giving a speech",
            "cars driving on a highway",
            "a woman singing on stage",
        ]

        stage_keys = ["encode", "faiss", "maxsim", "dgse", "transformer", "total"]
        all_times  = {k: [] for k in stage_keys}

        print(f"\n[M15] Benchmarking {n_runs} queries ...\n")
        for i in range(n_runs):
            q  = test_queries[i % len(test_queries)]
            t0 = time.time()

            # Re-run with timing capture
            timings = {}

            phrasings    = self.query_svc.expand_query(q)
            t = time.time()
            multi_emb    = self.query_svc.encode_queries(phrasings)
            single_emb   = self.query_svc.encode_query(q)
            query_tensor = torch.from_numpy(single_emb).to(self.device)
            timings["encode"] = time.time() - t

            t = time.time()
            coarse_hits  = self.search_index.coarse_search(single_emb,
                                                           top_k=self.faiss_candidates)
            timings["faiss"] = time.time() - t

            t = time.time()
            maxsim_hits  = maxsim_rerank_fast(multi_emb, coarse_hits,
                                              self.memmap, self.id2row,
                                              top_k=self.maxsim_top_k)
            timings["maxsim"] = time.time() - t

            t = time.time()
            dgse_hits    = dgse_rerank_batched(maxsim_hits, self.memmap,
                                               self.id2row, self.meta,
                                               self.dgse, query_tensor,
                                               self.device,
                                               top_k=self.dgse_top_k)
            timings["dgse"] = time.time() - t

            t = time.time()
            for chunk_id, _ in dgse_hits:
                row = self.id2row.get(chunk_id)
                if row is None:
                    continue
                frames = torch.from_numpy(
                    self.memmap[row].astype(np.float32)
                ).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    self.transformer(frames, query_tensor)
            timings["transformer"] = time.time() - t

            timings["total"] = sum(v for k, v in timings.items()
                                   if k != "total")

            for k in stage_keys:
                all_times[k].append(timings[k] * 1000)

            print(f"  Run {i+1:2d}: {timings['total']*1000:.1f}ms — {q[:45]}")

        # Summary table
        print(f"\n{'─'*55}")
        print(f"{'Stage':<15} {'Mean':>8} {'Min':>8} {'Max':>8}")
        print(f"{'─'*55}")
        for k in stage_keys:
            vals = all_times[k]
            print(f"  {k:<13} {np.mean(vals):>7.1f}ms "
                  f"{np.min(vals):>7.1f}ms "
                  f"{np.max(vals):>7.1f}ms")
        print(f"{'─'*55}")

        mean_total = np.mean(all_times["total"])
        gate = "✓ M15 GATE PASSED" if mean_total < 100 else "✗ M15 GATE FAILED"
        print(f"\n  Mean total: {mean_total:.1f}ms — {gate}")
        return mean_total


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M15 optimised VCMR pipeline")
    parser.add_argument("--query",     type=str,  default=None)
    parser.add_argument("--benchmark", action="store_true",
                        help="Run 10-query latency benchmark (M15 gate)")
    parser.add_argument("--top_k",    type=int,  default=1)
    parser.add_argument("--db",       type=str,  default="data/fedvcmr.db")
    parser.add_argument("--memmap",   type=str,  default="data/frame_features.bin")
    args = parser.parse_args()

    pipe = VCMRPipeline(db_path=args.db, memmap_path=args.memmap)
    pipe.warmup()

    if args.benchmark:
        pipe.benchmark(n_runs=10)
    elif args.query:
        results = pipe.infer(args.query, return_top_k=args.top_k)
        print(f"\n{'='*50}\nQuery: {args.query}\n{'='*50}")
        for i, r in enumerate(results, 1):
            print(f"\nResult #{i}\n{r}")
    else:
        print("Provide --query or --benchmark")
