"""
Build MSRVTT phone payload from randomly sampled MSRVTT videos.

Each video contributes all its sliding-window chunks (avg 3.2 chunks/video).
One mp4 per video is pushed; ExoPlayer clips to chunk t_start/t_end on playback.

Usage:
    python scripts/build_msrvtt_payload.py --max-videos 300 --seed 42
    python scripts/build_msrvtt_payload.py --append --max-videos 100 --seed 99
"""
import argparse
import json
import random
import shutil
import sqlite3
import struct
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH      = PROJECT_ROOT / "fedvcmr.db"
VIDEO_DIR    = PROJECT_ROOT / "MSRVTT" / "MSRVTT" / "videos" / "all"
OUTPUT_DIR   = PROJECT_ROOT / "phone_payload_msrvtt"

FRAMES  = 8
DIM     = 512
WEIGHTS = np.array([0.5, 0.75, 1.0, 1.25, 1.25, 1.0, 0.75, 0.5], dtype=np.float32)


def pool_embedding(features: np.ndarray) -> np.ndarray:
    feat32 = features.astype(np.float32)
    emb = np.sum(feat32 * WEIGHTS[:, np.newaxis], axis=0) / WEIGHTS.sum()
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-videos", type=int, default=300)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--append",     action="store_true",
                        help="Add new videos to existing payload (no wipe)")
    args = parser.parse_args()

    # ── 1. Load all MSRVTT video → chunks mapping ─────────────────────────────
    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()
    cur.execute("""
        SELECT c.chunk_id, c.video_id, c.t_start, c.t_end, c.cache_path, v.duration
        FROM chunks c
        JOIN videos v ON c.video_id = v.video_id
        ORDER BY c.video_id, c.t_start
    """)
    rows = cur.fetchall()
    conn.close()

    # Group by video_id, filter to those with video file + all cache files present
    from collections import defaultdict
    video_chunks = defaultdict(list)
    for chunk_id, video_id, t_start, t_end, cache_path, duration in rows:
        cache = Path(cache_path)
        if cache.exists() and (VIDEO_DIR / f"{video_id}.mp4").exists():
            video_chunks[video_id].append((chunk_id, t_start, t_end, cache_path, duration))

    print(f"Available MSRVTT videos with full cache: {len(video_chunks)}")

    # ── 2. Exclude already-pushed videos in append mode ───────────────────────
    existing_index   = []
    existing_vm      = None
    existing_blob    = b""
    existing_offsets = {}
    already_pushed   = set()

    if args.append:
        index_path = OUTPUT_DIR / "msrvtt_index.json"
        if not index_path.exists():
            print("ERROR: --append requested but no existing msrvtt_index.json. Run without --append first.")
            return
        with open(index_path) as f:
            existing_index = json.load(f)
        already_pushed = {e["video_id"] for e in existing_index}
        existing_vm = np.load(str(OUTPUT_DIR / "msrvtt_vector_map.npy"))
        with open(OUTPUT_DIR / "msrvtt_features_blob.bin", "rb") as f:
            existing_blob = f.read()
        with open(OUTPUT_DIR / "msrvtt_features_index.json") as f:
            existing_offsets = json.load(f)
        print(f"Append mode: {len(already_pushed)} videos already in payload")

    pool = [vid for vid in video_chunks if vid not in already_pushed]
    print(f"Pool after exclusion: {len(pool)} videos available")

    random.seed(args.seed)
    selected_vids = random.sample(pool, min(args.max_videos, len(pool)))
    print(f"Sampling {len(selected_vids)} new videos (seed={args.seed})")

    # ── 3. Build new chunk data ───────────────────────────────────────────────
    new_index_entries = []
    new_embeddings    = []
    new_blob_bytes    = bytearray()
    new_blob_offsets  = {}
    new_video_ids     = []
    skipped           = 0
    blob_base_offset  = len(existing_blob)

    for video_id in selected_vids:
        chunks = video_chunks[video_id]
        video_ok = True
        video_entries = []
        video_embeddings = []

        for chunk_id, t_start, t_end, cache_path, duration in chunks:
            features = np.load(cache_path)
            if features.shape != (FRAMES, DIM):
                print(f"  SKIP {chunk_id}: unexpected shape {features.shape}")
                video_ok = False
                break

            emb = pool_embedding(features)
            offset = blob_base_offset + len(new_blob_bytes)
            frames_f32 = features.astype(np.float32)

            video_entries.append({
                "chunk_id":       chunk_id,
                "video_id":       video_id,
                "start":          round(t_start, 3),
                "end":            round(t_end, 3),
                "source_dataset": "msrvtt"
            })
            video_embeddings.append((chunk_id, emb, offset, frames_f32))

        if not video_ok:
            skipped += 1
            continue

        for chunk_id, emb, offset, frames_f32 in video_embeddings:
            new_embeddings.append(emb)
            new_blob_offsets[chunk_id] = offset
            new_blob_bytes += struct.pack(f"{FRAMES * DIM}f", *frames_f32.flatten())

        new_index_entries.extend(video_entries)
        new_video_ids.append(video_id)

    if not new_embeddings:
        print("ERROR: no new chunks built.")
        return

    # ── 4. Merge with existing and write ──────────────────────────────────────
    if args.append:
        combined_index   = existing_index + new_index_entries
        combined_offsets = {**existing_offsets, **new_blob_offsets}
        combined_vm      = np.vstack([existing_vm, np.stack(new_embeddings).astype(np.float32)])
        combined_blob    = existing_blob + bytes(new_blob_bytes)
    else:
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        (OUTPUT_DIR / "msrvtt_videos").mkdir(parents=True)
        combined_index   = new_index_entries
        combined_offsets = new_blob_offsets
        combined_vm      = np.stack(new_embeddings).astype(np.float32)
        combined_blob    = bytes(new_blob_bytes)

    with open(OUTPUT_DIR / "msrvtt_index.json", "w") as f:
        json.dump(combined_index, f, indent=2)

    np.save(str(OUTPUT_DIR / "msrvtt_vector_map.npy"), combined_vm)

    with open(OUTPUT_DIR / "msrvtt_features_blob.bin", "wb") as f:
        f.write(combined_blob)

    with open(OUTPUT_DIR / "msrvtt_features_index.json", "w") as f:
        json.dump(combined_offsets, f)

    for video_id in new_video_ids:
        shutil.copy2(str(VIDEO_DIR / f"{video_id}.mp4"),
                     str(OUTPUT_DIR / "msrvtt_videos" / f"{video_id}.mp4"))

    # ── 5. Summary ────────────────────────────────────────────────────────────
    total_chunks = len(combined_index)
    total_videos = len({e["video_id"] for e in combined_index})
    total_mb = sum(p.stat().st_size for p in OUTPUT_DIR.rglob("*") if p.is_file()) / (1024 * 1024)
    mode_str = "appended" if args.append else "built"

    print(f"\nPayload {mode_str}: {total_chunks} total chunks across {total_videos} videos "
          f"({len(new_embeddings)} new chunks, {skipped} videos skipped)")
    print(f"  msrvtt_index.json:         {total_chunks} entries")
    print(f"  msrvtt_vector_map.npy:     {combined_vm.shape}  float32")
    print(f"  msrvtt_features_blob.bin:  {len(combined_blob) / (1024*1024):.1f} MB")
    print(f"  msrvtt_videos/ (new only): {len(new_video_ids)} .mp4 files")
    print(f"  Total payload size: {total_mb:.1f} MB")

    print(f"\n--- Push index files (always) ---")
    print(f"  adb push {OUTPUT_DIR / 'msrvtt_index.json'} /sdcard/fedvcmr/msrvtt_index.json")
    print(f"  adb push {OUTPUT_DIR / 'msrvtt_vector_map.npy'} /sdcard/fedvcmr/msrvtt_vector_map.npy")
    print(f"  adb push {OUTPUT_DIR / 'msrvtt_features_blob.bin'} /sdcard/fedvcmr/msrvtt_features_blob.bin")
    print(f"  adb push {OUTPUT_DIR / 'msrvtt_features_index.json'} /sdcard/fedvcmr/msrvtt_features_index.json")

    if args.append:
        print(f"\n--- Push new videos only ({len(new_video_ids)} files) ---")
        for vid in new_video_ids:
            print(f"  adb push {OUTPUT_DIR / 'msrvtt_videos' / (vid + '.mp4')} /sdcard/fedvcmr/msrvtt_videos/{vid}.mp4")
    else:
        print(f"\n--- Push all videos ---")
        print(f"  adb push {OUTPUT_DIR}\\msrvtt_videos /sdcard/fedvcmr/msrvtt_videos")


if __name__ == "__main__":
    main()
