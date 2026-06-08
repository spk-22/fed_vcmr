"""
Build phone payload from Charades-STA test-set videos (never used in training).

Each Charades video becomes one chunk: chunk_id={vid}_c0, start=0.0, end=duration.

Modes:
    Fresh build (default) — wipes and rebuilds from scratch:
        python scripts/build_phone_payload.py --max-videos 100 --seed 42

    Append — adds new videos without touching already-pushed ones:
        python scripts/build_phone_payload.py --append --max-videos 100 --seed 99

    In append mode the existing index.json is the exclusion list — any video_id
    already present is skipped automatically, so there are zero duplicates.
    Only the new mp4s need to be pushed; existing ones stay on the phone.
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
CACHE_DIR    = PROJECT_ROOT / "cache" / "charades_frames"
VIDEO_DIR    = PROJECT_ROOT / "Charades" / "Charades_v1_480"
OUTPUT_DIR   = PROJECT_ROOT / "phone_payload"

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
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--append",     action="store_true",
                        help="Add new videos to existing payload (no wipe)")
    args = parser.parse_args()

    # ── 1. Load all available Charades test videos ────────────────────────────
    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()
    cur.execute("""
        SELECT video_id, MAX(duration) AS duration
        FROM charades_sta_segments
        GROUP BY video_id
    """)
    all_videos = cur.fetchall()
    conn.close()

    available = {
        vid: dur for vid, dur in all_videos
        if (CACHE_DIR / f"{vid}.npy").exists() and (VIDEO_DIR / f"{vid}.mp4").exists()
    }

    # ── 2. Exclude already-pushed videos in append mode ───────────────────────
    existing_index   = []
    existing_vm      = None
    existing_blob    = b""
    existing_offsets = {}
    already_pushed   = set()

    if args.append:
        index_path = OUTPUT_DIR / "index.json"
        if not index_path.exists():
            print("ERROR: --append requested but no existing index.json found. Run without --append first.")
            return

        with open(index_path) as f:
            existing_index = json.load(f)
        already_pushed = {e["video_id"] for e in existing_index}

        existing_vm = np.load(str(OUTPUT_DIR / "vector_map.npy"))  # (N, 512)

        with open(OUTPUT_DIR / "features_blob.bin", "rb") as f:
            existing_blob = f.read()

        with open(OUTPUT_DIR / "features_index.json") as f:
            existing_offsets = json.load(f)

        print(f"Append mode: {len(already_pushed)} videos already in payload")

    pool = [(vid, dur) for vid, dur in available.items() if vid not in already_pushed]
    print(f"Pool after exclusion: {len(pool)} videos available")

    random.seed(args.seed)
    selected = random.sample(pool, min(args.max_videos, len(pool)))
    print(f"Sampling {len(selected)} new videos (seed={args.seed})")

    # ── 3. Build new chunk data ───────────────────────────────────────────────
    new_index_entries = []
    new_embeddings    = []
    new_blob_bytes    = bytearray()
    new_blob_offsets  = {}
    new_video_ids     = []
    skipped           = 0
    blob_base_offset  = len(existing_blob)   # new entries start after existing blob

    for video_id, duration in selected:
        features = np.load(str(CACHE_DIR / f"{video_id}.npy"))
        if features.shape != (FRAMES, DIM):
            print(f"  SKIP {video_id}: unexpected shape {features.shape}")
            skipped += 1
            continue

        chunk_id = f"{video_id}_c0"
        emb      = pool_embedding(features)
        new_embeddings.append(emb)

        offset = blob_base_offset + len(new_blob_bytes)
        new_blob_offsets[chunk_id] = offset
        frames_f32 = features.astype(np.float32)
        new_blob_bytes += struct.pack(f"{FRAMES * DIM}f", *frames_f32.flatten())

        new_index_entries.append({
            "chunk_id":       chunk_id,
            "video_id":       video_id,
            "start":          0.0,
            "end":            round(duration, 3),
            "source_dataset": "charades_sta"
        })
        new_video_ids.append(video_id)

    if not new_embeddings:
        print("ERROR: no new chunks built.")
        return

    # ── 4. Merge with existing and write ──────────────────────────────────────
    if args.append:
        # Combine index
        combined_index   = existing_index + new_index_entries
        combined_offsets = {**existing_offsets, **new_blob_offsets}
        combined_vm      = np.vstack([existing_vm, np.stack(new_embeddings).astype(np.float32)])
        combined_blob    = existing_blob + bytes(new_blob_bytes)
    else:
        # Fresh build
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        (OUTPUT_DIR / "videos").mkdir(parents=True)
        combined_index   = new_index_entries
        combined_offsets = new_blob_offsets
        combined_vm      = np.stack(new_embeddings).astype(np.float32)
        combined_blob    = bytes(new_blob_bytes)

    with open(OUTPUT_DIR / "index.json", "w") as f:
        json.dump(combined_index, f, indent=2)

    np.save(str(OUTPUT_DIR / "vector_map.npy"), combined_vm)

    with open(OUTPUT_DIR / "features_blob.bin", "wb") as f:
        f.write(combined_blob)

    with open(OUTPUT_DIR / "features_index.json", "w") as f:
        json.dump(combined_offsets, f)

    # Copy only new mp4s
    for video_id in new_video_ids:
        shutil.copy2(str(VIDEO_DIR / f"{video_id}.mp4"),
                     str(OUTPUT_DIR / "videos" / f"{video_id}.mp4"))

    # ── 5. Summary ────────────────────────────────────────────────────────────
    total_mb = sum(p.stat().st_size for p in OUTPUT_DIR.rglob("*") if p.is_file()) / (1024 * 1024)
    mode_str = "appended" if args.append else "built"

    print(f"\nPayload {mode_str}: {len(combined_index)} total chunks "
          f"({len(new_embeddings)} new, {skipped} skipped)")
    print(f"  index.json:         {len(combined_index)} entries")
    print(f"  vector_map.npy:     {combined_vm.shape}  float32")
    print(f"  features_blob.bin:  {len(combined_blob) / (1024*1024):.1f} MB")
    print(f"  videos/ (new only): {len(new_video_ids)} .mp4 files")
    print(f"  Total payload size: {total_mb:.1f} MB")

    print(f"\n--- Push index files (always) ---")
    print(f"  adb push {OUTPUT_DIR / 'index.json'} /sdcard/fedvcmr/index.json")
    print(f"  adb push {OUTPUT_DIR / 'vector_map.npy'} /sdcard/fedvcmr/vector_map.npy")
    print(f"  adb push {OUTPUT_DIR / 'features_blob.bin'} /sdcard/fedvcmr/features_blob.bin")
    print(f"  adb push {OUTPUT_DIR / 'features_index.json'} /sdcard/fedvcmr/features_index.json")

    if args.append:
        print(f"\n--- Push new videos only ({len(new_video_ids)} files) ---")
        for vid in new_video_ids:
            print(f"  adb push {OUTPUT_DIR / 'videos' / (vid + '.mp4')} /sdcard/fedvcmr/videos/{vid}.mp4")
    else:
        print(f"\n--- Push all videos ---")
        print(f"  adb push {OUTPUT_DIR / 'videos'} /sdcard/fedvcmr/videos")


if __name__ == "__main__":
    main()
