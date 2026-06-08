"""Pack individual .npy feature files into a single binary blob for fast on-device access.

Output:
  features_blob.bin  — flat float32 array: [chunk0_frame0_dim0..dim511, chunk0_frame1_dim0..., ...]
  features_index.json — {"chunk_id": byte_offset, ...}

Each chunk = 8 frames × 512 dims × 4 bytes = 16384 bytes.
"""
import json
import numpy as np
from pathlib import Path

FEATURES_DIR = Path("phone_payload/features") if Path("phone_payload/features").exists() else None
OUT_DIR = Path("phone_payload")

def main():
    # Find features dir
    feat_dir = FEATURES_DIR
    if feat_dir is None or not feat_dir.exists():
        # Try alternate location
        feat_dir = Path("outputs/features")
    if not feat_dir.exists():
        print(f"Features directory not found. Checking what we have...")
        # List .npy files that might be features
        import subprocess
        result = subprocess.run(["adb", "shell", "ls", "/sdcard/fedvcmr/features/"],
                              capture_output=True, text=True)
        print("Features are on phone only. Pulling them first...")
        feat_dir = Path("phone_payload/features")
        feat_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["adb", "pull", "/sdcard/fedvcmr/features/", str(feat_dir)], check=True)

    npy_files = sorted(feat_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} feature files")

    index = {}
    blob_data = bytearray()

    for f in npy_files:
        chunk_id = f.stem
        arr = np.load(f).astype(np.float32)  # (8, 512) -> float32
        assert arr.shape == (8, 512), f"Unexpected shape {arr.shape} for {f.name}"

        index[chunk_id] = len(blob_data)
        blob_data.extend(arr.tobytes())

    blob_path = OUT_DIR / "features_blob.bin"
    index_path = OUT_DIR / "features_index.json"

    with open(blob_path, "wb") as f:
        f.write(blob_data)

    with open(index_path, "w") as f:
        json.dump(index, f)

    print(f"Blob: {blob_path} ({len(blob_data) / 1024:.0f} KB)")
    print(f"Index: {index_path} ({len(index)} chunks)")
    print(f"Each chunk: {8 * 512 * 4} bytes at known offset -> zero-parse random access")

if __name__ == "__main__":
    main()
