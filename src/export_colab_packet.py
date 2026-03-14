import os
import sqlite3
import json
import shutil
from pathlib import Path

def export_packet():
    print("Preparing Phase 4 Colab Export Packet...")
    
    # Paths
    PROJECT_ROOT = Path("d:/fed_vcmr/fed_vcmr")
    DB_PATH = PROJECT_ROOT / "fedvcmr.db"
    CACHED_FEATURES = PROJECT_ROOT / "cache" / "frame_features"
    EXPORT_ROOT = PROJECT_ROOT / "data" / "activitynet" / "colab_export"
    EXPORT_FEATURES = EXPORT_ROOT / "features"
    ANNOTATION_PATH = PROJECT_ROOT / "data" / "activitynet" / "annotations.json"
    
    EXPORT_FEATURES.mkdir(parents=True, exist_ok=True)
    
    # 1. Connect to DB and get ActivityNet chunk metadata
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Select all chunks related to ActivityNet
    cursor.execute("""
        SELECT chunk_id, video_id, t_start, t_end, cache_path 
        FROM chunks 
        WHERE chunk_id LIKE 'AN_%'
    """)
    chunk_rows = cursor.fetchall()
    
    # 2. Load the ground truth annotations to match sentences
    with open(ANNOTATION_PATH, 'r', encoding='utf-8') as f:
        anet_data = json.load(f)
    
    # Create a lookup for GT: video_id -> list of {start, end, caption}
    gt_lookup = {}
    for sample in anet_data.get('samples', []):
        v_id = sample.get('video_id', '')
        captions = sample.get('captions', {}).get('temporal', [])
        gt_lookup[v_id] = captions

    # 3. Build the export manifest
    manifest = []
    copied_count = 0
    
    print(f"Processing {len(chunk_rows)} chunks...")
    for chunk_id, v_id, t_start, t_end, cache_path in chunk_rows:
        src_path = Path(cache_path)
        if not src_path.exists():
            # Fallback check in case path changed
            src_path = CACHED_FEATURES / f"{chunk_id}.npy"
            
        if src_path.exists():
            # Copy to export folder
            dest_path = EXPORT_FEATURES / f"{chunk_id}.npy"
            if not dest_path.exists():
                shutil.copy2(src_path, dest_path)
                copied_count += 1
            
            # Find matching GT segments for THIS chunk's time range
            # A chunk is relevant to a GT if they overlap
            relevant_gt = []
            if v_id in gt_lookup:
                for gt in gt_lookup[v_id]:
                    # Temporal IoU or simple overlap check
                    gt_start = gt['start_time']
                    gt_end = gt['end_time']
                    
                    # Check overlap
                    overlap_start = max(t_start, gt_start)
                    overlap_end = min(t_end, gt_end)
                    if overlap_end > overlap_start:
                        relevant_gt.append({
                            "caption": gt['caption'],
                            "gt_start": gt_start,
                            "gt_end": gt_end
                        })
            
            if relevant_gt:
                manifest.append({
                    "chunk_id": chunk_id,
                    "video_id": v_id,
                    "t_start": t_start,
                    "t_end": t_end,
                    "matches": relevant_gt
                })

    # 4. Save manifest
    with open(EXPORT_ROOT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    conn.close()
    
    print(f"\nExport Detailed:")
    print(f"- Total chunks in manifest: {len(manifest)}")
    print(f"- Total .npy files copied: {copied_count}")
    print(f"- Destination: {EXPORT_ROOT}")
    print("\nNext Step: Upload the 'colab_export' folder to your Google Drive and run the Phase 4 notebook.")

if __name__ == "__main__":
    export_packet()
