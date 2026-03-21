import os
import json
import sqlite3
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.config import (
    VIDEO_ROOT, ANNOTATION_PATH, CACHE_ROOT, DB_PATH,
    CHUNK_DURATION, CHUNK_STRIDE, FRAMES_PER_CHUNK,
    DEVICE, DIMENSION
)
from src.video_utils import sample_frame_indices, extract_frames_pyav
from src.backbone import MobileCLIPWrapper
from src.metadata_store import MetadataStore

class IngestionPipeline:
    def __init__(self):
        self.backbone = MobileCLIPWrapper(device=DEVICE)
        self.db = MetadataStore(str(DB_PATH))
        self.feature_dir = CACHE_ROOT / "frame_features"

    def process_metadata_streaming(self):
        """Stream annotations using ijson to handle large files and build a manifest."""
        import ijson
        print(f"Streaming metadata from {ANNOTATION_PATH}...")
        
        # Priority: Load 1K-A test manifest to process them first
        test_manifest_path = r"c:\prism\MSRVTT\MSRVTT\structured-symlinks\val_list_jsfusion.txt"
        test_vids = set()
        if os.path.exists(test_manifest_path):
            with open(test_manifest_path, 'r') as f:
                test_vids = {line.strip() for line in f if line.strip()}
        
        # Map of image_id to captions
        caption_map = {}
        with open(ANNOTATION_PATH, 'rb') as f:
            for item in ijson.items(f, 'annotations.item'):
                img_id = item['image_id']
                if img_id not in caption_map:
                    caption_map[img_id] = []
                caption_map[img_id].append(item['caption'])
        
        # Load all video info from JSON
        all_img_info = []
        with open(ANNOTATION_PATH, 'rb') as f:
            for img_info in ijson.items(f, 'images.item'):
                all_img_info.append(img_info)
        
        # Sort so test vids are first
        all_img_info.sort(key=lambda x: x['id'] not in test_vids)
        
        for img_info in all_img_info:
            video_id = img_info['id']
            possible_paths = [
                VIDEO_ROOT / f"{video_id}.mp4",
                VIDEO_ROOT / f"{video_id}.avi"
            ]
            video_path = None
            for p in possible_paths:
                if p.exists():
                    video_path = p
                    break
            
            if video_path:
                captions = caption_map.get(video_id, [])
                yield video_id, video_path, captions

    def decode_worker(self, video_id, video_path, captions, queue):
        """Worker function to decode a single video and put result into queue."""
        try:
            import av
            container = av.open(str(video_path))
            duration = float(container.duration) / 1000000.0
            container.close()
            
            chunks_data = []
            t_start = 0.0
            chunk_idx = 0
            while t_start < duration:
                t_end = min(t_start + CHUNK_DURATION, duration)
                chunk_id = f"{video_id}_c{chunk_idx}"
                
                chunk_duration = t_end - t_start
                rel_timestamps = sample_frame_indices(chunk_duration, 1.0, FRAMES_PER_CHUNK)
                abs_timestamps = [t_start + ts for ts in rel_timestamps]
                
                # Extract frames (heavy CPU task)
                frames = extract_frames_pyav(str(video_path), abs_timestamps)
                chunks_data.append({
                    'chunk_id': chunk_id,
                    't_start': t_start,
                    't_end': t_end,
                    'frames': frames
                })
                
                if t_end == duration:
                    break
                t_start += CHUNK_STRIDE
                chunk_idx += 1
            
            queue.put((video_id, video_path, duration, captions, chunks_data))
        except Exception as e:
            print(f"Error decoding {video_id}: {e}")
            queue.put(None)

    def run_ingestion(self):
        """Run the full ingestion using a producer-consumer pattern for speed."""
        from concurrent.futures import ThreadPoolExecutor
        from queue import Queue
        import threading
        
        print("Starting optimized resumable ingestion...")
        
        # Get list of already completed videos
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT video_id FROM videos WHERE duration > 0")
        completed_vids = {row[0] for row in cursor.fetchall()}
        conn.close()

        # Producer: thread pool for decoding
        queue = Queue(maxsize=20) # Buffer for decoded videos
        
        def producer():
            with ThreadPoolExecutor(max_workers=4) as executor:
                for video_id, video_path, captions in self.process_metadata_streaming():
                    if video_id in completed_vids:
                        continue
                    executor.submit(self.decode_worker, video_id, video_path, captions, queue)
                
                # Push None to signal end
                executor.shutdown(wait=True)
                queue.put("DONE")

        producer_thread = threading.Thread(target=producer)
        producer_thread.start()

        # Consumer: Main thread for model inference (GPU)
        count = 0
        while True:
            item = queue.get()
            if item == "DONE":
                break
            if item is None:
                continue
            
            video_id, video_path, duration, captions, chunks_data = item
            print(f"[{count+1}] Encoding {video_id} ({len(chunks_data)} chunks)...")
            
            try:
                # Add video and captions to metadata
                self.db.add_video(video_id, str(video_path), duration, "rest")
                for caption in captions:
                    self.db.add_caption(video_id, caption)

                # Process chunks
                for cdata in chunks_data:
                    features = self.backbone.encode_images(cdata['frames'])
                    
                    # Persist
                    cache_path = self.feature_dir / f"{cdata['chunk_id']}.npy"
                    np.save(cache_path, features.astype(np.float16))
                    self.db.add_chunk(cdata['chunk_id'], video_id, cdata['t_start'], cdata['t_end'], 1.0, str(cache_path))
                
                count += 1
                if count % 10 == 0:
                    print(f"Completed {count} new videos.")
            except Exception as e:
                print(f"Error encoding {video_id}: {e}")

        producer_thread.join()
        print("Full ingestion complete.")

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run_ingestion()
