import os
import json
import sqlite3
import numpy as np
from pathlib import Path
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from src.config import (
    ACTIVITYNET_VIDEO_ROOT, ACTIVITYNET_ANNOTATION_PATH, CACHE_ROOT, DB_PATH,
    CHUNK_DURATION, CHUNK_STRIDE, FRAMES_PER_CHUNK,
    DEVICE
)
from src.video_utils import sample_frame_indices, extract_frames_pyav
from src.backbone import MobileCLIPWrapper
from src.metadata_store import MetadataStore

class ActivityNetIngestionPipeline:
    def __init__(self):
        self.backbone = MobileCLIPWrapper(device=DEVICE)
        self.db = MetadataStore(str(DB_PATH))
        self.feature_dir = CACHE_ROOT / "frame_features"
        os.makedirs(self.feature_dir, exist_ok=True)

    def process_metadata_streaming(self):
        """Yields (video_id, video_path, captions) for downloaded ActivityNet videos."""
        print(f"Loading metadata from {ACTIVITYNET_ANNOTATION_PATH}...")
        
        with open(ACTIVITYNET_ANNOTATION_PATH, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
            
        if 'samples' in annotations and isinstance(annotations['samples'], list):
            items = annotations['samples']
        else:
            print("Error: Annotations format does not have a 'samples' list.")
            return

        for sample in items:
            video_id = sample.get('video_id', '')
            
            # Check if we successfully downloaded this video
            video_path = ACTIVITYNET_VIDEO_ROOT / f"{video_id}.mp4"
            if not video_path.exists():
                continue
                
            # ActivityNet captions are stored under timestamps/sentences
            captions = sample.get('sentences', [])
            if not captions:
                # Sometimes it might just be 'sentence'
                sentences = sample.get('sentence', [])
                if isinstance(sentences, str):
                    captions = [sentences]
                elif isinstance(sentences, list):
                    captions = sentences
                    
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
                chunk_id = f"AN_{video_id}_c{chunk_idx}" # Prefix to avoid collision
                
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
        print("Starting ActivityNet ingestion...")
        
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
                        print(f"Skipping already ingested video: {video_id}")
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
                self.db.add_video(video_id, str(video_path), duration, "activitynet")
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
        print("Full ActivityNet ingestion complete.")

if __name__ == "__main__":
    pipeline = ActivityNetIngestionPipeline()
    pipeline.run_ingestion()
