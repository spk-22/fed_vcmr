import av
import numpy as np
from PIL import Image
from typing import List, Tuple
import torch

def sample_frame_indices(duration: float, fps: float, num_frames: int) -> List[float]:
    """Sample num_frames evenly spaced timestamps."""
    if duration <= 0:
        return [0.0] * num_frames
    return np.linspace(0, duration, num_frames).tolist()

def extract_frames_pyav(video_path: str, timestamps: List[float]) -> List[Image.Image]:
    """Extract frames at specific timestamps using PyAV."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    
    frames = []
    # Sort timestamps to minimize seeking
    sorted_ts = sorted(timestamps)
    
    for ts in sorted_ts:
        # Seek to the timestamp (in microseconds for av)
        container.seek(int(ts * 1000000), any_frame=False, backward=True, stream=stream)
        found = False
        for frame in container.decode(video=0):
            # Check if this frame is close enough or the first one after seek
            # PyAV seek is not always exact to the frame, but decode starts from nearest keyframe
            if frame.time >= ts:
                frames.append(frame.to_image())
                found = True
                break
        if not found:
            # If we reached end of stream, use the last frame
            pass 
            
    container.close()
    
    # Reorder frames if timestamps were shuffled (unlikely for our use case)
    # and pad if any frames were missed
    while len(frames) < len(timestamps):
        if frames:
            frames.append(frames[-1])
        else:
            # Fallback for empty video or error
            frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))
            
    return frames[:len(timestamps)]
