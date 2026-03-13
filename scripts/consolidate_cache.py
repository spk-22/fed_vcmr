import numpy as np
import sqlite3
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_chunk(args):
    idx, chunk_id, cache_path, out_shape = args
    if os.path.exists(cache_path):
        data = np.load(cache_path).astype('float16')
        return idx, data
    return idx, None

def consolidate():
    conn = sqlite3.connect('fedvcmr.db')
    chunks = conn.execute(
        'SELECT chunk_id, cache_path FROM chunks ORDER BY chunk_id'
    ).fetchall()
    conn.close()

    n = len(chunks)
    print(f"Consolidating {n} chunks using ThreadPool...")
    
    bin_path = 'cache/frame_features.bin'
    out = np.memmap(bin_path,
                    dtype='float16',
                    mode='w+',
                    shape=(n, 8, 512))

    # Using a thread pool to read files in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        args_list = [(i, chunk_id, cache_path, (8, 512)) for i, (chunk_id, cache_path) in enumerate(chunks)]
        
        for idx, data in tqdm(executor.map(process_chunk, args_list), total=n):
            if data is not None:
                out[idx] = data

    out.flush()
    print(f"Done: {n} chunks consolidated into {bin_path}")

if __name__ == "__main__":
    consolidate()
