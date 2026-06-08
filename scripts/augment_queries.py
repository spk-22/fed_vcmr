import json
import os
import random

# Config
VAL_JSON = 'ActivityNet/val_1.json'
VIDEO_DIR = 'ActivityNet/videos'
TARGET_VIDS = 733
OUTPUT_FILE = 'benchmark_queries_10k.json'

EXPANSIONS = {
    'someone ':  ['a person ', 'a human '],
    'a person ': ['someone ', 'a human '],
    'people ':   ['a group ', 'individuals '],
    'a man ':    ['a guy ', 'a person '],
}

TEMPLATES = [
    "a video of %s",
    "a video showing %s",
    "a clip of %s",
    "someone %s"
]

def augment():
    with open(VAL_JSON, 'r') as f:
        data = json.load(f)
    
    existing_vids = set([f[:-4] for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])
    all_ids = sorted(data.keys())
    
    selected_vids = []
    for vid in all_ids:
        if vid in existing_vids:
            selected_vids.append(vid)
            if len(selected_vids) >= TARGET_VIDS:
                break
    
    selected_set = set(selected_vids)
    base_queries = []
    for vid in selected_vids:
        for ann in data[vid]['sentences']:
            # ActivityNet sentences are sometimes list of [start, end, text] or dict
            # In val_1.json they are typically tied to timestamps in 'sentences' and 'timestamps'
            pass

    # Re-reading val_1.json structure
    # Based on standard ActivityNet Captions, it's:
    # { vid: { 'duration': X, 'timestamps': [[s, e], ...], 'sentences': ["s1", ...] } }
    
    flat_base = []
    for vid in selected_vids:
        entry = data[vid]
        for i in range(len(entry['sentences'])):
            flat_base.append({
                "query": entry['sentences'][i],
                "video_id": vid,
                "t_start": entry['timestamps'][i][0],
                "t_end": entry['timestamps'][i][1],
                "augmentation": "baseline",
                "base_query": entry['sentences'][i]
            })

    print(f"Base queries: {len(flat_base)}")
    
    augmented = list(flat_base)
    
    # Pass 2: Prefix Swap
    for q in flat_base:
        for prefix, replacements in EXPANSIONS.items():
            if q['query'].lower().startswith(prefix):
                for rep in replacements:
                    new_q = rep + q['query'][len(prefix):]
                    augmented.append({
                        "query": new_q,
                        "video_id": q['video_id'],
                        "t_start": q['t_start'],
                        "t_end": q['t_end'],
                        "augmentation": "prefix_swap",
                        "base_query": q['base_query']
                    })
    
    print(f"Total after Prefix Swap: {len(augmented)}")

    # Pass 4: Template Wrap
    current_count = len(augmented)
    if current_count < 10000:
        needed = 10000 - current_count
        temp_pool = list(flat_base)
        
        while len(augmented) < 10000:
            random.shuffle(temp_pool)
            for q in temp_pool:
                if len(augmented) >= 10000:
                    break
                template = random.choice(TEMPLATES)
                new_q = template % q['query']
                # Check for duplicates
                augmented.append({
                    "query": new_q,
                    "video_id": q['video_id'],
                    "t_start": q['t_start'],
                    "t_end": q['t_end'],
                    "augmentation": "template_wrap",
                    "base_query": q['base_query']
                })

    print(f"Total after Template Wrap: {len(augmented)}")
    
    # If still not enough, repeat with more templates or variations
    if len(augmented) < 10000:
        # Just duplicate some with minor changes
        pass

    final_10k = augmented[:10000]
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_10k, f, indent=2)
    
    print(f"Saved 10,000 queries to {OUTPUT_FILE}")

if __name__ == "__main__":
    augment()
