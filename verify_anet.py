import json
import os

def verify():
    path = os.path.join('ActivityNet', 'val_1.json')
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return

    with open(path, 'r') as f:
        data = json.load(f)
    
    sample_id = list(data.keys())[0]
    sample = data[sample_id]
    
    print(f"Total videos: {len(data)}")
    print(f"Keys for {sample_id}: {list(sample.keys())}")
    print(f"Sample timestamps: {sample.get('timestamps', [])[:3]}")
    print(f"Sample sentences: {sample.get('sentences', [])[:3]}")

if __name__ == "__main__":
    verify()
