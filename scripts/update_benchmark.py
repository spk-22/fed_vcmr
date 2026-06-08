import json

with open('benchmark_queries_10k.json', 'r') as f:
    queries = json.load(f)

custom = [
    {
        "query": "a woman in a bathroom using mouthwash",
        "video_id": "phone_20260502_734",
        "t_start": 0,
        "t_end": 31,
        "augmentation": "custom"
    },
    {
        "query": "a man in a blue shirt at a bathroom sink",
        "video_id": "phone_20260502_735",
        "t_start": 0,
        "t_end": 27,
        "augmentation": "custom"
    }
]

updated = custom + queries
with open('benchmark_queries_updated.json', 'w') as f:
    json.dump(updated, f, indent=2)
