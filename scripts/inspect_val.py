import json

with open("ActivityNet/val_1.json") as f:
    db = json.load(f)

# Find videos with clear single-activity descriptions, manageable duration
candidates = []
for vid, entry in db.items():
    sents = entry.get("sentences", [])
    ts = entry.get("timestamps", [])
    dur = entry.get("duration", 0)
    if len(sents) >= 2 and 30 < dur < 300:
        for i, s in enumerate(sents):
            seg = ts[i] if i < len(ts) else [0, dur]
            seg_len = seg[1] - seg[0]
            if seg_len > 10:  # meaningful segment
                candidates.append((vid, seg, s, dur))

print(f"Total candidates: {len(candidates)}")
for vid, t, s, dur in candidates[:40]:
    print(f"{vid} [{t[0]:.0f}-{t[1]:.0f}s / {dur:.0f}s] {s[:70]}")
