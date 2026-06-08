import json

db1 = json.load(open('ActivityNet/val_1.json'))
try:
    db2 = json.load(open('ActivityNet/val_2.json'))
    db1.update(db2)
except Exception:
    pass

vids  = ['v__Zq8ugolzlA', 'v_juLxWt_3omw', 'v__RCe4Q0p1aA', 'v_Po8gmt7hVTY', 'v_uqiMw7tQ1Cc']
preds = [(60.4, 80.5), (16.0, 28.1), (25.6, 64.0), (85.9, 92.0), (0.0, 55.0)]

for vid, (ps, pe) in zip(vids, preds):
    e = db1.get(vid)
    if not e:
        print(f'{vid}: NOT in annotations')
        continue
    ts    = e.get('timestamps', [])
    sents = e.get('sentences', [])
    dur   = e.get('duration', 0)
    print(f'\n{vid}  dur={dur:.1f}s  pred={ps:.1f}-{pe:.1f}s')
    best_iou, best_t = -1, None
    for t, s in zip(ts, sents):
        inter = max(0, min(pe, t[1]) - max(ps, t[0]))
        union = max(pe, t[1]) - min(ps, t[0])
        iou = inter / union if union > 0 else 0
        if iou > best_iou:
            best_iou, best_t = iou, (t, s)
        print(f'  [{t[0]:.0f}-{t[1]:.0f}s] IoU={iou:.3f}  {s[:60]}')
    if best_t:
        print(f'  >> BEST: [{best_t[0][0]:.0f}-{best_t[0][1]:.0f}s] IoU={best_iou:.3f}')
