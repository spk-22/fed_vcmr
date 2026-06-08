"""Parse logcat output, pull video from phone, extract midpoint JPG."""
import sys, re, subprocess, os, cv2

query, log_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
safe = re.sub(r'[^A-Za-z0-9]', '_', query)[:40]

# Parse "rank=N | video=X | segment=Ss-Es | score=F"
pat = re.compile(r'rank=(\d+)\s*\|\s*video=(\S+)\s*\|\s*segment=([\d.]+)s-([\d.]+)s\s*\|\s*score=([\d.]+)')
hits = []
with open(log_path, errors='ignore') as f:
    for line in f:
        m = pat.search(line)
        if m and query in line:
            hits.append((int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4)), float(m.group(5))))

if not hits:
    print(f"[!] No hits parsed from logcat. Log contents:")
    print(open(log_path).read())
    sys.exit(1)

hits.sort(key=lambda h: h[0])
print(f'\n=== Query: "{query}" ===')
for rank, vid, ts, te, sc in hits[:3]:
    mid = (ts + te) / 2.0
    print(f'  Rank#{rank}: {vid} @ {mid:.1f}s (seg {ts:.1f}-{te:.1f}s, score={sc:.4f})')

    # Pull video
    tmp_vid = f'C:/prism/pc_verification/tmp/{vid}.mp4'
    if not os.path.exists(tmp_vid):
        r = subprocess.run(['adb', 'pull', f'//sdcard/fedvcmr/user_videos/{vid}.mp4', tmp_vid],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f'    [!] pull failed: {r.stderr.strip()}'); continue

    # Extract midpoint frame
    cap = cv2.VideoCapture(tmp_vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f'    [!] frame read failed at {mid:.1f}s'); continue
    out = f'{out_dir}/rank{rank}_{safe}_{vid}_{mid:.1f}s.jpg'
    cv2.imwrite(out, frame)
    print(f'    -> {out}')
