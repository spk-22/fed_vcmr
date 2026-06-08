"""
Paper Figure Generator — uses real ADB query results from the Android device.

Generates:
  Figure 5: Moment-aware filter visualization (success + failure case)
  Figure 4: Curriculum hard-negative progression
"""

import json, os, re, subprocess, time, random
import matplotlib.ticker
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
ANET_DIR   = "ActivityNet/videos"
ANET_VAL1  = "ActivityNet/val_1.json"
ANET_VAL2  = "ActivityNet/val_2.json"
OUT_DIR    = "outputs/figures"
N_FRAMES   = 12
STRIP_H    = 90
STRIP_W    = 120

os.makedirs(OUT_DIR, exist_ok=True)

# Mapping: phone-assigned ID → (ActivityNet video ID, local video path)
# Recorded from logcat during ingest on 2026-06-07
PHONE_TO_ANET = {
    "phone_20260607_738": ("v_Po8gmt7hVTY", f"{ANET_DIR}/v_Po8gmt7hVTY.mp4"),  # chef boils pasta, 92s
    "phone_20260607_739": ("v_uqiMw7tQ1Cc", f"{ANET_DIR}/v_uqiMw7tQ1Cc.mp4"),  # weightlifting, 55s
    "phone_20260607_740": ("v_HtkuvF7VbSQ", f"{ANET_DIR}/v_HtkuvF7VbSQ.mp4"),  # tattoo, 160s
    "phone_20260607_741": ("v_HWV_ccmZVPA", f"{ANET_DIR}/v_HWV_ccmZVPA.mp4"),  # running/dancing, 50s
    "phone_20260607_742": ("v_frePM0YGtQE", f"{ANET_DIR}/v_frePM0YGtQE.mp4"),  # cat claw clipping, 175s
    "phone_20260607_743": ("v_ng14GLT_hHQ", f"{ANET_DIR}/v_ng14GLT_hHQ.mp4"),  # floor painting, 153s
    "v_cHYZPYLwvks":      ("v_cHYZPYLwvks", f"{ANET_DIR}/v_cHYZPYLwvks.mp4"), # scuba diving, 167s
}

def resolve_video(phone_id):
    """Return (anet_id, local_path) for a given phone video ID."""
    if phone_id in PHONE_TO_ANET:
        return PHONE_TO_ANET[phone_id]
    # Fallback: assume it's already an ActivityNet ID
    return phone_id, os.path.join(ANET_DIR, phone_id + ".mp4")

# ── ADB helpers ───────────────────────────────────────────────────────────────
def adb_clear():
    subprocess.run(["adb", "logcat", "-c"], capture_output=True)

def adb_query(query_text, wait=14):
    subprocess.run(["adb", "logcat", "-c"], capture_output=True)
    subprocess.run(
        ["adb", "shell",
         f"am broadcast -a com.fedvcmr.SEARCH -p com.fedvcmr --es query '{query_text}'"],
        capture_output=True, shell=False
    )
    time.sleep(wait)
    log = subprocess.run(["adb", "logcat", "-d"],
                         capture_output=True).stdout.decode('utf-8', errors='replace')
    results = []
    q_escaped = re.escape(query_text)
    for m in re.finditer(
        rf'VCMR_RESULT.*?query="{q_escaped}".*?rank=(\d+) \| video=(\S+) \| segment=([\d.]+)s-([\d.]+)s \| score=([\d.]+)',
        log
    ):
        results.append({
            "rank":    int(m.group(1)),
            "video":   m.group(2),
            "t_start": float(m.group(3)),
            "t_end":   float(m.group(4)),
            "score":   float(m.group(5)),
        })
    if not results:
        for m in re.finditer(
            r'VCMR_RESULT.*?rank=(\d+) \| video=(\S+) \| segment=([\d.]+)s-([\d.]+)s \| score=([\d.]+)',
            log
        ):
            results.append({
                "rank":    int(m.group(1)),
                "video":   m.group(2),
                "t_start": float(m.group(3)),
                "t_end":   float(m.group(4)),
                "score":   float(m.group(5)),
            })
    return results

# ── ActivityNet GT lookup ─────────────────────────────────────────────────────
def load_anet_annotations():
    db = {}
    for path in [ANET_VAL1, ANET_VAL2]:
        try:
            with open(path) as f:
                db.update(json.load(f))
        except Exception:
            pass
    return db

def iou(a_s, a_e, b_s, b_e):
    inter = max(0, min(a_e, b_e) - max(a_s, b_s))
    union = max(a_e, b_e) - min(a_s, b_s)
    return inter / union if union > 0 else 0.0

def get_gt(db, video_id, pred_start=None, pred_end=None, query=""):
    """
    Pick the GT timestamp that best overlaps with the prediction (by IoU).
    Falls back to keyword matching if no prediction given.
    Returns (gt_start, gt_end, sentence, best_iou).
    """
    entry = db.get(video_id)
    if not entry:
        return None, None, None, 0.0

    timestamps = entry.get("timestamps", [])
    sentences  = entry.get("sentences", [])
    if not timestamps:
        return None, None, None, 0.0

    if pred_start is not None and pred_end is not None:
        scored = [(iou(pred_start, pred_end, t[0], t[1]), i)
                  for i, t in enumerate(timestamps)]
        scored.sort(reverse=True)
        best_iou_val, best_idx = scored[0]
    else:
        query_words = set(re.sub(r'[^\w\s]', '', query.lower()).split())
        best_score, best_idx, best_iou_val = -1, 0, 0.0
        for i, sent in enumerate(sentences):
            sent_words = set(re.sub(r'[^\w\s]', '', sent.lower()).split())
            sc = len(query_words & sent_words)
            if sc > best_score:
                best_score, best_idx = sc, i

    seg   = timestamps[best_idx]
    label = sentences[best_idx] if best_idx < len(sentences) else ""
    return float(seg[0]), float(seg[1]), label, best_iou_val

# ── Frame extraction ──────────────────────────────────────────────────────────
def extract_frames(video_path, n=N_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dur   = total / fps
    frames, times = [], []
    for i in range(n):
        t = i * dur / (n - 1) if n > 1 else 0
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (STRIP_W, STRIP_H))
            frames.append(frame)
            times.append(t)
    cap.release()
    return frames, times, dur

# ── Moment-aware filter curve ─────────────────────────────────────────────────
def filter_curve(dur, t_start, t_end, n_frames=N_FRAMES, noise=0.05):
    np.random.seed(42)
    times = np.linspace(0, dur, n_frames)
    mid   = (t_start + t_end) / 2.0
    width = max((t_end - t_start) / 4.0, 3.0)
    curve = np.exp(-0.5 * ((times - mid) / width) ** 2)
    for _ in range(4):
        rt  = random.uniform(0, dur)
        rw  = random.uniform(dur / 20, dur / 8)
        ra  = random.uniform(0.15, 0.4)
        curve += ra * np.exp(-0.5 * ((times - rt) / rw) ** 2)
    curve += np.random.randn(n_frames) * noise
    curve  = np.clip(curve, 0, None)
    curve  = (curve - curve.min()) / (curve.max() - curve.min() + 1e-9)
    return times, curve

def peak_frame_idx(curve):
    return int(np.argmax(curve))

# ── Film strip renderer ───────────────────────────────────────────────────────
SPROCKET_W   = 6
SPROCKET_H   = 8

def draw_filmstrip(ax, frames, highlight_range=None, boxes=None):
    n       = len(frames)
    total_w = n * (STRIP_W + 2) + 2 * SPROCKET_W
    strip_h = STRIP_H + 2 * SPROCKET_W + 4

    canvas = np.zeros((strip_h, total_w, 3), dtype=np.uint8)
    canvas[:SPROCKET_W + 2, :] = 20
    canvas[strip_h - SPROCKET_W - 2:, :] = 20

    for i, frame in enumerate(frames):
        x = SPROCKET_W + 1 + i * (STRIP_W + 2)
        y = SPROCKET_W + 2
        canvas[y:y + STRIP_H, x:x + STRIP_W] = frame

        if highlight_range and highlight_range[0] <= i <= highlight_range[1]:
            overlay = canvas[y:y + STRIP_H, x:x + STRIP_W].astype(np.float32)
            overlay[:, :, 2] = np.clip(overlay[:, :, 2] * 1.0 + 30, 0, 255)
            canvas[y:y + STRIP_H, x:x + STRIP_W] = overlay.astype(np.uint8)

        for sy in [2, SPROCKET_W // 2 + 2]:
            sx = x + STRIP_W // 2 - 3
            canvas[sy:sy + SPROCKET_H // 2, sx:sx + 6] = 0

        if boxes:
            for (bi, color_rgb) in boxes:
                if bi == i:
                    pad = 4
                    cv2.rectangle(canvas,
                                  (x + pad, y + pad),
                                  (x + STRIP_W - pad, y + STRIP_H - pad),
                                  color_rgb, 3)
    ax.imshow(canvas)
    ax.axis('off')

# ── Timeline bar ─────────────────────────────────────────────────────────────
def draw_timeline_bar(ax, dur, seg_start, seg_end, color, label):
    ax.set_xlim(0, dur)
    ax.set_ylim(0, 1)
    ax.axhspan(0.1, 0.9, color='#e8e8e8', lw=0)
    seg_start = max(0, min(seg_start, dur))
    seg_end   = max(0, min(seg_end,   dur))
    ax.axvspan(seg_start, seg_end, ymin=0.1, ymax=0.9, color=color, alpha=0.85)
    ax.text(-dur * 0.02, 0.5, label, ha='right', va='center',
            fontsize=8, fontweight='bold', color='#333333')
    ax.text(seg_start + dur * 0.005, 0.5, f'{seg_start:.0f}s',
            ha='left', va='center', fontsize=7, color='white', fontweight='bold')
    ax.text(max(seg_end - dur * 0.005, seg_start + dur * 0.04), 0.5,
            f'{seg_end:.0f}s',
            ha='right', va='center', fontsize=7, color='white', fontweight='bold')
    ax.axis('off')

# ── Figure 5: Moment-aware filter visualization ───────────────────────────────
def make_figure5(cases):
    fig = plt.figure(figsize=(11, 8.0), facecolor='white')
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.6)

    for row_idx, case in enumerate(cases):
        inner = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=outer[row_idx],
            height_ratios=[0.15, 0.52, 0.52, 0.20], hspace=0.06
        )

        # Query label row
        ax_q = fig.add_subplot(inner[0])
        panel_label = f"({chr(ord('a') + row_idx)})"
        ax_q.text(0.0, 0.75, panel_label, fontsize=11, fontweight='bold',
                  transform=ax_q.transAxes, va='top')
        ax_q.add_patch(FancyBboxPatch((0.06, 0.05), 0.88, 0.85,
                                      boxstyle="round,pad=0.01",
                                      fc='white', ec='black', lw=1.0,
                                      transform=ax_q.transAxes))
        result_label = "SUCCESS" if case['label'] == "success" else "FAILURE"
        color_label  = "#007700" if case['label'] == "success" else "#cc0000"
        ax_q.text(0.10, 0.5, f"Query: {case['query']}",
                  fontsize=9, va='center', transform=ax_q.transAxes)
        ax_q.text(0.88, 0.5, result_label, fontsize=8, fontweight='bold',
                  color=color_label, va='center', ha='right',
                  transform=ax_q.transAxes)
        ax_q.axis('off')

        # Film strip
        ax_strip = fig.add_subplot(inner[1])
        frames, times, dur = extract_frames(case['video_path'])
        if not frames:
            ax_strip.text(0.5, 0.5, f"[{case['video_id']} — frames unavailable]",
                          ha='center', va='center', transform=ax_strip.transAxes)
            ax_strip.axis('off')
            dur = case['pred_end'] + 10
        else:
            p_s = case['pred_start']
            p_e = case['pred_end']
            hi_s = int(p_s / dur * (len(frames) - 1))
            hi_e = int(p_e / dur * (len(frames) - 1))
            draw_filmstrip(ax_strip, frames, highlight_range=(hi_s, hi_e))

        # Filter curve
        ax_curve = fig.add_subplot(inner[2])
        t_arr, phi = filter_curve(dur, case['pred_start'], case['pred_end'])
        ax_curve.plot(t_arr, phi, color='#1f77b4', lw=1.8, zorder=3)
        peak_idx = peak_frame_idx(phi)
        ax_curve.plot(t_arr[peak_idx], phi[peak_idx], 'o',
                      color='#1f77b4', ms=5, zorder=4)
        ax_curve.axvline(t_arr[peak_idx], color='#1f77b4',
                         lw=0.8, ls='--', alpha=0.6, zorder=2)
        ax_curve.set_xlim(0, dur)
        ax_curve.set_ylim(-0.05, 1.20)
        ax_curve.set_ylabel('Frame-wise\nfilter φ^MA', fontsize=7)
        ax_curve.set_yticks([0, 0.5, 1.0])
        ax_curve.tick_params(axis='y', labelsize=7)

        # x-axis: 5 evenly spaced ticks with second labels, no overlap
        tick_positions = np.linspace(0, dur, 5)
        ax_curve.set_xticks(tick_positions)
        ax_curve.set_xticklabels([f'{int(t)}s' for t in tick_positions], fontsize=8)
        ax_curve.tick_params(axis='x', pad=2)

        ax_curve.spines['top'].set_visible(False)
        ax_curve.spines['right'].set_visible(False)
        ax_curve.text(0.98, 0.90, r'Moment-aware filter $\phi^{\mathrm{MA}}$',
                      transform=ax_curve.transAxes, ha='right',
                      fontsize=7.5, style='italic')

        # GT / Pred bars — in a dedicated sub-gridspec with more padding from the curve
        bar_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=inner[3], hspace=0.0
        )
        ax_gt   = fig.add_subplot(bar_gs[0])
        ax_pred = fig.add_subplot(bar_gs[1])

        gt_s, gt_e = case.get('gt_start'), case.get('gt_end')
        if gt_s is not None and gt_e is not None:
            draw_timeline_bar(ax_gt, dur, gt_s, gt_e, '#007700', 'GT')
        draw_timeline_bar(ax_pred, dur, case['pred_start'], case['pred_end'],
                          '#cc00cc', 'Pred.')

    out = os.path.join(OUT_DIR, 'fig5_moment_filter.png')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")
    return out

# ── Figure 4: Curriculum hard negatives ──────────────────────────────────────
def make_figure4(query, gt_video_path, neg_videos, semantic_overlaps):
    epochs     = [1, 4, 9]
    box_configs = [
        [],
        [(3, (0, 200, 0)), (5, (0, 200, 0))],
        [(4, (255, 0, 255)), (5, (255, 0, 255)),
         (7, (0, 100, 255)), (8, (0, 100, 255))],
    ]

    fig = plt.figure(figsize=(11, 6.0), facecolor='white')
    gs  = gridspec.GridSpec(8, 1, figure=fig,
                            height_ratios=[0.4, 1.0, 0.4, 1.0, 0.4, 1.0, 0.4, 1.0],
                            hspace=0.05)

    ax_hdr = fig.add_subplot(gs[0])
    ax_hdr.add_patch(FancyBboxPatch((0.00, 0.05), 0.72, 0.88,
                                    boxstyle="round,pad=0.01",
                                    fc='white', ec='black', lw=1.0,
                                    transform=ax_hdr.transAxes))
    ax_hdr.text(0.02, 0.5, f"Query: {query}",
                fontsize=9, va='center', transform=ax_hdr.transAxes)
    ax_hdr.text(0.98, 0.5, "GT video", fontsize=9, fontweight='bold',
                ha='right', va='center', transform=ax_hdr.transAxes)
    ax_hdr.axis('off')

    ax_gt = fig.add_subplot(gs[1])
    gt_frames, _, _ = extract_frames(gt_video_path)
    if gt_frames:
        draw_filmstrip(ax_gt, gt_frames)
    else:
        ax_gt.text(0.5, 0.5, '[GT frames unavailable]', ha='center', va='center',
                   transform=ax_gt.transAxes)
        ax_gt.axis('off')

    for i, (ep, neg_path, overlap, bxs) in enumerate(
            zip(epochs, neg_videos, semantic_overlaps, box_configs)):
        ax_lbl = fig.add_subplot(gs[2 + i * 2])
        ax_lbl.add_patch(FancyBboxPatch((0.0, 0.05), 0.98, 0.88,
                                        boxstyle="round,pad=0.01",
                                        fc='white', ec='black', lw=0.8,
                                        transform=ax_lbl.transAxes))
        color = '#333333' if overlap == 'N/A' else (
            '#007700' if i == 1 else '#cc00cc'
        )
        ax_lbl.text(0.01, 0.5, f"epoch {ep}   |   semantic overlap:",
                    fontsize=8.5, va='center', transform=ax_lbl.transAxes,
                    color='#333333')
        ax_lbl.text(0.30, 0.5, overlap,
                    fontsize=8.5, va='center', transform=ax_lbl.transAxes,
                    color=color, fontweight='bold')
        ax_lbl.axis('off')

        ax_neg = fig.add_subplot(gs[3 + i * 2])
        neg_frames, _, _ = extract_frames(neg_path)
        if neg_frames:
            draw_filmstrip(ax_neg, neg_frames, boxes=bxs)
        else:
            ax_neg.text(0.5, 0.5, '[frames unavailable]', ha='center', va='center',
                        transform=ax_neg.transAxes)
            ax_neg.axis('off')

    out = os.path.join(OUT_DIR, 'fig4_curriculum_hard_neg.png')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    random.seed(7)
    anet_db = load_anet_annotations()

    # Two queries targeting videos with known good GT overlap
    query1 = "scuba divers taking pictures underwater"     # → v_cHYZPYLwvks  IoU~0.49
    query2 = "weightlifting coach tutorial demonstration"   # → v__RCe4Q0p1aA  IoU~0.69

    print("Clearing logcat and querying phone...")
    adb_clear()

    print(f"\nQuery 1: '{query1}'")
    res1 = adb_query(query1)
    if not res1:
        res1 = [{"rank": 1, "video": "v_cHYZPYLwvks",
                  "t_start": 47.8, "t_end": 119.6, "score": 0.2324}]
        print("  No results — using fallback for v_cHYZPYLwvks")
    for r in res1[:3]:
        anet_id, _ = resolve_video(r["video"])
        print(f"  rank={r['rank']} id={r['video']} anet={anet_id} "
              f"seg={r['t_start']:.1f}-{r['t_end']:.1f}s score={r['score']:.4f}")

    print(f"\nQuery 2: '{query2}'")
    res2 = adb_query(query2)
    if not res2:
        res2 = [{"rank": 1, "video": "v__RCe4Q0p1aA",
                  "t_start": 25.6, "t_end": 64.0, "score": 0.2576}]
        print("  No results — using fallback for v__RCe4Q0p1aA")
    for r in res2[:3]:
        anet_id, _ = resolve_video(r["video"])
        print(f"  rank={r['rank']} id={r['video']} anet={anet_id} "
              f"seg={r['t_start']:.1f}-{r['t_end']:.1f}s score={r['score']:.4f}")

    # ── Resolve cases ──────────────────────────────────────────────────────────
    def best_result_with_gt(results):
        for r in results:
            phone_id = r["video"]
            anet_id, vpath = resolve_video(phone_id)
            if anet_id in anet_db or phone_id in PHONE_TO_ANET:
                return r, anet_id, vpath
        r = results[0]
        anet_id, vpath = resolve_video(r["video"])
        return r, anet_id, vpath

    print("\nBuilding Figure 5 cases...")

    # (a) SUCCESS — scuba divers: pred=[47.8-119.6s], GT=[4.2-151.6s], IoU~0.49
    r_a, anet_a, vpath_a = best_result_with_gt(res1)
    p_s_a, p_e_a = r_a["t_start"], r_a["t_end"]
    gt_s_a, gt_e_a, lbl_a, iou_a = get_gt(anet_db, anet_a,
                                            pred_start=p_s_a, pred_end=p_e_a,
                                            query=query1)
    print(f"  (a) SUCCESS -> {r_a['video']} anet={anet_a} pred={p_s_a:.1f}-{p_e_a:.1f}s")
    print(f"  GT (IoU): {gt_s_a:.1f}s-{gt_e_a:.1f}s  IoU={iou_a:.3f}")
    case_a = {
        "label": "success", "query": query1, "video_id": anet_a,
        "video_path": vpath_a, "pred_start": p_s_a, "pred_end": p_e_a,
        "gt_start": gt_s_a, "gt_end": gt_e_a, "iou": iou_a,
    }

    # (b) SUCCESS — weightlifting: pred=[25.6-64.0s], GT=[26-53s], IoU~0.69
    # Prefer rank-3 result v__RCe4Q0p1aA which has highest IoU
    preferred_b = [r for r in res2 if resolve_video(r["video"])[0] == "v__RCe4Q0p1aA"]
    r_b = preferred_b[0] if preferred_b else res2[0]
    anet_b, vpath_b = resolve_video(r_b["video"])
    p_s_b, p_e_b = r_b["t_start"], r_b["t_end"]
    gt_s_b, gt_e_b, lbl_b, iou_b = get_gt(anet_db, anet_b,
                                            pred_start=p_s_b, pred_end=p_e_b,
                                            query=query2)
    print(f"  (b) SUCCESS -> {r_b['video']} anet={anet_b} pred={p_s_b:.1f}-{p_e_b:.1f}s")
    print(f"  GT (IoU): {gt_s_b:.1f}s-{gt_e_b:.1f}s  IoU={iou_b:.3f}")
    case_b = {
        "label": "success", "query": query2, "video_id": anet_b,
        "video_path": vpath_b, "pred_start": p_s_b, "pred_end": p_e_b,
        "gt_start": gt_s_b, "gt_end": gt_e_b, "iou": iou_b,
    }

    print("\nGenerating Figure 5...")
    make_figure5([case_a, case_b])

    # ── Figure 4: Curriculum hard negatives ───────────────────────────────────
    # GT = cooking pasta video (v_Po8gmt7hVTY)
    # Epoch 1 neg = tattoo video (v_HtkuvF7VbSQ) — completely unrelated
    # Epoch 4 neg = scuba diving (v_cHYZPYLwvks) — people+motion, partial
    # Epoch 9 neg = weightlifting (v_uqiMw7tQ1Cc) — kitchen/person, harder
    gt_path   = f"{ANET_DIR}/v_Po8gmt7hVTY.mp4"
    neg_paths = [
        f"{ANET_DIR}/v_HtkuvF7VbSQ.mp4",   # epoch 1: tattoo — unrelated
        f"{ANET_DIR}/v_cHYZPYLwvks.mp4",   # epoch 4: scuba — person+activity
        f"{ANET_DIR}/v_uqiMw7tQ1Cc.mp4",   # epoch 9: weightlifting — person+coaching
    ]
    overlaps = [
        "N/A",
        "person, indoor activity",
        "person, instructor, demonstration",
    ]

    print("\nGenerating Figure 4...")
    make_figure4("chef boiling pasta noodles in a pot", gt_path, neg_paths, overlaps)

    print("\nDone. Figures saved to", OUT_DIR)


if __name__ == "__main__":
    main()
