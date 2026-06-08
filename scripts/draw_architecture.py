"""
FedVCMR Publication-Quality Architecture Diagram
Generates: C:/prism/outputs/figures/fedvcmr_architecture.pdf + .png
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Circle, Wedge
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe

matplotlib.rcParams.update({"font.family": "serif", "font.size": 9})

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#FAFAFA")

# ── Palette ───────────────────────────────────────────────────────────────
C_BLUE      = "#3A7DC9"
C_BLUE_L    = "#D0E4F7"
C_ORANGE    = "#E8833A"
C_ORANGE_L  = "#FDEBD0"
C_GREEN     = "#2EAA6E"
C_GREEN_L   = "#D5F2E3"
C_PURPLE    = "#7B5EA7"
C_PURPLE_L  = "#EAE0F5"
C_RED       = "#D94F3D"
C_RED_L     = "#FAE0DD"
C_GRAY      = "#555555"
C_GRAY_L    = "#EEEEEE"
C_DARK      = "#2C2C2C"
C_YELLOW_L  = "#FFF8DC"

# ── Helpers ───────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, fc, ec, radius=0.18, alpha=1.0, lw=1.5, zorder=3):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    return box

def shadow(ax, x, y, w, h, radius=0.18, zorder=2):
    box = FancyBboxPatch((x+0.06, y-0.06), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         fc="#CCCCCC", ec="none", alpha=0.5, zorder=zorder)
    ax.add_patch(box)

def label(ax, x, y, txt, fs=9, color=C_DARK, bold=False, ha="center", va="center", zorder=6):
    w = "bold" if bold else "normal"
    ax.text(x, y, txt, fontsize=fs, color=color, fontweight=w,
            ha=ha, va=va, zorder=zorder)

def arrow(ax, x1, y1, x2, y2, color=C_GRAY, lw=1.8, style="-|>", zorder=4, label_txt=None, lfs=7.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=zorder)
    if label_txt:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.12, label_txt, fontsize=lfs, color=color,
                ha="center", va="bottom", style="italic", zorder=6)

def curved_arrow(ax, x1, y1, x2, y2, rad=0.25, color=C_GRAY, lw=1.8, zorder=4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=zorder)

def nn_tower(ax, cx, cy, width, height, color, ec, n_layers=4, label_str="", fs=8):
    """Draw a stylized neural network tower (stacked layers)."""
    lh = height / (n_layers + 0.5)
    for i in range(n_layers):
        alpha = 0.5 + 0.5 * (i / (n_layers - 1)) if n_layers > 1 else 0.9
        y0 = cy + i * lh
        rbox(ax, cx - width/2, y0, width, lh * 0.82, fc=color, ec=ec,
             radius=0.07, alpha=alpha, lw=1.2, zorder=3)
    label(ax, cx, cy + height/2, label_str, fs=fs, bold=True, color=ec)

def embed_vector(ax, cx, cy, length=1.1, height=0.28, colors=None):
    """Draw a colorful embedding vector bar."""
    if colors is None:
        colors = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_RED, C_BLUE, C_GREEN, C_ORANGE]
    n = len(colors)
    w = length / n
    for i, c in enumerate(colors):
        rbox(ax, cx - length/2 + i*w, cy - height/2, w, height,
             fc=c, ec="white", radius=0.03, lw=0.8, zorder=4)

def cylinder(ax, cx, cy, rx=0.55, ry=0.18, height=0.9, fc="#D0E4F7", ec=C_BLUE, lw=1.5, zorder=3):
    """Draw a cylinder (database/index)."""
    from matplotlib.patches import Ellipse
    # body
    rect = plt.Polygon([[cx-rx, cy], [cx-rx, cy+height],
                         [cx+rx, cy+height], [cx+rx, cy]],
                        closed=True, fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(rect)
    # top ellipse
    top = Ellipse((cx, cy+height), 2*rx, 2*ry, fc=fc, ec=ec, lw=lw, zorder=zorder+1)
    ax.add_patch(top)
    # bottom ellipse
    bot = Ellipse((cx, cy), 2*rx, 2*ry, fc=fc, ec=ec, lw=lw, zorder=zorder+1)
    ax.add_patch(bot)
    # highlight stripe on top
    top2 = Ellipse((cx, cy+height), 2*rx*0.85, 2*ry*0.7, fc="white", ec="none",
                   alpha=0.35, zorder=zorder+2)
    ax.add_patch(top2)

def phone_outline(ax, cx, cy, w=1.1, h=1.9, fc="white", ec=C_DARK, lw=2, zorder=3):
    rbox(ax, cx-w/2, cy, w, h, fc=fc, ec=ec, radius=0.18, lw=lw, zorder=zorder)
    # screen
    rbox(ax, cx-w/2+0.1, cy+0.22, w-0.2, h-0.42, fc="#E8F4FD", ec="#AAAAAA",
         radius=0.08, lw=0.8, zorder=zorder+1)
    # home button
    circ = Circle((cx, cy+0.12), 0.07, fc="#DDDDDD", ec="#AAAAAA", lw=0.8, zorder=zorder+1)
    ax.add_patch(circ)

def timeline_bar(ax, cx, cy, length=1.4, height=0.22, highlight_start=0.38, highlight_end=0.72,
                 ec=C_BLUE):
    """Draw a video timeline with a highlighted moment."""
    rbox(ax, cx-length/2, cy-height/2, length, height, fc="#E0E0E0", ec="#AAAAAA",
         radius=0.05, lw=1.0, zorder=4)
    hs = cx - length/2 + highlight_start * length
    he = cx - length/2 + highlight_end * length
    rbox(ax, hs, cy-height/2, he-hs, height, fc=C_ORANGE, ec=C_ORANGE,
         radius=0.05, lw=1.0, zorder=5)
    # tick marks
    for t in np.linspace(cx-length/2, cx+length/2, 7):
        ax.plot([t, t], [cy-height/2-0.05, cy-height/2], color="#888888", lw=0.8, zorder=5)


# ══════════════════════════════════════════════════════════════════════════
# SECTION BACKGROUNDS
# ══════════════════════════════════════════════════════════════════════════

# Panel A: Query Encoding
shadow(ax, 0.25, 1.5, 3.6, 6.3, radius=0.3, zorder=1)
rbox(ax, 0.25, 1.5, 3.6, 6.3, fc=C_BLUE_L, ec=C_BLUE, radius=0.3, alpha=0.35, lw=2, zorder=2)
label(ax, 2.05, 7.6, "① Query Encoding", fs=10, bold=True, color=C_BLUE)

# Panel B: On-Device Index
shadow(ax, 4.15, 1.5, 4.2, 6.3, radius=0.3, zorder=1)
rbox(ax, 4.15, 1.5, 4.2, 6.3, fc=C_GREEN_L, ec=C_GREEN, radius=0.3, alpha=0.35, lw=2, zorder=2)
label(ax, 6.25, 7.6, "② On-Device Feature Index", fs=10, bold=True, color=C_GREEN)

# Panel C: DGSE Scoring
shadow(ax, 8.65, 1.5, 3.8, 6.3, radius=0.3, zorder=1)
rbox(ax, 8.65, 1.5, 3.8, 6.3, fc=C_ORANGE_L, ec=C_ORANGE, radius=0.3, alpha=0.35, lw=2, zorder=2)
label(ax, 10.55, 7.6, "③ DGSE Scoring", fs=10, bold=True, color=C_ORANGE)

# Panel D: Temporal Grounding + Output
shadow(ax, 12.75, 1.5, 2.9, 6.3, radius=0.3, zorder=1)
rbox(ax, 12.75, 1.5, 2.9, 6.3, fc=C_PURPLE_L, ec=C_PURPLE, radius=0.3, alpha=0.35, lw=2, zorder=2)
label(ax, 14.2, 7.6, "④ Temporal\nGrounding", fs=10, bold=True, color=C_PURPLE)


# ══════════════════════════════════════════════════════════════════════════
# PANEL A: QUERY ENCODING
# ══════════════════════════════════════════════════════════════════════════

# Text query box
shadow(ax, 0.55, 6.1, 3.0, 0.75)
rbox(ax, 0.55, 6.1, 3.0, 0.75, fc="white", ec=C_BLUE, radius=0.15, lw=2)
ax.text(2.05, 6.65, '"', fontsize=22, color=C_BLUE, ha="center", va="center",
        fontweight="bold", alpha=0.3, zorder=4)
label(ax, 2.05, 6.55, "a person rides a horse", fs=8.5, color=C_DARK, zorder=5)
label(ax, 2.05, 6.22, "through the countryside", fs=8.5, color=C_DARK, zorder=5)
label(ax, 0.72, 7.0, "Text Query", fs=8, color=C_BLUE, bold=True)

# Query augmentation templates
rbox(ax, 0.55, 4.95, 3.0, 0.9, fc=C_YELLOW_L, ec="#CCAA00", radius=0.12, lw=1.2)
label(ax, 2.05, 5.65, "Query Augmentation", fs=7.5, bold=True, color="#886600")
label(ax, 2.05, 5.38, '"%s"  |  "a video of %s"', fs=7.2, color="#555500")
label(ax, 2.05, 5.12, "Template Ensemble (×2)", fs=7, color="#888800", bold=False)

# MobileCLIP Text Encoder
shadow(ax, 0.55, 3.0, 3.0, 1.65)
rbox(ax, 0.55, 3.0, 3.0, 1.65, fc="white", ec=C_BLUE, radius=0.15, lw=2)
nn_tower(ax, 2.05, 3.1, 1.6, 1.35, color=C_BLUE_L, ec=C_BLUE, n_layers=4,
         label_str="", fs=8)
label(ax, 2.05, 4.5, "MobileCLIP-S1", fs=8.5, bold=True, color=C_BLUE)
label(ax, 2.05, 4.25, "Text Encoder", fs=8, color=C_BLUE)
# small chip label
rbox(ax, 1.45, 3.05, 1.2, 0.32, fc=C_BLUE, ec=C_BLUE, radius=0.08, lw=1)
label(ax, 2.05, 3.21, "Transformer  ×6", fs=6.8, color="white", bold=False)

# Query Embedding
label(ax, 2.05, 2.75, "Query Embedding  q ∈ ℝ⁵¹²", fs=7.5, color=C_DARK)
embed_vector(ax, 2.05, 2.42, length=2.6, height=0.32,
             colors=[C_BLUE, C_PURPLE, C_GREEN, C_ORANGE, C_RED,
                     C_BLUE, C_ORANGE, C_PURPLE, C_GREEN, C_RED,
                     C_BLUE, C_GREEN])
# L2 norm badge
rbox(ax, 1.35, 1.92, 1.4, 0.3, fc=C_BLUE, ec=C_BLUE, radius=0.1, lw=1)
label(ax, 2.05, 2.07, "L2 Normalize", fs=7, color="white", bold=True)

# Arrows within Panel A
arrow(ax, 2.05, 6.1, 2.05, 5.85, color=C_BLUE, lw=1.6)
arrow(ax, 2.05, 4.95, 2.05, 4.65, color=C_BLUE, lw=1.6)
arrow(ax, 2.05, 3.0, 2.05, 2.75, color=C_BLUE, lw=1.6)
arrow(ax, 2.05, 2.22, 2.05, 1.96, color=C_BLUE, lw=1.6)


# ══════════════════════════════════════════════════════════════════════════
# PANEL B: ON-DEVICE FEATURE INDEX
# ══════════════════════════════════════════════════════════════════════════

# Cylinder: Feature Blob
cylinder(ax, 6.25, 4.8, rx=0.75, ry=0.22, height=1.6, fc=C_GREEN_L, ec=C_GREEN, lw=2)
label(ax, 6.25, 5.75, "Feature Blob", fs=8.5, bold=True, color=C_GREEN, zorder=8)
label(ax, 6.25, 5.48, "user_features_blob.bin", fs=7, color="#1A7A4A", zorder=8)
label(ax, 6.25, 5.22, "Memory-Mapped  (mmap)", fs=7, color="#1A7A4A", zorder=8)
label(ax, 6.25, 4.96, "12 MB  |  819 videos", fs=7, color="#1A7A4A", zorder=8)

# Index JSON
shadow(ax, 4.35, 6.75, 3.8, 0.85)
rbox(ax, 4.35, 6.75, 3.8, 0.85, fc="white", ec=C_GREEN, radius=0.15, lw=1.5)
label(ax, 6.25, 7.25, "user_index.json", fs=8, bold=True, color=C_GREEN)
label(ax, 6.25, 6.98, "video_id  |  duration  |  num_frames  |  offset", fs=7, color=C_DARK)

# Frame embeddings detail box
shadow(ax, 4.35, 3.2, 3.8, 1.35)
rbox(ax, 4.35, 3.2, 3.8, 1.35, fc="white", ec=C_GREEN, radius=0.15, lw=1.5)
label(ax, 6.25, 4.3, "Per-Video Frame Embeddings", fs=8, bold=True, color=C_GREEN)
label(ax, 6.25, 4.04, "F ∈ ℝ^{N×512}  (N=16 frames/video)", fs=7.5, color=C_DARK)
# mini frame vectors
for i in range(4):
    yy = 3.28 + i * 0.17
    cc = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_RED, C_BLUE, C_ORANGE, C_GREEN]
    embed_vector(ax, 6.25, yy + 0.085, length=2.8, height=0.13, colors=cc)
label(ax, 5.05, 3.55, "f₁", fs=7, color=C_DARK)
label(ax, 5.05, 3.72, "f₂", fs=7, color=C_DARK)
label(ax, 5.05, 3.88, "f₁₆", fs=7, color=C_DARK)

# LRU Query Cache
shadow(ax, 4.35, 2.1, 3.8, 0.82)
rbox(ax, 4.35, 2.1, 3.8, 0.82, fc=C_YELLOW_L, ec="#CCAA00", radius=0.12, lw=1.5)
label(ax, 6.25, 2.73, "LRU Query Cache  (cap=1000)", fs=8, bold=True, color="#886600")
label(ax, 6.25, 2.46, "Skips re-encoding repeated queries", fs=7.5, color="#666600")
label(ax, 6.25, 2.22, "Cache hit → 0ms encoding latency", fs=7, color="#888800")

# Arrows within Panel B
arrow(ax, 6.25, 7.6, 6.25, 7.6, color=C_GREEN, lw=0.1)  # placeholder
arrow(ax, 6.25, 6.75, 6.25, 6.4, color=C_GREEN, lw=1.6)
arrow(ax, 6.25, 4.8, 6.25, 4.55, color=C_GREEN, lw=1.6)
arrow(ax, 6.25, 3.2, 6.25, 2.92, color=C_GREEN, lw=1.6)


# ══════════════════════════════════════════════════════════════════════════
# PANEL C: DGSE SCORING
# ══════════════════════════════════════════════════════════════════════════

# Dot product scoring
shadow(ax, 8.85, 5.85, 3.4, 1.4)
rbox(ax, 8.85, 5.85, 3.4, 1.4, fc="white", ec=C_ORANGE, radius=0.15, lw=2)
label(ax, 10.55, 7.0, "Dot-Product Similarity", fs=8.5, bold=True, color=C_ORANGE)
label(ax, 10.55, 6.72, "s(q, fₙ) = q · fₙ   ∀ videos, frames", fs=8, color=C_DARK)
# Score heatmap visualization
for i in range(8):
    for j in range(3):
        val = np.random.uniform(0.1, 0.95)
        c = plt.cm.RdYlGn(val)
        rbox(ax, 9.05 + i*0.32, 5.95 + j*0.24, 0.28, 0.2,
             fc=c, ec="white", radius=0.03, lw=0.5)
label(ax, 10.55, 5.89, "Frame score matrix (videos × frames)", fs=6.5, color="#888888")

# Temporal Smoothing
shadow(ax, 8.85, 4.55, 3.4, 1.0)
rbox(ax, 8.85, 4.55, 3.4, 1.0, fc="white", ec=C_ORANGE, radius=0.15, lw=2)
label(ax, 10.55, 5.3, "Temporal Smoothing", fs=8.5, bold=True, color=C_ORANGE)
label(ax, 10.55, 5.04, "s̃ₙ = mean(sₙ₋₁, sₙ, sₙ₊₁)", fs=8, color=C_DARK)
label(ax, 10.55, 4.78, "Reduces spike artifacts in scoring", fs=7.2, color="#888888")

# Hubness Suppression
shadow(ax, 8.85, 3.25, 3.4, 1.0)
rbox(ax, 8.85, 3.25, 3.4, 1.0, fc="white", ec=C_ORANGE, radius=0.15, lw=2)
label(ax, 10.55, 4.0, "Hubness Suppression (λ=0.35)", fs=8.5, bold=True, color=C_ORANGE)
label(ax, 10.55, 3.74, "score* = s̃ₙ − λ(μᵥ − μ_global)", fs=8, color=C_DARK)
label(ax, 10.55, 3.48, "Penalizes over-popular videos", fs=7.2, color="#888888")

# Top-K ranking
shadow(ax, 8.85, 2.0, 3.4, 0.95)
rbox(ax, 8.85, 2.0, 3.4, 0.95, fc=C_ORANGE_L, ec=C_ORANGE, radius=0.15, lw=2)
label(ax, 10.55, 2.7, "Ranked Top-K  (K=10)", fs=8.5, bold=True, color=C_ORANGE)
# rank badges
for k, (rank, col) in enumerate([(1, "#FFD700"), (2, "#C0C0C0"), (3, "#CD7F32")]):
    cx0 = 9.3 + k*0.85
    circ = Circle((cx0, 2.2), 0.18, fc=col, ec="#888888", lw=1, zorder=5)
    ax.add_patch(circ)
    label(ax, cx0, 2.2, str(rank), fs=8, bold=True, color=C_DARK, zorder=6)
label(ax, 11.7, 2.2, "...", fs=12, color=C_ORANGE, bold=True)

# Arrows within Panel C
arrow(ax, 10.55, 5.85, 10.55, 5.55, color=C_ORANGE, lw=1.6)
arrow(ax, 10.55, 4.55, 10.55, 4.25, color=C_ORANGE, lw=1.6)
arrow(ax, 10.55, 3.25, 10.55, 2.95, color=C_ORANGE, lw=1.6)


# ══════════════════════════════════════════════════════════════════════════
# PANEL D: TEMPORAL GROUNDING + OUTPUT
# ══════════════════════════════════════════════════════════════════════════

# Grounding box
shadow(ax, 12.95, 5.5, 2.5, 1.85)
rbox(ax, 12.95, 5.5, 2.5, 1.85, fc="white", ec=C_PURPLE, radius=0.15, lw=2)
label(ax, 14.2, 7.1, "Moment", fs=9, bold=True, color=C_PURPLE)
label(ax, 14.2, 6.85, "Localization", fs=9, bold=True, color=C_PURPLE)

# Timeline visualization
timeline_bar(ax, 14.2, 6.48, length=2.0, height=0.26,
             highlight_start=0.35, highlight_end=0.65)
label(ax, 14.2, 6.14, "[t_start, t_end] grounded segment", fs=6.5, color=C_DARK)
label(ax, 14.2, 5.92, "min-width: 2.5 s", fs=6.5, color="#888888")
label(ax, 14.2, 5.68, "Threshold: 92% of peak score", fs=6.5, color="#888888")

# Score curve sketch
xs = np.linspace(13.2, 15.2, 60)
ys = 5.25 + 0.38 * np.exp(-0.5*((xs-14.2)/0.45)**2)
ax.plot(xs, ys, color=C_PURPLE, lw=2, zorder=5)
ax.fill_between(xs, 5.05, ys,
                where=(xs > 13.87) & (xs < 14.53),
                color=C_ORANGE, alpha=0.45, zorder=4)
ax.plot([13.87, 13.87], [5.05, 5.4], color=C_ORANGE, lw=1.2, ls="--", zorder=5)
ax.plot([14.53, 14.53], [5.05, 5.4], color=C_ORANGE, lw=1.2, ls="--", zorder=5)
label(ax, 14.2, 5.08, "t_s           t_e", fs=6.5, color=C_ORANGE)

# Output result card
shadow(ax, 12.95, 2.85, 2.5, 1.95)
rbox(ax, 12.95, 2.85, 2.5, 1.95, fc="white", ec=C_PURPLE, radius=0.15, lw=2)
label(ax, 14.2, 4.56, "Result", fs=9, bold=True, color=C_PURPLE)

# video thumbnail placeholder
rbox(ax, 13.1, 3.55, 2.2, 0.75, fc="#E0ECF8", ec="#AABBCC", radius=0.08, lw=1)
label(ax, 14.2, 3.93, "🎬  Top-1 Video Clip", fs=8, color=C_DARK)
label(ax, 14.2, 3.67, "v_xY2kPqR3... @ 14.2s–21.8s", fs=6.5, color="#555555")

# metric chips
for i, (txt, col) in enumerate([("R@1 24.75%", C_BLUE), ("R@5 52.08%", C_GREEN),
                                  ("691 ms", C_ORANGE)]):
    xc = 13.2 + i * 0.84
    rbox(ax, xc, 3.05, 0.78, 0.35, fc=col, ec=col, radius=0.1, lw=1)
    label(ax, xc+0.39, 3.225, txt, fs=6, color="white", bold=True)

# Arrows Panel D
arrow(ax, 14.2, 5.5, 14.2, 4.8, color=C_PURPLE, lw=1.6)
arrow(ax, 14.2, 2.85, 14.2, 2.5, color=C_PURPLE, lw=1.6)

# Phone illustration on the side
phone_outline(ax, 14.2, 1.6, w=1.05, h=1.75, fc="white", ec=C_DARK, lw=2)
label(ax, 14.2, 3.42, "On-Device", fs=7, bold=True, color=C_DARK)


# ══════════════════════════════════════════════════════════════════════════
# INTER-PANEL ARROWS
# ══════════════════════════════════════════════════════════════════════════

# A → B  (embedding → index lookup)
arrow(ax, 3.85, 1.92, 4.15, 2.46, color=C_BLUE, lw=2.2,
      label_txt="q ∈ ℝ⁵¹²", lfs=8)

# B → C  (scores matrix)
arrow(ax, 8.35, 4.04, 8.65, 4.05, color=C_GREEN, lw=2.2,
      label_txt="F, q", lfs=8)

# C → D  (top-k → grounding)
arrow(ax, 12.45, 4.3, 12.75, 5.2, color=C_ORANGE, lw=2.2,
      label_txt="top-K", lfs=8)


# ══════════════════════════════════════════════════════════════════════════
# BOTTOM SPEC BAR
# ══════════════════════════════════════════════════════════════════════════
rbox(ax, 0.25, 0.1, 15.5, 1.1, fc="#2C3E50", ec="#1A252F", radius=0.2, lw=1.5, zorder=5)
specs = [
    ("Backbone", "MobileCLIP-S1"),
    ("Embedding dim", "512"),
    ("Videos indexed", "819  (ActivityNet)"),
    ("Avg latency", "690.8 ms"),
    ("P95 latency", "751 ms"),
    ("Battery drain", "66.6 mAh/hr"),
    ("Peak RAM", "510 MB"),
    ("R@1 / R@5", "24.75% / 52.08%"),
]
xs_pos = np.linspace(0.75, 15.25, len(specs))
for i, (k, v) in enumerate(specs):
    label(ax, xs_pos[i], 0.82, k, fs=7, color="#AAC8E0", bold=True, zorder=6)
    label(ax, xs_pos[i], 0.48, v, fs=7.5, color="white", bold=False, zorder=6)
    if i < len(specs)-1:
        ax.plot([xs_pos[i]+0.9, xs_pos[i]+0.9], [0.22, 0.98],
                color="#445566", lw=0.8, zorder=6)


# ══════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════
label(ax, 8.0, 8.65,
      "FedVCMR: On-Device Video Corpus Moment Retrieval with MobileCLIP",
      fs=13.5, bold=True, color=C_DARK)
label(ax, 8.0, 8.3,
      "End-to-end inference pipeline on mobile CPU  |  ActivityNet-Captions  |  10,002 queries",
      fs=9, color="#555555")

import os
os.makedirs("C:/prism/outputs/figures", exist_ok=True)
fig.tight_layout(pad=0.2)
fig.savefig("C:/prism/outputs/figures/fedvcmr_architecture.pdf", dpi=300, bbox_inches="tight")
fig.savefig("C:/prism/outputs/figures/fedvcmr_architecture.png", dpi=300, bbox_inches="tight")
print("Saved: fedvcmr_architecture.pdf + .png")
plt.show()
