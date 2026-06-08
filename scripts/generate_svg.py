"""
FedVCMR Architecture Diagram — Clean 3-row layout inspired by AdaFilter.

Layout:
  TRACK A (top-blue):   Text Query → MobileCLIP-S1 Text Encoder → q ∈ ℝ⁵¹²
  TRACK B (mid-orange): Feature Index (mmap) → per-video frame embeddings
  PIPELINE (bottom):    Dot Product → Temporal Smoothing → Hubness Suppression
                         → Top-K Ranking → Temporal Grounding

Run: .venv/Scripts/python scripts/generate_svg.py
"""
import os, math
os.makedirs("C:/prism/outputs/figures", exist_ok=True)

OUT = "C:/prism/outputs/figures/fedvcmr_architecture.svg"

W, H = 1600, 820

# ── Palette ──────────────────────────────────────────────────────────────────
BG  = "#F8F9FA"; WH = "#FFFFFF"; DK = "#1C2833"
B1="#154360"; B2="#2471A3"; B3="#D6EAF8"; B4="#A9CCE3"
O1="#6E2C00"; O2="#CA6F1E"; O3="#FAD7A0"; O4="#F5CBA7"
G1="#0B5345"; G2="#1E8449"; G3="#A9DFBF"; G4="#D5F5E3"
P1="#4A235A"; P2="#7D3C98"; P3="#D7BDE2"; P4="#EDE0F5"
N1="#2C3E50"; N2="#7F8C8D"; N3="#BDC3C7"; N4="#ECF0F1"

svg = []; add = svg.append

def R(x,y,w,h,fill=WH,stroke="none",sw=1.5,rx=5):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}"/>'

def T(x,y,s,sz=12,fill=DK,a="middle",b=False,i=False):
    w=' font-weight="bold"' if b else ""; it=' font-style="italic"' if i else ""
    esc_s = str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    return f'<text x="{x}" y="{y}" font-size="{sz}" fill="{fill}" text-anchor="{a}" font-family="Arial,Helvetica,sans-serif"{w}{it}>{esc_s}</text>'

def L(x1,y1,x2,y2,stroke=DK,sw=1.5,dash=""):
    d=f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}/>'

def A(x1,y1,x2,y2,color=DK,sw=2,m="arr"):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#{m})"/>'

def PA(d,fill="none",stroke=DK,sw=1.5):
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

def hsv(h,s=0.68,v=0.88):
    h6=h*6; i=int(h6)%6; f=h6-int(h6)
    p=v*(1-s); q_=v*(1-s*f); t_=v*(1-s*(1-f))
    m=[(v,t_,p),(q_,v,p),(p,v,t_),(p,q_,v),(t_,p_,v),(v,p,q_)] if False else \
      [(v,t_,p),(q_,v,p),(p,v,t_),(p,q_,v),(t_,p,v),(v,p,q_)]
    r,g,b=m[i]; return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

def heat(s):
    s=max(0.,min(1.,s))
    if s<0.35: r,g,b=int(220+35*s/0.35),int(220+35*s/0.35),255
    elif s<0.65: f=(s-0.35)/0.3; r,g,b=255,int(255*(1-f*0.6)),int(255*(1-f))
    else: f=(s-0.65)/0.35; r,g,b=255,int(102*(1-f)),int(0)
    return f"rgb({r},{g},{b})"

# ── SVG header + markers ──────────────────────────────────────────────────────
add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
    f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
for mid,col in [("arr",DK),("ab",B2),("ao",O2),("ag",G2),("ap",P2)]:
    add(f'<marker id="{mid}" markerWidth="9" markerHeight="6" refX="8" refY="3" orient="auto">'
        f'<polygon points="0 0,9 3,0 6" fill="{col}"/></marker>')
add('</defs>' if False else '')  # defs opened implicitly via markers above — wrap properly:

# Re-emit properly
svg.clear()
add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
    f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs>')
for mid,col in [("arr",DK),("ab",B2),("ao",O2),("ag",G2),("ap",P2)]:
    add(f'<marker id="{mid}" markerWidth="9" markerHeight="6" refX="8" refY="3" orient="auto">'
        f'<polygon points="0 0,9 3,0 6" fill="{col}"/></marker>')
add('</defs>')
add(R(0,0,W,H,fill=BG,rx=0))

PAD = 14

# ═══════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════
add(T(W//2, 32, "FedVCMR: On-Device Video Corpus Moment Retrieval", sz=22, b=True))
add(T(W//2, 52, "MobileCLIP-S1 text encoder  ·  DGSE scoring  ·  Temporal grounding  ·  OnePlus 11R (Snapdragon 8+ Gen 1)", sz=10.5, fill=N2))

# ═══════════════════════════════════════════════════════════════
# ROW Y POSITIONS (streamlined)
# ═══════════════════════════════════════════════════════════════
TA_Y, TA_H = 62,  128   # Track A: text encoder
TB_Y, TB_H = 203, 110   # Track B: video corpus
PL_Y, PL_H = 336, 390   # Pipeline row (5 stages)
SB_Y, SB_H = 738,  40   # Stats bar
# Total used: 778  (H=820 → room for margins)

# ═══════════════════════════════════════════════════════════════
# TRACK A — Text Encoder (clean horizontal flow)
# ═══════════════════════════════════════════════════════════════
add(R(PAD, TA_Y, W-2*PAD, TA_H, fill=B3, stroke=B2, sw=2, rx=8))
add(T(PAD+12, TA_Y+12, "① Text Encoder (MobileCLIP-S1)", sz=11, fill=B1, b=True, a="start"))

# Input: query text
X_POS = PAD + 24
QX, QY, QW, QH = X_POS, TA_Y+26, 110, 50
add(R(QX, QY, QW, QH, fill=WH, stroke=B2, sw=1.5, rx=4))
add(T(QX+QW//2, QY+10, "Text Query", sz=9, fill=B1, b=True))
add(T(QX+QW//2, QY+26, '"a person rides', sz=7.5, fill=DK, i=True))
add(T(QX+QW//2, QY+36, 'a horse"', sz=7.5, fill=DK, i=True))
add(A(QX+QW+2, QY+QH//2, QX+QW+16, QY+QH//2, color=B2, sw=1.8, m="ab"))
X_POS += QW + 22

# Tokenizer
BPX, BPY, BPW, BPH = X_POS, TA_Y+28, 70, 45
add(R(BPX, BPY, BPW, BPH, fill=B2, stroke=B1, sw=1.2, rx=4))
add(T(BPX+BPW//2, BPY+12, "Tokenizer", sz=9, fill=WH, b=True))
add(T(BPX+BPW//2, BPY+27, "(77 tokens)", sz=7.5, fill=B4))
add(A(BPX+BPW+2, BPY+BPH//2, BPX+BPW+16, BPY+BPH//2, color=B2, sw=1.8, m="ab"))
X_POS += BPW + 22

# 6 Transformer blocks (stacked 3D look)
TRX, TRY, TRW, TRH = X_POS, TA_Y+20, 50, 58
for ti in range(6):
    bx = TRX + ti * (TRW + 6)
    for off in reversed([0, 3, 6]):
        fc = B2 if off == 0 else (B4 if off == 3 else B3)
        add(R(bx + off, TRY + off, TRW - off, TRH - off, fill=fc, stroke=B1, sw=1, rx=3))
    if ti < 5:
        add(A(bx + TRW + 2, TRY + TRH // 2, bx + TRW + 8, TRY + TRH // 2, color=B2, sw=1.5, m="ab"))

add(T(TRX + 3 * (TRW + 6), TA_Y + TRH + 10, "6 Transformer Layers (Attn + FFN)", sz=8, fill=B1))

# Projection + Normalization → q vector
PRJ_X = TRX + 6 * (TRW + 6) + 20
PRJ_BOX_X, PRJ_BOX_Y, PRJ_BOX_W, PRJ_BOX_H = PRJ_X, TA_Y + 28, 75, 45
add(R(PRJ_BOX_X, PRJ_BOX_Y, PRJ_BOX_W, PRJ_BOX_H, fill=B1, stroke=B1, sw=1, rx=4))
add(T(PRJ_BOX_X + PRJ_BOX_W // 2, PRJ_BOX_Y + 12, "Linear Proj", sz=9, fill=WH, b=True))
add(T(PRJ_BOX_X + PRJ_BOX_W // 2, PRJ_BOX_Y + 27, "+ L2 Norm", sz=8, fill=B4))
add(A(PRJ_BOX_X + PRJ_BOX_W + 2, PRJ_BOX_Y + PRJ_BOX_H // 2, PRJ_BOX_X + PRJ_BOX_W + 16, PRJ_BOX_Y + PRJ_BOX_H // 2, color=B2, sw=1.8, m="ab"))

# Final q embedding visualization
Q_FINAL_X = PRJ_BOX_X + PRJ_BOX_W + 22
Q_FINAL_Y = TA_Y + 32
Q_FINAL_W = 140
Q_FINAL_H = 32
for ci in range(36):
    add(R(Q_FINAL_X + ci * (Q_FINAL_W // 36), Q_FINAL_Y, Q_FINAL_W // 36, Q_FINAL_H, 
          fill=hsv((ci / 36.) % 1., 0.7, 0.9), stroke="none", rx=0))
add(R(Q_FINAL_X, Q_FINAL_Y, Q_FINAL_W, Q_FINAL_H, fill="none", stroke=B1, sw=1.2, rx=3))
add(T(Q_FINAL_X + Q_FINAL_W // 2, Q_FINAL_Y - 8, "q ∈ ℝ⁵¹²", sz=9, fill=B1, b=True))
add(T(Q_FINAL_X + Q_FINAL_W // 2, Q_FINAL_Y + Q_FINAL_H + 8, "(L2-normalized)", sz=8, fill=B1, i=True))

# ═══════════════════════════════════════════════════════════════
# TRACK B — Video Feature Index (simplified)
# ═══════════════════════════════════════════════════════════════
add(R(PAD, TB_Y, W-2*PAD, TB_H, fill=O3, stroke=O2, sw=2, rx=8))
add(T(PAD+12, TB_Y+12, "② Video Feature Index (mmap, 12 MB)", sz=11, fill=O1, b=True, a="start"))

# Cylinder representation
CX = PAD + 80; CY = TB_Y + 30; CW = 72; CH = 65
add(R(CX - CW // 2, CY + 8, CW, CH - 16, fill=O2, stroke=O1, sw=1.5, rx=0))
add(f'<ellipse cx="{CX}" cy="{CY+8}" rx="{CW//2}" ry="8" fill="{O2}" stroke="{O1}" stroke-width="1.5"/>')
add(f'<ellipse cx="{CX}" cy="{CY+CH-8}" rx="{CW//2}" ry="8" fill="{O1}" stroke="{O1}" stroke-width="1.5"/>')
add(T(CX, CY + 32, "features", sz=8.5, fill=WH, b=True))
add(T(CX, CY + 43, "_blob.bin", sz=8, fill=WH, i=True))
add(T(CX, CY + 53, "12 MB", sz=9.5, fill=WH, b=True))
add(A(CX + CW // 2 + 2, CY + CH // 2, CX + CW // 2 + 18, CY + CH // 2, color=O2, sw=1.8, m="ao"))

# Frame embedding grid (simplified, conceptual)
FEX = CX + CW // 2 + 30; FEY = TB_Y + 20
CW2 = 11; CH2 = 14; CG2 = 1; VG2 = 4
FE_COLS = 20
FE_ROWS = 4
for vi in range(FE_ROWS):
    for fi in range(FE_COLS):
        add(R(FEX + fi * (CW2 + CG2), FEY + vi * (CH2 + VG2), CW2, CH2,
              fill=hsv((vi * 0.22 + fi * 0.049) % 1., 0.6, 0.85), stroke=WH, sw=0.3, rx=1))
add(R(FEX, FEY, FE_COLS * (CW2 + CG2), FE_ROWS * (CH2 + VG2), fill="none", stroke=O2, sw=1.5, rx=2))
add(T(FEX + FE_COLS * (CW2 + CG2) // 2, FEY - 8, "Frame Embeddings", sz=9, fill=O1, b=True))
add(T(FEX + FE_COLS * (CW2 + CG2) // 2, FEY + FE_ROWS * (CH2 + VG2) + 10,
      "819 videos × 16 frames × 512-dim", sz=8, fill=O1, i=True))

# ═══════════════════════════════════════════════════════════════
# PIPELINE — 5-Stage Retrieval & Grounding
# ═══════════════════════════════════════════════════════════════
add(R(PAD, PL_Y, W-2*PAD, PL_H, fill=G4, stroke=G2, sw=2, rx=8))
add(T(W//2, PL_Y+14, "DGSE Scoring + Temporal Grounding Pipeline", sz=12, fill=G1, b=True))

PL_INNER_W = W - 2*PAD - 20
NSEC = 5
SW = PL_INNER_W // NSEC  # ~310 per section

def sec_box(si, title, desc="", title_col=G1, fill=WH, stroke=G2):
    sx = PAD + 10 + si * SW
    sy = PL_Y + 28
    sw_ = SW - 8
    sh = PL_H - 42
    add(R(sx, sy, sw_, sh, fill=fill, stroke=stroke, sw=1.5, rx=5))
    add(T(sx + sw_ // 2, sy + 14, title, sz=10, fill=title_col, b=True))
    if desc:
        add(T(sx + sw_ // 2, sy + 30, desc, sz=8.5, fill=DK, i=True))
    return sx, sy, sw_, sh

# ── Stage 1: Dot Product ─────────────────────────────────────────────────────
S1X, S1Y, S1W, S1H = sec_box(0, "1. Dot Product")
add(T(S1X + S1W // 2, S1Y + 46, "s(q, fᵢ) = q · fᵢ", sz=10, fill=DK, i=True))
add(T(S1X + S1W // 2, S1Y + 60, "∀ videos, frames", sz=8, fill=N2))
# Show q (small) and F (small grid) and scores
QS_X = S1X + 16; QS_Y = S1Y + 75; QS_W = 12; QS_H = 60
for ci in range(15):
    add(R(QS_X + ci, QS_Y + ci, 2, QS_H - 2*ci, fill=hsv((ci / 15.) % 1., 0.7, 0.88), stroke="none", rx=0))
add(T(QS_X + 10, QS_Y + QS_H + 8, "q", sz=8, fill=B2, b=True))

# Score bars
SC_X = QS_X + 30; SC_Y = S1Y + 75
for si in range(6):
    sc = 0.08 + (0.8 if si == 1 else 0.3 if si == 3 else 0.1) * math.exp(-((si - 1) ** 2) / 2.5)
    add(R(SC_X, SC_Y + si * 12, int(sc * 50), 10, fill=heat(sc), stroke="none", rx=1))
add(T(SC_X + 35, SC_Y + 6 * 12 + 8, "scores", sz=8, fill=G1, b=True))

# Arrow
add(A(S1X + S1W + 2, PL_Y + PL_H // 2, S1X + S1W + 10, PL_Y + PL_H // 2, color=G2, sw=2.5, m="ag"))

# ── Stage 2: Temporal Smoothing ──────────────────────────────────────────────
S2X, S2Y, S2W, S2H = sec_box(1, "2. Temporal", "Smoothing")
add(T(S2X + S2W // 2, S2Y + 50, "s̃ₑ = mean(sₑ₋₁, sₑ, sₑ₊₁)", sz=9.5, fill=DK, i=True))
add(T(S2X + S2W // 2, S2Y + 64, "3-frame window", sz=8, fill=N2))
# Bar chart
BC_X = S2X + 14; BC_Y = S2Y + 80
braw = [0.12, 0.16, 0.25, 0.42, 0.68, 0.80, 0.71, 0.48, 0.28, 0.14, 0.10]
bsmth = [(braw[max(0, k - 1)] + braw[k] + braw[min(10, k + 1)]) / 3 for k in range(11)]
BW2, BHM, BG3 = 12, 60, 2
for k, (rv, sv) in enumerate(zip(braw, bsmth)):
    bx = BC_X + k * (BW2 + BG3)
    add(R(bx, BC_Y + BHM - int(rv * BHM), BW2, int(rv * BHM), fill=N3, stroke="none", rx=1))
    add(R(bx + 2, BC_Y + BHM - int(sv * BHM), BW2 - 4, int(sv * BHM), fill=G2, stroke="none", rx=1))
add(L(BC_X, BC_Y + BHM, BC_X + 11 * (BW2 + BG3), BC_Y + BHM, stroke=N2, sw=1))
add(T(S2X + S2W // 2, BC_Y + BHM + 12, "gray→green", sz=7, fill=N2))

add(A(S2X + S2W + 2, PL_Y + PL_H // 2, S2X + S2W + 10, PL_Y + PL_H // 2, color=G2, sw=2.5, m="ag"))

# ── Stage 3: Hubness Suppression ─────────────────────────────────────────────
S3X, S3Y, S3W, S3H = sec_box(2, "3. Hubness", "Suppression")
add(T(S3X + S3W // 2, S3Y + 50, "score* = s̃ - λ·Δμ", sz=9.5, fill=DK, i=True))
add(T(S3X + S3W // 2, S3Y + 63, "λ=0.35, penalize hubs", sz=7.5, fill=N2, i=True))
# Before/after bars
HA_X = S3X + 10; HA_Y = S3Y + 80
for li, ((lbl, h_bef, h_aft), col) in enumerate(zip(
        [("hub", 60, 35), ("v₂", 45, 43), ("v₃", 32, 31)],
        [O2, G2, G2])):
    hy = HA_Y + li * 20
    add(R(HA_X + 8, hy, h_bef, 12, fill=N3, stroke=N3, sw=0, rx=2))
    add(R(HA_X + 8, hy + 1, h_aft, 11, fill=col, stroke="none", rx=2))
    add(T(HA_X + 3, hy + 9, lbl, sz=8, fill=DK, a="end"))
add(T(S3X + S3W // 2, HA_Y + 3 * 20 + 10, "before→after", sz=7, fill=N2))

add(A(S3X + S3W + 2, PL_Y + PL_H // 2, S3X + S3W + 10, PL_Y + PL_H // 2, color=G2, sw=2.5, m="ag"))

# ── Stage 4: Top-K Ranking ────────────────────────────────────────────────────
S4X, S4Y, S4W, S4H = sec_box(3, "4. Top-K", "Ranking (K=10)")
rk_data = [(0.254, G2, "v₂"), (0.198, G3, "v₄"), (0.161, "#58D68D", "v₆"), (0.130, N3, "v₁")]
RKX = S4X + 12; RKY = S4Y + 52
for ri, (sc, rc, vid) in enumerate(rk_data):
    ry = RKY + ri * 22
    bw = int(sc / 0.254 * (S4W - 45))
    add(R(RKX + 25, ry, bw, 16, fill=rc, stroke="none", rx=2))
    add(T(RKX + 21, ry + 12, f"#{ri + 1}", sz=8, fill=G1, a="end", b=True))
    add(T(RKX + 32 + bw // 2, ry + 11, vid, sz=8.5, fill=DK if rc == N3 else WH, b=True))
    add(T(RKX + 32 + bw + 2, ry + 11, f"{sc:.3f}", sz=7.5, fill=N2, a="start"))

add(A(S4X + S4W + 2, PL_Y + PL_H // 2, S4X + S4W + 10, PL_Y + PL_H // 2, color=P2, sw=2.5, m="ap"))

# ── Stage 5: Temporal Grounding ──────────────────────────────────────────────
S5X, S5Y, S5W, S5H = sec_box(4, "5. Temporal", "Grounding", title_col=P1, fill=P4, stroke=P2)
add(T(S5X + S5W // 2, S5Y + 50, "threshold = 92% of peak", sz=8.5, fill=N2))
add(T(S5X + S5W // 2, S5Y + 62, "min window = 2.5 s", sz=7.5, fill=N2))
# Score curve
CVX = S5X + 10; CVY = S5Y + 78; CVW = S5W - 20; CVH = 50
add(L(CVX, CVY, CVX, CVY + CVH, stroke=N2, sw=1))
add(L(CVX, CVY + CVH, CVX + CVW, CVY + CVH, stroke=N2, sw=1))
add(T(CVX - 3, CVY + 4, "s", sz=8, fill=P2, a="end", i=True))
add(T(CVX + CVW // 2, CVY + CVH + 8, "t", sz=8, fill=N2, i=True))
PEAK = 0.42
N_CV = 40
pts = []
for k in range(N_CV + 1):
    tx = k / N_CV
    sc_ = 0.07 + 0.83 * math.exp(-((tx - PEAK) ** 2) / 0.018)
    pts.append(f"{CVX + tx * CVW:.1f},{CVY + CVH - sc_ * CVH:.1f}")
add(PA("M " + " L ".join(pts), stroke=P2, sw=2, fill="none"))
# Threshold line
THRESH = 0.07 + 0.83 * 0.92
THY = CVY + CVH - THRESH * CVH
add(L(CVX, THY, CVX + CVW, THY, stroke=P1, sw=1, dash="4,3"))
# Segment fill (window)
TS = PEAK - 0.12
TE = PEAK + 0.12
seg = []
seg.append(f"{CVX + TS * CVW:.1f},{CVY + CVH:.1f}")
for k in range(N_CV + 1):
    tx = k / N_CV
    if TS <= tx <= TE:
        sc_ = 0.07 + 0.83 * math.exp(-((tx - PEAK) ** 2) / 0.018)
        seg.append(f"{CVX + tx * CVW:.1f},{CVY + CVH - sc_ * CVH:.1f}")
seg.append(f"{CVX + TE * CVW:.1f},{CVY + CVH:.1f}")
add(PA("M " + " L ".join(seg) + " Z", fill=P3, stroke="none"))
add(T(CVX + CVW // 2, CVY + CVH + 24, "[tₛ, tₑ] = retrieved moment", sz=7.5, fill=P1, b=True))

# ═══════════════════════════════════════════════════════════════
# SIMPLIFIED VERTICAL CONNECTORS (q and F flowing down)
# ═══════════════════════════════════════════════════════════════

# q flows down from Track A to Stage 1
q_x = Q_FINAL_X + Q_FINAL_W // 2
q_from_y = TA_Y + TA_H + 2
q_to_y = PL_Y + 28
q_mid_y = (q_from_y + q_to_y) // 2
add(PA(f"M {q_x},{q_from_y} L {q_x},{q_mid_y} L {S1X + S1W//3},{q_mid_y} L {S1X + S1W//3},{q_to_y}",
       stroke=B2, sw=2, fill="none"))
add(A(S1X + S1W // 3, q_to_y - 2, S1X + S1W // 3, q_to_y + 2, color=B2, sw=2, m="ab"))
add(T(q_x + 8, q_mid_y - 4, "q", sz=9.5, fill=B2, b=True, i=True))

# F flows down from Track B to Stage 1
f_x = FEX + FE_COLS * (CW2 + CG2) // 2
f_from_y = TB_Y + TB_H + 2
f_to_y = PL_Y + 28
f_mid_y = (f_from_y + f_to_y) // 2
add(PA(f"M {f_x},{f_from_y} L {f_x},{f_mid_y} L {S1X + S1W*2//3},{f_mid_y} L {S1X + S1W*2//3},{f_to_y}",
       stroke=O2, sw=2, fill="none"))
add(A(S1X + S1W * 2 // 3, f_to_y - 2, S1X + S1W * 2 // 3, f_to_y + 2, color=O2, sw=2, m="ao"))
add(T(f_x + 8, f_mid_y - 4, "F", sz=9.5, fill=O2, b=True, i=True))

# ═══════════════════════════════════════════════════════════════
# STATS BAR (compact)
# ═══════════════════════════════════════════════════════════════
add(R(PAD, SB_Y, W - 2 * PAD, SB_H, fill=DK, stroke="none", rx=6))
stats = [
    ("Backbone", "MobileCLIP-S1"),
    ("Embedding", "512-dim"),
    ("Corpus", "819 videos"),
    ("Avg Latency", "691 ms"),
    ("P95 Latency", "751 ms"),
    ("Battery", "66.6 mAh/hr"),
    ("Peak RAM", "510 MB"),
    ("R@1 / R@5", "24.75% / 52.08%")
]
NW2 = (W - 2 * PAD) // len(stats)
for si, (lbl, val) in enumerate(stats):
    sx = PAD + si * NW2 + NW2 // 2
    add(T(sx, SB_Y + 12, lbl, sz=7.5, fill=N3))
    add(T(sx, SB_Y + 28, val, sz=9, fill=WH, b=True))

add('</svg>')

with open(OUT,"w",encoding="utf-8") as f:
    f.write("\n".join(svg))
print(f"Saved: {OUT}")
print(f"Run:   start {OUT}")
