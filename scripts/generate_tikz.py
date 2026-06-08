"""
Generate FedVCMR Architecture Diagram using TikZ (LaTeX)

TikZ produces publication-quality diagrams perfect for ML/research papers.

Run:
  python scripts/generate_tikz.py
  
Then compile the .tex file:
  pdflatex outputs/figures/fedvcmr_architecture.tex
  
Or use latexmk:
  latexmk -pdf outputs/figures/fedvcmr_architecture.tex
"""

import os
os.makedirs("C:/prism/outputs/figures", exist_ok=True)

tikz_code = r"""
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{amsmath}
\usetikzlibrary{shapes,arrows,positioning,calc,decorations.markings}

\definecolor{blue1}{RGB}{20, 67, 96}
\definecolor{blue2}{RGB}{36, 113, 163}
\definecolor{blue3}{RGB}{173, 216, 230}
\definecolor{orange1}{RGB}{110, 44, 0}
\definecolor{orange2}{RGB}{202, 111, 30}
\definecolor{orange3}{RGB}{250, 215, 160}
\definecolor{green1}{RGB}{11, 83, 69}
\definecolor{green2}{RGB}{30, 132, 73}
\definecolor{green3}{RGB}{169, 223, 191}
\definecolor{purple1}{RGB}{74, 35, 90}
\definecolor{purple2}{RGB}{125, 60, 152}
\definecolor{purple3}{RGB}{215, 189, 226}
\definecolor{gray1}{RGB}{44, 62, 80}
\definecolor{gray2}{RGB}{127, 140, 141}
\definecolor{gray3}{RGB}{189, 195, 199}

\tikzset{
  block/.style={rectangle, draw=none, fill=white, text centered, minimum width=2.5cm, minimum height=0.8cm, font=\small},
  blockb/.style={rectangle, draw=blue2, fill=blue3, text centered, minimum width=2.8cm, minimum height=0.9cm, font=\small, line width=1.2pt},
  blocko/.style={rectangle, draw=orange2, fill=orange3, text centered, minimum width=2.8cm, minimum height=0.9cm, font=\small, line width=1.2pt},
  blockg/.style={rectangle, draw=green2, fill=green3, text centered, minimum width=2.5cm, minimum height=0.85cm, font=\small, line width=1.2pt},
  blockp/.style={rectangle, draw=purple2, fill=purple3, text centered, minimum width=2.5cm, minimum height=0.85cm, font=\small, line width=1.2pt},
  component/.style={rectangle, draw=blue2, fill=white, text centered, minimum width=1.6cm, minimum height=0.6cm, font=\footnotesize, line width=0.8pt},
  componento/.style={rectangle, draw=orange2, fill=white, text centered, minimum width=1.6cm, minimum height=0.6cm, font=\footnotesize, line width=0.8pt},
  componentg/.style={rectangle, draw=green2, fill=white, text centered, minimum width=1.6cm, minimum height=0.6cm, font=\footnotesize, line width=0.8pt},
  arrow/.style={->, thick, line width=1.5pt},
  arrowb/.style={->, line width=1.5pt, color=blue2},
  arrowo/.style={->, line width=1.5pt, color=orange2},
  arrowg/.style={->, line width=1.5pt, color=green2},
  arrowp/.style={->, line width=1.5pt, color=purple2},
  stageblock/.style={rectangle, draw, fill=white, text centered, minimum width=2.2cm, minimum height=3.5cm, font=\small, line width=1pt},
}

\begin{document}

\begin{tikzpicture}[node distance=0.8cm]

% ============ TITLE ============
\node (title) at (0, 0) [anchor=north, font=\Large\bfseries] {FedVCMR: On-Device Video Corpus Moment Retrieval};
\node (subtitle) at (0, -0.4) [anchor=north, font=\small\color{gray2}] {MobileCLIP-S1 · DGSE Scoring · Temporal Grounding · ARM Cortex-A55};

% ============ TRACK A: TEXT ENCODER ============
\node (track_a_label) at (-5, -1.5) [anchor=west, font=\footnotesize\bfseries\color{blue1}] {① Text Encoder};
\node (query) at (-4.5, -2.2) [component] {Query};
\node (tokenizer) at (-2.8, -2.2) [component] {Tokenizer};
\node (trans1) at (-1.2, -2.2) [component] {Attn};
\node (trans2) at (0.3, -2.2) [component] {Attn};
\node (proj) at (1.8, -2.2) [component] {Linear};
\node (q_final) at (3.3, -2.2) [component, fill=blue3] {$\mathbf{q} \in \mathbb{R}^{512}$};

\draw [arrowb] (query) -- (tokenizer);
\draw [arrowb] (tokenizer) -- (trans1);
\draw [arrowb] (trans1) -- (trans2);
\draw [arrowb] (trans2) -- (proj);
\draw [arrowb] (proj) -- (q_final);

% ============ TRACK B: VIDEO INDEX ============
\node (track_b_label) at (-5, -3.5) [anchor=west, font=\footnotesize\bfseries\color{orange1}] {② Video Index};
\node (cylinder) at (-3.5, -4.2) [block, draw=orange2, fill=orange1, minimum width=1.2cm, minimum height=1cm, font=\footnotesize\color{white}] {$\mathcal{F}$};
\node (fmatrix) at (-0.5, -4.2) [block, draw=orange2, fill=orange3, text=black, minimum width=3cm, minimum height=0.7cm, font=\small] {819 videos $\times$ 16 frames $\times$ 512-dim};

\draw [arrowo] (cylinder) -- (fmatrix);

% ============ PIPELINE: 5 STAGES ============
\node (pipeline_label) at (-5, -5.5) [anchor=west, font=\footnotesize\bfseries\color{green1}] {Pipeline};

% Stage 1: Dot Product
\node[stageblock, draw=green2, fill=white] (stage1) at (-4.5, -7.5) {};
\node at (-4.5, -6.2) [font=\small\bfseries\color{green1}] {1. Dot Product};
\node at (-4.5, -6.7) [font=\small, align=center] {$s(q,f_i) = q \cdot f_i$};
\node[draw, fill=gray3, minimum size=0.8cm] at (-4.5, -8.2) {};
\node at (-4.5, -9.2) [font=\footnotesize] {scores};

% Stage 2: Temporal Smoothing
\node[stageblock, draw=green2, fill=white] (stage2) at (-2.2, -7.5) {};
\node at (-2.2, -6.2) [font=\small\bfseries\color{green1}] {2. Temporal};
\node at (-2.2, -6.5) [font=\small\bfseries\color{green1}] {Smoothing};
\node at (-2.2, -7) [font=\small, align=center] {$\tilde{s}_e = \text{mean}(s_{e-1},s_e,s_{e+1})$};
\node[draw, fill=green2, minimum width=1.5cm, minimum height=0.4cm] at (-2.2, -8.1) {};
\node at (-2.2, -9.2) [font=\footnotesize] {$\tilde{s}$};

% Stage 3: Hubness Suppression
\node[stageblock, draw=green2, fill=white] (stage3) at (0.1, -7.5) {};
\node at (0.1, -6.2) [font=\small\bfseries\color{green1}] {3. Hubness};
\node at (0.1, -6.5) [font=\small\bfseries\color{green1}] {Suppress};
\node at (0.1, -7) [font=\small, align=center] {$s^* = \tilde{s} - \lambda \Delta\mu$};
\node[draw, fill=green3, minimum width=1.5cm, minimum height=0.4cm] at (0.1, -8.1) {};
\node at (0.1, -9.2) [font=\footnotesize] {adjusted};

% Stage 4: Top-K Ranking
\node[stageblock, draw=green2, fill=white] (stage4) at (2.4, -7.5) {};
\node at (2.4, -6.2) [font=\small\bfseries\color{green1}] {4. Top-K};
\node at (2.4, -6.5) [font=\small\bfseries\color{green1}] {Ranking};
\node at (2.4, -7) [font=\small, align=center] {K=10};
\draw[fill=green2] (1.7, -8) rectangle (3.1, -8.4);
\draw[fill=green3] (1.7, -8.5) rectangle (3.1, -8.8);
\node at (2.4, -9.2) [font=\footnotesize] {ranked};

% Stage 5: Temporal Grounding
\node[stageblock, draw=purple2, fill=white] (stage5) at (4.7, -7.5) {};
\node at (4.7, -6.2) [font=\small\bfseries\color{purple1}] {5. Temporal};
\node at (4.7, -6.5) [font=\small\bfseries\color{purple1}] {Grounding};
\node at (4.7, -7) [font=\small, align=center] {threshold=92\%};
\node at (4.7, -8) [font=\small, align=center, text width=1.8cm] {peak detection};
\node at (4.7, -9.2) [font=\footnotesize] {$[t_s, t_e]$};

% Arrows between stages
\draw [arrowg] (stage1.east) -- (stage2.west);
\draw [arrowg] (stage2.east) -- (stage3.west);
\draw [arrowg] (stage3.east) -- (stage4.west);
\draw [arrowp] (stage4.east) -- (stage5.west);

% ============ DATA FLOW INPUTS ============
% q flows down
\draw [arrowb, dashed] (q_final) -- (q_final |- stage1.north) node[midway, right, font=\footnotesize\color{blue1}, inner sep=2pt] {$q$};

% F flows down  
\draw [arrowo, dashed] (fmatrix) -- (fmatrix |- stage1.north) node[midway, left, font=\footnotesize\color{orange1}, inner sep=2pt] {$\mathcal{F}$};

% ============ OUTPUT ============
\node (output) at (4.7, -10.5) [font=\small\bfseries\color{purple1}] {Retrieved Moment};
\node at (4.7, -11) [font=\footnotesize, align=center] {video $v^* \in [t_s, t_e]$};

% ============ STATS ============
\node at (-5, -11.8) [font=\tiny, align=left, text width=10cm] {
  \textbf{System Specs:} MobileCLIP-S1 · 512-dim embeddings · 819 videos (ActivityNet) · \\
  691ms avg latency · 751ms p95 · R@1/R@5: 24.75\%/52.08\%
};

\end{tikzpicture}

\end{document}
"""

OUT = "C:/prism/outputs/figures/fedvcmr_architecture.tex"
with open(OUT, "w", encoding="utf-8") as f:
    f.write(tikz_code)

print(f"✓ Saved: {OUT}")
print(f"✓ Compile with: pdflatex {OUT}")
print(f"✓ Or use: latexmk -pdf {OUT}")
print(f"✓ Output: C:/prism/outputs/figures/fedvcmr_architecture.pdf")
