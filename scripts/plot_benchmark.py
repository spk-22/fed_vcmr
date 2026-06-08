"""
Publication-quality benchmark visualization for FedVCMR.
Usage: python scripts/plot_benchmark.py
Reads:  C:/prism/benchmark_metrics.csv  (pulled from phone after run)
Output: C:/prism/outputs/figures/benchmark_*.pdf + .png
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
})

CSV_PATH = "C:/prism/benchmark_metrics.csv"
OUT_DIR  = "C:/prism/outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Forward-fill charge readings (only written every 50 queries)
df["charge_mah"] = df["charge_mah"].replace(-1, np.nan).ffill()
df["drain_mah"] = df["drain_mah"].replace(-1, np.nan).ffill()
df["elapsed_min"] = df["elapsed_sec"] / 60.0

# Rolling smoothing for noisy series
WINDOW = 50
df["latency_smooth"] = df["latency_ms"].rolling(WINDOW, min_periods=1).mean()
df["memory_smooth"]  = df["memory_mb"].rolling(WINDOW, min_periods=1).mean()

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"


# ── Figure 1: Query Count vs Time ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df["elapsed_min"], df["query_num"], color=BLUE)
ax.set_xlabel("Elapsed Time (min)")
ax.set_ylabel("Queries Processed")
ax.set_title("Query Throughput Over Time")
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/benchmark_query_vs_time.pdf")
fig.savefig(f"{OUT_DIR}/benchmark_query_vs_time.png")
plt.close(fig)
print("Saved: benchmark_query_vs_time")


# ── Figure 2: Response Time Trends ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df["query_num"], df["latency_ms"],      color=BLUE,   alpha=0.25, linewidth=0.6, label="Per-query")
ax.plot(df["query_num"], df["latency_smooth"],  color=ORANGE, linewidth=2, label=f"Rolling avg (n={WINDOW})")
ax.axhline(df["latency_ms"].mean(), color=RED, linestyle="--", linewidth=1.2,
           label=f"Mean = {df['latency_ms'].mean():.0f} ms")
ax.set_xlabel("Query Number")
ax.set_ylabel("Latency (ms)")
ax.set_title("Response Time Trends")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/benchmark_latency_trend.pdf")
fig.savefig(f"{OUT_DIR}/benchmark_latency_trend.png")
plt.close(fig)
print("Saved: benchmark_latency_trend")


# ── Figure 3: Memory Usage Over Time ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df["query_num"], df["memory_mb"],      color=GREEN, alpha=0.25, linewidth=0.6, label="Memory (MB)")
ax.plot(df["query_num"], df["memory_smooth"],  color=GREEN, linewidth=2,               label=f"Rolling avg (n={WINDOW})")
ax.plot(df["query_num"], df["peak_ram_mb"],    color=RED,   linewidth=1.4, linestyle="--", label="Peak RAM (MB)")
ax.set_xlabel("Query Number")
ax.set_ylabel("Memory (MB)")
ax.set_title("Memory Usage Over Time")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/benchmark_memory.pdf")
fig.savefig(f"{OUT_DIR}/benchmark_memory.png")
plt.close(fig)
print("Saved: benchmark_memory")


# ── Figure 4: Battery Drain (mAh) vs Duration ─────────────────────────────
bat_df = df.dropna(subset=["drain_mah"]).copy()
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(bat_df["elapsed_min"], bat_df["drain_mah"], color=RED, linewidth=2)
ax.set_xlabel("Elapsed Time (min)")
ax.set_ylabel("Cumulative Drain (mAh)")
ax.set_title("Battery Drain During Benchmark")
if len(bat_df) > 1:
    total_drain = bat_df["drain_mah"].iloc[-1]
    dur = bat_df["elapsed_min"].iloc[-1]
    rate = total_drain / (dur / 60) if dur > 0 else 0
    ax.annotate(f"Total: {total_drain:.1f} mAh\n({rate:.1f} mAh/hr)",
                xy=(bat_df["elapsed_min"].iloc[-1], total_drain),
                xytext=(-80, -30), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/benchmark_battery.pdf")
fig.savefig(f"{OUT_DIR}/benchmark_battery.png")
plt.close(fig)
print("Saved: benchmark_battery")


# ── Summary stats ─────────────────────────────────────────────────────────
print("\n-- Summary --------------------------------------------------")
print(f"Total queries   : {len(df)}")
print(f"Avg latency     : {df['latency_ms'].mean():.1f} ms")
print(f"P95 latency     : {df['latency_ms'].quantile(0.95):.1f} ms")
print(f"Peak RAM        : {df['peak_ram_mb'].max()} MB")
print(f"Avg memory      : {df['memory_mb'].mean():.1f} MB")
if bat_df["drain_mah"].notna().any():
    total_drain = bat_df["drain_mah"].max()
    dur = df["elapsed_min"].max()
    print(f"Total battery drain: {total_drain:.1f} mAh")
    print(f"Drain rate      : {total_drain / (dur / 60):.2f} mAh/hr")
print("Figures saved to:", OUT_DIR)
