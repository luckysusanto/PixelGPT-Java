"""
Alignment Statistics Visualizer
=================================
Reads the 3 CSVs from statistic_report/ and produces 4 PNG figures.

Usage:
    python visualize_alignment.py

Expects:
    statistic_report/alignment_stats.csv
    statistic_report/mismatch_by_length.csv
    statistic_report/crosstok_correlation.csv

Output:
    statistic_report/fig1_mismatch_and_ratio.png
    statistic_report/fig2_over_under_seg.png
    statistic_report/fig3_mismatch_by_quartile.png
    statistic_report/fig4_alignment_rate.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_DIR  = "statistic_report"
OUTPUT_DIR = "statistic_report"

TOK_COLORS = {
    "tok_grapheme": "#2D6A4F",
    "tok_llama2":   "#1D6FA4",
    "tok_komodo":   "#E07B39",
    "tok_mt5":      "#9B2D8E",
}
TOK_LABELS = {
    "tok_grapheme": "Grapheme (ref)",
    "tok_llama2":   "LLaMA-2",
    "tok_komodo":   "Komodo",
    "tok_mt5":      "mT5",
}
DS_ORDER  = ["java", "bali_java-tok", "bali_bali-tok", "sunda", "lampung"]
DS_LABELS = ["Java", "Bali\n(Java tok)", "Bali\n(Bali tok)", "Sunda", "Lampung"]
TOK_ORDER = ["tok_grapheme", "tok_llama2", "tok_komodo", "tok_mt5"]
Q_ORDER   = ["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"]

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})

# =============================================================================
# LOAD
# =============================================================================
summary  = pd.read_csv(f"{INPUT_DIR}/alignment_stats.csv")
binned   = pd.read_csv(f"{INPUT_DIR}/mismatch_by_length.csv")
crosstok = pd.read_csv(f"{INPUT_DIR}/crosstok_correlation.csv")

def get_val(df, ds, tok, col):
    row = df.loc[(df.dataset == ds) & (df.tokenizer == tok)]
    return row[col].values[0] if len(row) > 0 else np.nan

# =============================================================================
# FIGURE 1 — Mismatch Rate (Stat 1) & Length Ratio (Stat 2)
# =============================================================================
fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle("Stats 1 & 2 — Mismatch Rate and Length Ratio vs tok_grapheme",
              fontsize=13, fontweight="bold", y=1.02)

x      = np.arange(len(DS_ORDER))
n_tok  = len(TOK_ORDER)
width  = 0.18
offset = np.linspace(-(n_tok - 1) / 2 * width, (n_tok - 1) / 2 * width, n_tok)

# Panel A: mismatch rate
ax = axes[0]
for i, tok in enumerate(TOK_ORDER):
    vals = [get_val(summary, ds, tok, "mismatch_rate") for ds in DS_ORDER]
    ax.bar(x + offset[i], vals, width, color=TOK_COLORS[tok],
           label=TOK_LABELS[tok], alpha=0.88, edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(DS_LABELS, fontsize=9)
ax.set_ylabel("Mismatch Rate (fraction)")
ax.set_ylim(0, 1.08)
ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_title("A. Mismatch Rate\n(fraction where len(tok) ≠ len(tok_grapheme))")
ax.legend(fontsize=8, framealpha=0.5)

# Panel B: ratio mean ± std
ax = axes[1]
for i, tok in enumerate(TOK_ORDER):
    means = [get_val(summary, ds, tok, "ratio_mean") for ds in DS_ORDER]
    stds  = [get_val(summary, ds, tok, "ratio_std")  for ds in DS_ORDER]
    ax.bar(x + offset[i], means, width, color=TOK_COLORS[tok],
           label=TOK_LABELS[tok], alpha=0.88, edgecolor="white", linewidth=0.5)
    ax.errorbar(x + offset[i], means, yerr=stds, fmt="none",
                color="black", capsize=2, linewidth=0.8, alpha=0.7)

ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5,
           label="Perfect alignment (ratio=1)")
ax.set_xticks(x)
ax.set_xticklabels(DS_LABELS, fontsize=9)
ax.set_ylabel("len(tok) / len(tok_grapheme)  (mean ± std)")
ax.set_title("B. Length Ratio\n(>1 = more tokens than grapheme, <1 = fewer)")
ax.legend(fontsize=8, framealpha=0.5)

fig1.tight_layout()
path1 = f"{OUTPUT_DIR}/fig1_mismatch_and_ratio.png"
fig1.savefig(path1, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {path1}")

# =============================================================================
# FIGURE 2 — Over vs Under Segmentation (Stat 3)
# =============================================================================
fig2, axes = plt.subplots(1, len(DS_ORDER), figsize=(16, 4.5), sharey=True)
fig2.suptitle("Stat 3 — Over- vs Under-Segmentation relative to tok_grapheme",
              fontsize=13, fontweight="bold", y=1.02)

for col, (ds, ds_label) in enumerate(zip(DS_ORDER, DS_LABELS)):
    ax  = axes[col]
    sub = summary[summary.dataset == ds].set_index("tokenizer")

    over_vals    = [sub.loc[t, "over_seg_rate"]  if t in sub.index else 0 for t in TOK_ORDER]
    under_vals   = [sub.loc[t, "under_seg_rate"] if t in sub.index else 0 for t in TOK_ORDER]
    aligned_vals = [max(0, 1 - o - u) for o, u in zip(over_vals, under_vals)]
    tok_labels   = [TOK_LABELS[t] for t in TOK_ORDER]

    ypos = np.arange(len(TOK_ORDER))
    ax.barh(ypos, over_vals,    color="#E07B39", alpha=0.85)
    ax.barh(ypos, under_vals,   left=over_vals,  color="#1D6FA4", alpha=0.85)
    ax.barh(ypos, aligned_vals,
            left=[o + u for o, u in zip(over_vals, under_vals)],
            color="#2D6A4F", alpha=0.85)

    ax.set_yticks(ypos)
    ax.set_yticklabels(tok_labels if col == 0 else [], fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Fraction", fontsize=8)
    ax.set_title(ds_label, fontsize=10)
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.6, alpha=0.4)

handles = [
    mpatches.Patch(color="#E07B39", alpha=0.85, label="Over-segmented (more tokens than grapheme)"),
    mpatches.Patch(color="#1D6FA4", alpha=0.85, label="Under-segmented (fewer tokens than grapheme)"),
    mpatches.Patch(color="#2D6A4F", alpha=0.85, label="Aligned (same length as grapheme)"),
]
fig2.legend(handles=handles, loc="lower center", ncol=3,
            fontsize=9, bbox_to_anchor=(0.5, -0.08), framealpha=0.6)
fig2.tight_layout()
path2 = f"{OUTPUT_DIR}/fig2_over_under_seg.png"
fig2.savefig(path2, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {path2}")

# =============================================================================
# FIGURE 3 — Mismatch by Sequence Length Quartile (Stat 4)
# =============================================================================
fig3, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
fig3.suptitle("Stat 4 — Mismatch Rate by tok_grapheme Length Quartile",
              fontsize=13, fontweight="bold", y=1.01)

axes_flat = axes.flatten()
for idx, (ds, ds_label) in enumerate(zip(DS_ORDER, DS_LABELS)):
    ax  = axes_flat[idx]
    sub = binned[binned.dataset == ds]

    # Build x-axis labels from grapheme_range column
    q_map = {}
    ref_rows = sub[sub.tokenizer == "tok_grapheme"]
    for _, row in ref_rows.iterrows():
        q_map[row["quartile"]] = row["grapheme_range"]
    x_labels = [f"{q}\n({q_map.get(q, '?')})" for q in Q_ORDER]

    for tok in TOK_ORDER:
        tok_sub = sub[sub.tokenizer == tok].set_index("quartile")
        vals = []
        for q in Q_ORDER:
            if q in tok_sub.index and tok_sub.loc[q, "n_samples"] > 0:
                vals.append(tok_sub.loc[q, "mismatch_rate"])
            else:
                vals.append(np.nan)
        ax.plot(range(len(Q_ORDER)), vals, marker="o", markersize=5,
                color=TOK_COLORS[tok], label=TOK_LABELS[tok],
                linewidth=1.8, alpha=0.9)

    ax.set_xticks(range(len(Q_ORDER)))
    ax.set_xticklabels(x_labels, fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_title(ds_label)
    ax.set_ylabel("Mismatch Rate" if idx % 3 == 0 else "")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7, alpha=0.4)
    if idx == 0:
        ax.legend(fontsize=8, framealpha=0.5)

axes_flat[-1].set_visible(False)
fig3.tight_layout()
path3 = f"{OUTPUT_DIR}/fig3_mismatch_by_quartile.png"
fig3.savefig(path3, bbox_inches="tight")
plt.close(fig3)
print(f"Saved: {path3}")

# =============================================================================
# FIGURE 4 — Alignment Rate vs tok_grapheme (Stat 5)
# =============================================================================
non_grapheme = ["tok_llama2", "tok_komodo", "tok_mt5"]

fig4, ax = plt.subplots(figsize=(9, 4.5))
fig4.suptitle("Stat 5 — Fraction of Samples Where len(tok_X) == len(tok_grapheme)",
              fontsize=11, fontweight="bold")

x        = np.arange(len(DS_ORDER))
width    = 0.22
offsets3 = np.array([-width, 0, width])

for i, tok in enumerate(non_grapheme):
    vals = []
    for ds in DS_ORDER:
        row = crosstok.loc[(crosstok.dataset == ds) & (crosstok.tokenizer == tok)]
        vals.append(row["aligned_rate"].values[0] if len(row) > 0 else 0)

    ax.bar(x + offsets3[i], vals, width,
           color=TOK_COLORS[tok], label=TOK_LABELS[tok],
           alpha=0.85, edgecolor="white")

    for xi, v in enumerate(vals):
        ax.text(xi + offsets3[i], v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5, color="dimgray")

ax.set_xticks(x)
ax.set_xticklabels(DS_LABELS, fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Aligned Rate (fraction of all samples)")
ax.legend(fontsize=9)

fig4.tight_layout()
path4 = f"{OUTPUT_DIR}/fig4_alignment_rate.png"
fig4.savefig(path4, bbox_inches="tight")
plt.close(fig4)
print(f"Saved: {path4}")

print("\nAll figures saved.")