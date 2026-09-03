"""Generate Figure: quality-cost Pareto plot for composition methods."""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# (label, wall_ms, KL, marker, color, xytext_offset in points)
methods = [
    ("M1",  3.1,   0.329, "o",  "#4d4d4d", (-28,  10)),   # gray
    ("M4",  3.5,   0.308, "D",  "#1b9e77", ( 12, -12)),   # green
    ("M2",  14.1,  0.405, "s",  "#d62728", ( 12,  10)),   # red
    ("M3",  14.0,  0.404, "v",  "#ff7f0e", ( 12, -14)),   # orange, triangle-down
    ("M5",  13.4,  0.367, "P",  "#7570b3", (-30, -14)),   # purple
]
oracle_x, oracle_y = 344.9, 0.0

fig, ax = plt.subplots(figsize=(8.5, 5.0))

# Oracle reference line
ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)

# Improvement arrows first (behind markers)
ax.annotate("", xy=(3.5, 0.308), xytext=(3.1, 0.329),
            arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=2.5, alpha=0.9,
                            shrinkA=8, shrinkB=8))
ax.annotate("", xy=(13.4, 0.367), xytext=(14.1, 0.405),
            arrowprops=dict(arrowstyle="->", color="#7570b3", lw=2.5, alpha=0.9,
                            shrinkA=8, shrinkB=8))

# Method markers
for label, ms, kl, marker, color, (dx, dy) in methods:
    ax.scatter(ms, kl, s=260, marker=marker, color=color, edgecolor="black",
               linewidth=1.3, zorder=4)
    ax.annotate(label, xy=(ms, kl), xytext=(dx, dy), textcoords="offset points",
                fontsize=13, fontweight="bold", color=color,
                ha="center", va="center", zorder=5)

# Oracle star
ax.scatter(oracle_x, oracle_y, s=340, marker="*", color="white",
           edgecolor="black", linewidth=1.3, zorder=4)
ax.annotate("full prefill\n(KL = 0 by def.)", xy=(oracle_x, oracle_y),
            xytext=(-6, 35), textcoords="offset points",
            fontsize=10, color="#333333", ha="center",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9, alpha=0.7))

# Improvement direction is communicated by the arrows and by the legend + caption.

# Legend — two columns, plenty of horizontal room
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#4d4d4d",
           markeredgecolor="black", markersize=12, label="M1  naive concat"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#d62728",
           markeredgecolor="black", markersize=12, label="M2  RoPE-shifted concat"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#ff7f0e",
           markeredgecolor="black", markersize=12, label="M3  RoPE-shift, mid layers unshifted"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#1b9e77",
           markeredgecolor="black", markersize=12, label="M4  M1 + H3 corrector"),
    Line2D([0], [0], marker="P", color="w", markerfacecolor="#7570b3",
           markeredgecolor="black", markersize=12, label="M5  M2 + H3 corrector"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
           markeredgecolor="black", markersize=14, label="full prefill (oracle)"),
]
ax.legend(handles=legend_handles, loc="upper center",
          bbox_to_anchor=(0.5, -0.16), ncol=2,
          frameon=True, fancybox=False, edgecolor="black")

ax.set_xlabel("Composition wall-clock (ms, CPU, |B|=19)")
ax.set_ylabel("Held-out next-token KL (lower is better)")
ax.set_title("Quality-cost trade-off across composition methods", pad=12)
ax.set_xscale("log")
ax.set_xlim(1.8, 800)
ax.set_ylim(-0.06, 0.50)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout(pad=1.5)
plt.savefig("paper/pareto.pdf", format="pdf", bbox_inches="tight")
plt.savefig("paper/pareto.png", format="png", dpi=200, bbox_inches="tight")
print("Wrote paper/pareto.pdf and paper/pareto.png")
