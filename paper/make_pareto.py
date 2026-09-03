"""Generate Figure: quality-cost Pareto plot for composition methods."""

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# (method label, wall-clock ms, held-out aggregate KL, marker, color, label_offset)
# label_offset: (dx_log_units, dy_kl_units) for the text label
methods = [
    ("M1",   3.1,   0.329, "o", "#4d4d4d", (0.10, +0.020)),
    ("M2",   14.1,  0.405, "s", "#d95f02", (0.10, +0.020)),
    ("M3",   14.0,  0.404, "^", "#e6ab02", (0.15, -0.010)),
    ("M4",   3.5,   0.308, "D", "#1b9e77", (0.10, -0.030)),
    ("M5",   13.4,  0.367, "P", "#7570b3", (0.10, -0.030)),
]
prefill_baseline = ("full prefill oracle", 344.9, 0.0)

fig, ax = plt.subplots(figsize=(7.0, 4.6))

# Reference: full prefill baseline as a horizontal reference line + a labeled point
ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.2, alpha=0.7)
ax.scatter(prefill_baseline[1], prefill_baseline[2], s=220, marker="*",
           color="white", edgecolor="black", linewidth=1.2, zorder=3,
           label="full prefill (oracle)")
ax.annotate("full prefill\n(KL = 0 by def.)", xy=(prefill_baseline[1], 0.0),
            xytext=(-40, 20), textcoords="offset points",
            fontsize=9, color="#333333", ha="right",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8, alpha=0.7))

# Draw improvement arrows first so they sit under the markers
ax.annotate("", xy=(3.5, 0.308), xytext=(3.1, 0.329),
            arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=2.0, alpha=0.85))
ax.annotate("", xy=(13.4, 0.367), xytext=(14.1, 0.405),
            arrowprops=dict(arrowstyle="->", color="#7570b3", lw=2.0, alpha=0.85))

# Methods
for label, ms, kl, marker, color, (dxlog, dy) in methods:
    ax.scatter(ms, kl, s=220, marker=marker, color=color, edgecolor="black",
               linewidth=1.1, label=f"{label}", zorder=4)
    # place text label at log-multiplied x
    ax.annotate(label, xy=(ms, kl), xytext=(ms * (1 + dxlog), kl + dy),
                fontsize=11, fontweight="bold", color=color,
                ha="left", va="center")

# Legend with method descriptions in expanded form
legend_labels = [
    "M1  naive concat",
    "M2  RoPE-shifted concat",
    "M3  RoPE-shift ex mid layers",
    "M4  M1 + H3 corrector",
    "M5  M2 + H3 corrector",
    "full prefill (oracle)",
]
# Rebuild legend manually
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#4d4d4d", markeredgecolor="black", markersize=11),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#d95f02", markeredgecolor="black", markersize=11),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#e6ab02", markeredgecolor="black", markersize=11),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#1b9e77", markeredgecolor="black", markersize=11),
    Line2D([0], [0], marker="P", color="w", markerfacecolor="#7570b3", markeredgecolor="black", markersize=11),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="white", markeredgecolor="black", markersize=14),
]
ax.legend(handles, legend_labels, loc="upper center",
          bbox_to_anchor=(0.5, -0.18), ncol=3,
          frameon=True, fancybox=False, edgecolor="black")

ax.set_xlabel("Composition wall-clock (ms, CPU, |B|=19)")
ax.set_ylabel("Held-out next-token KL (lower is better)")
ax.set_title("Quality-cost trade-off across composition methods", pad=12)
ax.set_xscale("log")
ax.set_xlim(2, 700)
ax.set_ylim(-0.05, 0.50)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout(pad=1.5)
plt.savefig("paper/pareto.pdf", format="pdf", bbox_inches="tight")
plt.savefig("paper/pareto.png", format="png", dpi=200, bbox_inches="tight")
print("Wrote paper/pareto.pdf and paper/pareto.png")
