"""Generate Figure: quality-cost Pareto plot for composition methods.

Data from results/results_v3_n5.txt (KL held-out aggregate) and
results/results_cost.txt (wall-clock).
"""

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

# (method label, wall-clock ms, held-out aggregate KL, marker, color)
methods = [
    ("M1 (naive)",       3.1,   0.329, "o", "#888888"),
    ("M2 (RoPE)",        14.1,  0.405, "s", "#d95f02"),
    ("M3 (RoPE, ex mid)",14.0,  0.404, "^", "#e6ab02"),
    ("M4 (naive + H3)",  3.5,   0.308, "D", "#1b9e77"),
    ("M5 (RoPE + H3)",   13.4,  0.367, "P", "#7570b3"),
]
prefill_baseline = ("full prefill", 344.9, 0.0)

fig, ax = plt.subplots(figsize=(6.0, 4.0))

for label, ms, kl, marker, color in methods:
    ax.scatter(ms, kl, s=110, marker=marker, color=color, edgecolor="black",
               linewidth=0.8, label=label, zorder=3)

# Reference: full prefill baseline (KL = 0 by definition, at high cost)
ax.axhline(0.0, color="grey", linestyle=":", linewidth=1, alpha=0.6)
ax.text(200, 0.005, f"full prefill oracle: {prefill_baseline[1]:.1f} ms, KL=0",
        fontsize=9, color="grey")

# Highlight the Pareto arrow from M1 -> M4 and M2 -> M5
ax.annotate("", xy=(3.5, 0.308), xytext=(3.1, 0.329),
            arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1.5, alpha=0.7))
ax.annotate("", xy=(13.4, 0.367), xytext=(14.1, 0.405),
            arrowprops=dict(arrowstyle="->", color="#7570b3", lw=1.5, alpha=0.7))

ax.set_xlabel("Composition wall-clock (ms, CPU, |B|=19)")
ax.set_ylabel("Held-out next-token KL (lower is better)")
ax.set_title("Quality-cost trade-off across composition methods")
ax.set_xscale("log")
ax.set_xlim(2, 400)
ax.set_ylim(-0.03, 0.5)
ax.grid(True, alpha=0.3, which="both")
ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")

plt.tight_layout()
plt.savefig("paper/pareto.pdf", format="pdf", bbox_inches="tight")
plt.savefig("paper/pareto.png", format="png", dpi=200, bbox_inches="tight")
print("Wrote paper/pareto.pdf and paper/pareto.png")
