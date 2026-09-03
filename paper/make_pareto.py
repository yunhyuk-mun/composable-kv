"""Generate Figure: horizontal bar chart of composition methods.

Two-in-one figure:
  - Left panel: held-out next-token KL per method (main quality metric).
  - Right panel: composition wall-clock per method (cost metric).

Same method ordering + colors on both panels, so the reader can eye up
a method's quality-vs-cost trade-off directly.
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
})

# Order: best KL at the top of the bar chart
methods = [
    ("M4  M1 + H3 corrector",       0.308, 3.5,  "#1b9e77"),  # green
    ("M1  naive concat",            0.329, 3.1,  "#4d4d4d"),  # gray
    ("M5  M2 + H3 corrector",       0.367, 13.4, "#7570b3"),  # purple
    ("M3  RoPE-shift ex mid",       0.404, 14.0, "#ff7f0e"),  # orange
    ("M2  RoPE-shifted concat",     0.405, 14.1, "#d62728"),  # red
]

labels     = [m[0] for m in methods]
kl_values  = [m[1] for m in methods]
ms_values  = [m[2] for m in methods]
colors     = [m[3] for m in methods]
prefill_ms = 344.9

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                gridspec_kw={"width_ratios": [1.6, 1.0]})

# --- LEFT: held-out KL ---
bars = axL.barh(labels, kl_values, color=colors, edgecolor="black", linewidth=0.9)
for bar, v in zip(bars, kl_values):
    axL.text(v + 0.006, bar.get_y() + bar.get_height() / 2,
             f"{v:.3f}", va="center", fontsize=10)

axL.set_xlim(0, max(kl_values) * 1.18)
axL.axvline(0, color="black", linewidth=0.7)
axL.set_xlabel("Held-out next-token KL vs full-prefill oracle  (lower is better)")
axL.set_title("Quality  (Table 2 aggregate)", pad=10)
axL.invert_yaxis()  # best on top
axL.grid(True, axis="x", alpha=0.3)
axL.spines["top"].set_visible(False)
axL.spines["right"].set_visible(False)

# Baseline references indicated by dashed lines instead of arrows
# (M1 baseline dashed grey; M2 baseline dashed red) so the bar length
# difference below each baseline reads as the H3 improvement.
axL.axvline(0.329, color="#4d4d4d", linestyle="--", linewidth=0.9, alpha=0.55)
axL.text(0.329, -0.55, "M1 baseline", color="#4d4d4d", fontsize=8.5,
         ha="center", va="center")
axL.axvline(0.405, color="#d62728", linestyle="--", linewidth=0.9, alpha=0.55)
axL.text(0.405, -0.55, "M2 baseline", color="#d62728", fontsize=8.5,
         ha="center", va="center")

# --- RIGHT: cost bar chart ---
bars_r = axR.barh(labels, ms_values, color=colors, edgecolor="black", linewidth=0.9)
for bar, v in zip(bars_r, ms_values):
    axR.text(v + 0.4, bar.get_y() + bar.get_height() / 2,
             f"{v:.1f} ms", va="center", fontsize=10)

axR.axvline(prefill_ms, color="#555555", linestyle="--", linewidth=1.2, alpha=0.8)
axR.text(prefill_ms, -0.7, f"full prefill\n{prefill_ms:.0f} ms",
         fontsize=9, ha="center", color="#333333")

axR.set_xlim(0, prefill_ms * 1.15)
axR.set_xlabel("Composition wall-clock  (ms, CPU, |B|=19)")
axR.set_title("Cost  (Table 3)", pad=10)
axR.invert_yaxis()
axR.set_yticklabels([])
axR.grid(True, axis="x", alpha=0.3)
axR.spines["top"].set_visible(False)
axR.spines["right"].set_visible(False)

plt.suptitle("Quality-cost trade-off across composition methods (Qwen-2.5-0.5B)",
             y=1.03, fontsize=13)
plt.tight_layout()
plt.savefig("paper/pareto.pdf", format="pdf", bbox_inches="tight")
plt.savefig("paper/pareto.png", format="png", dpi=200, bbox_inches="tight")
print("Wrote paper/pareto.pdf and paper/pareto.png")
