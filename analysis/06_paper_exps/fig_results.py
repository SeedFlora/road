# Fig. 3: (a) test mAP50 of the three random and three group-aware split draws (unpaired, mean marked);
# (b) crossover twin injection as differences with 95% bootstrap intervals (anchors and placebo).
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                            "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"], "font.size": 8,
                            "axes.spines.top": False, "axes.spines.right": False})
HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results_paper.json").read_text())
E1 = R["E1"]; E3 = R["E3"]
BLUE, ORANGE, GREY = "#1f5fa8", "#d9730d", "#777777"
fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.75), gridspec_kw={"width_ratios": [1, 1.3]})
for x, kind, col in ((0, "group", ORANGE), (1, "random", BLUE)):
    v = [E1[f"{kind}_s{s}"]["map50"] for s in (0, 1, 2)]
    ax[0].scatter([x + d for d in (-0.08, 0, 0.08)], v, color=col, s=16, zorder=3)
    ax[0].hlines(np.mean(v), x - 0.22, x + 0.22, color="k", lw=1)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["group-\naware", "random"]); ax[0].set_xlim(-0.5, 1.5)
ax[0].set_ylabel("test mAP50"); ax[0].set_title("(a) split draws", fontsize=8)
rows = [("anchors: leaked $-$ unleaked", E3["diff_leaked_minus_unleaked"], BLUE),
        ("anchors: unleaked $-$ reference", E3["diff_unleaked_minus_base"], GREY),
        ("placebo: model A $-$ model B", E3["placebo_twinA_minus_twinB"], GREY)]
for k, (lab, d, col) in enumerate(rows):
    y = len(rows) - 1 - k
    ax[1].errorbar([d["point"]], [y], xerr=[[d["point"] - d["ci95"][0]], [d["ci95"][1] - d["point"]]], fmt="o", color=col, ms=4, capsize=2, lw=1)
ax[1].axvline(0, color="k", lw=0.6, ls="--")
for k, (lab, d, col) in enumerate(rows):
    ax[1].text(-0.034, len(rows) - 1 - k + 0.2, lab, fontsize=7, va="bottom", ha="left", bbox=dict(fc="white", ec="none", pad=0.4))
ax[1].set_yticks([]); ax[1].spines["left"].set_visible(False); ax[1].set_xlim(-0.035, 0.03)
ax[1].set_xlabel("difference in mAP50 (own AP)"); ax[1].set_title("(b) crossover", fontsize=8)
ax[1].set_ylim(-0.5, len(rows) - 0.1)
plt.tight_layout(pad=0.2, w_pad=0.6)
out = HERE.parent.parent / "repro" / "generated" / "figures"
out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "fig_results.pdf", bbox_inches="tight", pad_inches=0.01); plt.savefig(out / "fig_results.png", dpi=220, bbox_inches="tight")
print("saved")
