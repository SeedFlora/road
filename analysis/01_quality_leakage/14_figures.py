import os, json, numpy as np, pandas as pd, matplotlib
from pathlib import Path
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
OUT = os.path.join(ROOT, "analysis", "01_quality_leakage"); FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)
SURF, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"; C1, C2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": T2, "xtick.color": T2, "ytick.color": T2,
                     "text.color": T1, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
S = json.load(open(os.path.join(OUT, "split_summary.json")))
# (a) leakage on the test split
mets = [("leak_rate_nd", "verified near-duplicate\nin train"), ("frac_maxsim_to_train_ge_0.90", "DINOv2 max-sim\nto train >= 0.90"), ("frac_maxsim_to_train_ge_0.85", "DINOv2 max-sim\nto train >= 0.85")]
fig, ax = plt.subplots(figsize=(7.2, 3.6)); x = np.arange(len(mets)); w = 0.36
for k, (kind, col, lab) in enumerate([("random", C1, "Random split"), ("group", C2, "Group split (ours)")]):
    v = [S["leakage"][kind]["test"][m] * 100 for m, _ in mets]
    b = ax.bar(x + (k - 0.5) * (w + 0.02), v, w, color=col, label=lab)
    for xi, vi in zip(x + (k - 0.5) * (w + 0.02), v): ax.text(xi, vi + 1.2, f"{vi:.1f}%", ha="center", va="bottom", fontsize=9, color=T1)
ax.set_xticks(x, [m[1] for m in mets]); ax.set_ylabel("% of test images"); ax.set_ylim(0, 115); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper center", ncol=2); ax.set_title("Test-set leakage: random vs near-duplicate-group split (labeled images, n=1065)", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_leakage_test.png"), dpi=160); plt.close(fig)
# (b) inlier histogram of candidate pairs
df = pd.read_csv(os.path.join(OUT, "candidate_pairs_verified.csv"))
fig, ax = plt.subplots(figsize=(7.2, 3.4))
bins = np.concatenate([np.arange(0, 60, 2), np.arange(60, 620, 20)])
ax.hist(df.n_inl.clip(upper=600), bins=np.arange(0, 202, 2), color=C1, edgecolor=SURF, lw=0.5)
ax.set_yscale("log"); ax.set_xlim(0, 200)
for t, lab, ha, dx in [(15, "accept >= 15", "right", -2), (30, "strong >= 30", "left", 2)]:
    ax.axvline(t, color=T2, lw=1, ls="--"); ax.text(t + dx, 2.5e4, lab, color=T2, fontsize=9, ha=ha)
ax.set_xlabel("RANSAC-verified moving SIFT inliers (upper 50% of frame; >200 clipped out of view)"); ax.set_ylabel("candidate pairs (log)")
ax.set_title(f"Geometric verification of {len(df):,} DINOv2 candidate pairs", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_inlier_hist.png"), dpi=160); plt.close(fig)
# (c) share of images with >=1 verified near-duplicate, per resolution group
g = pd.read_csv(os.path.join(OUT, "near_duplicate_groups.csv")); nd = pd.read_csv(os.path.join(OUT, "near_duplicate_pairs.csv"))
g["has"] = g.id.isin(set(nd.id_a) | set(nd.id_b)); r = g.groupby("res").agg(n=("id", "size"), s=("has", "mean")).reset_index()
fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.bar(r.res.astype(str), r.s * 100, color=C1, width=0.55)
for i, row in r.iterrows(): ax.text(i, row.s * 100 + 1.5, f"{row.s*100:.0f}% (n={int(row.n)})", ha="center", fontsize=9, color=T1)
ax.set_ylim(0, 105); ax.set_ylabel("% images with a near-duplicate"); ax.set_xlabel("resolution group (px)"); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("Near-duplicate prevalence by source (resolution group)", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_nd_by_res.png"), dpi=160); plt.close(fig)
print("ok")
