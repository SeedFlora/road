import os, numpy as np, pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform
ROOT = Path(__file__).resolve().parents[2]
OUT = os.path.join(ROOT, "analysis", "01_quality_leakage")
ids = np.load(os.path.join(OUT, "emb_ids.npy"))
E = np.load(os.path.join(OUT, "emb_dinov2s_top_cls.npy")); E /= np.linalg.norm(E, axis=1, keepdims=True)
S = (E @ E.T).astype(np.float64); np.fill_diagonal(S, 1.0)
D = np.clip(1 - S, 0, 2); np.fill_diagonal(D, 0); D = (D + D.T) / 2
cond = squareform(D, checks=False)
Sm = S.copy(); np.fill_diagonal(Sm, -1)
def stats(lab, name):
    _, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    sz = cnt[inv]
    # residual cross-group pairs with sim >= 0.90 / 0.92
    same = lab[:, None] == lab[None, :]
    r90 = ((Sm >= 0.90) & ~same).sum() // 2; r92 = ((Sm >= 0.92) & ~same).sum() // 2
    print(f"{name:28s} groups={len(cnt):5d} multi={np.sum(cnt>1):5d} imgs_in_multi={np.mean(sz>1)*100:5.1f}% max={cnt.max():5d} p99size={np.quantile(sz,0.99):.0f} cross>=.90:{r90:6d} cross>=.92:{r92:5d}")
for t in [0.95, 0.93, 0.92, 0.91, 0.90, 0.88]:
    n, lab = connected_components(csr_matrix(Sm >= t), directed=False); stats(lab, f"CC sim>={t}")
for method in ["complete", "average"]:
    Z = linkage(cond, method=method)
    np.save(os.path.join(OUT, f"linkage_top_cls_{method}.npy"), Z)
    for t in ([0.08, 0.10, 0.12, 0.15, 0.18, 0.20] if method == "complete" else [0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.25]):
        lab = fcluster(Z, t=t, criterion="distance"); stats(lab, f"{method} d<={t} (s>={1-t:.2f})")
