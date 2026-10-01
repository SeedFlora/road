import os, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = os.path.join(ROOT, "analysis", "01_quality_leakage")
ids = np.load(os.path.join(OUT, "emb_ids.npy"))
meta = pd.read_csv(os.path.join(OUT, "meta.csv")).set_index("id").loc[ids].reset_index()
def norm(x): return x / np.linalg.norm(x, axis=1, keepdims=True)
for name in ["top_cls", "top_mean", "full_cls", "bot_cls"]:
    E = norm(np.load(os.path.join(OUT, f"emb_dinov2s_{name}.npy")))
    S = E @ E.T; np.fill_diagonal(S, -1)
    nn = S.max(1)
    iu = np.triu_indices(len(E), 1)
    allv = S[iu]
    print(name, "NN sim quantiles:", np.round(np.quantile(nn, [0.01, .1, .25, .5, .75, .9, .99]), 3),
          "| random pair median %.3f p99.9 %.4f" % (np.median(allv), np.quantile(allv, 0.999)))
    for t in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
        print(f"   pairs>={t}: {(allv>=t).sum():7d}   imgs with NN>={t}: {(nn>=t).sum()}")
