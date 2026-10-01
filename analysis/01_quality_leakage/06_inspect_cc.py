import os, numpy as np, pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from pairmontage import grid, OUT, meta
FIG = os.path.join(OUT, "figs")
ids = np.load(os.path.join(OUT, "emb_ids.npy"))
E = np.load(os.path.join(OUT, "emb_dinov2s_top_cls.npy")); E /= np.linalg.norm(E, axis=1, keepdims=True)
S = E @ E.T; np.fill_diagonal(S, -1)
n, lab = connected_components(csr_matrix(S >= 0.90), directed=False)
u, cnt = np.unique(lab, return_counts=True); order = u[np.argsort(-cnt)]
rng = np.random.default_rng(0)
for rank in range(4):
    g = np.where(lab == order[rank])[0]
    res = meta.loc[ids[g], "res"].value_counts().to_dict()
    print(rank, len(g), res, "labeled frac %.2f" % meta.loc[ids[g], "labeled"].mean())
    pick = rng.choice(g, min(40, len(g)), replace=False)
    grid(list(ids[pick]), os.path.join(FIG, f"cc90_rank{rank}.jpg"), T=160, ncol=8, texts=[f"{ids[p]} r{meta.loc[ids[p],'res']}" for p in pick])
