import os, numpy as np, pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix, csr_matrix
from pairmontage import OUT, meta
ids = np.load(os.path.join(OUT, "emb_ids.npy")); N = len(ids)
df = pd.read_csv(os.path.join(OUT, "candidate_pairs_verified.csv"))
res = meta.loc[ids, "res"].values
df["ratio"] = df.n_inl / df.n_good.clip(lower=1)
x = df[((df.n_inl >= 15) | (df.sim_top >= .96)) & (res[df.a] != res[df.b])]
print(x[["id_a", "id_b", "n_good", "n_inl", "n_static", "ratio", "sim_top", "sim_full"]])
acc = df[(df.n_inl >= 15) | (df.sim_top >= .96)]
print(acc[["n_good", "n_inl", "ratio", "sim_top"]].describe().round(3))
def cc(e):
    A = coo_matrix((np.ones(len(e)), (e.a, e.b)), shape=(N, N)); return connected_components(A, directed=False)[1]
def report(e, name):
    g = cc(e); u, cnt = np.unique(g, return_counts=True); sz = cnt[g]
    print(f"{name:40s} edges={len(e):6d} groups={len(u)} imgs_in_multi={np.mean(sz>1)*100:.1f}% max={cnt.max()} top5={sorted(cnt.tolist())[-5:]} crossres={(res[e.a]!=res[e.b]).sum()}")
    return g
for weak_lo, strong in [(15, 30), (15, 25), (13, 30), (15, 40)]:
    E = df[(df.n_inl >= weak_lo) | (df.sim_top >= .96)].copy()
    strong_m = (E.n_inl >= strong) | (E.sim_top >= .96)
    A = csr_matrix((np.ones(len(E)), (E.a, E.b)), shape=(N, N)); A = ((A + A.T) > 0).astype(np.float32)
    deg = np.asarray(A.sum(1)).ravel()
    tri = np.asarray(A[E.a.values].multiply(A[E.b.values]).sum(1)).ravel()  # common neighbours
    keep = strong_m | (tri >= 1) | (np.minimum(deg[E.a], deg[E.b]) == 1)
    report(E[keep], f"weak>={weak_lo} strong>={strong} tri/leaf")
    keep2 = strong_m | (tri >= 2) | (np.minimum(deg[E.a], deg[E.b]) == 1)
    report(E[keep2], f"weak>={weak_lo} strong>={strong} tri2/leaf")
