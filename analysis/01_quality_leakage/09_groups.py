# Near-duplicate groups = connected components of geometrically verified edges
import os, json, numpy as np, pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from pairmontage import pair_montage, grid, OUT, meta
FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)
ids = np.load(os.path.join(OUT, "emb_ids.npy")); N = len(ids)
df = pd.read_csv(os.path.join(OUT, "candidate_pairs_verified.csv"))
res = meta.loc[ids, "res"].values; lab_ = meta.loc[ids, "labeled"].values
from scipy.sparse import csr_matrix
STRONG = 30
def build(tin, tsim, filt=True):
    """accepted edges: moving inliers >= tin or sim_top >= tsim. For grouping, weak edges (inl < STRONG and sim < tsim)
    are kept only if supported by >=1 common verified neighbour (triangle) or if one endpoint is a leaf (degree 1):
    prevents single spurious edges from bridging two different rides (anti-chaining)."""
    E = df[(df.n_inl >= tin) | (df.sim_top >= tsim)].copy()
    if filt:
        strong = (E.n_inl >= STRONG) | (E.sim_top >= tsim)
        A = csr_matrix((np.ones(len(E)), (E.a, E.b)), shape=(N, N)); A = ((A + A.T) > 0).astype(np.float32)
        deg = np.asarray(A.sum(1)).ravel()
        tri = np.asarray(A[E.a.values].multiply(A[E.b.values]).sum(1)).ravel()
        E["strong"] = strong.values; E["tri"] = tri; E["used_for_grouping"] = (strong.values | (tri >= 1) | (np.minimum(deg[E.a], deg[E.b]) == 1))
        e = E[E.used_for_grouping]
    else:
        E["used_for_grouping"] = True; e = E
    A = coo_matrix((np.ones(len(e)), (e.a, e.b)), shape=(N, N))
    n, g = connected_components(A, directed=False)
    return E, g
out = {}
for tin in []:
    e, g = build(tin, 0.96)
    u, cnt = np.unique(g, return_counts=True); sz = cnt[g]
    print(f"inl>={tin}|sim>=.96: edges={len(e)} groups={len(u)} multi={np.sum(cnt>1)} imgs_in_multi={np.mean(sz>1)*100:.1f}% max={cnt.max()} top5={sorted(cnt)[-5:]} cross-res edges={(res[e.a]!=res[e.b]).sum()}")
TIN, TSIM = 15, 0.96
e, g = build(TIN, TSIM)
print("accepted near-dup pairs:", len(e), "used for grouping:", int(e.used_for_grouping.sum()))
# capped union: the weak edges that were held back are merged afterwards, strongest first, as long as the merged group
# stays <= CAP images. Merging never hurts leakage safety; the cap keeps groups small enough for a stratified split.
CAP = 250
core = g.copy()
parent = np.arange(g.max() + 1); size = np.bincount(g).astype(int)
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
held = e[~e.used_for_grouping].sort_values("n_inl", ascending=False)
n_merged = n_refused = 0
for a, b in zip(held.a.values, held.b.values):
    ra, rb = find(g[a]), find(g[b])
    if ra == rb: continue
    if size[ra] + size[rb] <= CAP: parent[rb] = ra; size[ra] += size[rb]; n_merged += 1
    else: n_refused += 1
g = np.array([find(x) for x in g])
_, g = np.unique(g, return_inverse=True)
print(f"capped union: merged {n_merged}, refused {n_refused} (cap {CAP})")
e = e.copy(); e["res_a"] = res[e.a]; e["res_b"] = res[e.b]
e.to_csv(os.path.join(OUT, "near_duplicate_pairs.csv"), index=False)
u, cnt = np.unique(g, return_counts=True); sz = cnt[g]
grp = pd.DataFrame({"id": ids, "group": g, "group_size": sz, "core_group": core, "res": res, "labeled": lab_})
grp.to_csv(os.path.join(OUT, "near_duplicate_groups.csv"), index=False)
summ = {"rule": f"weak edges (<{STRONG} inl) kept for grouping only with triangle support or leaf endpoint; edge if SIFT(upper50%, RootSIFT, ratio .8, static<4px removed, MAGSAC F 1.5px) moving inliers >= {TIN} OR DINOv2 top60 cosine >= {TSIM}; groups = connected components",
        "n_pairs": int(len(e)), "n_pairs_used_for_grouping": int(e.used_for_grouping.sum()), "n_imgs_with_nd_partner": int(len(set(e.a) | set(e.b))), "share_imgs_with_nd_partner": float(len(set(e.a) | set(e.b)) / N), "capped_union": {"cap": CAP, "merged": int(n_merged), "refused": int(n_refused)}, "n_groups": int(len(u)), "n_multi_groups": int(np.sum(cnt > 1)), "share_in_multi": float(np.mean(sz > 1)), "max_group": int(cnt.max()),
        "size_hist": {str(k): int(v) for k, v in zip(*np.unique(np.minimum(cnt, 21), return_counts=True))},
        "cross_res_pairs": int((e.res_a != e.res_b).sum())}
per = []
for r in sorted(set(res)):
    m = res == r
    per.append({"res": int(r), "n": int(m.sum()), "pairs_within": int(((e.res_a == r) & (e.res_b == r)).sum()),
                "groups": int(len(np.unique(g[m]))), "share_in_multi": float(np.mean(sz[m] > 1)), "max_group": int(sz[m].max())})
summ["per_res"] = per
# groups spanning labeled + unlabeled
gl = grp.groupby("group").labeled.agg(["min", "max", "size"])
summ["groups_mixing_labeled_unlabeled"] = int(((gl["min"] == False) & (gl["max"] == True)).sum())
summ["unlabeled_imgs_in_groups_with_labeled"] = int(grp[grp.group.isin(gl[(gl["min"] == False) & (gl["max"] == True)].index) & (~grp.labeled)].shape[0])
json.dump(summ, open(os.path.join(OUT, "near_duplicate_summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
# montages: cross-res pairs, largest groups
x = e[e.res_a != e.res_b]
if len(x): pair_montage([(r.id_a, r.id_b, f"inl={r.n_inl} t={r.sim_top:.3f} {r.res_a}/{r.res_b}") for r in x.head(16).itertuples()], os.path.join(FIG, "nd_crossres_pairs.jpg"))
order = u[np.argsort(-cnt)]
rng = np.random.default_rng(0)
for k in range(3):
    m = np.where(g == order[k])[0]; pk = rng.choice(m, min(40, len(m)), replace=False)
    grid(list(ids[pk]), os.path.join(FIG, f"nd_group_rank{k}.jpg"), T=160, texts=[f"{ids[p]} r{res[p]}" for p in pk])
    print("rank", k, len(m), pd.Series(res[m]).value_counts().to_dict())
