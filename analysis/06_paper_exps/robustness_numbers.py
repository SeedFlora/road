# Extra numbers requested by the internal review (all CPU, low memory):
# (1) prevalence and random-split leakage counted with strong pairs only (>=30 inliers or cos>=0.96) and audit-weighted;
# (2) chaining: largest group of plain connected components of all accepted pairs vs our rule;
# (3) twins in the crossover: share with a direct verified edge to their anchor; twins per anchor;
# (4) composition of test sets (singleton share, images from groups >=20) and training-group coverage per split kind;
# (5) residual-leakage estimate for group-aware test sets from DINOv2 cos>=0.90 neighbours and the audit;
# (6) false positives of the audit that share a verified neighbour (2-hop) or final group;
# (7) extra grouping baseline: average-linkage clustering of the upper-60% DINOv2 embedding (and full-frame cos>=0.90 CC).
import os, json, numpy as np, pandas as pd
from pathlib import Path
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.cluster.hierarchy import fcluster

ROOT = Path(__file__).resolve().parents[2]
A = os.path.join(ROOT, "analysis"); QL = os.path.join(A, "01_quality_leakage"); HERE = os.path.join(A, "06_paper_exps")
ids = np.load(os.path.join(QL, "emb_ids.npy")); N = len(ids); pos = {int(i): k for k, i in enumerate(ids)}
meta = pd.read_csv(os.path.join(QL, "meta.csv")).set_index("id")
lab = meta.loc[ids, "labeled"].values; res = meta.loc[ids, "res"].values
nd = pd.read_csv(os.path.join(QL, "near_duplicate_pairs.csv"))
grp = pd.read_csv(os.path.join(QL, "near_duplicate_groups.csv")).set_index("id").loc[ids, "group"].values
ea = np.array([pos[i] for i in nd.id_a]); eb = np.array([pos[i] for i in nd.id_b])
strong = ((nd.n_inl >= 30) | (nd.sim_top >= 0.96)).values
out = {}
# (1)
def prevalence(mask):
    s = np.zeros(N, bool); s[ea[mask]] = True; s[eb[mask]] = True; return float(s.mean())
out["prevalence_all"] = prevalence(np.ones(len(nd), bool)); out["prevalence_strong"] = prevalence(strong)
au = json.load(open(os.path.join(HERE, "pair_audit_summary.json")))
def split_of(kind, seed):
    if seed == 0:
        sa = pd.read_csv(os.path.join(QL, "splits", "split_assignments.csv")).set_index("id")[f"{kind}_split"].to_dict()
        return sa
    sp = {}
    for s in ("train", "val", "test"):
        p = os.path.join(HERE, "ds", f"{kind}_s{seed}", "labels", s)
        for f in os.listdir(p):
            if Path(f).suffix.lower() == ".txt":
                sp[int(Path(f).stem)] = s
    return sp
def leak(sp, mask):
    test = {i for i, v in sp.items() if v == "test"}; hit = set()
    for a, b in zip(nd.id_a.values[mask], nd.id_b.values[mask]):
        if a in test and sp.get(b) == "train": hit.add(a)
        if b in test and sp.get(a) == "train": hit.add(b)
    return len(hit) / len(test)
# audit-weighted: each accepted pair is a true near-duplicate with its stratum precision (lower / upper); P(image leaked) = 1 - prod(1-p)
pl = {"strong": (au["ours_strong"]["share_same_lower"], au["ours_strong"]["share_same_upper"]),
      "weak": (au["ours_weak"]["share_same_lower"], au["ours_weak"]["share_same_upper"])}
def weighted_leak(sp, b):
    test = {i for i, v in sp.items() if v == "test"}; logq = {}
    for a, c, st in zip(nd.id_a.values, nd.id_b.values, strong):
        p = pl["strong" if st else "weak"][b]
        for x, y in ((a, c), (c, a)):
            if x in test and sp.get(y) == "train": logq[x] = logq.get(x, 0) + np.log(1 - p + 1e-12)
    return float(sum(1 - np.exp(v) for v in logq.values()) / len(test))
def weighted_prev(b):
    logq = np.zeros(N)
    for a, c, st in zip(ea, eb, strong):
        p = pl["strong" if st else "weak"][b]; logq[a] += np.log(1 - p + 1e-12); logq[c] += np.log(1 - p + 1e-12)
    return float((1 - np.exp(logq)).mean())
out["prevalence_weighted"] = [weighted_prev(0), weighted_prev(1)]
for kind in ("random", "group"):
    L_all, L_str, L_w0, L_w1 = [], [], [], []
    for seed in (0, 1, 2):
        sp = split_of(kind, seed)
        L_all.append(leak(sp, np.ones(len(nd), bool))); L_str.append(leak(sp, strong))
        L_w0.append(weighted_leak(sp, 0)); L_w1.append(weighted_leak(sp, 1))
    out[f"{kind}_test_leak_all"] = L_all; out[f"{kind}_test_leak_strong"] = L_str; out[f"{kind}_test_leak_weighted"] = [float(np.mean(L_w0)), float(np.mean(L_w1))]
# (2) chaining
n, g_cc = connected_components(coo_matrix((np.ones(len(ea)), (ea, eb)), shape=(N, N)), directed=False)
out["plain_cc_largest_all"] = int(np.bincount(g_cc).max()); out["plain_cc_largest_labeled"] = int(np.bincount(g_cc[lab]).max())
out["ours_largest_all"] = int(np.bincount(grp).max())
# (3) twins
adj = {}
for a, b in zip(nd.id_a.values, nd.id_b.values): adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
tw = {}
for half in ("A", "B"):
    anc = [int(x) for x in open(os.path.join(HERE, f"inj_anchors{half}_ids.txt")).read().split()]
    twins = [int(x) for x in open(os.path.join(HERE, f"inj_twins{half}_ids.txt")).read().split()]
    gof = dict(zip(ids, grp)); ga = {}
    for a in anc: ga.setdefault(gof[a], []).append(a)
    direct = sum(1 for t in twins if any(a in adj.get(t, ()) for a in ga.get(gof[t], [])))
    per = pd.Series([gof[t] for t in twins]).value_counts()
    no_direct_anchor = sum(1 for a in anc if not any(t in adj.get(a, ()) for t in twins))
    tw[half] = {"twins": len(twins), "direct": direct, "anchors": len(anc), "anchors_without_direct_twin": no_direct_anchor,
                "mean_twins_per_anchor": float(len(twins) / len(anc)), "share_anchors_one_twin": float((per == 1).sum() / len(anc))}
out["twins"] = tw
# (4) composition
lab_ids = [int(i) for i in ids[lab]]; gofl = dict(zip(ids, grp)); gsize = pd.Series([gofl[i] for i in lab_ids]).value_counts()
comp = {}
for kind in ("random", "group"):
    rows = []
    for seed in (0, 1, 2):
        sp = split_of(kind, seed); test = [i for i, v in sp.items() if v == "test"]; train = [i for i, v in sp.items() if v == "train"]
        sz = np.array([gsize[gofl[i]] for i in test])
        rows.append({"test_singleton_share": float((sz == 1).mean()), "test_from_groups_ge20": float((sz >= 20).mean()), "max_group_size_of_test_images": int(sz.max()),
                     "train_groups_covered": int(len({gofl[i] for i in train}))})
    comp[kind] = rows
out["composition"] = comp; out["labeled_groups_total"] = int(len(gsize))
def overlap(kind):
    T = [set(i for i, v in split_of(kind, s).items() if v == "test") for s in (0, 1, 2)]
    return [len(T[a] & T[b]) / len(T[a]) for a, b in ((0, 1), (0, 2), (1, 2))]
out["test_overlap_group"] = overlap("group"); out["test_overlap_random"] = overlap("random")
# (5) residual leakage estimate
E = np.load(os.path.join(QL, "emb_dinov2s_top_cls.npy")).astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True)
rl = []
for seed in (0, 1, 2):
    sp = split_of("group", seed); te = [pos[i] for i, v in sp.items() if v == "test"]; tr = [pos[i] for i, v in sp.items() if v == "train"]
    ms = (E[te] @ E[tr].T).max(1); rl.append(float((ms >= 0.90).mean()))
cos_lo, cos_hi = au["cos_only"]["share_same_lower"], au["cos_only"]["share_same_upper"]
out["group_test_share_cos90_neighbor"] = rl
out["residual_leak_estimate"] = [float(np.mean(rl) * cos_lo), float(np.mean(rl) * cos_hi)]
rs = []
for seed in (0, 1, 2):
    sp = split_of("random", seed); te = [pos[i] for i, v in sp.items() if v == "test"]; tr = [pos[i] for i, v in sp.items() if v == "train"]
    rs.append(float(((E[te] @ E[tr].T).max(1) >= 0.90).mean()))
out["random_test_share_cos90_neighbor"] = rs
# (6) audit false positives among accepted pairs
pa = pd.read_csv(os.path.join(HERE, "pair_audit.csv")); fp = pa[pa.stratum.str.startswith("ours") & (pa.same_scene == "0")]
two_hop = sum(1 for r in fp.itertuples() if len((adj.get(r.id_a, set()) - {r.id_b}) & (adj.get(r.id_b, set()) - {r.id_a})) > 0)
out["audit_fp_accepted"] = int(len(fp)); out["audit_fp_with_common_neighbor"] = int(two_hop)
# (7) extra baseline groupings for dedup_baselines: average-linkage clustering and full-frame cos>=0.90
Z = np.load(os.path.join(QL, "linkage_top_cls_average.npy"))
extra = {}
for cut in (0.15, 0.18):
    extra[f"avg_linkage_{cut}"] = fcluster(Z, t=cut, criterion="distance")
Ef = np.load(os.path.join(QL, "emb_dinov2s_full_cls.npy")).astype(np.float32); Ef /= np.linalg.norm(Ef, axis=1, keepdims=True)
P = []
for s0 in range(0, N, 1000):
    Sb = Ef[s0:s0 + 1000] @ Ef.T; r = np.arange(s0, min(N, s0 + 1000))[:, None]; ii, jj = np.where((Sb >= 0.90) & (np.arange(N)[None, :] > r)); P.append(np.stack([ii + s0, jj], 1))
P = np.concatenate(P); _, gfull = connected_components(coo_matrix((np.ones(len(P)), (P[:, 0], P[:, 1])), shape=(N, N)), directed=False)
extra["full_cos_0.90"] = gfull
np.savez(os.path.join(HERE, "extra_groupings.npz"), ids=ids, **extra)
json.dump(out, open(os.path.join(HERE, "robustness_numbers.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
