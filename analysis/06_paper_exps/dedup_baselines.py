# Compare grouping rules used before splitting: pHash Hamming thresholds and DINOv2 cosine connected components
# versus the geometry-verified grouping. For each rule we build the same stratified 70/15/15 group split (3 seeds) and
# measure residual leakage = share of labeled test images that still have a geometry-verified near-duplicate in train.
import os, json, numpy as np, pandas as pd
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]; A = os.path.join(ROOT, "analysis"); QL = os.path.join(A, "01_quality_leakage")
OUT = os.path.join(A, "06_paper_exps")
meta = pd.read_csv(os.path.join(QL, "meta.csv")); bx = pd.read_csv(os.path.join(QL, "boxes.csv"))
ids = np.load(os.path.join(QL, "emb_ids.npy")); N = len(ids); pos = {int(i): k for k, i in enumerate(ids)}
ours = pd.read_csv(os.path.join(QL, "near_duplicate_groups.csv")).set_index("id").loc[ids, "group"].values
nd = pd.read_csv(os.path.join(QL, "near_duplicate_pairs.csv"))
# pHash (64-bit DCT) computed in analysis/phash_groups.py, reordered to emb_ids order. Low-memory: blockwise pair search.
pb = np.load(os.path.join(A, "phash_bits.npy")); pid = np.load(os.path.join(A, "phash_ids.npy"))
pb = pb[np.argsort(pid)][np.searchsorted(np.sort(pid), ids)].astype(np.float32)
E = np.load(os.path.join(QL, "emb_dinov2s_top_cls.npy")).astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True)
PH = {6: [], 8: [], 10: []}; CS = {0.95: [], 0.90: []}
for s0 in range(0, N, 1000):
    blk = slice(s0, min(N, s0 + 1000))
    Dh = 64 - (pb[blk] @ pb.T + (1 - pb[blk]) @ (1 - pb).T)
    Sb = E[blk] @ E.T
    r = np.arange(blk.start, blk.stop)[:, None]; c = np.arange(N)[None, :]; upper = c > r
    for t in PH:
        ii, jj = np.where((Dh <= t) & upper); PH[t].append(np.stack([ii + s0, jj], 1))
    for t in CS:
        ii, jj = np.where((Sb >= t) & upper); CS[t].append(np.stack([ii + s0, jj], 1))
    del Dh, Sb, upper


def cc_pairs(P):
    P = np.concatenate(P) if len(P) else np.zeros((0, 2), int)
    n, g = connected_components(coo_matrix((np.ones(len(P)), (P[:, 0], P[:, 1])), shape=(N, N)), directed=False)
    return g, len(P)


rules = {"geometry-verified (ours)": (ours, len(nd))}
for t in PH:
    rules[f"pHash <= {t}"] = cc_pairs(PH[t])
for t in CS:
    rules[f"DINOv2 cos >= {t:.2f}"] = cc_pairs(CS[t])
X = np.load(os.path.join(OUT, "extra_groupings.npz"))
assert (X["ids"] == ids).all()
rules["DINOv2 full-frame cos >= 0.90"] = (X["full_cos_0.90"], -1)
rules["DINOv2 average linkage, cut 0.15"] = (X["avg_linkage_0.15"], -1)
rules["DINOv2 average linkage, cut 0.18"] = (X["avg_linkage_0.18"], -1)
rules["none (random split)"] = (np.arange(N), 0)


def ham(a, b):
    return (pb[a] != pb[b]).sum(1)


lab = meta.set_index("id").loc[ids, "labeled"].values
res = meta.set_index("id").loc[ids, "res"].values
cnt = bx.groupby(["id", "cls"]).size().unstack(fill_value=0).reindex(index=ids, columns=range(5), fill_value=0).values
RES = sorted(set(res[lab]))
F = np.column_stack([np.ones(N)] + [(res == r).astype(float) for r in RES] + [cnt[:, c] for c in range(5)])[lab]
W = np.array([4.0] + [1.0] * len(RES) + [1.0] * 5); FR = np.array([0.7, 0.15, 0.15])
lab_idx = np.where(lab)[0]
# verified edges among labeled images (reference for leakage)
ea = np.array([pos[i] for i in nd.id_a]); eb = np.array([pos[i] for i in nd.id_b])
strong_all = ((nd.n_inl >= 30) | (nd.sim_top >= 0.96)).values
keep = lab[ea] & lab[eb]; ea, eb, st = ea[keep], eb[keep], strong_all[keep]


def split(groups, seed, reps=60):
    g = groups[lab]; u, inv = np.unique(g, return_inverse=True)
    GF = np.zeros((len(u), F.shape[1])); np.add.at(GF, inv, F)
    tot = GF.sum(0); target = FR[:, None] * tot[None, :]
    cost = lambda S_: float((W * ((S_ - target) / np.maximum(tot, 1)) ** 2).sum())
    rng = np.random.default_rng(seed); best = None
    for rep in range(reps):
        order = np.argsort(-GF[:, 0] * np.exp(rng.normal(0, 0.3, len(GF))))
        S_ = np.zeros((3, GF.shape[1])); asg = np.empty(len(GF), int)
        for k in order:
            c = []
            for s in range(3):
                S_[s] += GF[k]; c.append(cost(S_)); S_[s] -= GF[k]
            s = int(np.argmin(c)); asg[k] = s; S_[s] += GF[k]
        c = cost(S_)
        if best is None or c < best[0]:
            best = (c, asg.copy(), S_.copy())
    sp = np.full(N, -1); sp[lab_idx] = best[1][inv]
    frac = best[2][:, 0] / tot[0]
    return sp, frac


rows = []
for name, (g, npairs) in rules.items():
    gl = g[lab]; _, c = np.unique(gl, return_counts=True)
    leak, fr, leak_s = [], [], []
    for seed in (0, 1, 2):
        if name.startswith("none"):
            perm = np.random.default_rng(seed).permutation(lab_idx); sp = np.full(N, -1)
            n = len(perm); sp[perm[:int(round(.7 * n))]] = 0; sp[perm[int(round(.7 * n)):int(round(.85 * n))]] = 1; sp[perm[int(round(.85 * n)):]] = 2
            frac = np.array([.7, .15, .15])
        else:
            sp, frac = split(g, seed)
        test = sp == 2
        hit = np.zeros(N, bool)
        m1 = test[ea] & (sp[eb] == 0); hit[ea[m1]] = True
        m2 = test[eb] & (sp[ea] == 0); hit[eb[m2]] = True
        leak.append(hit[test].mean()); fr.append(frac.tolist())
        hs = np.zeros(N, bool); m1s = m1 & st; m2s = m2 & st; hs[ea[m1s]] = True; hs[eb[m2s]] = True; leak_s.append(hs[test].mean())
    rows.append({"rule": name, "pairs": int(npairs) if npairs >= 0 else None, "groups_labeled": int(len(c)), "largest_group": int(c.max()),
                 "share_labeled_in_multi": round(float((c[c > 1]).sum() / c.sum()), 4),
                 "test_leak_mean": round(float(np.mean(leak)), 4), "test_leak_min": round(float(np.min(leak)), 4), "test_leak_max": round(float(np.max(leak)), 4), "test_leak_strong_mean": round(float(np.mean(leak_s)), 4),
                 "achieved_image_fractions_seed0": [round(x, 3) for x in fr[0]]})
    print(rows[-1], flush=True)
# agreement of pHash / DINOv2 pairs with geometric verification (among candidate pairs that were verified)
cand = pd.read_csv(os.path.join(QL, "candidate_pairs_verified.csv"))
ca = np.array([pos[i] for i in cand.id_a]); cb = np.array([pos[i] for i in cand.id_b])
cand["ham"] = ham(ca, cb); acc = set(zip(nd.id_a, nd.id_b))
cand["ours"] = [(a, b) in acc for a, b in zip(cand.id_a, cand.id_b)]
agree = {}
for t in (6, 8, 10):
    m = cand.ham <= t; agree[f"pHash<={t}"] = {"candidate_pairs": int(m.sum()), "share_verified": round(float(cand.ours[m].mean()), 4)}
for t in (0.95, 0.90):
    m = cand.sim_top >= t; agree[f"cos>={t}"] = {"candidate_pairs": int(m.sum()), "share_verified": round(float(cand.ours[m].mean()), 4)}
# how many verified pairs would each baseline find (recall w.r.t. verified pairs)
na = np.array([pos[i] for i in nd.id_a]); nb = np.array([pos[i] for i in nd.id_b])
for t in (6, 8, 10):
    agree[f"pHash<={t}"]["recall_of_verified_pairs"] = round(float((ham(na, nb) <= t).mean()), 4)
for t in (0.95, 0.90):
    agree[f"cos>={t}"]["recall_of_verified_pairs"] = round(float(((E[na] * E[nb]).sum(1) >= t).mean()), 4)
cand.to_csv(os.path.join(OUT, "candidate_pairs_with_phash.csv"), index=False)
json.dump({"rules": rows, "pair_agreement": agree}, open(os.path.join(OUT, "dedup_baselines.json"), "w"), indent=1)
print(json.dumps(agree, indent=1))
