# Leakage-aware (group) split and plain random split of LABELED images, 70/15/15, seed 0
import os, json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = os.path.join(ROOT, "analysis", "01_quality_leakage"); SP = os.path.join(OUT, "splits")
os.makedirs(SP, exist_ok=True)
SEED = 0; FR = np.array([0.70, 0.15, 0.15]); SPL = ["train", "val", "test"]
meta = pd.read_csv(os.path.join(OUT, "meta.csv")); bx = pd.read_csv(os.path.join(OUT, "boxes.csv"))
grp = pd.read_csv(os.path.join(OUT, "near_duplicate_groups.csv"))[["id", "group"]]
nd = pd.read_csv(os.path.join(OUT, "near_duplicate_pairs.csv"))
meta = meta.merge(grp, on="id")
L = meta[meta.labeled].copy()
RES = sorted(L.res.unique())
# per-image feature vector: [1, onehot(res), box count per class]
cc = bx.groupby(["id", "cls"]).size().unstack(fill_value=0).reindex(columns=range(5), fill_value=0)
L = L.join(cc, on="id")
feat = np.column_stack([np.ones(len(L))] + [(L.res == r).values.astype(float) for r in RES] + [L[c].values.astype(float) for c in range(5)])
W = np.array([4.0] + [1.0] * len(RES) + [1.0] * 5)  # image-count term weighted higher
G = pd.DataFrame(feat, index=L.group.values).groupby(level=0).sum()
gids = G.index.values; GF = G.values; tot = GF.sum(0); target = FR[:, None] * tot[None, :]
def cost(S):  # S: 3 x F
    return float((W * ((S - target) / np.maximum(tot, 1)) ** 2).sum())
rng = np.random.default_rng(SEED)
best = None
for rep in range(300):
    size = GF[:, 0] * np.exp(rng.normal(0, 0.3, len(GF)))  # large groups first, randomised tie-breaking
    order = np.argsort(-size)
    S = np.zeros((3, GF.shape[1])); assign = np.empty(len(GF), int)
    for g in order:
        c = []
        for s in range(3):
            S[s] += GF[g]; c.append(cost(S)); S[s] -= GF[g]
        s = int(np.argmin(c)); assign[g] = s; S[s] += GF[g]
    # local improvement: single-group moves
    for it in range(3):
        improved = False
        for g in rng.permutation(len(GF)):
            s0 = assign[g]; cur = cost(S)
            for s in range(3):
                if s == s0: continue
                S[s0] -= GF[g]; S[s] += GF[g]
                if cost(S) < cur - 1e-12: assign[g] = s; improved = True; break
                S[s] -= GF[g]; S[s0] += GF[g]
        if not improved: break
    c = cost(S)
    if best is None or c < best[0]: best = (c, assign.copy(), S.copy())
print("best cost", best[0])
gmap = dict(zip(gids, best[1]))
L["group_split"] = L.group.map(gmap).map(dict(enumerate(SPL)))
# plain random split
perm = np.random.default_rng(SEED).permutation(L.id.values)
n = len(perm); ntr = int(round(0.70 * n)); nva = int(round(0.15 * n))
rs = {i: "train" for i in perm[:ntr]}; rs.update({i: "val" for i in perm[ntr:ntr + nva]}); rs.update({i: "test" for i in perm[ntr + nva:]})
L["random_split"] = L.id.map(rs)
for kind in ["group", "random"]:
    for s in SPL:
        ids_ = np.sort(L[L[f"{kind}_split"] == s].id.values)
        np.savetxt(os.path.join(SP, f"{kind}_{s}_ids.txt"), ids_, fmt="%d")
U = meta[~meta.labeled]
np.savetxt(os.path.join(SP, "unlabeled_ids.txt"), np.sort(U.id.values), fmt="%d")
# unlabeled images that share a near-duplicate group with group-val/test labeled images must not be used for training
g_valtest = set(L[L.group_split != "train"].group)
safe = U[~U.group.isin(g_valtest)].id.values; unsafe = U[U.group.isin(g_valtest)].id.values
np.savetxt(os.path.join(SP, "unlabeled_safe_for_group_train_ids.txt"), np.sort(safe), fmt="%d")
L[["id", "file", "res", "group", "group_split", "random_split", "n_boxes"]].to_csv(os.path.join(SP, "split_assignments.csv"), index=False)
# ---------- balance report ----------
rep = {}
for kind in ["group", "random"]:
    d = {}
    for s in SPL:
        sub = L[L[f"{kind}_split"] == s]
        d[s] = {"n_images": int(len(sub)), "frac_images": round(len(sub) / len(L), 4),
                "res_share": {int(r): round(float((sub.res == r).mean()), 4) for r in RES},
                "boxes": {int(c): int(sub[c].sum()) for c in range(5)},
                "box_class_share": {int(c): round(float(sub[c].sum() / sub[list(range(5))].values.sum()), 4) for c in range(5)},
                "n_groups": int(sub.group.nunique())}
    rep[kind] = d
# ---------- leakage ----------
lab_set = set(L.id)
E = nd[nd.id_a.isin(lab_set) & nd.id_b.isin(lab_set)]
ids_all = np.load(os.path.join(OUT, "emb_ids.npy"))
Et = np.load(os.path.join(OUT, "emb_dinov2s_top_cls.npy")); Et /= np.linalg.norm(Et, axis=1, keepdims=True)
pos = {i: k for k, i in enumerate(ids_all)}
def leak(kind):
    sp = dict(zip(L.id, L[f"{kind}_split"]))
    out = {}
    tr = [i for i in L.id if sp[i] == "train"]; Ttr = Et[[pos[i] for i in tr]]
    for s in ["val", "test"]:
        ev = [i for i in L.id if sp[i] == s]
        evs = set(ev)
        # near-duplicate partner (verified edge) in train
        has = set()
        for a, b in zip(E.id_a.values, E.id_b.values):
            if a in evs and sp.get(b) == "train": has.add(a)
            if b in evs and sp.get(a) == "train": has.add(b)
        # soft proxy: max DINOv2 upper-60% cosine similarity to any train image
        ms = (Et[[pos[i] for i in ev]] @ Ttr.T).max(1)
        out[s] = {"n": len(ev), "n_with_nd_in_train": len(has), "leak_rate_nd": round(len(has) / len(ev), 4),
                  "frac_maxsim_to_train_ge_0.90": round(float((ms >= 0.90).mean()), 4),
                  "frac_maxsim_to_train_ge_0.85": round(float((ms >= 0.85).mean()), 4),
                  "median_maxsim_to_train": round(float(np.median(ms)), 4)}
        out[s]["_leaky_ids"] = sorted(has)
    return out
lk = {k: leak(k) for k in ["group", "random"]}
test_ids = np.sort(L[L.random_split == "test"].id.values)
leaky = np.array(lk["random"]["test"]["_leaky_ids"], int)
np.savetxt(os.path.join(SP, "random_test_leaky_ids.txt"), np.sort(leaky), fmt="%d")
np.savetxt(os.path.join(SP, "random_test_clean_ids.txt"), np.setdiff1d(test_ids, leaky), fmt="%d")
for k in lk:
    for s in lk[k]: lk[k][s].pop("_leaky_ids")
summary = {"balance": rep, "leakage": lk, "n_labeled": int(len(L)), "n_unlabeled": int(len(U)),
           "n_unlabeled_safe_for_group_train": int(len(safe)), "n_unlabeled_sharing_group_with_group_valtest": int(len(unsafe)),
           "random_test_leaky": int(len(leaky)), "random_test_clean": int(len(test_ids) - len(leaky)),
           "group_split_cost": best[0], "n_groups_labeled": int(L.group.nunique()), "max_group_labeled": int(L.group.value_counts().max())}
json.dump(summary, open(os.path.join(OUT, "split_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
