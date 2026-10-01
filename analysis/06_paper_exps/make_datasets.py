# Datasets for the paper experiments.
# E1: extra split draws (seeds 1,2) of the group-aware and random 70/15/15 splits (same algorithm as 01_quality_leakage/12_splits.py).
# E3: controlled twin injection on the seed-0 group split: anchors = one image per multi-image near-duplicate group in val+test;
#     "twin" arm moves up to 5 group-mates of each anchor into train, "ctrl" arm moves the same number of held-out singletons
#     (matched by source resolution); both arms drop the same random train images so train size is constant.
import os, json, shutil, yaml, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMG = os.path.join(ROOT, "RDDC 2024_image"); LAB = os.path.join(ROOT, "RDDC 2024_label")
QL = os.path.join(ROOT, "analysis", "01_quality_leakage"); OUT = os.path.join(ROOT, "analysis", "06_paper_exps")
NAMES = {0: "longitudinal", 1: "lateral", 2: "alligator", 3: "pothole", 4: "others"}
FR = np.array([0.70, 0.15, 0.15]); SPL = ["train", "val", "test"]

meta = pd.read_csv(os.path.join(QL, "meta.csv")); bx = pd.read_csv(os.path.join(QL, "boxes.csv"))
grp = pd.read_csv(os.path.join(QL, "near_duplicate_groups.csv"))[["id", "group"]]
nd = pd.read_csv(os.path.join(QL, "near_duplicate_pairs.csv"))
meta = meta.merge(grp, on="id"); L = meta[meta.labeled].copy(); RES = sorted(L.res.unique())
cc = bx.groupby(["id", "cls"]).size().unstack(fill_value=0).reindex(columns=range(5), fill_value=0); L = L.join(cc, on="id")
feat = np.column_stack([np.ones(len(L))] + [(L.res == r).values.astype(float) for r in RES] + [L[c].values.astype(float) for c in range(5)])
W = np.array([4.0] + [1.0] * len(RES) + [1.0] * 5)
G = pd.DataFrame(feat, index=L.group.values).groupby(level=0).sum()
gids = G.index.values; GF = G.values; tot = GF.sum(0); target = FR[:, None] * tot[None, :]


def cost(S):
    return float((W * ((S - target) / np.maximum(tot, 1)) ** 2).sum())


def group_split(seed):
    rng = np.random.default_rng(seed); best = None
    for rep in range(300):
        size = GF[:, 0] * np.exp(rng.normal(0, 0.3, len(GF))); order = np.argsort(-size)
        S = np.zeros((3, GF.shape[1])); assign = np.empty(len(GF), int)
        for g in order:
            c = []
            for s in range(3):
                S[s] += GF[g]; c.append(cost(S)); S[s] -= GF[g]
            s = int(np.argmin(c)); assign[g] = s; S[s] += GF[g]
        for it in range(3):
            improved = False
            for g in rng.permutation(len(GF)):
                s0 = assign[g]; cur = cost(S)
                for s in range(3):
                    if s == s0:
                        continue
                    S[s0] -= GF[g]; S[s] += GF[g]
                    if cost(S) < cur - 1e-12:
                        assign[g] = s; improved = True; break
                    S[s] -= GF[g]; S[s0] += GF[g]
            if not improved:
                break
        c = cost(S)
        if best is None or c < best[0]:
            best = (c, assign.copy())
    gmap = dict(zip(gids, best[1]))
    return L.group.map(gmap).map(dict(enumerate(SPL))).values


def random_split(seed):
    perm = np.random.default_rng(seed).permutation(L.id.values)
    n = len(perm); ntr = int(round(0.70 * n)); nva = int(round(0.15 * n))
    rs = {i: "train" for i in perm[:ntr]}
    rs.update({i: "val" for i in perm[ntr:ntr + nva]}); rs.update({i: "test" for i in perm[ntr + nva:]})
    return L.id.map(rs).values


lab = set(L.id); E = nd[nd.id_a.isin(lab) & nd.id_b.isin(lab)]


def leak_rate(sp):
    out = {}
    for s in ["val", "test"]:
        ev = {i for i, v in sp.items() if v == s}; has = set()
        for a, b in zip(E.id_a.values, E.id_b.values):
            if a in ev and sp[b] == "train":
                has.add(a)
            if b in ev and sp[a] == "train":
                has.add(b)
        out[s] = round(len(has) / len(ev), 4)
    return out


fmeta = meta.set_index("id")


def build(name, parts):
    """parts: dict split -> list of ids. Link or copy images, copy labels, write data.yaml."""
    ds = os.path.join(OUT, "ds", name)
    for s, ids in parts.items():
        di = os.path.join(ds, "images", s); dl = os.path.join(ds, "labels", s)
        os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
        for i in ids:
            f = fmeta.loc[i, "file"]; dst = os.path.join(di, f)
            if not os.path.exists(dst):
                src = os.path.join(IMG, f)
                try:
                    os.link(src, dst)
                except OSError:
                    # Dataset mounts and the output folder can be on different filesystems.
                    shutil.copy2(src, dst)
            shutil.copyfile(os.path.join(LAB, f"{i}.txt"), os.path.join(dl, f"{i}.txt"))
        images = [f for f in os.listdir(di) if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}]
        labels = [f for f in os.listdir(dl) if Path(f).suffix.lower() == ".txt"]
        assert len(images) == len(ids) == len(labels), (name, s)
    dsf = ds.replace(os.sep, "/"); y = {"path": dsf, "names": NAMES}
    for s in parts:
        y[s] = dsf + f"/images/{s}"
    if "val" not in parts:
        y["val"] = y["test"]
    with open(os.path.join(ds, "data.yaml"), "w") as fh:
        yaml.safe_dump(y, fh, sort_keys=False)
    return os.path.join(ds, "data.yaml")



def main():
    global report
    report = {"E1": {}, "E3": {}}
    free0 = shutil.disk_usage(ROOT).free
    # ---------------- E1 ----------------
    old = pd.read_csv(os.path.join(QL, "splits", "split_assignments.csv")).set_index("id")
    for seed in (1, 2):
        for kind, fn in (("group", group_split), ("random", random_split)):
            sp = dict(zip(L.id, fn(seed)))
            parts = {s: sorted(i for i, v in sp.items() if v == s) for s in SPL}
            y = build(f"{kind}_s{seed}", parts)
            t0 = set(old.index[old[f"{kind}_split"] == "test"])
            report["E1"][f"{kind}_s{seed}"] = {
                "yaml": y, "n": {s: len(v) for s, v in parts.items()}, "leak_rate": leak_rate(sp),
                "test_overlap_with_seed0": round(len(t0 & set(parts["test"])) / len(parts["test"]), 4),
                "res_share_test": {int(r): round(float(np.mean([fmeta.loc[i, "res"] == r for i in parts["test"]])), 4) for r in RES}}
            np.savetxt(os.path.join(OUT, f"{kind}_s{seed}_test_ids.txt"), parts["test"], fmt="%d")
    # ---------------- E3 ----------------
    rng = np.random.default_rng(0)
    g0 = old["group_split"].to_dict(); res = fmeta["res"].to_dict(); gof = dict(zip(L.id, L.group))
    H = [i for i in L.id if g0[i] in ("val", "test")]; TR = [i for i in L.id if g0[i] == "train"]
    Hg = pd.Series({i: gof[i] for i in H}); sizes = Hg.value_counts()
    multi = sizes[sizes >= 2].index; single_ids = [i for i in H if sizes[gof[i]] == 1]
    anchors, twins = [], []
    for g in sorted(multi):
        mem = sorted(Hg.index[Hg == g]); a = int(rng.choice(mem)); rest = [m for m in mem if m != a]
        rng.shuffle(rest); anchors.append(a); twins += rest[:5]
    tw_res = pd.Series([res[i] for i in twins]).value_counts(); ctrl = []; pool = pd.Series({i: res[i] for i in single_ids})
    short = 0
    for r, k in tw_res.items():
        cand = list(pool.index[pool == r]); rng.shuffle(cand); take = cand[:k]; ctrl += take; short += k - len(take)
    if short:
        rest = [i for i in single_ids if i not in set(ctrl)]; rng.shuffle(rest); ctrl += rest[:short]
    assert len(ctrl) == len(twins)
    removed = [int(x) for x in rng.choice(TR, size=len(twins), replace=False)]
    keep = sorted(set(TR) - set(removed))
    placebo = sorted(set(H) - set(anchors) - set(twins) - set(ctrl))
    arms = {"base": sorted(TR), "twin": sorted(keep + twins), "ctrl": sorted(keep + ctrl)}
    for arm, tr in arms.items():
        report["E3"][arm] = {"yaml": build(f"inj_{arm}", {"train": tr, "test": sorted(anchors)}), "n_train": len(tr)}
    for nm, ids in (("anchors", anchors), ("twins", twins), ("ctrl", ctrl), ("removed", removed), ("placebo", placebo)):
        np.savetxt(os.path.join(OUT, f"inj_{nm}_ids.txt"), sorted(ids), fmt="%d")
    Li = L.set_index("id")
    report["E3"]["counts"] = {
        "heldout": len(H), "multi_groups": int(len(multi)), "anchors": len(anchors), "twins": len(twins), "ctrl": len(ctrl),
        "ctrl_res_unmatched": int(short), "placebo": len(placebo),
        "anchor_res": {int(k): int(v) for k, v in pd.Series([res[i] for i in anchors]).value_counts().items()},
        "twin_res": {int(k): int(v) for k, v in tw_res.items()},
        "anchor_boxes": int(Li.loc[anchors, "n_boxes"].sum())}
    report["disk_used_MB"] = round((free0 - shutil.disk_usage(ROOT).free) / 1e6, 2)
    json.dump(report, open(os.path.join(OUT, "datasets_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
