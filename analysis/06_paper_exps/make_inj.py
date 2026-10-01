# E3 (crossover twin injection) on the seed-0 group split.
# Anchors = one image per multi-image near-duplicate group in group val+test (392). Anchors are split into halves A/B
# (balanced by source resolution, box count and number of twins). Model "twinA" gets the group-mates (twins, <=5) of
# A-anchors added to train, model "twinB" those of B-anchors; both drop the same random train images, so train size and
# the kind of added data are identical. Each anchor is thus scored once as leaked and once as unleaked.
import os, json, shutil, numpy as np, pandas as pd
from pathlib import Path
import make_datasets as md  # reuses build(), metadata (E1 datasets already exist and are skipped by os.path.exists)

OUT = md.OUT; L = md.L; fmeta = md.fmeta
old = pd.read_csv(os.path.join(md.QL, "splits", "split_assignments.csv")).set_index("id")
g0 = old["group_split"].to_dict(); res = fmeta["res"].to_dict(); gof = dict(zip(L.id, L.group)); nbox = dict(zip(L.id, L.n_boxes))
H = [i for i in L.id if g0[i] in ("val", "test")]; TR = [i for i in L.id if g0[i] == "train"]
Hg = pd.Series({i: gof[i] for i in H}); sizes = Hg.value_counts(); multi = sorted(sizes[sizes >= 2].index)
rng = np.random.default_rng(0)
anc, tw = [], {}
for g in multi:
    mem = sorted(Hg.index[Hg == g]); a = int(rng.choice(mem)); rest = [m for m in mem if m != a]; rng.shuffle(rest)
    anc.append(a); tw[a] = rest[:5]
anc = np.array(anc); rcls = np.array([res[a] if res[a] in (960, 640) else 0 for a in anc])
best = None
for it in range(5000):
    side = np.zeros(len(anc), bool)
    for r in np.unique(rcls):
        idx = np.where(rcls == r)[0]; rng.shuffle(idx); side[idx[: len(idx) // 2 + (it % 2 if len(idx) % 2 else 0)]] = True
    tA = sum(len(tw[a]) for a in anc[side]); tB = sum(len(tw[a]) for a in anc[~side])
    bA = sum(nbox[a] for a in anc[side]); bB = sum(nbox[a] for a in anc[~side])
    c = abs(tA - tB) * 10 + abs(bA - bB) + abs(side.sum() - (~side).sum()) * 5
    if best is None or c < best[0]:
        best = (c, side.copy())
side = best[1]; A = sorted(anc[side].tolist()); B = sorted(anc[~side].tolist())
TA = [t for a in A for t in tw[a]]; TB = [t for a in B for t in tw[a]]
k = min(len(TA), len(TB)); rng.shuffle(TA); rng.shuffle(TB); TA = sorted(TA[:k]); TB = sorted(TB[:k])
removed = sorted(int(x) for x in rng.choice(TR, size=k, replace=False)); keep = sorted(set(TR) - set(removed))
placebo = sorted(set(H) - set(anc.tolist()) - set(t for a in anc for t in tw[a]))
rep = {}
for nm in ("inj_twin", "inj_ctrl"):  # superseded single-control design: remove its hardlinked copies (originals untouched)
    p = os.path.join(OUT, "ds", nm)
    if os.path.isdir(p):
        if Path(p).resolve().parent != (Path(OUT) / "ds").resolve():
            raise RuntimeError(f"Refusing to remove a dataset outside the experiment directory: {p}")
        shutil.rmtree(p)
evalset = sorted(anc.tolist()) + placebo
for arm, tr in (("twinA", keep + TA), ("twinB", keep + TB)):
    rep[arm] = {"yaml": md.build(f"inj_{arm}", {"train": sorted(tr), "test": sorted(evalset)}), "n_train": len(tr)}
for nm, ids in (("anchorsA", A), ("anchorsB", B), ("twinsA", TA), ("twinsB", TB), ("removed", removed), ("placebo", placebo)):
    np.savetxt(os.path.join(OUT, f"inj_{nm}_ids.txt"), ids, fmt="%d")
for stale in ("inj_twins_ids.txt", "inj_ctrl_ids.txt"):
    if os.path.exists(os.path.join(OUT, stale)):
        os.remove(os.path.join(OUT, stale))
rep["counts"] = {"anchors": len(anc), "A": len(A), "B": len(B), "twins_per_arm": k, "placebo": len(placebo),
                 "A_res": pd.Series([res[a] for a in A]).value_counts().to_dict(), "B_res": pd.Series([res[a] for a in B]).value_counts().to_dict(),
                 "A_boxes": int(sum(nbox[a] for a in A)), "B_boxes": int(sum(nbox[a] for a in B)),
                 "twinsA_res": pd.Series([res[t] for t in TA]).value_counts().to_dict(), "twinsB_res": pd.Series([res[t] for t in TB]).value_counts().to_dict()}
rep["counts"] = json.loads(json.dumps(rep["counts"], default=int))
d = json.load(open(os.path.join(OUT, "datasets_report.json")))
d["E3"] = {"design": "crossover", **rep, "base": d["E3"].get("base")}
json.dump(d, open(os.path.join(OUT, "datasets_report.json"), "w"), indent=1)
print(json.dumps(d["E3"], indent=1, default=int))
