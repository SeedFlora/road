"""Evaluate all paper models.
E1: random vs group split, 3 split draws (seed 0 = pilot models, seeds 1-2 = this folder). Official ultralytics test metrics
    (best.pt, conf 0.001, IoU 0.7, 640 px) + own AP50 per source from cached predictions.
E3: crossover twin injection (last.pt). Each anchor is scored once by the model that saw its twins ("leaked") and once by
    the model that did not ("unleaked"); paired image-level bootstrap over anchors. Held-out singletons act as placebo.
Usage: python eval_all.py [--device cpu|0]"""
import argparse, os, sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent; A = ROOT / "analysis"; QL = A / "01_quality_leakage"; PILOT = A / "04_pilot"
sys.path.insert(0, str(PILOT))
B = 2000; RS = np.linspace(0, 1, 101); NAMES = ["longitudinal", "lateral", "alligator", "pothole", "others"]
DEVICE = os.environ.get("ROAD_DEVICE", "0")


def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    return inter / ((a[:, 2:] - a[:, :2]).prod(1)[:, None] + (b[:, 2:] - b[:, :2]).prod(1)[None] - inter + 1e-12)


def gt_of(i):
    p = ROOT / "RDDC 2024_label" / f"{i}.txt"
    rows = [list(map(float, l.split())) for l in p.read_text().splitlines() if len(l.split()) == 5]
    g = np.array(rows).reshape(-1, 5)
    if len(g):
        c, x, y, w, h = g.T; g = np.stack([c, x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
    return g


def predict(weights, ids, img_dir):
    from ultralytics import YOLO
    m = YOLO(str(weights)); dets = {}
    paths = [str(Path(img_dir) / meta.loc[i, "file"]) for i in ids]
    for k in range(0, len(paths), 64):
        for i, r in zip(ids[k:k + 64], m.predict(paths[k:k + 64], imgsz=640, conf=0.001, iou=0.7, max_det=300, verbose=False, batch=16, device=DEVICE)):
            bx = r.boxes
            dets[i] = np.concatenate([bx.xyxyn.cpu().numpy(), bx.conf.cpu().numpy()[:, None], bx.cls.cpu().numpy()[:, None]], 1) if len(bx) else np.zeros((0, 6))
    return dets


def tp_table(ids, dets):
    """per class: conf, tp, image index; npos per image and class (greedy matching at IoU 0.5)."""
    per = {c: {"conf": [], "tp": [], "img": []} for c in range(5)}; npos = np.zeros((len(ids), 5))
    for k, i in enumerate(ids):
        g = gt_of(i); d = dets[i]
        for c in range(5):
            gc = g[g[:, 0] == c][:, 1:5] if len(g) else np.zeros((0, 4)); dc = d[d[:, 5] == c] if len(d) else np.zeros((0, 6))
            npos[k, c] = len(gc)
            if not len(dc):
                continue
            dc = dc[np.argsort(-dc[:, 4])]; mt = np.zeros(len(gc), bool); tp = np.zeros(len(dc))
            I = iou(dc[:, :4], gc) if len(gc) else None
            for j in range(len(dc)):
                if I is None:
                    break
                cand = np.where(~mt & (I[j] >= 0.5))[0]
                if len(cand):
                    q = cand[np.argmax(I[j, cand])]; mt[q] = True; tp[j] = 1
            per[c]["conf"].append(dc[:, 4]); per[c]["tp"].append(tp); per[c]["img"].append(np.full(len(dc), k))
    for c in range(5):
        cf = np.concatenate(per[c]["conf"]) if per[c]["conf"] else np.zeros(0); o = np.argsort(-cf)
        per[c] = {kk: (np.concatenate(v)[o] if v else np.zeros(0)) for kk, v in per[c].items()}; per[c]["img"] = per[c]["img"].astype(int)
    return per, npos


def ap_per_class(per, npos, w):
    out = []
    for c in range(5):
        P = w @ npos[:, c]
        if P <= 0:
            out.append(np.nan); continue
        dw = w[per[c]["img"]]; tp = per[c]["tp"]; ctp = np.cumsum(dw * tp); cfp = np.cumsum(dw * (1 - tp))
        rec = ctp / P; den = ctp + cfp; prec = np.where(den > 0, ctp / np.maximum(den, 1e-12), 0)
        prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
        idx = np.searchsorted(rec, RS, side="left")
        q = np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0) if len(prec) else np.zeros(101)
        out.append(q.mean())
    return np.array(out)


def official(weights, data, split="test", source_list=None):
    from ultralytics import YOLO
    m = YOLO(str(weights))
    r = m.val(data=str(data), split=split, imgsz=640, batch=16, conf=0.001, iou=0.7, workers=4, plots=False, verbose=False, device=DEVICE)
    return {"map50": float(r.box.map50), "map50_95": float(r.box.map), "precision": float(r.box.mp), "recall": float(r.box.mr),
            "ap50_per_class": [float(x) for x in r.box.ap50], "ap50_95_per_class": [float(x) for x in r.box.ap]}


def ids_of(p):
    return [int(x) for x in Path(p).read_text().split()]


def main():
    global DEVICE, meta, GROUP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEVICE, help="Ultralytics device, e.g. 0 or cpu (default: ROAD_DEVICE or 0).")
    args = parser.parse_args()
    DEVICE = args.device
    res = {"E1": {}, "E3": {}}
    # ---------------- E1 ----------------
    draws = {("group", 0): (PILOT / "runs/group_y11n_640_e40/weights/best.pt", A / "yolo_ds_group/data.yaml", QL / "splits/group_test_ids.txt", A / "yolo_ds_group/images/test"),
             ("random", 0): (PILOT / "runs/random_y11n_640_e40/weights/best.pt", A / "yolo_ds_random/data.yaml", QL / "splits/random_test_ids.txt", A / "yolo_ds_random/images/test")}
    for kind in ("group", "random"):
        for s in (1, 2):
            draws[(kind, s)] = (HERE / f"runs/{kind}_s{s}_y11n/weights/best.pt", HERE / f"ds/{kind}_s{s}/data.yaml", HERE / f"{kind}_s{s}_test_ids.txt", HERE / f"ds/{kind}_s{s}/images/test")
    models = {arm: HERE / f"runs/inj_{arm}_y11n/weights/last.pt" for arm in ("base", "twinA", "twinB")}
    missing = [str(w) for w in [*(v[0] for v in draws.values()), *models.values()] if not w.is_file()]
    if missing:
        raise FileNotFoundError("Full paper evaluation requires all 9 checkpoints. Train both pilot splits and run train_queue.py first:\n"
                                + "\n".join(missing))
    required = [QL / "meta.csv", QL / "near_duplicate_groups.csv", QL / "splits/split_assignments.csv",
                HERE / "inj_anchorsA_ids.txt", HERE / "inj_anchorsB_ids.txt"]
    required.extend(p for _, data, idf, imgdir in draws.values() for p in (data, idf, imgdir))
    required.append(HERE / "ds/inj_twinA/images/test")
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing evaluation inputs; prepare the analysis and YOLO datasets first:\n" + "\n".join(missing))
    meta = pd.read_csv(QL / "meta.csv").set_index("id")
    GROUP = pd.read_csv(QL / "near_duplicate_groups.csv").set_index("id")["group"].to_dict()
    # pickle is used only for prediction caches this script writes itself into eval_cache/ (never external files)
    cache = HERE / "eval_cache"; cache.mkdir(exist_ok=True)
    for (kind, s), (w, data, idf, imgdir) in draws.items():
        key = f"{kind}_s{s}"; o = official(w, data)
        ids = ids_of(idf); pk = cache / f"{key}.pkl"
        dets = pickle.load(open(pk, "rb")) if pk.exists() else predict(w, ids, imgdir)
        pickle.dump(dets, open(pk, "wb"))
        per, npos = tp_table(ids, dets); resv = np.array([meta.loc[i, "res"] for i in ids])
        o["own_map50"] = float(np.nanmean(ap_per_class(per, npos, np.ones(len(ids)))))
        o["own_map50_by_source"] = {int(r): float(np.nanmean(ap_per_class(per, npos, (resv == r).astype(float)))) for r in (960, 640, 1080, 1600)}
        gid = np.array([GROUP[i] for i in ids]); ug, ginv = np.unique(gid, return_inverse=True); rng_g = np.random.default_rng(0); bs = []
        for b in range(1000):
            wg = np.bincount(rng_g.integers(0, len(ug), len(ug)), minlength=len(ug)).astype(float); bs.append(np.nanmean(ap_per_class(per, npos, wg[ginv])))
        o["own_map50_group_boot_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        res["E1"][key] = o; print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if not isinstance(v, (list, dict))}, flush=True)
    E1 = res["E1"]; summ = {}
    for kind in ("group", "random"):
        ks = [k for k in E1 if k.startswith(kind)]
        for met in ("map50", "map50_95"):
            v = np.array([E1[k][met] for k in ks]); summ[f"{kind}_{met}"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "n": len(v), "values": v.tolist()}
        summ[f"{kind}_ap50_per_class_mean"] = np.mean([E1[k]["ap50_per_class"] for k in ks], 0).tolist()
    for met in ("map50", "map50_95"):
        g = np.array(summ[f"group_{met}"]["values"]); r = np.array(summ[f"random_{met}"]["values"])
        if len(g) == len(r):
            d = r - g; summ[f"inflation_{met}"] = {"mean_abs": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else None,
                                                  "mean_rel_pct": float((d / g).mean() * 100), "per_draw": d.tolist()}
    res["E1_summary"] = summ
    # ---------------- E3 ----------------
    A_ids = ids_of(HERE / "inj_anchorsA_ids.txt"); B_ids = ids_of(HERE / "inj_anchorsB_ids.txt")
    split0 = pd.read_csv(QL / "splits/split_assignments.csv").set_index("id")
    H = split0.index[split0.group_split.isin(["val", "test"])]
    gs = split0.loc[H, "group"].value_counts(); plac = sorted(int(i) for i in H if gs[split0.loc[i, "group"]] == 1)
    (HERE / "inj_placebo_ids.txt").write_text("\n".join(map(str, plac)))  # the 866 held-out labeled singletons used as placebo
    img_dir = HERE / "ds/inj_twinA/images/test"
    allids = sorted(A_ids + B_ids)
    D = {}
    for arm, w in models.items():
        pk = cache / f"inj_{arm}.pkl"
        D[arm] = pickle.load(open(pk, "rb")) if pk.exists() else {**predict(w, allids, img_dir), **predict(w, plac, HERE / "ds/inj_twinA/images/test")}
        pickle.dump(D[arm], open(pk, "wb"))
    if len(D) == 3:
        rng = np.random.default_rng(0); Aset = set(A_ids)
        leaked = {i: (D["twinA"][i] if i in Aset else D["twinB"][i]) for i in allids}
        unleak = {i: (D["twinB"][i] if i in Aset else D["twinA"][i]) for i in allids}
        base = {i: D["base"][i] for i in allids}
        T = {nm: tp_table(allids, d) for nm, d in (("leaked", leaked), ("unleaked", unleak), ("base", base))}
        TP = {nm: tp_table(plac, {i: D[arm][i] for i in plac}) for nm, arm in (("twinA", "twinA"), ("twinB", "twinB"), ("base", "base"))}
        one = np.ones(len(allids))
        # weight placebo images so that their source mix matches the anchors'
        ares = pd.Series([meta.loc[i, "res"] for i in allids]).value_counts(normalize=True); pres_ = np.array([meta.loc[i, "res"] for i in plac])
        pshare = pd.Series(pres_).value_counts(normalize=True)
        onep = np.array([ares.get(r, 0.0) / pshare[r] for r in pres_])
        pt = {nm: ap_per_class(*T[nm], one) for nm in T}; ptp = {nm: ap_per_class(*TP[nm], onep) for nm in TP}
        bs = {"leaked_minus_unleaked": [], "unleaked_minus_base": [], "leaked_minus_base": []}; bsp = {"twinA_minus_twinB": [], "twin_mean_minus_base": []}
        bs_cls = []
        for b in range(B):
            w = np.bincount(rng.integers(0, len(allids), len(allids)), minlength=len(allids)).astype(float)
            m = {nm: ap_per_class(*T[nm], w) for nm in T}
            bs["leaked_minus_unleaked"].append(np.nanmean(m["leaked"]) - np.nanmean(m["unleaked"]))
            bs["unleaked_minus_base"].append(np.nanmean(m["unleaked"]) - np.nanmean(m["base"]))
            bs["leaked_minus_base"].append(np.nanmean(m["leaked"]) - np.nanmean(m["base"]))
            bs_cls.append(m["leaked"] - m["unleaked"])
            wp = np.bincount(rng.integers(0, len(plac), len(plac)), minlength=len(plac)).astype(float) * onep
            mp = {nm: ap_per_class(*TP[nm], wp) for nm in TP}
            bsp["twinA_minus_twinB"].append(np.nanmean(mp["twinA"]) - np.nanmean(mp["twinB"]))
            bsp["twin_mean_minus_base"].append((np.nanmean(mp["twinA"]) + np.nanmean(mp["twinB"])) / 2 - np.nanmean(mp["base"]))
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        e3 = {"n_anchors": len(allids), "n_anchor_boxes": int(sum(len(gt_of(i)) for i in allids)), "n_placebo": len(plac),
              "anchor_map50": {nm: float(np.nanmean(v)) for nm, v in pt.items()}, "anchor_ap50_per_class": {nm: v.tolist() for nm, v in pt.items()},
              "placebo_map50": {nm: float(np.nanmean(v)) for nm, v in ptp.items()}}
        e3["diff_leaked_minus_unleaked"] = {"point": e3["anchor_map50"]["leaked"] - e3["anchor_map50"]["unleaked"], "ci95": ci(bs["leaked_minus_unleaked"]), "p_le_0": float(np.mean(np.array(bs["leaked_minus_unleaked"]) <= 0))}
        e3["diff_unleaked_minus_base"] = {"point": e3["anchor_map50"]["unleaked"] - e3["anchor_map50"]["base"], "ci95": ci(bs["unleaked_minus_base"])}
        e3["diff_leaked_minus_base"] = {"point": e3["anchor_map50"]["leaked"] - e3["anchor_map50"]["base"], "ci95": ci(bs["leaked_minus_base"])}
        bc = np.array(bs_cls); e3["per_class_leaked_minus_unleaked"] = {NAMES[c]: {"point": float(pt["leaked"][c] - pt["unleaked"][c]), "ci95": ci(bc[:, c])} for c in range(5)}
        isA = np.array([i in Aset for i in allids], float)
        TA_ = {nm: tp_table(allids, {i: D[arm][i] for i in allids}) for nm, arm in (("twinA", "twinA"), ("twinB", "twinB"))}
        hA = np.nanmean(ap_per_class(*TA_["twinA"], isA)) - np.nanmean(ap_per_class(*TA_["twinB"], isA))
        hB = np.nanmean(ap_per_class(*TA_["twinB"], 1 - isA)) - np.nanmean(ap_per_class(*TA_["twinA"], 1 - isA))
        e3["per_half"] = {"A_anchors_modelA_minus_modelB": float(hA), "B_anchors_modelB_minus_modelA": float(hB)}
        e3["placebo_weighting"] = "source-weighted to match anchor source mix"
        e3["placebo_twinA_minus_twinB"] = {"point": e3["placebo_map50"]["twinA"] - e3["placebo_map50"]["twinB"], "ci95": ci(bsp["twinA_minus_twinB"])}
        e3["placebo_twin_minus_base"] = {"point": (e3["placebo_map50"]["twinA"] + e3["placebo_map50"]["twinB"]) / 2 - e3["placebo_map50"]["base"], "ci95": ci(bsp["twin_mean_minus_base"])}
        # official ultralytics numbers on the anchor halves
        for arm in ("twinA", "twinB", "base"):
            for half, ids in (("A", A_ids), ("B", B_ids)):
                lst = cache / f"list_{half}.txt"; lst.write_text("\n".join(str(img_dir / meta.loc[i, "file"]).replace("\\", "/") for i in ids))
                y = cache / f"data_{half}.yaml"; y.write_text(f"path: {str(HERE).replace(chr(92), '/')}\ntrain: {str(lst).replace(chr(92), '/')}\nval: {str(lst).replace(chr(92), '/')}\ntest: {str(lst).replace(chr(92), '/')}\nnames: {dict(enumerate(NAMES))}\n")
                e3.setdefault("official_half", {})[f"{arm}_{half}"] = official(models[arm], y)
        res["E3"] = e3; print(json.dumps({k: v for k, v in e3.items() if k.startswith(("diff", "placebo", "anchor_map50"))}, indent=1))
    (HERE / "results_paper.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
