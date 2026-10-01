"""Image-level bootstrap CIs for mAP50 (COCO-style 101-pt, IoU 0.5, own reimplementation) from cached predictions.
Matching is per image, so TP flags are fixed; a bootstrap resample just reweights images."""
import pickle, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
SPL = HERE.parent / "01_quality_leakage" / "splits"
B = 2000; RS = np.linspace(0, 1, 101)

def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    return inter / ((a[:, 2:] - a[:, :2]).prod(1)[:, None] + (b[:, 2:] - b[:, :2]).prod(1)[None] - inter + 1e-12)

def build(pk):
    files = list(pk["gts"].keys())
    per = {c: {"conf": [], "tp": [], "img": []} for c in range(5)}
    npos = np.zeros((len(files), 5))
    for i, f in enumerate(files):
        g, d = pk["gts"][f], pk["dets"][f]
        for c in range(5):
            gc = g[g[:, 0] == c][:, 1:5] if len(g) else np.zeros((0, 4))
            dc = d[d[:, 5] == c] if len(d) else np.zeros((0, 6))
            npos[i, c] = len(gc)
            if not len(dc): continue
            dc = dc[np.argsort(-dc[:, 4])]
            m = np.zeros(len(gc), bool); tp = np.zeros(len(dc))
            I = iou(dc[:, :4], gc) if len(gc) else None
            for j in range(len(dc)):
                if I is None: break
                cand = np.where(~m & (I[j] >= 0.5))[0]
                if len(cand):
                    k = cand[np.argmax(I[j, cand])]; m[k] = True; tp[j] = 1
            per[c]["conf"].append(dc[:, 4]); per[c]["tp"].append(tp); per[c]["img"].append(np.full(len(dc), i))
    for c in range(5):
        cf = np.concatenate(per[c]["conf"]) if per[c]["conf"] else np.zeros(0)
        o = np.argsort(-cf)
        per[c] = {k: (np.concatenate(v)[o] if v else np.zeros(0)) for k, v in per[c].items()}
        per[c]["img"] = per[c]["img"].astype(int)
    return files, per, npos

def map50(per, npos, w):
    aps = []
    for c in range(5):
        P = w @ npos[:, c]
        if P <= 0: continue
        dw = w[per[c]["img"]]; tp = per[c]["tp"]
        ctp = np.cumsum(dw * tp); cfp = np.cumsum(dw * (1 - tp))
        rec = ctp / P; den = ctp + cfp
        prec = np.where(den > 0, ctp / np.maximum(den, 1e-12), 0)
        prec = np.maximum.accumulate(prec[::-1])[::-1]
        idx = np.searchsorted(rec, RS, side="left")
        q = np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0) if len(prec) else np.zeros(101)
        aps.append(q.mean())
    return float(np.mean(aps))

def boot(per, npos, mask, rng):
    idx = np.where(mask)[0]; n = len(idx)
    base = np.zeros(len(mask)); base[idx] = 1
    point = map50(per, npos, base)
    vals = np.empty(B)
    for b in range(B):
        w = np.zeros(len(mask)); np.add.at(w, idx[rng.integers(0, n, n)], 1)
        vals[b] = map50(per, npos, w)
    return point, vals

rng = np.random.default_rng(0)
out = {}
gp = pickle.load(open(HERE / "preds_group_test.pkl", "rb")); rp = pickle.load(open(HERE / "preds_random_test.pkl", "rb"))
gf, gper, gnp = build(gp); rf, rper, rnp = build(rp)
gres = np.array([gp["file2res"][f] for f in gf]); rres = np.array([rp["file2res"][f] for f in rf]); rid = np.array([rp["file2id"][f] for f in rf])
leaky = set(int(x) for x in (SPL / "random_test_leaky_ids.txt").read_text().split())
isleaky = np.array([i in leaky for i in rid])
sets = {
    "group_test": (gper, gnp, np.ones(len(gf), bool)),
    "group_test_res960": (gper, gnp, gres == 960), "group_test_res640": (gper, gnp, gres == 640),
    "group_test_res1080": (gper, gnp, gres == 1080), "group_test_res1600": (gper, gnp, gres == 1600),
    "random_test": (rper, rnp, np.ones(len(rf), bool)),
    "random_test_res960": (rper, rnp, rres == 960), "random_test_res640": (rper, rnp, rres == 640),
    "random_leaky": (rper, rnp, isleaky), "random_clean": (rper, rnp, ~isleaky),
    "random_leaky_res960": (rper, rnp, isleaky & (rres == 960)), "random_clean_res960": (rper, rnp, ~isleaky & (rres == 960)),
    "random_leaky_res640": (rper, rnp, isleaky & (rres == 640)), "random_clean_res640": (rper, rnp, ~isleaky & (rres == 640)),
}
V = {}
for k, (per, npos, m) in sets.items():
    p, v = boot(per, npos, m, rng); V[k] = v
    out[k] = {"n_images": int(m.sum()), "map50": p, "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    print(k, out[k])
def diff(a, b):
    d = V[a] - V[b]
    return {"point": out[a]["map50"] - out[b]["map50"], "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_le_0": float((d <= 0).mean())}
out["diff_random_minus_group"] = diff("random_test", "group_test")
out["diff_random_minus_group_res960"] = diff("random_test_res960", "group_test_res960")
out["diff_leaky_minus_clean"] = diff("random_leaky", "random_clean")
out["diff_leaky_minus_clean_res960"] = diff("random_leaky_res960", "random_clean_res960")
out["diff_leaky_minus_clean_res640"] = diff("random_leaky_res640", "random_clean_res640")
for k in [k for k in out if k.startswith("diff")]: print(k, out[k])
out["note"] = "Own COCO-style AP50 (101-pt, greedy conf-ordered matching, IoU>=0.5) on predictions at conf 0.001, iou 0.7, imgsz 640; ~6-7% lower than ultralytics' mAP50 values. Unpaired image-level bootstrap, B=2000, seed 0."
(HERE / "bootstrap_map50.json").write_text(json.dumps(out, indent=1))
