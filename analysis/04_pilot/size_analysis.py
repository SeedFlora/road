"""AP50 by object size (COCO-style area ranges measured at the 640 px network input) + recall@conf0.25.
Usage: python size_analysis.py <split> <weights> <tag> [ids_file]
Area bins on normalized area a=w*h (square images resized to 640): small a<32^2/640^2, medium <96^2/640^2, large otherwise.
COCO-style: GT outside the range are 'ignore'; detections matched to ignored GT, and unmatched detections outside the range, are dropped.
"""
import sys, os, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEVICE = os.environ.get("ROAD_DEVICE", "0")
NAMES = ["longitudinal", "lateral", "alligator", "pothole", "others"]
S, M = (32/640)**2, (96/640)**2
BINS = {"all": (0, 10), "small": (0, S), "medium": (S, M), "large": (M, 10)}

def iou(a, b):  # a Nx4, b Mx4 xyxy
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1); ab = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (aa[:, None] + ab[None] - inter + 1e-12)

def ap_coco(tp, conf, npos):
    if npos == 0: return float("nan")
    if len(tp) == 0: return 0.0
    o = np.argsort(-conf); tp = tp[o]
    ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
    rec = ctp / npos; prec = ctp / np.maximum(ctp + cfp, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    rs = np.linspace(0, 1, 101); q = np.zeros(101)
    idx = np.searchsorted(rec, rs, side="left")
    for k, i in enumerate(idx):
        q[k] = prec[i] if i < len(prec) else 0
    return float(q.mean())

def evaluate(gts, dets, lo, hi, cls, thr=0.5):
    tps, confs, npos = [], [], 0
    for k in gts:
        g = gts[k]; d = dets[k]
        g = g[g[:, 0] == cls]; d = d[d[:, 5] == cls]
        ga = (g[:, 3] - g[:, 1]) * (g[:, 4] - g[:, 2]) if len(g) else np.zeros(0)
        gign = ~((ga >= lo) & (ga < hi))
        npos += int((~gign).sum())
        if len(d) == 0: continue
        d = d[np.argsort(-d[:, 4])]
        da = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
        matched = np.zeros(len(g), bool)
        I = iou(d[:, :4], g[:, 1:5]) if len(g) else np.zeros((len(d), 0))
        for j in range(len(d)):
            best, bi = thr, -1
            # prefer non-ignored GT (COCO order: non-ignored first)
            for pref in (False, True):
                for gi in range(len(g)):
                    if matched[gi] or gign[gi] != pref: continue
                    if I[j, gi] >= best: best, bi = I[j, gi], gi
                if bi >= 0: break
            if bi >= 0:
                matched[bi] = True
                if gign[bi]: continue  # ignored
                tps.append(1); confs.append(d[j, 4])
            else:
                if not (lo <= da[j] < hi): continue  # unmatched outside range -> ignore
                tps.append(0); confs.append(d[j, 4])
    return ap_coco(np.array(tps, float), np.array(confs, float), npos), npos

def main():
    split, weights, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    sa = pd.read_csv(ROOT / "01_quality_leakage/splits/split_assignments.csv")
    t = sa[sa[f"{split}_split"] == "test"]
    if len(sys.argv) > 4:
        ids = set(int(x) for x in open(sys.argv[4]).read().split()); t = t[t.id.isin(ids)]
    from ultralytics import YOLO
    model = YOLO(weights)
    gts, dets = {}, {}
    files = [str(ROOT / f"yolo_ds_{split}/images/test/{f}") for f in t.file]
    for k in range(0, len(files), 32):
        rs = model.predict(files[k:k+32], imgsz=640, conf=0.001, iou=0.7, max_det=300, device=DEVICE, verbose=False, half=False)
        for f, r in zip(files[k:k+32], rs):
            dets[f] = np.concatenate([r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None], r.boxes.cls.cpu().numpy()[:, None]], 1) if len(r.boxes) else np.zeros((0, 6))
    for f in files:
        lab = ROOT / f"yolo_ds_{split}" / "labels" / "test" / (Path(f).stem + ".txt")
        a = np.loadtxt(lab, ndmin=2) if os.path.getsize(lab) else np.zeros((0, 5))
        g = np.stack([a[:, 0], a[:, 1]-a[:, 3]/2, a[:, 2]-a[:, 4]/2, a[:, 1]+a[:, 3]/2, a[:, 2]+a[:, 4]/2], 1) if len(a) else np.zeros((0, 5))
        gts[f] = g
    import pickle
    pickle.dump({"split": split, "weights": weights, "gts": gts, "dets": dets,
                 "file2id": dict(zip(files, t.id.tolist())), "file2res": dict(zip(files, t.res.tolist()))},
                open(HERE / f"preds_{tag}.pkl", "wb"))
    out = {"tag": tag, "n_images": len(files), "bins_norm_area": {"small": S, "medium": M}, "ap50": {}, "n_gt": {}, "recall_conf025_iou05": {}}
    for b, (lo, hi) in BINS.items():
        aps, ns = {}, {}
        for c in range(5):
            ap, n = evaluate(gts, dets, lo, hi, c); aps[NAMES[c]] = ap; ns[NAMES[c]] = n
        v = [x for x in aps.values() if not np.isnan(x)]
        aps["mean"] = float(np.mean(v)) if v else float("nan")
        out["ap50"][b] = aps; out["n_gt"][b] = ns
        # recall at conf 0.25, class-aware, IoU>=0.5
        hit = tot = 0
        for f in files:
            g = gts[f]; d = dets[f]; d = d[d[:, 4] >= 0.25]
            if not len(g): continue
            ga = (g[:, 3]-g[:, 1])*(g[:, 4]-g[:, 2]); sel = (ga >= lo) & (ga < hi)
            for gi in np.where(sel)[0]:
                tot += 1
                dd = d[d[:, 5] == g[gi, 0]]
                if len(dd) and iou(g[gi:gi+1, 1:5], dd[:, :4]).max() >= 0.5: hit += 1
        out["recall_conf025_iou05"][b] = {"recall": hit / max(tot, 1), "n_gt": tot}
    (HERE / f"size_{tag}.json").write_text(json.dumps(out, indent=1))
    for b in BINS:
        print(b, "n_gt", sum(out["n_gt"][b].values()), "mAP50", round(out["ap50"][b]["mean"], 4),
              {k[:4]: (round(v, 3) if not np.isnan(v) else None) for k, v in out["ap50"][b].items() if k != "mean"},
              "R@0.25", round(out["recall_conf025_iou05"][b]["recall"], 3))

if __name__ == "__main__":
    main()
