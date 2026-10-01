import os, json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = os.path.join(ROOT, "analysis", "01_quality_leakage")
NAMES = {0: "longitudinal", 1: "lateral", 2: "alligator", 3: "pothole", 4: "others"}
bx = pd.read_csv(os.path.join(OUT, "boxes.csv")); meta = pd.read_csv(os.path.join(OUT, "meta.csv"))
bx["x1"] = bx.cx - bx.w / 2; bx["x2"] = bx.cx + bx.w / 2; bx["y1"] = bx.cy - bx.h / 2; bx["y2"] = bx.cy + bx.h / 2
def iou(a, b):
    ix = np.maximum(0, np.minimum(a[:, None, 1], b[None, :, 1]) - np.maximum(a[:, None, 0], b[None, :, 0]))
    iy = np.maximum(0, np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 2], b[None, :, 2]))
    inter = ix * iy; ar = (a[:, 1] - a[:, 0]) * (a[:, 3] - a[:, 2]); br = (b[:, 1] - b[:, 0]) * (b[:, 3] - b[:, 2])
    return inter / (ar[:, None] + br[None, :] - inter + 1e-12)
dups, cross, contain = [], [], []
for i, g in bx.groupby("id"):
    if len(g) < 2: continue
    B = g[["x1", "x2", "y1", "y2"]].values; c = g.cls.values; k = g.k.values
    M = iou(B, B)
    # containment: intersection / smaller area
    ix = np.maximum(0, np.minimum(B[:, None, 1], B[None, :, 1]) - np.maximum(B[:, None, 0], B[None, :, 0]))
    iy = np.maximum(0, np.minimum(B[:, None, 3], B[None, :, 3]) - np.maximum(B[:, None, 2], B[None, :, 2]))
    ar = (B[:, 1] - B[:, 0]) * (B[:, 3] - B[:, 2]); ios = ix * iy / (np.minimum(ar[:, None], ar[None, :]) + 1e-12)
    for p in range(len(g)):
        for q in range(p + 1, len(g)):
            if c[p] == c[q] and M[p, q] > 0.9: dups.append((i, k[p], k[q], int(c[p]), M[p, q]))
            if c[p] != c[q] and M[p, q] > 0.5: cross.append((i, k[p], k[q], int(min(c[p], c[q])), int(max(c[p], c[q])), M[p, q]))
            if c[p] == c[q] and M[p, q] <= 0.9 and ios[p, q] > 0.9: contain.append((i, k[p], k[q], int(c[p])))
dups = pd.DataFrame(dups, columns=["id", "k1", "k2", "cls", "iou"]); cross = pd.DataFrame(cross, columns=["id", "k1", "k2", "c_lo", "c_hi", "iou"])
contain = pd.DataFrame(contain, columns=["id", "k1", "k2", "cls"])
exact = dups[dups.iou > 0.999]
degen = bx[(bx.w < 0.005) | (bx.h < 0.005)]
oob = bx[(bx.x1 < -0.001) | (bx.y1 < -0.001) | (bx.x2 > 1.001) | (bx.y2 > 1.001)]
huge = bx[bx.w * bx.h > 0.9]
many = meta[meta.n_boxes > 15]
cross["pair"] = cross.c_lo.map(NAMES) + "-" + cross.c_hi.map(NAMES)
res = {
    "n_boxes": int(len(bx)), "n_labeled_images": int(meta.labeled.sum()),
    "duplicate_boxes_same_class_iou_gt_0.9": {"pairs": int(len(dups)), "exact_iou_1": int(len(exact)), "images": int(dups.id.nunique()), "by_class": {NAMES[k]: int(v) for k, v in dups.cls.value_counts().sort_index().items()}, "example_ids": dups.id.unique()[:10].tolist()},
    "same_class_nested_boxes_ios_gt_0.9": {"pairs": int(len(contain)), "images": int(contain.id.nunique()), "by_class": {NAMES[k]: int(v) for k, v in contain.cls.value_counts().sort_index().items()}},
    "cross_class_overlap_iou_gt_0.5": {"pairs": int(len(cross)), "images": int(cross.id.nunique()), "by_pair": {k: int(v) for k, v in cross.pair.value_counts().items()}, "example_ids": cross.id.unique()[:10].tolist()},
    "degenerate_boxes_w_or_h_lt_0.005": {"n": int(len(degen)), "images": int(degen.id.nunique()), "by_class": {NAMES[k]: int(v) for k, v in degen.cls.value_counts().sort_index().items()}, "example_ids": degen.id.unique()[:10].tolist()},
    "out_of_bounds_boxes": {"n": int(len(oob)), "images": int(oob.id.nunique())},
    "boxes_area_gt_0.9": {"n": int(len(huge)), "by_class": {NAMES[k]: int(v) for k, v in huge.cls.value_counts().sort_index().items()}, "example_ids": huge.id.unique()[:10].tolist()},
    "images_gt_15_boxes": {"n": int(len(many)), "ids": many.sort_values("n_boxes", ascending=False).id.tolist()[:20], "max": int(meta.n_boxes.max())},
}
json.dump(res, open(os.path.join(OUT, "label_qa_summary.json"), "w"), indent=1)
dups.to_csv(os.path.join(OUT, "label_qa_duplicate_boxes.csv"), index=False); cross.to_csv(os.path.join(OUT, "label_qa_crossclass_overlaps.csv"), index=False)
degen.to_csv(os.path.join(OUT, "label_qa_degenerate_boxes.csv"), index=False)
print(json.dumps(res, indent=1))
