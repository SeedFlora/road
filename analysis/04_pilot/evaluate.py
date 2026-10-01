"""Evaluate a trained YOLO model on test split, per resolution domain, and (random) leaky/clean subsets.
Usage: python evaluate.py group|random <weights.pt>
"""
import sys, os, json, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEVICE = os.environ.get("ROAD_DEVICE", "0")
SPL = ROOT / "01_quality_leakage" / "splits"
NAMES = ["longitudinal", "lateral", "alligator", "pothole", "others"]

def make_list(split, ids, tag):
    """Write a txt of image paths (inside yolo_ds_<split>/images/test) + a data yaml pointing test to it."""
    d = HERE / "eval_lists"; d.mkdir(exist_ok=True)
    ds = ROOT / f"yolo_ds_{split}"
    files = sorted(os.listdir(ds / "images" / "test"))
    stem2f = {Path(f).stem: f for f in files}
    paths = [f"{ds.as_posix()}/images/test/{stem2f[str(i)]}" for i in ids]
    for p in paths:
        assert os.path.exists(p), p
    txt = d / f"{tag}.txt"; txt.write_text("\n".join(paths) + "\n")
    yml = d / f"{tag}.yaml"
    yml.write_text(
        f"path: {ds.as_posix()}\ntrain: {ds.as_posix()}/images/train\nval: {ds.as_posix()}/images/val\n"
        f"test: {txt.as_posix()}\nnames:\n" + "".join(f"  {k}: {v}\n" for k, v in enumerate(NAMES)))
    return yml, len(paths)

def run_val(model, data, split, name, plots=False):
    m = model.val(data=str(data), split=split, imgsz=640, batch=16, conf=0.001, iou=0.7, half=False,
                  device=DEVICE, workers=4, plots=plots, project=str(HERE / "eval"), name=name, exist_ok=True,
                  verbose=False)
    box = m.box
    out = {"precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50),
           "map50_95": float(box.map), "map75": float(box.map75),
           "n_images": int(sum(1 for _ in open(data.parent / (data.stem + ".txt")))) if str(data).endswith(".yaml") and (data.parent / (data.stem + ".txt")).exists() else None,
           "instances_per_class": {NAMES[i]: int(n) for i, n in enumerate(m.nt_per_class)},
           "speed_ms": {k: float(v) for k, v in m.speed.items()}}
    per = []
    for j, c in enumerate(box.ap_class_index):
        per.append({"cls": NAMES[int(c)], "p": float(box.p[j]), "r": float(box.r[j]),
                    "ap50": float(box.ap50[j]), "ap50_95": float(box.ap[j]),
                    "instances": int(m.nt_per_class[int(c)])})
    out["per_class"] = per
    cm = getattr(m, "confusion_matrix", None)
    if cm is not None:
        out["confusion_matrix_pred_by_gt"] = cm.matrix.astype(int).tolist()  # rows=pred (5=bg), cols=gt (5=bg)
    return out

def speed_test(model, split, n=300):
    import torch
    ds = ROOT / f"yolo_ds_{split}" / "images" / "test"
    files = sorted(str(ds / f) for f in os.listdir(ds))[:n]
    res = {}
    use_cuda = str(DEVICE).lower() not in ("cpu", "mps") and torch.cuda.is_available()
    for half in ((False, True) if use_cuda else (False,)):
        for f in files[:20]:  # warm-up
            model.predict(f, imgsz=640, half=half, device=DEVICE, verbose=False, conf=0.25)
        pre, inf, post = [], [], []
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        for f in files:
            r = model.predict(f, imgsz=640, half=half, device=DEVICE, verbose=False, conf=0.25)[0]
            pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
        wall = (time.time() - t0) / len(files) * 1000
        res["fp16" if half else "fp32"] = {"preprocess_ms": float(np.mean(pre)), "inference_ms": float(np.mean(inf)),
                                           "inference_ms_median": float(np.median(inf)),
                                           "postprocess_ms": float(np.mean(post)),
                                           "wall_ms_incl_imread": float(wall), "n_images": len(files), "batch": 1}
    return res

def main():
    split, weights = sys.argv[1], sys.argv[2]
    from ultralytics import YOLO
    model = YOLO(weights)
    sa = pd.read_csv(SPL / "split_assignments.csv")
    col = f"{split}_split"
    test = sa[sa[col] == "test"]
    R = {"split": split, "weights": weights}
    data = ROOT / f"yolo_ds_{split}" / "data.yaml"
    R["val"] = run_val(model, data, "val", f"{split}_val", plots=False)
    R["test"] = run_val(model, data, "test", f"{split}_test", plots=True)
    R["test"]["n_images"] = int(len(test))
    # per domain (resolution group)
    R["per_domain"] = []
    for res in [960, 640, 1080, 1600, 720]:
        ids = test[test.res == res].id.tolist()
        if not ids:
            continue
        yml, n = make_list(split, ids, f"{split}_test_res{res}")
        r = run_val(model, yml, "test", f"{split}_test_res{res}")
        r.update({"domain": str(res), "n_images": n, "n_boxes": int(test[test.res == res].n_boxes.sum())})
        R["per_domain"].append(r)
    if split == "random":
        R["subsets"] = {}
        for sub in ("leaky", "clean"):
            ids = [int(x) for x in (SPL / f"random_test_{sub}_ids.txt").read_text().split()]
            yml, n = make_list(split, ids, f"random_test_{sub}")
            r = run_val(model, yml, "test", f"random_test_{sub}")
            r["n_images"] = n
            r["res_counts"] = test[test.id.isin(ids)].res.value_counts().to_dict()
            R["subsets"][sub] = r
            # domain-stratified control: leaky vs clean within the two big resolution groups
            for res in (960, 640):
                ids_r = test[test.id.isin(ids) & (test.res == res)].id.tolist()
                if len(ids_r) >= 20:
                    yml, n = make_list(split, ids_r, f"random_test_{sub}_res{res}")
                    r2 = run_val(model, yml, "test", f"random_test_{sub}_res{res}")
                    r2["n_images"] = n
                    R["subsets"][f"{sub}_res{res}"] = r2
    R["speed_batch1"] = speed_test(model, split)
    (HERE / f"eval_{split}.json").write_text(json.dumps(R, indent=1))
    print(json.dumps({k: R[k] for k in ("val", "test")}, indent=1)[:3000])

if __name__ == "__main__":
    main()
