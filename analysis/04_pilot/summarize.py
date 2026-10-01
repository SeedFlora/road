import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
N = ["longitudinal", "lateral", "alligator", "pothole", "others"]
cmout = {}
for sp in ("group", "random"):
    R = json.load(open(HERE / f"eval_{sp}.json"))
    cm = np.array(R["test"]["confusion_matrix_pred_by_gt"])
    gt_tot = cm[:, :5].sum(0)
    d = {"gt_total": gt_tot.tolist(),
         "correct_frac_of_gt": (np.diag(cm)[:5] / gt_tot).round(3).tolist(),
         "missed_frac_of_gt": (cm[5, :5] / gt_tot).round(3).tolist(),
         "misclassified_frac_of_gt": ((cm[:5, :5].sum(0) - np.diag(cm)[:5]) / gt_tot).round(3).tolist(),
         "background_fp_per_pred_class": cm[:5, 5].tolist(),
         "top_confusions": sorted([(int(cm[p, g]), f"gt {N[g]} -> pred {N[p]}") for p in range(5) for g in range(5) if p != g], reverse=True)[:5]}
    cmout[sp] = d; print(sp, json.dumps(d))
(HERE / "confusion_summary.json").write_text(json.dumps(cmout, indent=1))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for sp, col in (("group", "#1f77b4"), ("random", "#d62728")):
    r = pd.read_csv(HERE / f"runs/{sp}_y11n_640_e40/results.csv"); r.columns = [c.strip() for c in r.columns]
    ax[0].plot(r.epoch, r["metrics/mAP50(B)"], color=col, label=f"{sp} split (own val)")
    ax[1].plot(r.epoch, r["metrics/mAP50-95(B)"], color=col, label=f"{sp} split (own val)")
    print(sp, "best val epoch", int(r.loc[r["metrics/mAP50-95(B)"].idxmax(), "epoch"]), "last train box/cls loss", float(r["train/box_loss"].iloc[-1]), float(r["train/cls_loss"].iloc[-1]),
          "last val box/cls loss", float(r["val/box_loss"].iloc[-1]), float(r["val/cls_loss"].iloc[-1]))
for a, t in zip(ax, ("val mAP50", "val mAP50-95")):
    a.set_xlabel("epoch"); a.set_title(t); a.grid(alpha=.3); a.legend()
fig.suptitle("YOLO11n, imgsz 640, 40 epochs: validation curves (each split's own val set)")
(HERE / "figs").mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(HERE / "figs/val_curves_group_vs_random.png", dpi=110)
