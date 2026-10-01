# Build YOLO datasets (images = hardlinks where supported, otherwise copies; labels = copied txt)
import os, shutil, json, yaml, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; IMG = os.path.join(ROOT, "RDDC 2024_image"); LAB = os.path.join(ROOT, "RDDC 2024_label")
OUT = os.path.join(ROOT, "analysis", "01_quality_leakage"); SP = os.path.join(OUT, "splits")
meta = pd.read_csv(os.path.join(OUT, "meta.csv")).set_index("id")
NAMES = {0: "longitudinal", 1: "lateral", 2: "alligator", 3: "pothole", 4: "others"}
report = {}
free0 = shutil.disk_usage(ROOT).free
for kind in ["group", "random"]:
    ds = os.path.join(ROOT, "analysis", f"yolo_ds_{kind}")
    info = {}
    for s in ["train", "val", "test"]:
        ids = np.loadtxt(os.path.join(SP, f"{kind}_{s}_ids.txt"), dtype=int).tolist()
        di = os.path.join(ds, "images", s); dl = os.path.join(ds, "labels", s)
        os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
        for i in ids:
            f = meta.loc[i, "file"]; src = os.path.join(IMG, f); dst = os.path.join(di, f)
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)
                except OSError:
                    # Dataset mounts and the output folder can be on different filesystems.
                    shutil.copy2(src, dst)
            shutil.copyfile(os.path.join(LAB, f"{i}.txt"), os.path.join(dl, f"{i}.txt"))
        # verification
        imgs = sorted(x for x in os.listdir(di) if Path(x).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
        labs = sorted(x for x in os.listdir(dl) if Path(x).suffix.lower() == ".txt")
        stems_i = sorted(int(os.path.splitext(x)[0]) for x in imgs); stems_l = sorted(int(os.path.splitext(x)[0]) for x in labs)
        nl = [os.stat(os.path.join(di, x)).st_nlink for x in imgs]
        same = all(os.path.samefile(os.path.join(di, x), os.path.join(IMG, x)) for x in imgs[:200])
        info[s] = {"ids": len(ids), "images": len(imgs), "labels": len(labs), "ids_match_images": stems_i == sorted(ids), "ids_match_labels": stems_l == sorted(ids),
                   "min_st_nlink": int(min(nl)), "samefile_check_first200": bool(same)}
    dsf = ds.replace(os.sep, "/")
    y_abs = {"path": dsf, "train": dsf + "/images/train", "val": dsf + "/images/val", "test": dsf + "/images/test", "names": NAMES}
    with open(os.path.join(ds, "data.yaml"), "w") as fh: yaml.safe_dump(y_abs, fh, sort_keys=False)
    info["yaml"] = os.path.join(ds, "data.yaml")
    report[kind] = info
free1 = shutil.disk_usage(ROOT).free
report["disk_free_before_GB"] = round(free0 / 1e9, 3); report["disk_free_after_GB"] = round(free1 / 1e9, 3); report["disk_used_MB"] = round((free0 - free1) / 1e6, 2)
json.dump(report, open(os.path.join(OUT, "yolo_ds_report.json"), "w"), indent=1)
print(json.dumps(report, indent=1))
