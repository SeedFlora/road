# Build per-image metadata table and per-box table (read-only access to the dataset)
import os, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
IMG = os.path.join(ROOT, "RDDC 2024_image"); LAB = os.path.join(ROOT, "RDDC 2024_label")
OUT = os.path.join(ROOT, "analysis", "01_quality_leakage")
os.makedirs(OUT, exist_ok=True)
files = {}
for f in os.listdir(IMG):
    stem, ext = os.path.splitext(f)
    files[int(stem)] = f
rows, boxes = [], []
for i in sorted(files):
    f = files[i]
    with Image.open(os.path.join(IMG, f)) as im:
        w, h = im.size
    lp = os.path.join(LAB, f"{i}.txt")
    labeled = os.path.exists(lp)
    nb = 0
    if labeled:
        with open(lp) as fh:
            for k, line in enumerate(fh):
                p = line.split()
                if len(p) != 5: continue
                c = int(float(p[0])); cx, cy, bw, bh = map(float, p[1:])
                boxes.append((i, k, c, cx, cy, bw, bh)); nb += 1
    rows.append((i, f, os.path.splitext(f)[1].lower(), w, h, labeled, nb))
meta = pd.DataFrame(rows, columns=["id", "file", "ext", "w", "h", "labeled", "n_boxes"])
meta["res"] = meta["w"].astype(str)
bx = pd.DataFrame(boxes, columns=["id", "k", "cls", "cx", "cy", "w", "h"])
meta.to_csv(os.path.join(OUT, "meta.csv"), index=False)
bx.to_csv(os.path.join(OUT, "boxes.csv"), index=False)
print(meta.groupby(["res", "labeled"]).size().unstack())
print("labeled but 0 boxes:", ((meta.labeled) & (meta.n_boxes == 0)).sum())
print(bx.cls.value_counts().sort_index(), len(bx))
print((meta.w != meta.h).sum(), "non-square")
