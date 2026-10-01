import os, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[2]; IMG = os.path.join(ROOT, "RDDC 2024_image")
OUT = os.path.join(ROOT, "analysis", "01_quality_leakage"); FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)
meta = pd.read_csv(os.path.join(OUT, "meta.csv")).set_index("id"); bx = pd.read_csv(os.path.join(OUT, "boxes.csv"))
COL = {0: (255, 60, 60), 1: (60, 160, 255), 2: (255, 220, 0), 3: (0, 255, 120), 4: (255, 0, 255)}
AB = {0: "L", 1: "T", 2: "A", 3: "P", 4: "O"}
try: font = ImageFont.truetype("arialbd.ttf", 18); sm = ImageFont.truetype("arial.ttf", 13)
except Exception: font = sm = ImageFont.load_default()
def draw(i, T):
    im = Image.open(os.path.join(IMG, meta.loc[i, "file"])).convert("RGB").resize((T, T)); d = ImageDraw.Draw(im)
    for r in bx[bx.id == i].itertuples():
        x1, y1, x2, y2 = (r.cx - r.w / 2) * T, (r.cy - r.h / 2) * T, (r.cx + r.w / 2) * T, (r.cy + r.h / 2) * T
        d.rectangle([x1, y1, x2, y2], outline=COL[r.cls], width=2); d.text((x1 + 2, y1 + 1), AB[r.cls], fill=COL[r.cls], font=sm)
    return im
def grid(ids_, fn, T, ncol, boxes):
    nrow = int(np.ceil(len(ids_) / ncol)); c = Image.new("RGB", (ncol * (T + 4), nrow * (T + 24)), (30, 30, 30)); d = ImageDraw.Draw(c)
    for k, i in enumerate(ids_):
        r, q = divmod(k, ncol); x0, y0 = q * (T + 4), r * (T + 24)
        im = draw(i, T) if boxes else Image.open(os.path.join(IMG, meta.loc[i, "file"])).convert("RGB").resize((T, T))
        c.paste(im, (x0, y0 + 22)); d.text((x0 + 3, y0 + 1), f"{i} ({meta.loc[i,'res']}) n={meta.loc[i,'n_boxes']}", fill=(255, 255, 255), font=font)
    c.save(fn, quality=88)
rng = np.random.default_rng(0)
unl = meta[~meta.labeled].index.values; lab = meta[meta.labeled].index.values
u40 = rng.choice(unl, 40, replace=False)
for k in range(4): grid(list(u40[k * 10:(k + 1) * 10]), os.path.join(FIG, f"unlabeled_random_{k}.jpg"), 300, 5, False)
l30 = rng.choice(lab, 30, replace=False)
for k in range(5): grid(list(l30[k * 6:(k + 1) * 6]), os.path.join(FIG, f"labeled_random_boxes_{k}.jpg"), 420, 3, True)
np.savetxt(os.path.join(OUT, "visual_sample_unlabeled_ids.txt"), u40, fmt="%d"); np.savetxt(os.path.join(OUT, "visual_sample_labeled_ids.txt"), l30, fmt="%d")
# special cases
big = bx[bx.w * bx.h > 0.9].id.unique(); grid(list(rng.choice(big, 9, replace=False)), os.path.join(FIG, "labels_wholeimage_boxes.jpg"), 400, 3, True)
cr = pd.read_csv(os.path.join(OUT, "label_qa_crossclass_overlaps.csv")).id.unique(); grid(list(cr[:9]), os.path.join(FIG, "labels_crossclass_overlap.jpg"), 400, 3, True)
many = meta[meta.n_boxes > 15].sort_values("n_boxes", ascending=False).index[:6]; grid(list(many), os.path.join(FIG, "labels_many_boxes.jpg"), 420, 3, True)
dg = pd.read_csv(os.path.join(OUT, "label_qa_degenerate_boxes.csv")).id.unique(); grid(list(dg), os.path.join(FIG, "labels_degenerate.jpg"), 420, 3, True)
print("ok", u40[:5], l30[:5])
