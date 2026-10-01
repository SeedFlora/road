"""GT (left, green) vs prediction conf>=0.25 (right, red) montage for sampled test images of a resolution domain (CPU)."""
import sys, os, random
from pathlib import Path
import numpy as np, pandas as pd, cv2
from ultralytics import YOLO
HERE = Path(__file__).resolve().parent
split, weights, res, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 0
ROOT = HERE.parent
sa = pd.read_csv(ROOT / "01_quality_leakage/splits/split_assignments.csv")
t = sa[(sa[f"{split}_split"] == "test") & (sa.res == res)]
random.seed(seed); ids = random.sample(t.id.tolist(), min(n, len(t)))
model = YOLO(weights)
N = ["long", "lat", "alli", "poth", "oth"]
tiles = []
S = 480
for i in ids:
    f = t[t.id == i].file.iloc[0]
    img = cv2.imread(str(ROOT / f"yolo_ds_{split}/images/test/{f}"))
    h, w = img.shape[:2]
    gt = img.copy(); pr = img.copy()
    lab = ROOT / f"yolo_ds_{split}/labels/test/{Path(f).stem}.txt"
    for line in open(lab):
        c, cx, cy, bw, bh = map(float, line.split())
        x1, y1, x2, y2 = int((cx - bw/2)*w), int((cy - bh/2)*h), int((cx + bw/2)*w), int((cy + bh/2)*h)
        cv2.rectangle(gt, (x1, y1), (x2, y2), (0, 255, 0), max(2, w//300))
        cv2.putText(gt, N[int(c)], (x1, max(20, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, w/900, (0, 255, 0), max(2, w//400))
    r = model.predict(img, imgsz=640, conf=0.25, device="cpu", verbose=False)[0]
    for b, c, s in zip(r.boxes.xyxy.numpy(), r.boxes.cls.numpy(), r.boxes.conf.numpy()):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(pr, (x1, y1), (x2, y2), (0, 0, 255), max(2, w//300))
        cv2.putText(pr, f"{N[int(c)]}{s:.2f}", (x1, max(20, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, w/900, (0, 0, 255), max(2, w//400))
    pair = np.hstack([cv2.resize(gt, (S, S)), cv2.resize(pr, (S, S))])
    cv2.putText(pair, f"id {i}  GT | pred", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    tiles.append(pair)
while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[k:k+2]) for k in range(0, len(tiles), 2)]
out = HERE / "figs" / f"{split}_test_res{res}_gt_vs_pred_s{seed}.jpg"
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80]); print(out)
