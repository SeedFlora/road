import os, sys, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[2]; IMG = os.path.join(ROOT, "RDDC 2024_image")
OUT = os.path.join(ROOT, "analysis", "01_quality_leakage"); FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)
emb = sys.argv[1] if len(sys.argv) > 1 else "top_cls"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ids = np.load(os.path.join(OUT, "emb_ids.npy"))
meta = pd.read_csv(os.path.join(OUT, "meta.csv")).set_index("id")
E = np.load(os.path.join(OUT, f"emb_dinov2s_{emb}.npy")); E /= np.linalg.norm(E, axis=1, keepdims=True)
S = E @ E.T; np.fill_diagonal(S, -1)
nn = S.argmax(1); nns = S.max(1)
rng = np.random.default_rng(seed)
bands = [(0.97, 1.01), (0.95, 0.97), (0.93, 0.95), (0.91, 0.93), (0.89, 0.91), (0.87, 0.89), (0.85, 0.87), (0.82, 0.85)]
T = 240
try: font = ImageFont.truetype("arial.ttf", 16)
except Exception: font = ImageFont.load_default()
def tile(i):
    im = Image.open(os.path.join(IMG, meta.loc[i, "file"])).convert("RGB").resize((T, T))
    d = ImageDraw.Draw(im); d.line([(0, int(0.6 * T)), (T, int(0.6 * T))], fill=(255, 255, 0), width=1)
    return im
for lo, hi in bands:
    cand = np.where((nns >= lo) & (nns < hi))[0]
    pick = rng.choice(cand, size=min(10, len(cand)), replace=False)
    canvas = Image.new("RGB", (4 * T + 3 * 12, 5 * (T + 22)), (40, 40, 40))
    d = ImageDraw.Draw(canvas)
    for k, a in enumerate(pick):
        b = nn[a]; r, c = divmod(k, 2)
        x0 = c * (2 * T + 24); y0 = r * (T + 22)
        canvas.paste(tile(ids[a]), (x0, y0 + 20)); canvas.paste(tile(ids[b]), (x0 + T + 2, y0 + 20))
        d.text((x0 + 4, y0 + 2), f"#{k} {ids[a]}({meta.loc[ids[a],'res']}) - {ids[b]}({meta.loc[ids[b],'res']})  s={S[a,b]:.3f}", fill=(255, 255, 255), font=font)
    fn = os.path.join(FIG, f"band_{emb}_{lo:.2f}_{hi:.2f}_s{seed}.jpg")
    canvas.save(fn, quality=85); print(fn, len(cand))
