import os, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[2]; IMG = os.path.join(ROOT, "RDDC 2024_image")
OUT = os.path.join(ROOT, "analysis", "01_quality_leakage")
meta = pd.read_csv(os.path.join(OUT, "meta.csv")).set_index("id")
try: font = ImageFont.truetype("arial.ttf", 15)
except Exception: font = ImageFont.load_default()
def tile(i, T, line=True):
    im = Image.open(os.path.join(IMG, meta.loc[i, "file"])).convert("RGB").resize((T, T))
    if line:
        d = ImageDraw.Draw(im); d.line([(0, int(0.6 * T)), (T, int(0.6 * T))], fill=(255, 255, 0), width=1)
    return im
def pair_montage(pairs, fn, T=220, ncol=2):
    """pairs: list of (a, b, text)"""
    Path(fn).parent.mkdir(parents=True, exist_ok=True)
    n = len(pairs); nrow = int(np.ceil(n / ncol))
    canvas = Image.new("RGB", (ncol * (2 * T + 24), nrow * (T + 22)), (40, 40, 40)); d = ImageDraw.Draw(canvas)
    for k, (a, b, txt) in enumerate(pairs):
        r, c = divmod(k, ncol); x0 = c * (2 * T + 24); y0 = r * (T + 22)
        canvas.paste(tile(a, T), (x0, y0 + 20)); canvas.paste(tile(b, T), (x0 + T + 2, y0 + 20))
        d.text((x0 + 4, y0 + 2), f"#{k} {a}-{b} {txt}", fill=(255, 255, 255), font=font)
    canvas.save(fn, quality=85)
def grid(ids_, fn, T=200, ncol=8, texts=None):
    Path(fn).parent.mkdir(parents=True, exist_ok=True)
    n = len(ids_); nrow = int(np.ceil(n / ncol))
    canvas = Image.new("RGB", (ncol * (T + 4), nrow * (T + 20)), (40, 40, 40)); d = ImageDraw.Draw(canvas)
    for k, i in enumerate(ids_):
        r, c = divmod(k, ncol); x0 = c * (T + 4); y0 = r * (T + 20)
        canvas.paste(tile(i, T, line=False), (x0, y0 + 18))
        d.text((x0 + 3, y0 + 1), texts[k] if texts else f"{i}", fill=(255, 255, 255), font=font)
    canvas.save(fn, quality=85)
