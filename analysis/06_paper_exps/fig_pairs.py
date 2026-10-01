# Fig. 1: three example pairs (verified near-duplicate / pHash false match caused by the ego vehicle / DINOv2 look-alike).
# The ego licence plate of the 1080-px rig is blurred.
import os
from pathlib import Path
from PIL import Image, ImageFilter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"], "font.size": 8})
ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "RDDC 2024_image"
FIG = ROOT / "repro" / "generated" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
pairs = [("2946.png", "5346.png", "(a) verified near-duplicate\n44 inliers, cos 0.93, Hamming 18"),
         ("5006.jpg", "7683.jpg", "(b) pHash match, different spots\nHamming 6 (shared ego plate), 0 inliers"),
         ("4433.jpg", "5135.jpg", "(c) DINOv2 look-alike, different spots\ncos 0.92, 0 inliers")]
def load(f):
    im = Image.open(os.path.join(IMG, f)).convert("RGB").resize((400, 400))
    if f in ("5006.jpg", "7683.jpg"):
        box = (255, 195, 400, 300); im.paste(im.crop(box).filter(ImageFilter.GaussianBlur(10)), box)
    return im
fig, ax = plt.subplots(1, 6, figsize=(7.16, 1.78))
for k, (a, b, t) in enumerate(pairs):
    for j, f in enumerate((a, b)):
        x = ax[2 * k + j]; x.imshow(load(f)); x.set_xticks([]); x.set_yticks([])
        for s in x.spines.values(): s.set_edgecolor(["#1f5fa8", "#d9730d", "#d9730d"][k]); s.set_linewidth(1.8)
    ax[2 * k].set_xlabel(t, fontsize=8, loc="left", labelpad=2)
    ax[2 * k].text(0.03, 0.97, ["✓", "✗", "✗"][k], transform=ax[2 * k].transAxes, va="top", ha="left", fontsize=11, color="white", fontweight="bold", family="DejaVu Sans", bbox=dict(boxstyle="round,pad=0.12", fc=["#1f5fa8", "#d9730d", "#d9730d"][k], ec="none"))
plt.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.22, wspace=0.04)
plt.savefig(FIG / "fig_pairs.pdf", dpi=300)
plt.savefig(FIG / "fig_pairs.png", dpi=200)
