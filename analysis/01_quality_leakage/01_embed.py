# DINOv2 ViT-S/14 embeddings: upper-60% crop (ego-vehicle removed), full image, and bottom-30% (ego-vehicle / rig)
import os, time, numpy as np, pandas as pd, torch, timm
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
ROOT = Path(__file__).resolve().parents[2]
IMG = os.path.join(ROOT, "RDDC 2024_image")
OUT = os.path.join(ROOT, "analysis", "01_quality_leakage")
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
TOP_WH = (336, 210)   # upper 60% crop (W x H), multiples of 14
FULL_WH = (308, 308)
BOT_WH = (336, 98)    # bottom 30% crop (ego-vehicle / mount)

def to_t(im, wh):
    a = (np.asarray(im.resize(wh, Image.BICUBIC), np.float32) / 255. - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1).copy())

class DS(Dataset):
    def __init__(self, meta): self.m = meta
    def __len__(self): return len(self.m)
    def __getitem__(self, k):
        r = self.m.iloc[k]
        im = Image.open(os.path.join(IMG, r.file)).convert("RGB")
        W, H = im.size
        top = im.crop((0, 0, W, int(round(0.6 * H))))
        bot = im.crop((0, int(round(0.7 * H)), W, H))
        return to_t(top, TOP_WH), to_t(im, FULL_WH), to_t(bot, BOT_WH), int(r.id)

if __name__ == "__main__":
    meta = pd.read_csv(os.path.join(OUT, "meta.csv"))
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True).cuda().eval().half()
    npt = model.num_prefix_tokens
    dl = DataLoader(DS(meta), batch_size=32, num_workers=10, pin_memory=True)
    res = {k: [] for k in ["top_cls", "top_mean", "full_cls", "full_mean", "bot_cls"]}; ids = []
    t0 = time.time()
    with torch.no_grad():
        for b, (top, full, bot, i) in enumerate(dl):
            for name, x in (("top", top), ("full", full), ("bot", bot)):
                f = model.forward_features(x.cuda(non_blocking=True).half()).float()
                res[name + "_cls"].append(f[:, 0].cpu().numpy())
                if name != "bot": res[name + "_mean"].append(f[:, npt:].mean(1).cpu().numpy())
            ids.append(i.numpy())
            if b % 40 == 0: print(b, len(dl), f"{time.time()-t0:.0f}s", flush=True)
    ids = np.concatenate(ids)
    np.save(os.path.join(OUT, "emb_ids.npy"), ids)
    for k, v in res.items():
        np.save(os.path.join(OUT, f"emb_dinov2s_{k}.npy"), np.concatenate(v).astype(np.float32))
    print("done", len(ids), f"{time.time()-t0:.0f}s", torch.cuda.max_memory_allocated() / 1e9, "GB")
