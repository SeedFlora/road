# Geometric verification of candidate near-duplicate pairs: SIFT on the upper-60% crop + ratio test + RANSAC fundamental matrix
import os, time, numpy as np, pandas as pd, cv2
from pathlib import Path
from multiprocessing import Pool
ROOT = Path(__file__).resolve().parents[2]; IMG = os.path.join(ROOT, "RDDC 2024_image")
OUT = os.path.join(ROOT, "analysis", "01_quality_leakage"); TMP = os.path.join(OUT, "sift_cache")
W = 480; NF = 1000
def extract(args):
    i, f = args
    im = cv2.imdecode(np.fromfile(os.path.join(IMG, f), np.uint8), cv2.IMREAD_GRAYSCALE)
    H = im.shape[0]; im = im[: int(round(0.6 * H))]
    im = cv2.resize(im, (W, int(round(W * im.shape[0] / im.shape[1]))), interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=NF)
    kp, d = sift.detectAndCompute(im, None)
    if d is None: return i, np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    d = d / (np.abs(d).sum(1, keepdims=True) + 1e-7); d = np.sqrt(d)  # RootSIFT
    d = np.clip(d * 512, 0, 255).astype(np.uint8)
    return i, np.array([k.pt for k in kp], np.float32), d
_G = {}
def init():
    _G["xy"] = np.load(os.path.join(TMP, "xy.npy"), mmap_mode="r")
    _G["d"] = np.load(os.path.join(TMP, "desc.npy"), mmap_mode="r")
    _G["off"] = np.load(os.path.join(TMP, "off.npy"))
    _G["bf"] = cv2.BFMatcher(cv2.NORM_L2)
YMAX = 240.0  # keep keypoints in the upper 50% of the image only (crop is 480x288 = upper 60%); drops license plate / windscreen
def get(k):
    o = _G["off"]; xy = np.asarray(_G["xy"][o[k]:o[k + 1]]); d = np.asarray(_G["d"][o[k]:o[k + 1]], np.float32)
    keep = xy[:, 1] < YMAX
    return xy[keep], d[keep]
def match(chunk):
    out = []
    for a, b in chunk:
        xa, da = get(a); xb, db = get(b)
        if len(da) < 8 or len(db) < 8: out.append((a, b, 0, 0, 0)); continue
        m = _G["bf"].knnMatch(da, db, k=2)
        good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if len(good) < 8: out.append((a, b, len(good), 0, 0)); continue
        pa = xa[[g.queryIdx for g in good]]; pb = xb[[g.trainIdx for g in good]]
        mov = np.linalg.norm(pa - pb, axis=1) >= 4.0   # drop static matches (ego vehicle / plate / overlays)
        n_static = int((~mov).sum()); pa = pa[mov]; pb = pb[mov]
        if len(pa) < 8: out.append((a, b, len(good), 0, n_static)); continue
        try:
            F, mask = cv2.findFundamentalMat(pa, pb, cv2.USAC_MAGSAC, 1.5, 0.999, 2000)
        except cv2.error:
            try: F, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.5, 0.999)
            except cv2.error: mask = None
        out.append((a, b, len(good), int(mask.sum()) if mask is not None else 0, n_static))
    return out
if __name__ == "__main__":
    os.makedirs(TMP, exist_ok=True)
    ids = np.load(os.path.join(OUT, "emb_ids.npy"))
    meta = pd.read_csv(os.path.join(OUT, "meta.csv")).set_index("id")
    if not os.path.exists(os.path.join(TMP, "off.npy")):
        t0 = time.time()
        with Pool(18) as p: res = p.map(extract, [(k, meta.loc[i, "file"]) for k, i in enumerate(ids)], chunksize=32)
        res.sort(key=lambda r: r[0])
        off = np.cumsum([0] + [len(r[1]) for r in res])
        np.save(os.path.join(TMP, "xy.npy"), np.concatenate([r[1] for r in res]))
        np.save(os.path.join(TMP, "desc.npy"), np.concatenate([r[2] for r in res]))
        np.save(os.path.join(TMP, "off.npy"), off)
        print("extracted", off[-1], "kp, mean/img %.0f" % (off[-1] / len(ids)), f"{time.time()-t0:.0f}s", flush=True)
    E = np.load(os.path.join(OUT, "emb_dinov2s_top_cls.npy")); E /= np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T; np.fill_diagonal(S, -1)
    K = 15
    nn = np.argsort(-S, axis=1)[:, :K]
    pairs = set()
    for a in range(len(ids)):
        for b in nn[a]:
            if S[a, b] >= 0.75: pairs.add((min(a, b), max(a, b)))
    pairs = sorted(pairs); print("candidate pairs", len(pairs), flush=True)
    chunks = [pairs[k:k + 500] for k in range(0, len(pairs), 500)]
    t0 = time.time()
    with Pool(18, initializer=init) as p:
        res = []
        for k, r in enumerate(p.imap_unordered(match, chunks)):
            res.extend(r)
            if k % 50 == 0: print(k, len(chunks), f"{time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(res, columns=["a", "b", "n_good", "n_inl", "n_static"])
    df["id_a"] = ids[df.a]; df["id_b"] = ids[df.b]
    df["sim_top"] = S[df.a, df.b]
    F = np.load(os.path.join(OUT, "emb_dinov2s_full_cls.npy")); F /= np.linalg.norm(F, axis=1, keepdims=True)
    df["sim_full"] = (F[df.a] * F[df.b]).sum(1)
    df.to_csv(os.path.join(OUT, "candidate_pairs_verified.csv"), index=False)
    print(df.n_inl.describe()); print(f"{time.time()-t0:.0f}s")
