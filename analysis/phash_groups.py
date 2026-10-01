# Perceptual-hash every image, then check near-duplicate structure and ID ordering.
import os, random, json, collections
from pathlib import Path
import numpy as np, cv2
from concurrent.futures import ThreadPoolExecutor
root = Path(__file__).resolve().parents[1]
out = root / "analysis"
imgd = os.path.join(root, "RDDC 2024_image"); lbld = os.path.join(root, "RDDC 2024_label")
files = sorted(os.listdir(imgd), key=lambda f: int(os.path.splitext(f)[0]))
def feat(f):
    im = cv2.imread(os.path.join(imgd, f), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    h, w = im.shape
    small = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(small)[:8, :8].flatten()
    bits = (d > np.median(d[1:])).astype(np.uint8)
    full = cv2.imread(os.path.join(imgd, f)); H, W = full.shape[:2]
    return int(os.path.splitext(f)[0]), np.packbits(bits), (W, H)
with ThreadPoolExecutor(12) as ex: res = list(ex.map(feat, files))
ids = np.array([r[0] for r in res]); H = np.stack([r[1] for r in res]); sizes = [r[2] for r in res]
bits = np.unpackbits(H, axis=1).astype(np.int16)  # N x 64
N = len(ids)
# hamming distance matrix via dot products
D = 64 - (bits @ bits.T + (1 - bits) @ (1 - bits).T)
np.fill_diagonal(D, 99)
cons = [int(D[i, i+1]) for i in range(N-1)]
rng = np.random.default_rng(0); rp = [int(D[a, b]) for a, b in rng.integers(0, N, (5000, 2)) if a != b]
print("consecutive-ID hamming: median", np.median(cons), "frac<=10", np.mean(np.array(cons) <= 10))
print("random-pair hamming:    median", np.median(rp), "frac<=10", np.mean(np.array(rp) <= 10))
same_size_consec = np.mean([sizes[i] == sizes[i+1] for i in range(N-1)])
print("consecutive IDs share resolution:", round(float(same_size_consec), 3))
for thr in (4, 6, 8, 10, 12):
    # connected components under threshold (union-find)
    par = list(range(N))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    ii, jj = np.where(np.triu(D <= thr, 1))
    for a, b in zip(ii, jj): par[f(a)] = f(b)
    comp = collections.Counter(f(i) for i in range(N))
    sz = sorted(comp.values(), reverse=True)
    print(f"thr={thr}: pairs={len(ii)} groups={len(comp)} imgs_in_multi={sum(s for s in sz if s>1)} largest={sz[:8]}")
    if thr == 10:
        grp = {int(ids[i]): int(f(i)) for i in range(N)}
        json.dump({"threshold": thr, "group_of_id": grp, "size_of_id": {int(ids[i]): list(sizes[i]) for i in range(N)}}, open(out / "phash_groups_thr10.json", "w"))
np.save(out / "phash_bits.npy", bits); np.save(out / "phash_ids.npy", ids)
# example near-dup pairs for eyeballing
ii, jj = np.where(np.triu(D <= 6, 1)); sel = rng.choice(len(ii), min(6, len(ii)), replace=False)
print("example near-dup pairs (<=6):", [(int(ids[ii[k]]), int(ids[jj[k]]), int(D[ii[k], jj[k]])) for k in sel])
