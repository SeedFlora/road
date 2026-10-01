import os, numpy as np, pandas as pd
from pairmontage import pair_montage, OUT, meta
FIG = os.path.join(OUT, "figs")
ids = np.load(os.path.join(OUT, "emb_ids.npy"))
def load(n):
    E = np.load(os.path.join(OUT, f"emb_dinov2s_{n}.npy")); return E / np.linalg.norm(E, axis=1, keepdims=True)
T = load("top_cls"); F = load("full_cls")
St = T @ T.T; Sf = F @ F.T; np.fill_diagonal(St, -1); np.fill_diagonal(Sf, -1)
rng = np.random.default_rng(1)
# finer bands, all pairs (not only NN) sampled uniformly within band
iu = np.triu_indices(len(ids), 1)
for lo, hi in [(0.86, 0.88), (0.84, 0.86), (0.82, 0.84), (0.80, 0.82), (0.77, 0.80)]:
    m = np.where((St[iu] >= lo) & (St[iu] < hi))[0]
    pick = rng.choice(m, 16, replace=False)
    pairs = [(ids[iu[0][p]], ids[iu[1][p]], f"t={St[iu[0][p], iu[1][p]]:.3f} f={Sf[iu[0][p], iu[1][p]]:.3f} r={meta.loc[ids[iu[0][p]],'res']}/{meta.loc[ids[iu[1][p]],'res']}") for p in pick]
    pair_montage(pairs, os.path.join(FIG, f"calib_allpairs_top_{lo:.2f}_{hi:.2f}.jpg"))
    print(lo, hi, len(m))
# disagreement: full high but top low (ego-vehicle driven?) and top high but full low
d = Sf[iu] - St[iu]
m = np.where((Sf[iu] >= 0.88) & (St[iu] < 0.82))[0]; print("full>=.88 & top<.82:", len(m))
pick = rng.choice(m, min(16, len(m)), replace=False)
pair_montage([(ids[iu[0][p]], ids[iu[1][p]], f"t={St[iu[0][p], iu[1][p]]:.3f} f={Sf[iu[0][p], iu[1][p]]:.3f}") for p in pick], os.path.join(FIG, "calib_fullhigh_toplow.jpg"))
m = np.where((St[iu] >= 0.88) & (Sf[iu] < 0.82))[0]; print("top>=.88 & full<.82:", len(m))
pick = rng.choice(m, min(16, len(m)), replace=False)
pair_montage([(ids[iu[0][p]], ids[iu[1][p]], f"t={St[iu[0][p], iu[1][p]]:.3f} f={Sf[iu[0][p], iu[1][p]]:.3f}") for p in pick], os.path.join(FIG, "calib_tophigh_fulllow.jpg"))
