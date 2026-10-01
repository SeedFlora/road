import os, numpy as np, pandas as pd
from pairmontage import pair_montage, OUT
FIG = os.path.join(OUT, "figs")
df = pd.read_csv(os.path.join(OUT, "candidate_pairs_verified.csv"))
bins = [0, 8, 13, 16, 20, 25, 30, 40, 60, 100, 2000]
df["ib"] = pd.cut(df.n_inl, bins, right=False)
sb = pd.cut(df.sim_top, [0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 1.01], right=False)
print(pd.crosstab(df.ib, sb))
rng = np.random.default_rng(0)
for lo, hi in [(13, 16), (16, 20), (20, 25), (25, 30), (30, 40), (40, 60), (60, 2000)]:
    sub = df[(df.n_inl >= lo) & (df.n_inl < hi)]
    pk = sub.sample(16, random_state=lo)
    pair_montage([(r.id_a, r.id_b, f"inl={r.n_inl} good={r.n_good} t={r.sim_top:.3f}") for r in pk.itertuples()], os.path.join(FIG, f"calib_inl_{lo}_{hi}.jpg"))
# high DINO similarity but failed verification
sub = df[(df.sim_top >= 0.90) & (df.n_inl < 13)]; print("sim>=.90 & inl<13:", len(sub), "of", (df.sim_top >= 0.90).sum())
pk = sub.sample(16, random_state=1)
pair_montage([(r.id_a, r.id_b, f"inl={r.n_inl} good={r.n_good} t={r.sim_top:.3f}") for r in pk.itertuples()], os.path.join(FIG, "calib_simhigh_inllow.jpg"))
