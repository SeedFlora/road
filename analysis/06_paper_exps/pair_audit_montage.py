# Visual audit sample of pair decisions (to estimate precision of accepted pairs and of the baselines' extra pairs).
# Strata: ours_weak (15-29 moving inliers), ours_strong (>=30 inliers or cos>=0.96), phash_only (Hamming<=10, rejected by us),
# cos_only (DINOv2 cos>=0.90, rejected by us). 20 pairs each, seed 0. Pairs are shown in random order with a neutral code
# so the rater does not see the stratum; labels go into pair_audit.csv (same_scene = 1 if both frames show the same road spot).
import os, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]; IMG = os.path.join(ROOT, "RDDC 2024_image")
OUT = os.path.join(ROOT, "analysis", "06_paper_exps"); FIG = os.path.join(OUT, "audit"); os.makedirs(FIG, exist_ok=True)
AUDIT = os.path.join(OUT, "pair_audit.csv")
if os.path.exists(AUDIT):
    raise SystemExit(f"Audit already exists: {AUDIT}. Keep the saved ratings; move it explicitly before generating a new blank audit.")
QL = os.path.join(ROOT, "analysis", "01_quality_leakage")
meta = pd.read_csv(os.path.join(QL, "meta.csv")).set_index("id")
cand = pd.read_csv(os.path.join(OUT, "candidate_pairs_with_phash.csv"))
rng = np.random.default_rng(0)
acc = cand[cand.ours]; rej = cand[~cand.ours]
strata = {
    "ours_weak": acc[(acc.n_inl >= 15) & (acc.n_inl < 30) & (acc.sim_top < 0.96)],
    "ours_strong": acc[(acc.n_inl >= 30) | (acc.sim_top >= 0.96)],
    "phash_only": rej[rej.ham <= 10],
    "cos_only": rej[rej.sim_top >= 0.90],
}
rows = []
for k, d in strata.items():
    s = d.sample(n=min(20, len(d)), random_state=0)
    for r in s.itertuples():
        rows.append({"stratum": k, "id_a": int(r.id_a), "id_b": int(r.id_b), "n_inl": int(r.n_inl), "sim_top": round(float(r.sim_top), 3), "ham": int(r.ham), "stratum_size": len(d)})
df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
df["code"] = [f"Q{i:03d}" for i in range(len(df))]; df["same_scene"] = ""
df.to_csv(AUDIT, index=False)
T = 300
for p in range(0, len(df), 8):
    chunk = df.iloc[p:p + 8]; M = Image.new("RGB", (4 * T, 4 * T + 40 * 4), "white"); d = ImageDraw.Draw(M)
    for j, r in enumerate(chunk.itertuples()):
        x0 = (j % 2) * 2 * T; y0 = (j // 2) * (T + 40)
        for c, i in enumerate((r.id_a, r.id_b)):
            im = Image.open(os.path.join(IMG, meta.loc[i, "file"])).convert("RGB").resize((T, T))
            M.paste(im, (x0 + c * T, y0 + 40))
        d.rectangle([x0, y0 + 40, x0 + 2 * T - 1, y0 + 40 + T - 1], outline="red", width=3)
        d.text((x0 + 6, y0 + 12), f"{r.code}", fill="black")
    M.save(os.path.join(FIG, f"audit_{p // 8:02d}.jpg"), quality=88)
print(df.stratum.value_counts().to_dict(), {k: len(v) for k, v in strata.items()}, "montages:", (len(df) + 7) // 8)
