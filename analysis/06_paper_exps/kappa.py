# Cohen's kappa between rater 1 (pair_audit.csv) and rater 2 (pair_audit_rater2_TEMPLATE.csv, filled with 1 / 0 / u).
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
HERE = Path(__file__).resolve().parent
a = pd.read_csv(HERE / "pair_audit.csv").set_index("code")["same_scene"].astype(str)
b = pd.read_csv(HERE / "pair_audit_rater2_TEMPLATE.csv").set_index("code")["same_scene_rater2"].astype(str)
m = b.isin(["0", "1", "u"])
print("pairs rated by rater 2:", int(m.sum()))
print("agreement:", round(float((a[m] == b[m]).mean()), 3), " kappa (3 labels):", round(cohen_kappa_score(a[m], b[m]), 3))
sure = m & a.isin(["0", "1"]) & b.isin(["0", "1"])
print("kappa on pairs both raters were sure about:", round(cohen_kappa_score(a[sure], b[sure]), 3), "n =", int(sure.sum()))
