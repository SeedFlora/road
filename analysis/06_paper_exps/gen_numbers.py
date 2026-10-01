"""Write paper/numbers.tex (LaTeX macros) from the analysis JSON files, so every number in the paper is traceable.
Missing results are rendered as a red '??' so they cannot slip into a submission unnoticed."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "analysis"; HERE = A / "06_paper_exps"; OUT = ROOT / "paper" / "numbers.tex"
J = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None
nd = J(A / "01_quality_leakage/near_duplicate_summary.json"); dd = J(HERE / "dedup_baselines.json")
au = J(HERE / "pair_audit_summary.json"); ds = J(HERE / "datasets_report.json"); rp = J(HERE / "results_paper.json")
M = {}
TBD = r"\textcolor{red}{??}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def f3(x):
    return f"{x:.3f}"


def num(x):
    return f"{x:,}".replace(",", "{,}")


M["NDpairs"] = num(nd["n_pairs"]); M["NDshare"] = pct(nd["share_imgs_with_nd_partner"]); M["NDgroups"] = num(nd["n_groups"])
M["NDmulti"] = num(nd["n_multi_groups"]); M["NDcap"] = str(nd["capped_union"]["cap"]); M["NDcross"] = str(nd["cross_res_pairs"])
for r in nd["per_res"]:
    M[f"NDshareRes{ {640: 'A', 720: 'B', 960: 'C', 1080: 'D', 1600: 'E'}[r['res']] }".replace(" ", "")] = pct(r["share_in_multi"])
rows = {r["rule"]: r for r in dd["rules"]}
key = {"geometry-verified (ours)": "Ours", "pHash <= 6": "PhSix", "pHash <= 8": "PhEight", "pHash <= 10": "PhTen",
       "DINOv2 cos >= 0.95": "CosNF", "DINOv2 cos >= 0.90": "CosNinety", "none (random split)": "Rand"}
for k, s in key.items():
    r = rows[k]; M[f"Leak{s}"] = pct(r["test_leak_mean"]); M[f"LeakMin{s}"] = pct(r["test_leak_min"]); M[f"LeakMax{s}"] = pct(r["test_leak_max"])
    M[f"Pairs{s}"] = num(r["pairs"]); M[f"Largest{s}"] = num(r["largest_group"]); M[f"Groups{s}"] = num(r["groups_labeled"])
pa = dd["pair_agreement"]
M["RecallPhTen"] = pct(pa["pHash<=10"]["recall_of_verified_pairs"]); M["RecallPhSix"] = pct(pa["pHash<=6"]["recall_of_verified_pairs"])
M["RecallCosNinety"] = pct(pa["cos>=0.9"]["recall_of_verified_pairs"])
for s, k in (("OursWeak", "ours_weak"), ("OursStrong", "ours_strong"), ("PhOnly", "phash_only"), ("CosOnly", "cos_only")):
    M[f"Aud{s}Lo"] = f"{au[k]['same']}"; M[f"Aud{s}Hi"] = f"{au[k]['same'] + au[k]['uncertain']}"; M[f"Aud{s}N"] = str(au[k]["n"])
M["AudAccLo"] = pct(au["accepted_precision_lower"], 0); M["AudAccHi"] = pct(au["accepted_precision_upper"], 0)
e1 = ds["E1"]
M["LeakValGroupMax"] = pct(max(e1["group_s1"]["leak_rate"]["val"], e1["group_s2"]["leak_rate"]["val"]), 2)
c = ds["E3"]["counts"]; M["InjAnchors"] = str(c["anchors"]); M["InjTwins"] = str(c["twins_per_arm"]); M["InjA"] = str(c["A"]); M["InjB"] = str(c["B"])
M["InjBoxesHalf"] = str(c["A_boxes"])
if rp and rp.get("E1_summary"):
    s = rp["E1_summary"]
    for kind in ("group", "random"):
        for met, nm in (("map50", "Fifty"), ("map50_95", "FiftyNF")):
            v = s[f"{kind}_{met}"]; M[f"{kind.capitalize()}{nm}"] = f3(v["mean"]); M[f"{kind.capitalize()}{nm}Sd"] = f3(v["sd"]) if v["sd"] is not None else TBD
            M[f"{kind.capitalize()}{nm}N"] = str(v["n"])
        for ci, nm in enumerate(["Long", "Lat", "Alli", "Pot", "Oth"]):
            M[f"{kind.capitalize()}AP{nm}"] = f3(s[f"{kind}_ap50_per_class_mean"][ci])
    for met, nm in (("map50", "Fifty"), ("map50_95", "FiftyNF")):
        if f"inflation_{met}" in s:
            v = s[f"inflation_{met}"]; M[f"Infl{nm}"] = f3(v["mean_abs"]); M[f"Infl{nm}Rel"] = f"{v['mean_rel_pct']:.0f}"
            M[f"Infl{nm}Sd"] = f3(v["sd"]) if v["sd"] is not None else TBD
            M[f"Infl{nm}Min"] = f3(min(v["per_draw"])); M[f"Infl{nm}Max"] = f3(max(v["per_draw"]))
    E1 = rp["E1"]
    for kind in ("group", "random"):
        for r, nm in ((960, "C"), (640, "A"), (1080, "D"), (1600, "E")):
            v = [E1[k]["own_map50_by_source"][str(r)] for k in E1 if k.startswith(kind)]
            M[f"{kind.capitalize()}Src{nm}"] = f3(float(np.mean(v)))
if rp and rp.get("E3"):
    e = rp["E3"]
    for nm, k in (("Leaked", "leaked"), ("Unleaked", "unleaked"), ("Base", "base")):
        M[f"Inj{nm}"] = f3(e["anchor_map50"][k])
    d = e["diff_leaked_minus_unleaked"]; M["InjDiff"] = f3(d["point"]); M["InjDiffLo"] = f3(d["ci95"][0]); M["InjDiffHi"] = f3(d["ci95"][1])
    M["InjDiffRel"] = f"{100 * d['point'] / e['anchor_map50']['unleaked']:.0f}"
    d = e["diff_unleaked_minus_base"]; M["InjUB"] = f3(d["point"]); M["InjUBLo"] = f3(d["ci95"][0]); M["InjUBHi"] = f3(d["ci95"][1])
    d = e["placebo_twinA_minus_twinB"]; M["PlacDiff"] = f3(d["point"]); M["PlacDiffLo"] = f3(d["ci95"][0]); M["PlacDiffHi"] = f3(d["ci95"][1])
    d = e["placebo_twin_minus_base"]; M["PlacTB"] = f3(d["point"]); M["PlacTBLo"] = f3(d["ci95"][0]); M["PlacTBHi"] = f3(d["ci95"][1])
    M["NPlacebo"] = str(e["n_placebo"]); M["NAnchorBoxes"] = str(e["n_anchor_boxes"])
    for ci, nm in enumerate(["Long", "Lat", "Alli", "Pot", "Oth"]):
        v = list(e["per_class_leaked_minus_unleaked"].values())[ci]; M[f"InjCls{nm}"] = f3(v["point"])
        M[f"InjCls{nm}Lo"] = f3(v["ci95"][0]); M[f"InjCls{nm}Hi"] = f3(v["ci95"][1])
# ---- robustness numbers (internal review) ----
rb = J(HERE / "robustness_numbers.json")
if rb:
    M["PrevStrong"] = pct(rb["prevalence_strong"]); M["PrevWLo"] = pct(rb["prevalence_weighted"][0]); M["PrevWHi"] = pct(rb["prevalence_weighted"][1])
    M["LeakRandStrong"] = pct(np.mean(rb["random_test_leak_strong"])); M["LeakRandStrongMin"] = pct(min(rb["random_test_leak_strong"]))
    M["LeakRandStrongMax"] = pct(max(rb["random_test_leak_strong"]))
    M["LeakRandWLo"] = pct(rb["random_test_leak_weighted"][0]); M["LeakRandWHi"] = pct(rb["random_test_leak_weighted"][1])
    M["PlainCCLargest"] = num(rb["plain_cc_largest_all"])
    t = rb["twins"]; M["TwinDirectPct"] = pct((t["A"]["direct"] + t["B"]["direct"]) / (t["A"]["twins"] + t["B"]["twins"]), 0)
    M["TwinMean"] = f"{(t['A']['twins'] + t['B']['twins']) / (t['A']['anchors'] + t['B']['anchors']):.1f}"
    c = rb["composition"]
    M["GroupTestSingleton"] = pct(np.mean([r["test_singleton_share"] for r in c["group"]]), 0)
    M["RandTestSingleton"] = pct(np.mean([r["test_singleton_share"] for r in c["random"]]), 0)
    M["RandTestBigGroups"] = pct(np.mean([r["test_from_groups_ge20"] for r in c["random"]]), 0)
    M["GroupTestMaxGroup"] = str(max(r["max_group_size_of_test_images"] for r in c["group"]))
    M["GroupTrainGroups"] = num(int(np.mean([r["train_groups_covered"] for r in c["group"]])))
    M["RandTrainGroups"] = num(int(np.mean([r["train_groups_covered"] for r in c["random"]])))
    M["LabeledGroups"] = num(rb["labeled_groups_total"])
    M["OverlapGroupLo"] = pct(min(rb["test_overlap_group"]), 0); M["OverlapGroupHi"] = pct(max(rb["test_overlap_group"]), 0)
    M["OverlapRandLo"] = pct(min(rb["test_overlap_random"]), 0); M["OverlapRandHi"] = pct(max(rb["test_overlap_random"]), 0)
    M["GroupCosNinetyShare"] = pct(np.mean(rb["group_test_share_cos90_neighbor"]), 0); M["RandCosNinetyShare"] = pct(np.mean(rb["random_test_share_cos90_neighbor"]), 0)
    M["ResidLo"] = pct(rb["residual_leak_estimate"][0], 0); M["ResidHi"] = pct(rb["residual_leak_estimate"][1], 0)
    M["AudFP"] = str(rb["audit_fp_accepted"]); M["AudFPNeighbor"] = str(rb["audit_fp_with_common_neighbor"])
for k, s in (("DINOv2 full-frame cos >= 0.90", "FullNinety"), ("DINOv2 average linkage, cut 0.15", "AvgA"), ("DINOv2 average linkage, cut 0.18", "AvgB")):
    if k in rows:
        r = rows[k]; M[f"Leak{s}"] = pct(r["test_leak_mean"]); M[f"Largest{s}"] = num(r["largest_group"])
for k, s in key.items():
    if "test_leak_strong_mean" in rows[k]:
        M[f"LeakS{s}"] = pct(rows[k]["test_leak_strong_mean"])
for k, s in (("DINOv2 full-frame cos >= 0.90", "FullNinety"), ("DINOv2 average linkage, cut 0.15", "AvgA"), ("DINOv2 average linkage, cut 0.18", "AvgB")):
    if k in rows and "test_leak_strong_mean" in rows[k]:
        M[f"LeakS{s}"] = pct(rows[k]["test_leak_strong_mean"])
M["AudOursStrongUnc"] = str(au["ours_strong"]["uncertain"]); M["AudPhOnlyUnc"] = str(au["phash_only"]["uncertain"])
M["AudCosOnlyUnc"] = str(au["cos_only"]["uncertain"])
if rp and rp.get("E3") and "per_half" in rp["E3"]:
    M["HalfA"] = f3(rp["E3"]["per_half"]["A_anchors_modelA_minus_modelB"]); M["HalfB"] = f3(rp["E3"]["per_half"]["B_anchors_modelB_minus_modelA"])
if rp and rp.get("E1"):
    for kind in ("group", "random"):
        cis = [rp["E1"][k].get("own_map50_group_boot_ci95") for k in rp["E1"] if k.startswith(kind)]
        cis = [c for c in cis if c]
        if cis:
            M[f"{kind.capitalize()}CIWidth"] = f3(float(np.mean([c[1] - c[0] for c in cis])))
EXPECTED = ["HalfA", "HalfB", "GroupCIWidth", "RandomCIWidth"]
for kind in ("Group", "Random"):
    for nm in ("Fifty", "FiftyNF"):
        EXPECTED += [f"{kind}{nm}", f"{kind}{nm}Sd", f"{kind}{nm}N"]
    EXPECTED += [f"{kind}AP{c}" for c in ("Long", "Lat", "Alli", "Pot", "Oth")] + [f"{kind}Src{s}" for s in "CADE"]
for nm in ("Fifty", "FiftyNF"):
    EXPECTED += [f"Infl{nm}{x}" for x in ("", "Rel", "Sd", "Min", "Max")]
EXPECTED += ["InjLeaked", "InjUnleaked", "InjBase", "InjDiff", "InjDiffLo", "InjDiffHi", "InjDiffRel", "InjUB", "InjUBLo", "InjUBHi",
             "PlacDiff", "PlacDiffLo", "PlacDiffHi", "PlacTB", "PlacTBLo", "PlacTBHi", "NPlacebo", "NAnchorBoxes"]
EXPECTED += [f"InjCls{c}{x}" for c in ("Long", "Lat", "Alli", "Pot", "Oth") for x in ("", "Lo", "Hi")]
for k in EXPECTED:
    M.setdefault(k, TBD)
lines = ["% Auto-generated by analysis/06_paper_exps/gen_numbers.py -- do not edit by hand"]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
lines.append("% Fallback: any macro used in the paper but not defined above renders as red ??")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n"); print(len(M), "macros ->", OUT)
