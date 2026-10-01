# Paper: near-duplicate leakage in RDDC2024-ID (IEEE conference template, ≤6 pages, 20 references)

## Build
From the repository root, run `docker compose build paper` and `docker compose run --rm paper` to regenerate the numbers and result figure and compile `paper/main.pdf`. See the root [README](../README.md) for the dataset and full experiment commands.

For a local TeX installation:
1. `python analysis/06_paper_exps/gen_numbers.py` → writes `paper/numbers.tex` (every number in the paper is a macro; a red `??` means a result is missing).
2. Change to `paper/`, then run `pdflatex main.tex` twice.

Figures: `analysis/06_paper_exps/fig_pairs.py` (Fig. 1), TikZ in `main.tex` (Fig. 2), `analysis/06_paper_exps/fig_results.py` (Fig. 3, after `eval_all.py`).

## Where the numbers come from
| Topic | Script | Output |
|---|---|---|
| Near-duplicate pairs & groups | `analysis/01_quality_leakage/07_sift_verify.py`, `09_groups.py` | `near_duplicate_pairs.csv`, `near_duplicate_groups.csv` |
| Group/random splits (seed 0) | `01_quality_leakage/12_splits.py` | `splits/` |
| Extra split draws + crossover datasets | `06_paper_exps/make_datasets.py`, `make_inj.py` | `ds/`, `datasets_report.json` |
| Grouping-rule comparison (Table II) | `06_paper_exps/dedup_baselines.py` | `dedup_baselines.json` |
| Visual audit (80 pairs) | `06_paper_exps/pair_audit_montage.py` | `pair_audit.csv`, `pair_audit_summary.json` |
| Robustness numbers | `06_paper_exps/robustness_numbers.py` | `robustness_numbers.json` |
| Training (9 runs total: 2 pilot + 7 paper experiments) | `04_pilot/train.py`, `06_paper_exps/train_queue.py` | `runs/` |
| Evaluation (Table III, Fig. 3) | `06_paper_exps/eval_all.py` | `results_paper.json` |

## TODO for the authors before submission
- [ ] **Second rater for the visual audit** (reviewers will ask): open `analysis/06_paper_exps/audit/audit_00..09.jpg`, fill `same_scene_rater2` (1 = same road spot, 0 = different, u = cannot tell) in `pair_audit_rater2_TEMPLATE.csv` without looking at `pair_audit.csv`, then run `python kappa.py` and add agreement/kappa to Sec. IV-B.
- [ ] Check the first rater's labels in `pair_audit.csv` (they were produced by the AI assistant during drafting and must be confirmed by a human author).
- [ ] Author block: the file currently has the anonymous (double-blind) block. Replace it for camera-ready.
- [ ] Release link: code, group files and recorded split partitions are released at https://github.com/SeedFlora/road. Use an anonymized link for double-blind review and cite the appropriate release in the contributions.
- [ ] Choose the venue and check its template/page rules (IEEE IV: 6 pages incl. references, double-blind; VEHITS uses the SciTePress template and forbids preprints).
- [ ] Balance the last page at camera-ready with `\IEEEtriggeratref{n}`.
