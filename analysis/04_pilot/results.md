# 04_pilot: YOLO11n baselines on RDDC 2024 (group split vs random split)

All numbers below were computed in this folder. The raw images and labels were not modified. Datasets: `analysis/yolo_ds_group` and `analysis/yolo_ds_random` (hardlinked images; the audit built them and I verified them: 4971/1065/1065 labeled images per split, label files identical to the originals, st_nlink = 3). Ultralytics wrote only its usual `labels/*.cache` files inside those two dataset folders.

## Exact settings
- ultralytics 8.4.75, torch 2.11.0+cu128, Python 3.13.5, GPU reported as "NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8192 MiB" (Windows 11, WDDM).
- `YOLO("yolo11n.pt").train(data=..., imgsz=640, epochs=40, batch=16, workers=8, cache=False, seed=0, deterministic=True, device=0, plots=True)`. Everything else is default: optimizer=auto, which resolved to **AdamW lr=0.001111, momentum 0.9**; AMP on; mosaic with close_mosaic=10; standard HSV/flip/scale/translate augmentations. `best.pt` is chosen by ultralytics fitness on the split's own val set.
- I used `cache=False` instead of `cache='ram'`: on Windows, DataLoader workers are spawned processes and would each receive a pickled copy of a RAM cache of about 6 GB.
- Random-split run only: the *validation* DataLoader was capped at 8 workers instead of ultralytics' default 2 x workers = 16 (a `DetectionTrainer` subclass, `train.py --capval`). The training loader, augmentations and seed are unchanged, and val has no augmentation, so metrics are unaffected. The reason is under "Problems" below.
- Evaluation: `model.val(split='test', imgsz=640, batch=16, conf=0.001, iou=0.7, half=False)`. The confusion matrix uses ultralytics' defaults: conf 0.25, IoU 0.45.
- Per-domain and leaky/clean evaluation: txt lists of image paths in `eval_lists/` (labels resolve via images->labels) plus one yaml per list.

## 1. Group split (leakage-free near-duplicate grouping), the main baseline
Training time: **37.0 min** (40 epochs, about 51 s/epoch, 2.85 GB GPU memory). Best val epoch: 34.

| set | P | R | mAP50 | mAP50-95 | mAP75 |
|---|---|---|---|---|---|
| val (1065 img, 3031 boxes) | 0.313 | 0.273 | 0.208 | 0.089 | 0.061 |
| **test (1065 img, 3031 boxes)** | **0.280** | **0.250** | **0.184** | **0.080** | 0.061 |

Per class on test (AP50 / AP50-95, P, R, instances):

| class | AP50 | AP50-95 | P | R | n |
|---|---|---|---|---|---|
| longitudinal | 0.271 | 0.129 | 0.326 | 0.354 | 769 |
| lateral | 0.175 | 0.066 | 0.274 | 0.250 | 515 |
| alligator | 0.221 | 0.103 | 0.318 | 0.299 | 551 |
| pothole | 0.180 | 0.068 | 0.294 | 0.252 | 937 |
| others | 0.072 | 0.036 | 0.190 | 0.093 | 259 |

Confusion matrix on test (conf 0.25, IoU 0.45; `eval/group_test/confusion_matrix*.png`, `confusion_summary.json`):
- **Missed objects dominate.** The share of GT predicted as background: longitudinal 73.3%, lateral 84.7%, alligator 77.3%, pothole 87.1%, others 85.7%. Correctly detected: 24.2 / 11.7 / 18.9 / 11.6 / 5.8%.
- **Class confusion is small.** Misclassified share of GT: 2.5 / 3.7 / 3.8 / 1.3 / 8.5%. The largest confusions: 17 longitudinal predicted as alligator, 16 lateral as longitudinal, 10 others as alligator, 8 alligator as longitudinal, 7 pothole as alligator. "others" is the most confused class.
- **Background false positives** per predicted class: longitudinal 144, lateral 51, alligator 70, pothole 82, others 25.

GPU inference speed (YOLO11n, 640):
- Batched, from ultralytics val (batch 16, fp32): 1.1 ms preprocess + **2.3 ms inference** + 1.4 ms NMS per image.
- Batch 1, 300 test images:

| precision | inference, mean | inference, median | preprocess | postprocess | wall time incl. imread |
|---|---|---|---|---|---|
| fp32 | 12.3 ms | 12.5 ms | 4.3 ms | 1.4 ms | 30.5 ms |
| fp16 | 12.1 ms | – | – | – | – |

- fp16 gives no gain at batch 1. That suggests batch-1 latency here is dominated by per-layer launch overhead, not compute. This is an inference, not measured.

### Per domain (resolution group), group test
Ultralytics metrics per domain. The CI column is my own COCO-style AP50 with a 95% image-level bootstrap interval (see section 4).

| domain | images | boxes | P | R | mAP50 | mAP50-95 | own AP50 [95% CI] |
|---|---|---|---|---|---|---|---|
| 960 | 754 | 2243 | 0.289 | 0.258 | 0.196 | 0.086 | 0.183 [0.169, 0.206] |
| 640 | 190 | 491 | 0.144 | 0.187 | **0.107** | 0.048 | 0.104 [0.083, 0.146] |
| 1080 | 64 | 154 | 0.395 | 0.269 | 0.243 | 0.110 | 0.262 [0.208, 0.369] |
| 1600 | 45 | 121 | 0.208 | 0.241 | 0.169 | 0.074 | 0.166 [0.108, 0.262] |
| 720 | 12 | 22 | 0.936 | 0.083 | 0.112 | 0.043 | too small to be meaningful |

The 640-px source is clearly the hardest: roughly half the mAP50 of 960, and its CI does not overlap 960's. The 1080/1600/720 test sets are small, so their CIs are wide.

## 2. Random split: how much does near-duplicate leakage inflate mAP?
Training time: **34.5 min** (successful third attempt; see "Problems"). Best val epoch: 37.

| set | P | R | mAP50 | mAP50-95 | mAP75 |
|---|---|---|---|---|---|
| val (1065) | 0.321 | 0.286 | 0.222 | 0.102 | 0.078 |
| **test (1065 img, 2982 boxes)** | **0.332** | **0.267** | **0.222** | **0.103** | 0.078 |
| test, leaky subset (720 img, 2174 boxes) | 0.344 | 0.276 | 0.231 | 0.108 | – |
| test, clean subset (345 img, 808 boxes) | 0.292 | 0.255 | 0.208 | 0.096 | – |
| leaky, 960 only (594) | 0.353 | 0.266 | 0.232 | 0.111 | – |
| clean, 960 only (158) | 0.342 | 0.276 | 0.263 | 0.132 | – |
| leaky, 640 only (103) | 0.224 | 0.223 | 0.203 | 0.091 | – |
| clean, 640 only (87) | 0.249 | 0.202 | 0.181 | 0.079 | – |

Random test per class (AP50 / AP50-95):

| class | AP50 | AP50-95 |
|---|---|---|
| longitudinal | 0.327 | 0.154 |
| lateral | 0.214 | 0.084 |
| alligator | 0.267 | 0.144 |
| pothole | 0.201 | 0.082 |
| others | 0.101 | 0.050 |

Random test per domain (mAP50 / mAP50-95):

| domain | images | mAP50 | mAP50-95 |
|---|---|---|---|
| 960 | 752 | 0.234 | 0.112 |
| 640 | 190 | 0.173 | 0.076 |
| 1080 | 66 | 0.255 | 0.123 |
| 1600 | 45 | 0.189 | 0.072 |
| 720 | 12 | 0.070 | 0.034 |

**Findings**
- **Random split vs group split, same model and recipe:** test mAP50 0.222 vs 0.184 (+0.038, +21% relative). mAP50-95 0.103 vs 0.080 (+0.023, +28% relative). Bootstrap on my own AP50: difference +0.034, 95% CI [0.012, 0.059], share of resamples with difference <= 0 = 0.001. Within the 960 domain: +0.038, CI [0.011, 0.066]. **The random split significantly overstates performance.** Caveat: one training seed per split, so seed-to-seed variance is not in the CI.
- **Leaky vs clean inside the random test set is not a reliable measure of leakage here.** Overall leaky minus clean is +0.020, CI [-0.017, 0.056], not significant. The two subsets have very different domain mixes: clean is enriched in 640/1080/1600/720, leaky is 82.5% 960-px. Within 960 the sign flips (-0.029, CI [-0.087, 0.016]). Within 640 it is +0.067, CI [-0.012, 0.142], borderline. The whole-split comparison (group vs random) is the defensible leakage number.
- The model underfits: train box loss is still 1.69 at epoch 40 and val curves are still creeping up (`figs/val_curves_group_vs_random.png`). A higher-capacity or longer-trained model memorises more, so the leakage gap should be re-measured for the final model.

## 3. Error anatomy by object size (group test)
Own COCO-style AP50, with area bins measured at the 640 network input (small < 32^2 px, medium < 96^2 px):

| bin | GT boxes | mAP50 | recall at conf 0.25 |
|---|---|---|---|
| small | 291 (188 of them potholes) | 0.036 | 0.3% |
| medium | 1148 | 0.106 | 6.3% |
| large | 1592 | 0.214 | 21.2% |

The random-split model shows the same pattern (small 0.060, medium 0.118, large 0.279).

Test-time resolution probe (group model trained at 640, no retraining):

| inference imgsz | test mAP50 | test mAP50-95 |
|---|---|---|
| 640 | 0.184 | 0.080 |
| 800 | 0.174 | 0.066 |
| 960 | 0.139 | 0.050 |
| 1280 | 0.097 | 0.034 |

Only pothole AP50 rises slightly at 800 (0.180 to 0.187). Upscaling at inference alone hurts because of train/test scale mismatch. Any resolution benefit would require **training** at a higher resolution, or tiling (SAHI-style); that is untested.

## 4. Bootstrap details
`boot.py` / `bootstrap_map50.json`:
- Own AP50 implementation: 101-point interpolation, greedy conf-ordered matching at IoU >= 0.5, on predictions at conf 0.001 / iou 0.7 / imgsz 640.
- Image-level unpaired bootstrap, B = 2000, seed 0.
- This implementation gives values about 6-7% below ultralytics' mAP50 (group test 0.172 vs 0.184; random test 0.206 vs 0.222) because it matches and interpolates differently. Use it for CIs and relative comparisons only.
- Group test: 0.172 [0.159, 0.191]. Random test: 0.206 [0.194, 0.227].

## 5. Visual inspection
Files: `figs/group_test_res640_gt_vs_pred_s0.jpg`, `figs/group_test_res960_gt_vs_pred_s1.jpg`, `eval/group_test/val_batch*_pred.jpg` vs `*_labels.jpg`.
- The model reliably finds large, near-field longitudinal cracks on concrete and large potholes. It misses distant defects near the horizon (thin lateral cracks, far potholes) and dense clusters of small potholes on broken roads.
- In the 640-px source the frames look softer (possibly upscaled) and come from a different bike with a windscreen in view. Almost nothing is detected at conf 0.25.
- **Label inconsistency is visible and caps achievable mAP:**
  - Some annotators draw one large region box, others draw nested per-crack boxes of several classes inside it (e.g. id 6063).
  - Some "alligator" boxes cover mostly intact asphalt or shoulder (id 1199).
  - The audit already flagged 167 near-full-image boxes; the label-position histogram in `runs/group_y11n_640_e40/labels.jpg` shows them as a spike at (0.5, 0.5). 107 of the 121 boxes centred within 0.01 of the image centre have w > 0.9, mostly alligator.

## Problems encountered
1. **Random-split attempt 1** crashed at the first batch: `RuntimeError: Couldn't open shared file mapping ... error code 1455`, the Windows "paging file too small" error, i.e. the commit limit was exhausted. At the time, another agent's joblib job and my own CPU montage script were running. Log: `train_random_attempt1_crash.log`, run dir `runs/_failed_random_attempt1`.
2. **Attempt 2**, identical settings and nothing else running, crashed in the first validation pass with `RuntimeError: bad allocation`. Idle system commit is about 33 of 60 GB, and 8 training workers plus 16 val workers, each a spawned process importing torch+CUDA, exceeded the limit. Log: `train_random_attempt2_crash.log`.
3. **Attempt 3** used `--capval` (val loader at 8 workers) and completed. The group run had succeeded without the cap, so its margin was simply luckier.
4. The first image-size probe hung silently because it had no `if __name__ == '__main__':` guard: spawned val workers re-imported the script. I killed it and fixed the script.

## Implications for the paper plan
- Use the group split as the main protocol. Random-split numbers are inflated by about 0.03-0.04 mAP50 even for an underfit nano model; this is significant under a test-set bootstrap.
- Absolute performance is low (group test mAP50 0.18). The main error is missed small and medium objects, not class confusion. Candidate methods follow directly: higher training resolution or tiling, small-object necks or attention, stronger or larger backbones such as YOLOv8/10/11-s/m or RT-DETR, and longer schedules.
- Cross-source gap: the 640-px source is about half as accurate. Leave-one-source-out and domain generalisation or adaptation experiments are well motivated.
- Label noise and inconsistent box granularity are a real ceiling. Label cleaning or noise-robust training is a possible contribution. Report "others" separately, or consider merging it, since it is the weakest and most confused class.

## Files
| file | contents |
|---|---|
| `train.py`, `watch.py` | training (detached) and progress watcher |
| `evaluate.py` | test, per-domain, leaky/clean and speed evaluation -> `eval_group.json`, `eval_random.json` |
| `size_analysis.py` | AP by size -> `size_*.json`, cached predictions `preds_*.pkl` |
| `boot.py` | bootstrap CIs -> `bootstrap_map50.json` |
| `imgsz_probe.py` | test-time resolution probe -> `imgsz_probe_group_test.json` |
| `summarize.py` | -> `confusion_summary.json`, `figs/val_curves_group_vs_random.png` |
| `montage_domain.py` | GT vs prediction montages in `figs/` |
| `runs/group_y11n_640_e40/`, `runs/random_y11n_640_e40/` | weights (best.pt, last.pt), results.csv, training plots |
| `eval/` | ultralytics val outputs, including confusion matrices and PR curves for group_test and random_test |
