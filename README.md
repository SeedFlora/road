# Geometry-verified road damage splits

Reproduction code for **Quantifying Near-Duplicate Frame Leakage in Crowdsourced Road Damage Detection with Geometry-Verified Splits**, using RDDC2024-ID.

This repository includes the analysis and training code, exact recorded partitions, near-duplicate groups and pairs, aggregate results, training configurations/history, and LaTeX paper sources. Original images, raw labels, trained weights, embedding caches, and large run logs are obtained or regenerated separately.

## Build the paper without a GPU or dataset

```bash
git clone https://github.com/SeedFlora/road.git
cd road
docker compose build paper
docker compose run --rm paper
```

The command regenerates `paper/numbers.tex` and the result figure from the committed JSON results, then builds **`paper/main.pdf`** with two LaTeX passes. The pair figure is provided in `paper/figs/fig_pairs.pdf`; it can be regenerated after downloading the dataset. This step rebuilds the manuscript using recorded measurements; it does not retrain the models.

## Obtain the original dataset

Download the dataset from the [official RDDC2024-ID repository](https://github.com/endurancechallengeindonesia/RDDC2024-ID), following its download link and dataset terms. Extract/rename its image and label directories to:

```text
road/
  RDDC 2024_image/   # 9,048 images, e.g. 1.jpg
  RDDC 2024_label/   # 7,101 numeric YOLO label files + classes.txt
```

Keep filenames and annotations unchanged. The five class IDs are 0 longitudinal crack, 1 lateral crack, 2 alligator crack, 3 pothole, and 4 others. Unlabeled images participate in near-duplicate grouping but are excluded from detector training/evaluation. `repro/dataset_manifest.csv` records file sizes and SHA-256 hashes of the exact inputs used in the paper.

For data stored elsewhere, copy `.env.example` to `.env` and set `ROAD_IMAGE_DIR` and `ROAD_LABEL_DIR` to the extracted folders. Use forward slashes for Windows paths. Compose mounts the original dataset read-only. Generated YOLO datasets, models, and caches are written into this checkout.

## Re-run the experiments

Use Docker Desktop with Linux containers and NVIDIA GPU support, or Docker Engine with the NVIDIA Container Toolkit. The GPU reservation follows the [Docker Compose GPU documentation](https://docs.docker.com/compose/how-tos/gpu-support/). Allow several hours for all training runs and at least 30 GB of working disk space when datasets must be copied across filesystems, plus space for the CUDA image. The original runs used an 8 GiB RTX 3070 Ti Laptop GPU. Embedding extraction requires CUDA; classifier analysis and manuscript generation can run on CPU.

```bash
docker compose build experiments
docker compose run --rm experiments python reproduce.py check --data
```

To retrain/evaluate the detectors on the **exact recorded partitions**, using the released grouping and analysis results:

```bash
docker compose run --rm experiments python reproduce.py prepare
docker compose run --rm experiments python reproduce.py train
docker compose run --rm experiments python reproduce.py evaluate
docker compose run --rm paper
```

For the complete computational pipeline, including embeddings, geometric verification, regenerated splits, grouping baselines, all nine training runs, evaluation, figures, and manuscript:

```bash
docker compose run --rm experiments python reproduce.py all
```

Alternatively, run the full pipeline in stages:

```bash
docker compose run --rm experiments python reproduce.py audit
docker compose run --rm experiments python reproduce.py prepare --recompute-splits
docker compose run --rm experiments python reproduce.py baselines
docker compose run --rm experiments python reproduce.py train
docker compose run --rm experiments python reproduce.py evaluate
docker compose run --rm experiments python analysis/06_paper_exps/fig_pairs.py
docker compose run --rm paper
```

Use a fresh checkout/output directory for a full regeneration. The `all` and `prepare --recompute-splits` stages refuse existing generated datasets to prevent mixing partitions. Separate training/evaluation stages reuse caches and checkpoints. Training skips completed runs and resumes interrupted runs. Evaluation requires all nine checkpoints before writing results. The original recipe is YOLO11n, pretrained `yolo11n.pt`, image size 640, 40 epochs, batch 16, training seed 0, deterministic mode. Pretrained DINOv2/YOLO weights download on first use, so that stage needs internet access. E1 uses `best.pt` selected on each run's validation split; E3 uses `last.pt` with validation disabled. Three seeds refer to split draws; the training seed remains 0. CUDA, operating system, and GPU differences can change the retrained numerical results despite fixed seeds.

## Files and provenance

| Location | Purpose |
| --- | --- |
| `analysis/01_quality_leakage/` | Metadata, DINOv2 retrieval, SIFT verification, grouping, seed-0 splits |
| `analysis/04_pilot/` | Group/random seed-0 training and evaluation helpers |
| `analysis/06_paper_exps/` | Additional split draws, crossover injection, baseline comparison, evaluation and figure scripts |
| `repro/partitions.json` | Exact train/validation/test IDs for all nine dataset variants |
| `repro/reference_runs/` | Original effective training arguments and epoch result CSVs |
| `repro/reference_environment.json` | Original Python/package versions and GPU |
| `repro/supplementary/` | Saved source classification and manual ego-vehicle audit evidence |
| `paper/` | Manuscript, generated number macros, IEEE class and figures |

Training dependencies are pinned to the original direct package versions: Python 3.13.5, Ultralytics 8.4.75, PyTorch 2.11.0 / torchvision 0.26.0 with CUDA 12.8. The Python base image is pinned by digest. See `requirements.txt` and `repro/reference_environment.json`. CUDA wheels follow the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).

The optional grouped source-classification probe can be regenerated after `audit`:

```bash
docker compose run --rm experiments python repro/source_probe.py
```

It reproduces the full-frame 224×224 and upper-60% 336×210 DINOv2 probes separately, and writes `repro/generated/source_probe.csv`. Manual ego-vehicle labels and the pair audit remain recorded human-review inputs, not automatically regenerated measurements.

**Audit status:** the existing first-rater pair labels were produced during AI-assisted drafting and still need confirmation by a human author. A second-rater template is included at `analysis/06_paper_exps/pair_audit_rater2_TEMPLATE.csv`; fill it independently, then run `analysis/06_paper_exps/kappa.py`. The montage script protects existing annotated audit files from overwrite. See `paper/README.md` for the remaining manuscript submission tasks.

## Run without Docker

Create a Python 3.13 environment, install a TeX distribution for the manuscript, then install dependencies:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python reproduce.py check --data
python reproduce.py prepare
python reproduce.py train
python reproduce.py evaluate
python reproduce.py paper
```

For CPU-only detector training, install the PyTorch CPU wheels and set `ROAD_DEVICE=cpu` in your shell before running the Python commands. This does not reproduce the original CUDA embedding extraction. The Docker `paper` service needs neither CUDA nor the dataset.
