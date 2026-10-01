"""Reproduce the experiment results and figures, with explicit stages."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QL = ROOT / "analysis" / "01_quality_leakage"
EXPS = ROOT / "analysis" / "06_paper_exps"
NAMES = {0: "longitudinal", 1: "lateral", 2: "alligator", 3: "pothole", 4: "others"}


def run(script, *args):
    print(f"Running {script}", flush=True)
    subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, check=True)


def check(data=False, validate=False):
    required = [QL / "meta.csv", QL / "near_duplicate_groups.csv",
                QL / "near_duplicate_pairs.csv", QL / "splits/split_assignments.csv"]
    required += [EXPS / f"{name}.json" for name in (
        "datasets_report", "dedup_baselines", "pair_audit_summary", "results_paper", "robustness_numbers")]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError("Missing reproduction inputs: " + ", ".join(missing))
    for p in required:
        if p.suffix == ".json":
            json.loads(p.read_text(encoding="utf-8"))
    if validate:
        from repro.validate_partitions import validate_partitions
        print("Recorded partition invariants:", validate_partitions(ROOT), flush=True)
    if data:
        manifest = ROOT / "repro/dataset_manifest.csv"
        checked = 0
        with manifest.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                p = ROOT / row["path"]
                if not p.is_file():
                    raise RuntimeError(f"Missing dataset file: {row['path']}. See README.md for the download.")
                with p.open("rb") as dataset_file:
                    digest = hashlib.file_digest(dataset_file, "sha256").hexdigest()
                if p.stat().st_size != int(row["bytes"]) or digest != row["sha256"]:
                    raise RuntimeError(f"Dataset differs from the paper input: {row['path']}")
                checked += 1
        print(f"Verified {checked} original dataset files (SHA-256).", flush=True)
    print("Reproduction inputs are present.", flush=True)


def report():
    check()
    run("analysis/06_paper_exps/gen_numbers.py")
    run("analysis/06_paper_exps/fig_results.py")
    print("Generated result figures and number macros in repro/generated/.", flush=True)


def prepare(recompute=False):
    if recompute:
        require_fresh_outputs(datasets_only=True)
    check(data=True, validate=not recompute)
    if recompute:
        run("analysis/01_quality_leakage/12_splits.py")
        run("analysis/01_quality_leakage/13_build_yolo.py")
        run("analysis/06_paper_exps/make_datasets.py")
        run("analysis/06_paper_exps/make_inj.py")
        return
    # Restore the exact recorded partitions without rerunning split optimization.
    import yaml
    with (QL / "meta.csv").open(encoding="utf-8", newline="") as fh:
        filenames = {int(r["id"]): r["file"] for r in csv.DictReader(fh)}
    with (ROOT / "repro/partitions.json").open(encoding="utf-8") as fh:
        partitions = json.load(fh)
    with (ROOT / "repro/dataset_manifest.csv").open(encoding="utf-8", newline="") as fh:
        image_hashes = {Path(row["path"]).name: row["sha256"] for row in csv.DictReader(fh)
                        if Path(row["path"]).parent.name == "RDDC 2024_image"}
    for rel, parts in partitions.items():
        dest = ROOT / rel
        cfg = {"path": dest.as_posix(), "names": NAMES}
        for split, ids in parts.items():
            images = dest / "images" / split
            labels = dest / "labels" / split
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            expected_i = {filenames[i] for i in ids}
            expected_l = {f"{i}.txt" for i in ids}
            unexpected = {p.name for p in images.iterdir()} - expected_i
            unexpected |= {p.name for p in labels.glob("*.txt")} - expected_l
            if unexpected:
                raise RuntimeError(f"{rel}/{split} contains files from another split; use a fresh checkout/output directory.")
            for i in ids:
                src = ROOT / "RDDC 2024_image" / filenames[i]
                dst = images / filenames[i]
                if dst.exists():
                    if not src.samefile(dst):
                        with dst.open("rb") as generated_image:
                            digest = hashlib.file_digest(generated_image, "sha256").hexdigest()
                        if digest != image_hashes[filenames[i]]:
                            raise RuntimeError(f"Generated image differs from the original dataset: {dst}. Use a fresh output directory.")
                else:
                    # Read-only raw-data mounts can be on a different filesystem.
                    try:
                        dst.hardlink_to(src)
                    except OSError:
                        shutil.copy2(src, dst)
                shutil.copyfile(ROOT / "RDDC 2024_label" / f"{i}.txt", labels / f"{i}.txt")
            cfg[split] = (images).as_posix()
        if "val" not in cfg:
            cfg["val"] = cfg["test"]
        (dest / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(f"Prepared {rel}: " + ", ".join(f"{s}={len(ids)}" for s, ids in parts.items()), flush=True)


def audit():
    check(data=True)
    run("analysis/01_quality_leakage/00_meta.py")
    run("analysis/01_quality_leakage/01_embed.py")
    run("analysis/01_quality_leakage/07_sift_verify.py")
    run("analysis/01_quality_leakage/09_groups.py")


def baselines():
    required = [QL / "emb_ids.npy", QL / "emb_dinov2s_top_cls.npy", QL / "emb_dinov2s_full_cls.npy"]
    if not all(p.exists() for p in required):
        raise RuntimeError("Baselines need regenerated DINOv2 embeddings. Run the audit stage first.")
    run("analysis/phash_groups.py")
    run("analysis/01_quality_leakage/05_group_explore.py")
    run("analysis/06_paper_exps/robustness_numbers.py")
    run("analysis/06_paper_exps/dedup_baselines.py")


def train():
    run("analysis/04_pilot/train.py", "group")
    run("analysis/04_pilot/train.py", "random", "--capval")
    run("analysis/06_paper_exps/train_queue.py")


def evaluate():
    run("analysis/06_paper_exps/eval_all.py")


def require_fresh_outputs(datasets_only=False):
    paths = [ROOT / "analysis/yolo_ds_group", ROOT / "analysis/yolo_ds_random", EXPS / "ds"]
    if not datasets_only:
        paths += [QL / "sift_cache", ROOT / "analysis/04_pilot/runs", EXPS / "runs", EXPS / "eval_cache"]
    existing = [str(p.relative_to(ROOT)) for p in paths if p.exists() and any(p.iterdir())]
    if existing:
        raise RuntimeError("Regeneration requires a fresh checkout/output directory to avoid stale partitions/checkpoints: "
                           + ", ".join(existing) + ". Use prepare (without --recompute-splits) to restore recorded partitions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "report", "prepare", "audit", "baselines", "train", "evaluate", "all"))
    parser.add_argument("--data", action="store_true", help="Verify all original images/labels against the paper SHA-256 manifest.")
    parser.add_argument("--recompute-splits", action="store_true", help="Recompute split assignments rather than restoring recorded partitions.")
    args = parser.parse_args()
    if args.stage == "check":
        check(args.data, validate=True)
    elif args.stage == "prepare":
        prepare(args.recompute_splits)
    elif args.stage == "all":
        require_fresh_outputs()
        audit()
        prepare(recompute=True)
        baselines()
        train()
        evaluate()
        run("analysis/06_paper_exps/fig_pairs.py")
        report()
    else:
        globals()[args.stage]()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Reproduction failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
