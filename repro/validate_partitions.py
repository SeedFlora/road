"""Validate recorded paper partitions without raw images or generated YOLO folders."""
import csv
import json
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise RuntimeError("Recorded partition validation failed: " + message)


def _rows(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _saved_ids(path):
    ids = [int(value) for value in path.read_text(encoding="utf-8").split()]
    _require(len(ids) == len(set(ids)), f"duplicate IDs in {path.name}")
    return set(ids)


def validate_partitions(root=None):
    """Check recorded splits, crossover arms, and the saved evaluation subset.

    Returns counts suitable for a CLI status message. Raises RuntimeError when
    the artifacts disagree. Generated datasets and image checksums are not read.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    quality = root / "analysis" / "01_quality_leakage"
    experiments = root / "analysis" / "06_paper_exps"
    raw = json.loads((root / "repro" / "partitions.json").read_text(encoding="utf-8"))
    metadata = _rows(quality / "meta.csv")
    _require(len(metadata) == len({int(row["id"]) for row in metadata}), "duplicate metadata IDs")
    labeled = {int(row["id"]) for row in metadata if row["labeled"].lower() == "true"}
    _require(len(labeled) == 7101, f"expected 7101 labeled images, found {len(labeled)}")

    group_rows = _rows(quality / "near_duplicate_groups.csv")
    groups = {int(row["id"]): int(row["group"]) for row in group_rows}
    _require(len(groups) == len(group_rows), "duplicate near-duplicate group IDs")
    _require(labeled <= groups.keys(), "some labeled images have no near-duplicate group")
    assignment_rows = _rows(quality / "splits" / "split_assignments.csv")
    assignments = {int(row["id"]): row for row in assignment_rows}
    _require(len(assignments) == len(assignment_rows), "duplicate split assignment IDs")
    _require(set(assignments) == labeled, "saved split assignments do not cover the labeled images")

    partitions = {}
    for dataset, splits in raw.items():
        converted = {}
        for split, ids in splits.items():
            _require(all(type(image_id) is int for image_id in ids), f"noninteger IDs in {dataset}/{split}")
            converted[split] = set(ids)
            _require(len(converted[split]) == len(ids), f"duplicate IDs in {dataset}/{split}")
            _require(converted[split] <= labeled, f"unlabeled or unknown IDs in {dataset}/{split}")
        names = list(converted)
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                _require(not converted[first] & converted[second], f"image overlap in {dataset}: {first}/{second}")
        partitions[dataset] = converted

    e1 = [f"analysis/yolo_ds_{kind}" for kind in ("group", "random")]
    e1 += [f"analysis/06_paper_exps/ds/{kind}_s{seed}" for kind in ("group", "random") for seed in (1, 2)]
    reports = json.loads((experiments / "datasets_report.json").read_text(encoding="utf-8"))
    for dataset in e1:
        _require(dataset in partitions, f"missing E1 dataset {dataset}")
        splits = partitions[dataset]
        _require(set(splits) == {"train", "val", "test"}, f"unexpected split names in {dataset}")
        _require(set.union(*splits.values()) == labeled, f"{dataset} does not cover all labeled images")
        counts = {name: len(ids) for name, ids in splits.items()}
        _require(counts == {"train": 4971, "val": 1065, "test": 1065}, f"unexpected E1 sizes in {dataset}: {counts}")
        short_name = dataset.rsplit("/", 1)[-1]
        if dataset.startswith("analysis/yolo_ds_"):
            kind = short_name.rsplit("_", 1)[-1]
            for split, ids in splits.items():
                expected = {image_id for image_id, row in assignments.items() if row[f"{kind}_split"] == split}
                _require(ids == expected, f"{dataset}/{split} differs from split_assignments.csv")
        else:
            _require(splits["test"] == _saved_ids(experiments / f"{short_name}_test_ids.txt"), f"{dataset} differs from saved test IDs")
            _require(counts == reports["E1"][short_name]["n"], f"{dataset} disagrees with datasets_report.json")

    seed0 = partitions["analysis/yolo_ds_group"]
    split_groups = {split: {groups[image_id] for image_id in ids} for split, ids in seed0.items()}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        _require(not split_groups[first] & split_groups[second], f"seed-0 group overlap in {first}/{second}")

    saved = {name: _saved_ids(experiments / f"inj_{name}_ids.txt") for name in (
        "anchors", "anchorsA", "anchorsB", "twinsA", "twinsB", "removed", "placebo")}
    anchors_a, anchors_b = saved["anchorsA"], saved["anchorsB"]
    anchors = anchors_a | anchors_b
    twins_a, twins_b = saved["twinsA"], saved["twinsB"]
    removed, placebo = saved["removed"], saved["placebo"]
    heldout = seed0["val"] | seed0["test"]
    _require(not anchors_a & anchors_b, "crossover anchor halves overlap")
    _require(anchors == saved["anchors"] and len(anchors) == 392, "expected 392 saved crossover anchors")
    _require(anchors <= heldout, "crossover anchors are not held out in seed-0")
    _require(len(twins_a) == len(twins_b) == len(removed) == 395, "expected 395 twins and removed images per arm")
    _require(not twins_a & twins_b, "twin arms overlap")
    _require((twins_a | twins_b) <= heldout and not anchors & (twins_a | twins_b), "twins overlap anchors or original training data")
    _require(removed <= seed0["train"], "removed IDs are not original training images")
    groups_a = {groups[image_id] for image_id in anchors_a}
    groups_b = {groups[image_id] for image_id in anchors_b}
    _require(not groups_a & groups_b, "anchor halves share a near-duplicate group")
    _require({groups[image_id] for image_id in twins_a} <= groups_a, "twinA includes images outside anchor-A groups")
    _require({groups[image_id] for image_id in twins_b} <= groups_b, "twinB includes images outside anchor-B groups")

    arms = {}
    for name in ("base", "twinA", "twinB"):
        dataset = f"analysis/06_paper_exps/ds/inj_{name}"
        _require(dataset in partitions, f"missing E3 dataset {dataset}")
        arms[name] = partitions[dataset]
        _require(set(arms[name]) == {"train", "test"}, f"unexpected splits in {dataset}")
        _require(len(arms[name]["train"]) == 4971, f"unexpected training size in {dataset}")
        _require(not anchors & arms[name]["train"], f"anchors leak into {dataset} training")
    _require(arms["base"]["train"] == seed0["train"], "base arm differs from seed-0 training")
    _require(arms["base"]["test"] == anchors, "base arm test differs from anchors")
    keep = seed0["train"] - removed
    _require(arms["twinA"]["train"] == keep | twins_a, "twinA training does not match crossover injection")
    _require(arms["twinB"]["train"] == keep | twins_b, "twinB training does not match crossover injection")
    test = arms["twinA"]["test"]
    _require(test == arms["twinB"]["test"], "crossover test sets differ")
    _require(anchors <= test <= heldout, "crossover test does not contain held-out anchors")
    _require(len(test) == 1340, "expected 1340 images in each crossover dataset test set")
    _require(placebo <= test and not placebo & (anchors | twins_a | twins_b), "saved placebo is not a disjoint held-out test subset")

    # make_inj builds 948 additional test images. Evaluation uses only the 866
    # held-out labeled singletons and saves that narrower placebo ID list.
    labeled_group_sizes = {}
    for image_id in labeled:
        group = groups[image_id]
        labeled_group_sizes[group] = labeled_group_sizes.get(group, 0) + 1
    singletons = {image_id for image_id in heldout if labeled_group_sizes[groups[image_id]] == 1}
    _require(placebo == singletons and len(placebo) == 866, "saved placebo differs from the 866 held-out labeled singletons")
    counts = reports["E3"]["counts"]
    _require(counts["anchors"] == 392 and counts["twins_per_arm"] == 395, "E3 report disagrees with crossover sizes")
    _require(counts["A"] == len(anchors_a) and counts["B"] == len(anchors_b), "E3 report disagrees with anchor halves")
    _require(counts["placebo"] == len(test - anchors) == 948, "E3 report disagrees with the builder test subset")
    return {"labeled_images": len(labeled), "E1_partitions": len(e1), "anchors": len(anchors),
            "twins_per_arm": len(twins_a), "E3_train_images": len(keep | twins_a),
            "E3_test_images": len(test), "placebo_evaluated": len(placebo)}


if __name__ == "__main__":
    try:
        print(json.dumps(validate_partitions(), indent=2))
    except (RuntimeError, KeyError, ValueError, OSError) as error:
        raise SystemExit(str(error))
