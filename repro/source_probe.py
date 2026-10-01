"""Optional reproduction of the two grouped source-classification results.

Full frames use 224x224 RGB thumbnails resized with OpenCV INTER_AREA, matching
analysis/02_domain_shift/s01_meta_quality.py and s02_embed.py. Upper-60% features
come from the audit stage, which uses 336x210 PIL bicubic crops. The two feature
sets intentionally have different input sizes, as in the original experiment.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parents[1]
QL = ROOT / "analysis" / "01_quality_leakage"
OUT = ROOT / "analysis" / "02_domain_shift"
RES = [960, 640, 1080, 1600, 720]


def full_features(meta, device):
    """Recreate the original full-frame DINOv2 ViT-S/14 CLS-token features."""
    import cv2
    import timm
    import torch

    if device.isdigit():
        device = f"cuda:{device}"
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The original embedding recipe requires CUDA fp16; use --device cuda or cached features.")
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=0, img_size=224).to(device).eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    features = []
    with torch.no_grad():
        for start in range(0, len(meta), 32):
            thumbnails = []
            for filename in meta.file.iloc[start:start + 32]:
                path = ROOT / "RDDC 2024_image" / filename
                bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"Cannot read image: {path}")
                thumbnails.append(cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)[:, :, ::-1])
            x = torch.from_numpy(np.ascontiguousarray(thumbnails)).to(device).permute(0, 3, 1, 2).float() / 255.
            x = ((x - mean) / std).half()
            features.append(model(x).float().cpu().numpy())
            if start % 1024 == 0:
                print(f"Full-frame embeddings: {start}/{len(meta)}", flush=True)
    result = np.concatenate(features).astype(np.float32)
    np.save(OUT / "emb_dinov2s.npy", result)
    np.save(OUT / "emb_ids.npy", meta.id.to_numpy())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get("ROAD_DEVICE", "cuda"))
    parser.add_argument("--recompute-full", action="store_true", help="Regenerate full-frame embeddings even if cached.")
    parser.add_argument("--n-jobs", type=int, default=5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(QL / "meta.csv")
    ids = np.load(QL / "emb_ids.npy")
    if not np.array_equal(ids, meta.id.to_numpy()):
        raise RuntimeError("Metadata and audit embeddings have different image orders.")
    group_table = pd.read_csv(QL / "near_duplicate_groups.csv").set_index("id")
    groups = group_table.loc[ids, "group"].to_numpy()
    full_path = OUT / "emb_dinov2s.npy"
    if full_path.exists() and not args.recompute_full:
        if not np.array_equal(ids, np.load(OUT / "emb_ids.npy")):
            raise RuntimeError("Full-frame embedding cache has a different image order; use --recompute-full.")
        full = np.load(full_path)
    else:
        full = full_features(meta, args.device)
    features = {
        "dinov2s_full224": normalize(full),
        "dinov2s_top60_egoremoved": normalize(np.load(QL / "emb_dinov2s_top_cls.npy")),
    }
    y = meta.res.to_numpy()
    rows = []
    for name, x in features.items():
        if x.shape != (len(ids), 384):
            raise RuntimeError(f"Unexpected feature shape for {name}: {x.shape}")
        pipeline = make_pipeline(StandardScaler(), PCA(128, random_state=0),
                                 LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"))
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
        predicted = cross_val_predict(pipeline, x, y, cv=cv, groups=groups, n_jobs=args.n_jobs)
        rec = recall_score(y, predicted, labels=RES, average=None)
        row = {"features": name, "cv": "neardup_grouped5fold", "accuracy": accuracy_score(y, predicted),
               "balanced_accuracy": balanced_accuracy_score(y, predicted), "macro_f1": f1_score(y, predicted, average="macro"),
               **{f"recall_{resolution}": value for resolution, value in zip(RES, rec)}}
        rows.append(row)
        print(f"{name}: balanced_accuracy={row['balanced_accuracy']:.12f}", flush=True)
    destination = ROOT / "repro" / "generated" / "source_probe.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
