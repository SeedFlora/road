"""Pilot baseline: YOLO11n on the group or random split of RDDC 2024.
Usage: python train.py group|random [--capval] [--device cpu|0] [--workers 8]
"""
import argparse, os, json, time, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

def make_trainer():
    """DetectionTrainer whose *val* loader uses `workers` (not 2x) processes.
    Training loader (8 workers, augmentation RNG) is unchanged; val has no augmentation so metrics are unaffected.
    Needed because 8 train + 16 val spawned torch/CUDA workers exceeded the Windows commit limit (error 1455 / bad allocation)."""
    from ultralytics.models.yolo.detect import DetectionTrainer
    class ValCapTrainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            w = self.args.workers
            self.args.workers = max(0, w // 2)  # super() multiplies by 2 -> w
            try:
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            finally:
                self.args.workers = w
    return ValCapTrainer

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("group", "random"))
    parser.add_argument("--capval", action="store_true", help="Cap validation loader workers to the training count.")
    parser.add_argument("--device", default=os.environ.get("ROAD_DEVICE", "0"), help="Ultralytics device, e.g. 0 or cpu (default: ROAD_DEVICE or 0).")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    split, capval = args.split, args.capval
    data = str(ROOT / "analysis" / f"yolo_ds_{split}" / "data.yaml")
    if not Path(data).is_file():
        raise FileNotFoundError(f"Missing dataset: {data}. Prepare the pilot YOLO datasets first.")
    name = f"{split}_y11n_640_e40"
    run = HERE / "runs" / name
    status_path = HERE / f"train_{split}_status.json"
    last = run / "weights" / "last.pt"
    best = run / "weights" / "best.pt"
    if status_path.is_file() and last.is_file() and best.is_file():
        previous = json.loads(status_path.read_text())
        if previous.get("done") and Path(previous.get("save_dir", "")).name == name:
            print(f"Completed run already exists: {run}")
            return
    os.chdir(HERE)  # downloaded pretrained weights and AMP check weights land here
    from ultralytics import YOLO
    t0 = time.time()
    info = {"split": split, "capval": capval, "data": data, "device": args.device, "workers": args.workers,
            "start": datetime.datetime.now().isoformat(), "pid": os.getpid()}
    status_path.write_text(json.dumps(info, indent=1))
    model = YOLO(str(last) if last.is_file() else "yolo11n.pt")
    try:
        if last.is_file():
            model.train(resume=True, data=data, workers=args.workers, device=args.device,
                        **({"trainer": make_trainer()} if capval else {}))
        else:
            model.train(data=data, imgsz=640, epochs=40, batch=16, workers=args.workers, cache=False, seed=0,
                        deterministic=True, project=str(HERE / "runs"), name=name, exist_ok=True,
                        device=args.device, plots=True, verbose=True, **({"trainer": make_trainer()} if capval else {}))
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            info["error"] = str(e)[:500]
            status_path.write_text(json.dumps(info, indent=1))
            raise RuntimeError("The paper recipe requires batch=16. Use a GPU with more memory or --device cpu; "
                               "--workers 0 can reduce loader memory. Batch size was not changed.") from e
        raise
    info.update({"end": datetime.datetime.now().isoformat(), "train_minutes": (time.time() - t0) / 60,
                 "batch": 16, "save_dir": str(model.trainer.save_dir), "done": True})
    status_path.write_text(json.dumps(info, indent=1))

if __name__ == "__main__":
    main()
