
"""One paper experiment; invoked by train_queue.py with the fixed paper recipe."""
import argparse
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("data")
    parser.add_argument("val", choices=("0", "1"))
    parser.add_argument("workers", type=int)
    parser.add_argument("--device", default=os.environ.get("ROAD_DEVICE", "0"))
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("workers must be non-negative")
    data = str(Path(args.data).resolve())
    if not Path(data).is_file():
        raise FileNotFoundError(f"Missing dataset: {data}")
    os.chdir(HERE)  # YOLO auto-downloads yolo11n.pt here on a clean checkout
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    class ValCapTrainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            w = self.args.workers
            self.args.workers = max(0, w // 2)
            try:
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            finally:
                self.args.workers = w

    t0 = time.time()
    last = HERE / "runs" / args.name / "weights" / "last.pt"
    if last.is_file():  # continue an interrupted run with its original recipe
        model = YOLO(str(last))
        model.train(resume=True, data=data, device=args.device, workers=args.workers, trainer=ValCapTrainer)
    else:
        model = YOLO("yolo11n.pt")
        model.train(data=data, imgsz=640, epochs=40, batch=16, workers=args.workers, cache=False, seed=0,
                    deterministic=True, val=args.val == "1", project=str(HERE / "runs"), name=args.name,
                    exist_ok=True, device=args.device, plots=False, verbose=False, trainer=ValCapTrainer)
    (HERE / "runs" / args.name / "train_minutes.txt").write_text(str((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
