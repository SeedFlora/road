"""Sequential training queue for the paper experiments (same recipe as the pilot: YOLO11n, 640 px, 40 epochs, batch 16, seed 0).
E3 runs use val=False and are scored from last.pt (no checkpoint selection on held-out data).
E1 runs select best.pt on their own validation split, exactly like the pilot.
Usage: python train_queue.py [--device cpu|0] [--workers 8]
Skips completed runs and retries a crashed run once with fewer workers."""
import argparse, os, sys, json, time, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = HERE / "ds"
QUEUE = [
    ("inj_base_y11n", DS / "inj_base" / "data.yaml", False),
    ("inj_twinA_y11n", DS / "inj_twinA" / "data.yaml", False),
    ("inj_twinB_y11n", DS / "inj_twinB" / "data.yaml", False),
    ("group_s1_y11n", DS / "group_s1" / "data.yaml", True),
    ("random_s1_y11n", DS / "random_s1" / "data.yaml", True),
    ("group_s2_y11n", DS / "group_s2" / "data.yaml", True),
    ("random_s2_y11n", DS / "random_s2" / "data.yaml", True),
]
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get("ROAD_DEVICE", "0"), help="Ultralytics device, e.g. 0 or cpu (default: ROAD_DEVICE or 0).")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    missing = [str(data) for _, data, _ in QUEUE if not data.is_file()]
    if missing:
        raise FileNotFoundError("Missing paper datasets; run make_datasets.py and make_inj.py first:\n" + "\n".join(missing))
    status = {}
    failed = []
    for name, data, val in QUEUE:
        run = HERE / "runs" / name
        checkpoint = run / "weights" / ("best.pt" if val else "last.pt")
        if checkpoint.is_file() and (run / "weights" / "last.pt").is_file() and (run / "train_minutes.txt").is_file():
            status[name] = "exists"; continue
        worker_counts = (args.workers, max(0, args.workers // 2))
        for attempt, workers in enumerate(worker_counts):
            t0 = time.time()
            with open(HERE / f"log_{name}_a{attempt}.txt", "w") as log:
                rc = subprocess.call([sys.executable, str(HERE / "_worker.py"), name, str(data), "1" if val else "0", str(workers),
                                      "--device", args.device], stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE))
            status[name] = {"rc": rc, "attempt": attempt, "workers": workers, "minutes": round((time.time() - t0) / 60, 1)}
            complete = rc == 0 and checkpoint.is_file() and (run / "train_minutes.txt").is_file()
            if rc == 0 and not complete:
                status[name]["error"] = f"Worker exited without completed checkpoint: {checkpoint}"
            (HERE / "queue_status.json").write_text(json.dumps(status, indent=1))
            if complete:
                break
        else:
            failed.append(name)
    (HERE / "queue_status.json").write_text(json.dumps(status, indent=1))
    print(json.dumps(status, indent=1))
    if failed:
        print("Training failed for: " + ", ".join(failed) + ". See log_<run>_a*.txt.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
