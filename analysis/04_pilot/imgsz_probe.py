"""Exploratory: group model (trained at 640) evaluated on group TEST at larger inference sizes (no retraining)."""
import json, os
from pathlib import Path
HERE = Path(__file__).resolve().parent

def main():
    os.chdir(HERE)
    from ultralytics import YOLO
    m = YOLO("runs/group_y11n_640_e40/weights/best.pt")
    out = {}
    for s in (640, 800, 960, 1280):
        r = m.val(data=str(HERE.parent / "yolo_ds_group" / "data.yaml"), split="test", imgsz=s, batch=8,
                  conf=0.001, iou=0.7, half=False, device=os.environ.get("ROAD_DEVICE", "0"), workers=4, plots=False, verbose=False,
                  project=str(HERE / "eval"), name=f"group_test_imgsz{s}", exist_ok=True)
        out[s] = {"map50": float(r.box.map50), "map50_95": float(r.box.map), "precision": float(r.box.mp), "recall": float(r.box.mr),
                  "ap50_per_class": [float(x) for x in r.box.ap50], "inference_ms_per_img_batch8": float(r.speed["inference"])}
        print("RESULT", s, out[s], flush=True)
    (HERE / "imgsz_probe_group_test.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
