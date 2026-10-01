"""Block up to MAX seconds, printing training progress; return early when the process exits."""
import sys, time, json, os, csv, glob
from pathlib import Path
import psutil
HERE = Path(__file__).resolve().parent
split = sys.argv[1]; MAX = float(sys.argv[2]) if len(sys.argv) > 2 else 530
st = json.loads((HERE / f"train_{split}_status.json").read_text())
pid = st["pid"]
t0 = time.time()
def alive():
    try:
        p = psutil.Process(pid); return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
while alive() and time.time() - t0 < MAX:
    time.sleep(20)
st = json.loads((HERE / f"train_{split}_status.json").read_text())
print("alive:", alive(), "| status:", {k: v for k, v in st.items() if k != 'oom_batch16'})
rs = sorted(glob.glob(str(HERE / "runs" / f"{split}_*" / "results.csv")), key=os.path.getmtime)
if rs:
    rows = list(csv.DictReader(open(rs[-1])))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    for r in rows[-4:]:
        print({k: r[k] for k in ("epoch", "time", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "train/box_loss") if k in r})
    print("epochs done:", len(rows))
log = HERE / f"train_{split}.log"
if log.exists():
    txt = log.read_text(errors="ignore").replace("\r", "\n").splitlines()
    print("--- log tail ---"); print("\n".join(l[:200] for l in txt[-6:]))
