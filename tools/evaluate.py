# detection metrics on a split -> docs/metrics_test.json + plots
# python tools/evaluate.py --model models/ppe_yolo11s.pt --split test
#
# mAP is computed at conf=0.001 (the ultralytics default). a higher conf cuts the tail
# of the precision-recall curve and makes mAP look worse than it is
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="construction-ppe.yaml")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=str(ROOT / "docs" / "metrics_test.json"))
    a = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.model)
    m = model.val(data=a.data, split=a.split, imgsz=a.imgsz, device=a.device, plots=True)
    res = {"split": a.split, "conf": 0.001, "mAP50": round(float(m.box.map50), 4),
           "mAP50_95": round(float(m.box.map), 4), "per_class": {}}
    for i, c in enumerate(m.box.ap_class_index):
        prec, rec, ap50, ap = m.box.class_result(i)
        res["per_class"][model.names[int(c)]] = {"precision": round(float(prec), 3), "recall": round(float(rec), 3),
                                                 "AP50": round(float(ap50), 3), "AP50_95": round(float(ap), 3)}

    print(f"\nmAP50 {res['mAP50']:.3f}  mAP50-95 {res['mAP50_95']:.3f}")
    print(f"{'class':<12}{'P':>7}{'R':>7}{'AP50':>7}")
    for name, v in res["per_class"].items():
        print(f"{name:<12}{v['precision']:>7.3f}{v['recall']:>7.3f}{v['AP50']:>7.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nsaved: {a.out}, plots: {m.save_dir}")


if __name__ == "__main__":
    main()
