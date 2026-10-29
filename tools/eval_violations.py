# compares two ways to find people without a helmet on a labeled split:
#   rule      - person box with no helmet in its head zone (what the app uses)
#   no_helmet - the model's own no_helmet class
# python tools/eval_violations.py --model models/ppe_yolo11s.pt --data construction-ppe.yaml --split test
#
# ground truth per labeled person: no_helmet box in the head zone -> violation,
# helmet box in the head zone -> ok. persons with neither are skipped (head not labeled).
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector import in_head_zone, person_has_helmet  # noqa: E402

HEAD_RATIO, PERSON_CONF, HELMET_CONF = 0.35, 0.45, 0.5  # same defaults as the app


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def read_labels(path: Path, w: int, h: int):
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                c, x, y, bw, bh = int(p[0]), *map(float, p[1:5])
                out.append((c, ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)))
    return out


def resolve_split(data_yaml: str, split: str):
    cfg_path = Path(data_yaml)
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
        root = Path(cfg.get("path") or cfg_path.parent)
        if not root.is_absolute():
            root = (cfg_path.parent / root).resolve()
    else:  # bundled ultralytics dataset name, downloads on first use
        from ultralytics.data.utils import check_det_dataset
        cfg = check_det_dataset(data_yaml)
        root = Path(cfg["path"])
    names = {int(k): str(v).lower() for k, v in cfg["names"].items()}
    img_dir = Path(cfg[split]) if Path(cfg[split]).is_absolute() else root / cfg[split]
    lbl_dir = Path(str(img_dir).replace("images", "labels"))
    return names, img_dir, lbl_dir


def collect(model_path: str, data_yaml: str, split: str, device):
    from PIL import Image
    from ultralytics import YOLO

    names, img_dir, lbl_dir = resolve_split(data_yaml, split)
    cid = {v: k for k, v in names.items()}
    model = YOLO(model_path)
    mid = {str(v).lower(): k for k, v in model.names.items()}

    items = []
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for i, img in enumerate(images, 1):
        w, h = Image.open(img).size
        gt = read_labels(lbl_dir / f"{img.stem}.txt", w, h)
        r = model.predict(str(img), conf=0.01, verbose=False, device=device)[0]
        pred = [(int(c), tuple(map(float, b)), float(s))
                for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        items.append({
            "gt_person": [b for c, b in gt if c == cid["person"]],
            "gt_helmet": [b for c, b in gt if c == cid["helmet"]],
            "gt_nohelmet": [b for c, b in gt if c == cid["no_helmet"]],
            "person": [(b, s) for c, b, s in pred if c == mid["person"]],
            "helmet": [(b, s) for c, b, s in pred if c == mid["helmet"]],
            "nohelmet": [(b, s) for c, b, s in pred if c == mid["no_helmet"]],
        })
        print(f"\r{i}/{len(images)}", end="", flush=True)
    print()
    return items


def evaluate(items, method: str, person_conf=PERSON_CONF, helmet_conf=HELMET_CONF, nohelmet_conf=PERSON_CONF):
    tp = fp = fn = tn = extra_fp = missed_person = skipped = 0
    for it in items:
        persons = [(b, s) for b, s in it["person"] if s >= person_conf]
        nohelmets = [b for b, s in it["nohelmet"] if s >= nohelmet_conf]

        # greedy person matching by iou
        pairs = sorted(((iou(g, p[0]), gi, pi) for gi, g in enumerate(it["gt_person"])
                        for pi, p in enumerate(persons)), reverse=True)
        g2p, used = {}, set()
        for v, gi, pi in pairs:
            if v < 0.5 or gi in g2p or pi in used:
                continue
            g2p[gi] = pi
            used.add(pi)

        def rule_flag(pbox):
            return not person_has_helmet(pbox, it["helmet"], head_ratio=HEAD_RATIO, min_conf=helmet_conf)

        def class_flag(gbox):
            return any(in_head_zone(gbox, b, HEAD_RATIO) for b in nohelmets)

        for gi, g in enumerate(it["gt_person"]):
            is_viol = any(in_head_zone(g, b, HEAD_RATIO) for b in it["gt_nohelmet"])
            is_ok = any(in_head_zone(g, b, HEAD_RATIO) for b in it["gt_helmet"])
            if is_viol == is_ok:  # head not labeled (or both) -> unknown
                skipped += 1
                continue
            r = gi in g2p and rule_flag(persons[g2p[gi]][0])
            c = class_flag(g)
            flag = {"rule": r, "no_helmet": c, "rule_or_no_helmet": r or c}[method]
            if is_viol:
                tp += flag
                fn += not flag
                missed_person += gi not in g2p
            else:
                fp += flag
                tn += not flag

        # alarms that don't land on any labeled person count against precision
        if method in ("rule", "rule_or_no_helmet"):
            extra_fp += sum(1 for pi, (b, _) in enumerate(persons) if pi not in used and rule_flag(b))
        if method in ("no_helmet", "rule_or_no_helmet"):
            extra_fp += sum(1 for b in nohelmets
                            if not any(in_head_zone(g, b, HEAD_RATIO) for g in it["gt_person"]))

    prec = tp / max(tp + fp + extra_fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"violators": tp + fn, "compliant": fp + tn, "skipped_unlabeled_heads": skipped,
            "tp": tp, "fp_on_compliant": fp, "fp_elsewhere": extra_fp, "fn": fn,
            "violators_missed_because_person_not_detected": missed_person,
            "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3),
            "false_alarm_rate_on_compliant": round(fp / max(fp + tn, 1), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="construction-ppe.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(ROOT / "docs" / "violations_eval.json"))
    args = ap.parse_args()

    items = collect(args.model, args.data, args.split, args.device)
    res = {m: evaluate(items, m) for m in ("rule", "no_helmet", "rule_or_no_helmet")}
    # the no_helmet class at its best threshold on this split: optimistic, shown for fairness
    best = max(((t / 100, evaluate(items, "no_helmet", nohelmet_conf=t / 100)) for t in range(5, 95, 5)),
               key=lambda x: x[1]["f1"])
    res["no_helmet_best_threshold"] = {"threshold": best[0], **best[1]}
    res["settings"] = {"split": args.split, "images": len(items), "person_conf": PERSON_CONF,
                       "helmet_conf": HELMET_CONF, "head_zone_ratio": HEAD_RATIO, "iou_match": 0.5}

    print(f"\n{'method':26s} {'precision':>9s} {'recall':>7s} {'f1':>6s} {'false alarms on compliant':>26s}")
    for m in ("rule", "no_helmet", "no_helmet_best_threshold", "rule_or_no_helmet"):
        r = res[m]
        print(f"{m:26s} {r['precision']:9.3f} {r['recall']:7.3f} {r['f1']:6.3f} "
              f"{r['false_alarm_rate_on_compliant']:26.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
