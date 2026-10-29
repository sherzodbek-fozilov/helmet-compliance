# fine-tunes yolo11s on construction-ppe (11 classes incl. person / helmet / no_helmet).
# settings fit a gtx 1650 with 4 gb: if cuda runs out of memory use --batch 2 or --imgsz 512
# python tools/train.py --epochs 80
import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="yolo11s.pt", help="yolo11n.pt if s does not fit in memory")
    p.add_argument("--data", default="construction-ppe.yaml", help="downloads on first run (~180 mb)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0", help='"0" for the first gpu, "cpu" otherwise')
    p.add_argument("--name", default="safety_ppe")
    p.add_argument("--patience", type=int, default=20)
    a = p.parse_args()

    from ultralytics import YOLO

    YOLO(a.base).train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
        amp=True, cache=False, workers=4, patience=a.patience, name=a.name, project="runs/detect",
        # light augmentation, the dataset is ~1.1k training images
        degrees=5.0, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0, close_mosaic=10,
    )
    print(f"\nвеса: runs/detect/{a.name}/weights/best.pt (выбраны по val)")
    print(f"оценка: python tools/evaluate.py --model runs/detect/{a.name}/weights/best.pt")


if __name__ == "__main__":
    main()
