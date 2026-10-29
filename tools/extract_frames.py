# saves every n-th frame of a video, e.g. to label your own test frames
# python tools/extract_frames.py --video site.mp4 --every 30 --out frames_to_label
import argparse
from pathlib import Path

import cv2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--every", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path("frames_to_label"))
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть видео: {a.video}")
    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % a.every == 0:
            cv2.imwrite(str(a.out / f"{a.video.stem}_{idx:06d}.jpg"), frame)
            saved += 1
        idx += 1
    print(f"{a.video.name}: {idx} кадров, сохранено {saved} в {a.out}")


if __name__ == "__main__":
    main()
