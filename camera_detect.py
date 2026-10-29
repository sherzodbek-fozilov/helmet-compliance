# finds connected webcams for the camera dropdown
from __future__ import annotations

import subprocess
import sys


def detect_cameras(max_index: int = 6) -> list[dict]:
    """tries indexes 0..max_index-1, returns [{index, label, available}] for the ones that open"""
    import cv2

    # friendly names via wmic, windows only (missing on newer windows 11 builds, then generic names)
    names: dict[int, str] = {}
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["wmic", "path", "Win32_PnPEntity", "where",
                 "PNPClass='Camera'", "get", "Name"],
                stderr=subprocess.DEVNULL, timeout=5
            ).decode("cp866", errors="ignore")
            cam_names = [l.strip() for l in out.splitlines()
                         if l.strip() and l.strip().lower() != "name"]
            for i, n in enumerate(cam_names):
                names[i] = n
        except Exception:
            pass

    results = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
        available = cap.isOpened()
        if available:
            cap.release()
        label = names.get(idx) or f"Камера {idx}"
        results.append({"index": idx, "label": label, "available": available})

    return [c for c in results if c["available"]]


if __name__ == "__main__":
    cams = detect_cameras()
    if cams:
        for c in cams:
            print(f"[{c['index']}] {c['label']}")
    else:
        print("Камеры не найдены")
