# exclusion zones: people standing inside one are not counted as violators
# (e.g. a walkway or an office corner where helmets are not required)
from __future__ import annotations

import cv2
import numpy as np


def point_in_polygon(x: float, y: float, polygon: list[tuple[int, int]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def detection_in_zone(det, polygon: list[tuple[int, int]]) -> bool:
    """uses the bottom center of the box: where the person's feet touch the floor"""
    if len(polygon) < 3:
        return False
    return point_in_polygon((det.x1 + det.x2) / 2.0, float(det.y2), polygon)


def draw_zone(frame, polygon: list[tuple[int, int]], color=(60, 220, 90), alpha: float = 0.08) -> None:
    if len(polygon) < 3:
        return
    pts = np.array(polygon, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)
