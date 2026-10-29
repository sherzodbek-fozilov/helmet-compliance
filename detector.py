# yolo wrapper and the helmet rule: a person is a violation when no helmet
# is found in the top part of their box
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

from zones import detection_in_zone

PERSON, HELMET, NO_HELMET = "person", "helmet", "no_helmet"
SHOWN_CLASSES = {PERSON, HELMET, NO_HELMET}  # the model also knows vest/gloves/boots, not used here

RED, GREEN, DARK_GREEN = (0, 0, 255), (0, 200, 0), (40, 160, 60)


@dataclass(slots=True)
class Detection:
    class_name: str  # lowercased model class name
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: int | None = None
    violation: bool = False  # set for persons without a helmet

    @property
    def is_person(self) -> bool:
        return self.class_name == PERSON

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def in_head_zone(person: tuple, box: tuple, head_ratio: float) -> bool:
    """box center lies in the top head_ratio of the person box (x1, y1, x2, y2)"""
    px1, py1, px2, py2 = person
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return px1 <= cx <= px2 and py1 <= cy <= py1 + (py2 - py1) * head_ratio


def person_has_helmet(person: tuple, helmets: Iterable[tuple[tuple, float]], *,
                      head_ratio: float = 0.35, min_conf: float = 0.5,
                      max_area_ratio: float = 0.5) -> bool:
    """helmets: (box, confidence) pairs. shared by the app and tools/eval_violations.py"""
    p_area = max((person[2] - person[0]) * (person[3] - person[1]), 1.0)
    for box, conf in helmets:
        if conf < min_conf:
            continue
        # a "helmet" half the size of the person is a misdetection, not a helmet on a head
        if (box[2] - box[0]) * (box[3] - box[1]) > p_area * max_area_ratio:
            continue
        if in_head_zone(person, box, head_ratio):
            return True
    return False


class SafetyDetector:
    def __init__(self, model_path: Path | str, confidence_threshold: float = 0.45, *,
                 use_tracking: bool = True, tracker: str = "bytetrack.yaml",
                 head_zone_ratio: float = 0.35, helmet_conf_threshold: float = 0.5,
                 helmet_max_area_ratio: float = 0.5) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Не найден файл модели: {model_path}. Скачайте веса из Releases в папку models/.")

        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.model_path = model_path
        self.names = {i: str(n).lower() for i, n in self.model.names.items()}
        missing = {PERSON, HELMET} - set(self.names.values())
        if missing:
            raise ValueError(f"Модель {model_path.name} не знает классы {sorted(missing)}: "
                             "нужна модель, обученная на Construction-PPE (tools/train.py).")

        self.confidence_threshold = confidence_threshold
        self.use_tracking = use_tracking
        self.tracker = tracker
        self.head_zone_ratio = head_zone_ratio
        self.helmet_conf_threshold = helmet_conf_threshold
        self.helmet_max_area_ratio = helmet_max_area_ratio

    def _raw_detections(self, frame) -> list[Detection]:
        # helmets are filtered by their own threshold later, so let them through a bit lower
        conf = min(self.confidence_threshold, self.helmet_conf_threshold)
        if self.use_tracking:
            results = self.model.track(frame, persist=True, tracker=self.tracker, conf=conf, verbose=False)
        else:
            results = self.model(frame, conf=conf, verbose=False)

        out: list[Detection] = []
        for result in results:
            for box in result.boxes:
                name = self.names[int(box.cls[0])]
                if name not in SHOWN_CLASSES:
                    continue
                c = float(box.conf[0])
                if name != HELMET and c < self.confidence_threshold:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                tid = int(box.id[0]) if getattr(box, "id", None) is not None else None
                out.append(Detection(name, c, x1, y1, x2, y2, track_id=tid))
        return out

    def predict(self, frame) -> list[Detection]:
        dets = self._raw_detections(frame)
        helmets = [(d.box, d.confidence) for d in dets if d.class_name == HELMET]
        for d in dets:
            if d.is_person:
                d.violation = not person_has_helmet(
                    d.box, helmets, head_ratio=self.head_zone_ratio,
                    min_conf=self.helmet_conf_threshold, max_area_ratio=self.helmet_max_area_ratio)
        return dets

    # --- drawing. cv2.putText can't render cyrillic, so labels are ascii

    def draw_detections(self, frame, detections: Iterable[Detection], *, display_mode: str = "both",
                        zone_polygon: list[tuple[int, int]] | None = None):
        """display_mode: both / person (people only) / helmet (helmets + violators)"""
        for d in detections:
            if display_mode == "person" and not d.is_person:
                continue
            if display_mode == "helmet" and d.is_person and not d.violation:
                continue

            excluded = bool(zone_polygon) and d.is_person and detection_in_zone(d, zone_polygon)
            if d.is_person:
                color = RED if d.violation and not excluded else DARK_GREEN
            else:
                color = GREEN if d.class_name == HELMET else RED

            label = f"{d.class_name} {d.confidence:.2f}"
            if d.track_id is not None:
                label += f" #{d.track_id}"
            if d.is_person and d.violation:
                label += " (zone)" if excluded else " NO HELMET"
            _draw_box(frame, d.box, color, label)
        return frame

    @staticmethod
    def draw_overlay(frame, *, processed_frames: int, violation_events: int, current_fps: float,
                     source_fps: float, tracking_enabled: bool):
        rows = [
            f"frames {processed_frames} | violations {violation_events}",
            f"fps {current_fps:.1f} (source {source_fps:.1f}) | tracking {'on' if tracking_enabled else 'off'}",
        ]
        cv2.rectangle(frame, (10, 10), (520, 22 + 28 * len(rows)), (20, 20, 20), cv2.FILLED)
        for i, text in enumerate(rows):
            cv2.putText(frame, text, (20, 38 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    @staticmethod
    def resize_for_display(frame, max_width: int = 1280, max_height: int = 720):
        h, w = frame.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        if scale < 1.0:
            return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return frame


def _draw_box(frame, box, color, label: str) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    top = max(0, y1 - th - base - 8)
    cv2.rectangle(frame, (x1, top), (min(frame.shape[1] - 1, x1 + tw + 10), y1), color, cv2.FILLED)
    cv2.putText(frame, label, (x1 + 5, max(th + 4, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
