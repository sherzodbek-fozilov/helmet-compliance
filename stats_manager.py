# per-run statistics, summary txt and a detections chart
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RunStats:
    processed_frames: int = 0
    frames_with_detections: int = 0
    frames_with_violations: int = 0
    # violation events, not frames: with tracking each person is counted once
    violations_count: int = 0
    saved_violation_frames: int = 0
    detections_by_class: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    confidences: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    processing_fps_values: list[float] = field(default_factory=list)

    def avg_confidence_by_class(self) -> dict[str, float]:
        return {
            class_name: round(sum(values) / len(values), 3) if values else 0.0
            for class_name, values in self.confidences.items()
        }

    def avg_processing_fps(self) -> float:
        if not self.processing_fps_values:
            return 0.0
        return round(sum(self.processing_fps_values) / len(self.processing_fps_values), 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed_frames": self.processed_frames,
            "frames_with_detections": self.frames_with_detections,
            "frames_with_violations": self.frames_with_violations,
            "violations_count": self.violations_count,
            "saved_violation_frames": self.saved_violation_frames,
            "detections_by_class": dict(self.detections_by_class),
            "avg_confidence_by_class": self.avg_confidence_by_class(),
            "avg_processing_fps": self.avg_processing_fps(),
        }


class StatsManager:
    def __init__(self) -> None:
        self.stats = RunStats()

    def register_frame(self, has_detection: bool, has_violation: bool) -> None:
        self.stats.processed_frames += 1
        if has_detection:
            self.stats.frames_with_detections += 1
        if has_violation:
            self.stats.frames_with_violations += 1

    def register_detection(self, class_name: str, confidence: float) -> None:
        self.stats.detections_by_class[class_name] += 1
        self.stats.confidences[class_name].append(confidence)

    def register_violation_event(self) -> None:
        self.stats.violations_count += 1

    def register_saved_frame(self) -> None:
        self.stats.saved_violation_frames += 1

    def register_processing_fps(self, value: float) -> None:
        if value > 0:
            self.stats.processing_fps_values.append(value)

    def save_summary_txt(
        self,
        path: Path,
        *,
        video_path: Path | str,
        fps: float,
        confidence_threshold: float,
        scenario_name: str = "",
        use_tracking: bool = True,
        tracker: str = "bytetrack.yaml",
        annotated_video_path: Path | str | None = None,
    ) -> None:
        summary = self.stats.to_dict()
        with path.open("w", encoding="utf-8") as file:
            file.write("Итоговая статистика: контроль касок\n")
            file.write("====================================\n")
            if scenario_name:
                file.write(f"Сценарий: {scenario_name}\n")
            file.write(f"Видео: {video_path}\n")
            file.write(f"FPS видеоисточника: {round(fps, 2) if fps > 0 else 'не определен'}\n")
            file.write(f"Средний FPS обработки: {summary['avg_processing_fps']}\n")
            file.write(f"Порог уверенности: {confidence_threshold}\n")
            file.write(f"Трекинг: {'включен' if use_tracking else 'выключен'}\n")
            if use_tracking:
                file.write(f"Трекер: {tracker}\n")
            if annotated_video_path:
                file.write(f"Аннотированное видео: {annotated_video_path}\n")
            file.write(f"Обработано кадров: {summary['processed_frames']}\n")
            file.write(f"Кадров с детекциями: {summary['frames_with_detections']}\n")
            file.write(f"Кадров с нарушениями: {summary['frames_with_violations']}\n")
            file.write(f"Зафиксировано событий нарушения (человек без каски): {summary['violations_count']}\n")
            if use_tracking:
                file.write("Правило учёта: нарушение засчитывается, если человек без каски несколько кадров подряд; один трек считается один раз.\n")
            file.write(f"Сохранено кадров с нарушениями: {summary['saved_violation_frames']}\n")
            file.write("\nКоличество детекций по классам:\n")
            for class_name, count in summary["detections_by_class"].items():
                file.write(f"- {class_name}: {count}\n")
            file.write("\nСредняя уверенность по классам:\n")
            for class_name, avg_conf in summary["avg_confidence_by_class"].items():
                file.write(f"- {class_name}: {avg_conf}\n")
            file.write("\nJSON для отчета/презентации:\n")
            file.write(json.dumps(summary, ensure_ascii=False, indent=2))

    def save_chart_png(self, path: Path) -> None:
        """bar chart of detections per class"""
        try:
            import matplotlib
            matplotlib.use("Agg")  # runs inside flask threads; the tk backend crashes there on windows
            import matplotlib.pyplot as plt
        except Exception:
            return

        data = dict(self.stats.detections_by_class)
        if not data:
            data = {"No detections": 0}

        labels = list(data.keys())
        values = list(data.values())

        plt.figure(figsize=(8, 4.5))
        plt.bar(labels, values)
        plt.title("Количество детекций по классам")
        plt.xlabel("Класс")
        plt.ylabel("Количество")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
