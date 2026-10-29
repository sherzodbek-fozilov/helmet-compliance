# app settings; cli flags in main.py and the web form override some of them
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    model_path: Path = Path("models/ppe_yolo11s.pt")
    output_dir: Path = Path("output")
    input_dir: Path = Path("input_videos")

    confidence_threshold: float = 0.45
    use_tracking: bool = True
    tracker: str = "bytetrack.yaml"

    # a person has to be without a helmet this many frames in a row to become an event
    min_violation_frames: int = 5

    save_violation_frames: bool = True
    save_annotated_video: bool = True

    display_max_width: int = 1280
    display_max_height: int = 720

    db_name: str = "safety_control.sqlite3"

    @property
    def violations_dir(self) -> Path:
        return self.output_dir / "violations"

    @property
    def crops_dir(self) -> Path:
        return self.violations_dir / "crops"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def annotated_dir(self) -> Path:
        return self.output_dir / "annotated"

    @property
    def db_path(self) -> Path:
        return self.output_dir / self.db_name

    @property
    def persons_db_path(self) -> Path:
        return self.output_dir / "persons.sqlite3"

    def ensure_dirs(self) -> None:
        for d in (self.output_dir, self.input_dir, self.violations_dir, self.crops_dir,
                  self.reports_dir, self.annotated_dir):
            d.mkdir(parents=True, exist_ok=True)
