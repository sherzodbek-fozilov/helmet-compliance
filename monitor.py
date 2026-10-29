# turns per-frame "person without helmet" into violation events.
# used by both the batch pipeline (main.py) and live sessions (streaming.py)
from __future__ import annotations

from dataclasses import dataclass

from detector import Detection
from zones import detection_in_zone


@dataclass(slots=True)
class _Track:
    streak: int = 0       # consecutive frames without a helmet
    reported: bool = False
    last_frame: int = 0


class ViolationMonitor:
    """
    a single frame where the helmet detection dips below the threshold is not a
    violation: a track has to stay without a helmet for min_frames frames in a row.
    each track is reported once. without tracking ids it falls back to a frame
    streak plus a cooldown between events.
    """

    def __init__(self, min_frames: int = 5, fallback_cooldown_frames: int = 50,
                 forget_after: int = 90) -> None:
        self.min_frames = min_frames
        self.fallback_cooldown_frames = fallback_cooldown_frames
        self.forget_after = forget_after
        self.zone: list[tuple[int, int]] = []
        self._tracks: dict[int, _Track] = {}
        self._frame = 0
        self._untracked_streak = 0
        self._last_untracked_event = -10**9

    def set_zone(self, polygon: list[tuple[int, int]]) -> None:
        self.zone = list(polygon)

    def is_excluded(self, det: Detection) -> bool:
        return bool(self.zone) and detection_in_zone(det, self.zone)

    def update(self, detections: list[Detection]) -> list[Detection]:
        """call once per frame; returns persons that just became violation events"""
        self._frame += 1
        events: list[Detection] = []
        untracked_violation: Detection | None = None

        for d in detections:
            if not d.is_person:
                continue
            violating = d.violation and not self.is_excluded(d)
            if d.track_id is None:
                if violating and untracked_violation is None:
                    untracked_violation = d
                continue
            t = self._tracks.setdefault(d.track_id, _Track())
            t.last_frame = self._frame
            t.streak = t.streak + 1 if violating else 0
            if t.streak >= self.min_frames and not t.reported:
                t.reported = True
                events.append(d)

        if untracked_violation is not None:
            self._untracked_streak += 1
            if (self._untracked_streak >= self.min_frames
                    and self._frame - self._last_untracked_event >= self.fallback_cooldown_frames):
                self._last_untracked_event = self._frame
                events.append(untracked_violation)
        else:
            self._untracked_streak = 0

        # bytetrack ids are not reused, so old tracks can be dropped
        stale = [tid for tid, t in self._tracks.items() if self._frame - t.last_frame > self.forget_after]
        for tid in stale:
            del self._tracks[tid]
        return events
