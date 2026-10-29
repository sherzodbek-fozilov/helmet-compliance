# live processing of a camera / rtsp stream / file, streamed to the browser as mjpeg
# (multipart/x-mixed-replace, shown in a plain <img> tag)
from __future__ import annotations

import threading
import time
from datetime import datetime

import cv2
import numpy as np

from config import AppConfig
from detector import SafetyDetector
from event_logger import EventLogger
from monitor import ViolationMonitor
from person_db import PersonDB
from reid import ReIDExtractor
from stats_manager import RunStats
from zones import draw_zone


def parse_source(source: str):
    """"0" -> webcam index, anything else (rtsp://, http://, path) as is"""
    s = (source or "").strip()
    return int(s) if s.isdigit() else s


class LiveSession:
    def __init__(self, source, detector: SafetyDetector, config: AppConfig, *, is_camera: bool = True,
                 person_db: PersonDB | None = None, reid_extractor: ReIDExtractor | None = None,
                 event_logger: EventLogger | None = None, display_mode: str = "both") -> None:
        self.source = source
        self.detector = detector
        self.config = config
        self.is_camera = is_camera
        self.person_db = person_db
        self.reid_extractor = reid_extractor
        self.event_logger = event_logger
        self.display_mode = display_mode
        self.monitor = ViolationMonitor(min_frames=config.min_violation_frames)

        self.running = False
        self.thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self.run_id: int | None = None

        self.processed_frames = 0
        self.violation_frames = 0
        self.violation_events = 0
        self.frames_with_detections = 0
        self.saved_frames = 0

    @property
    def zone_polygon(self) -> list[tuple[int, int]]:
        return self.monitor.zone

    def set_zone(self, polygon: list[tuple[int, int]]) -> None:
        with self._lock:
            self.monitor.set_zone(polygon)

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    @property
    def _source_label(self) -> str:
        return "camera" if self.is_camera else str(self.source)

    def _loop(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self._render_error(f"Не удалось открыть источник: {self.source}")
            self.running = False
            return
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if self.event_logger is not None:
            self.run_id = self.event_logger.start_run(
                video_file=self._source_label, model_path=self.detector.model_path,
                confidence_threshold=self.detector.confidence_threshold, scenario_name="live",
                use_tracking=self.detector.use_tracking, tracker=self.detector.tracker)
        fps_values: list[float] = []

        while self.running:
            started = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            dets = self.detector.predict(frame)
            with self._lock:
                events = self.monitor.update(dets)
                zone = list(self.monitor.zone)

            vis = frame.copy()  # crops for re-id are taken from the clean frame
            self.detector.draw_detections(vis, dets, display_mode=self.display_mode, zone_polygon=zone)
            for d in events:
                self.violation_events += 1
                self._save_event(frame, vis, d, source_fps)

            self.processed_frames += 1
            self.frames_with_detections += bool(dets)
            self.violation_frames += any(d.violation for d in dets)

            cur_fps = 1.0 / max(time.perf_counter() - started, 1e-9)
            fps_values.append(cur_fps)
            self.detector.draw_overlay(vis, processed_frames=self.processed_frames,
                                       violation_events=self.violation_events, current_fps=cur_fps,
                                       source_fps=source_fps, tracking_enabled=self.detector.use_tracking)
            draw_zone(vis, zone)
            ok_enc, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok_enc:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()

        cap.release()
        self.running = False
        if self.event_logger is not None and self.run_id is not None:
            stats = RunStats(processed_frames=self.processed_frames,
                             frames_with_detections=self.frames_with_detections,
                             frames_with_violations=self.violation_frames,
                             violations_count=self.violation_events,
                             saved_violation_frames=self.saved_frames)
            stats.processing_fps_values = fps_values
            self.event_logger.finish_run(run_id=self.run_id, stats=stats, fps=source_fps, summary_path="",
                                         chart_path="", annotated_video_path=None)

    def _save_event(self, frame, vis, det, source_fps: float) -> None:
        ts = datetime.now()
        path = ""
        if self.config.save_violation_frames:
            self.config.ensure_dirs()
            full = self.config.violations_dir / f"live_{ts:%Y%m%d_%H%M%S_%f}_id{det.track_id or 'na'}.jpg"
            cv2.imwrite(str(full), vis)
            path = str(full)
            self.saved_frames += 1

        if self.event_logger is not None and self.run_id is not None:
            self.event_logger.log_violation(
                run_id=self.run_id, video_file=self._source_label, frame_number=self.processed_frames,
                time_seconds=self.processed_frames / source_fps if source_fps > 0 else 0.0,
                class_name=det.class_name, confidence=det.confidence, track_id=det.track_id,
                x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2, frame_image_path=path)

        if self.person_db is not None:
            self.person_db.find_or_create(frame_bgr=frame, person_bbox=det.box, confidence=det.confidence,
                                          source=self._source_label, reid_extractor=self.reid_extractor,
                                          event_time=ts, frame_save_path=path)

    def _render_error(self, message: str) -> None:
        img = np.zeros((360, 640, 3), dtype="uint8")
        cv2.putText(img, "ERROR", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        # cyrillic would render as ????, show the source instead of the message
        cv2.putText(img, f"cannot open source: {self.source}"[:60], (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def mjpeg_generator(self):
        deadline = time.time() + 10  # wait for the first frame
        while self.get_jpeg() is None and time.time() < deadline:
            time.sleep(0.05)
        while True:
            jpeg = self.get_jpeg()
            if jpeg is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            if not self.running:  # file finished: the last frame was sent above
                break
            time.sleep(0.03)
