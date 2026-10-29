# batch processing of a video file (cli + used by the web ui)
# python main.py --video site.mp4 [--show] [--no-track]
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from config import AppConfig
from detector import SafetyDetector
from event_logger import EventLogger
from monitor import ViolationMonitor
from stats_manager import StatsManager


def _video_writer(path: Path, fps: float, width: int, height: int):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else 25.0,
                             (width, height))
    return writer if writer.isOpened() else None


def process_video(*, video_path: Path, model_path: Path, scenario_name: str = "",
                  confidence_threshold: float = 0.45, display: bool = False,
                  save_annotated_video: bool = True, use_tracking: bool = True,
                  tracker: str = "bytetrack.yaml", config: AppConfig | None = None,
                  person_db=None, reid_extractor=None) -> int:
    """processes the whole file, logs events to sqlite and returns the run id"""
    config = config or AppConfig()
    config.ensure_dirs()
    if not video_path.exists():
        raise FileNotFoundError(f"Не найден видеофайл: {video_path}")

    logger = EventLogger(config.db_path)
    run_id = logger.start_run(video_file=video_path, model_path=model_path,
                              confidence_threshold=confidence_threshold, scenario_name=scenario_name,
                              use_tracking=use_tracking, tracker=tracker)
    cap = out = None
    annotated_path: Path | None = None
    stats = StatsManager()
    window = "helmet compliance"

    try:
        detector = SafetyDetector(model_path, confidence_threshold, use_tracking=use_tracking, tracker=tracker)
        monitor = ViolationMonitor(min_frames=config.min_violation_frames)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if save_annotated_video:
            annotated_path = config.annotated_dir / f"run_{run_id}_annotated.mp4"
            out = _video_writer(annotated_path, fps, width, height)
            if out is None:
                annotated_path = None
                print("Не удалось создать VideoWriter, видео с разметкой не будет сохранено.")
        if display:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        print(f"Запуск #{run_id}: {video_path}, порог {confidence_threshold}, "
              f"трекинг {'вкл' if use_tracking else 'выкл'}")

        while True:
            started = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            t_sec = frame_no / fps if fps > 0 else 0.0

            dets = detector.predict(frame)
            events = monitor.update(dets)
            for d in dets:
                stats.register_detection(d.class_name, d.confidence)

            vis = frame.copy()  # keep `frame` clean for person crops
            detector.draw_detections(vis, dets)

            for d in events:
                stats.register_violation_event()
                evidence = ""
                if config.save_violation_frames:
                    evidence = config.violations_dir / f"run_{run_id}_frame_{frame_no}_id{d.track_id or 'na'}.jpg"
                    cv2.imwrite(str(evidence), vis)
                    stats.register_saved_frame()
                logger.log_violation(run_id=run_id, video_file=video_path, frame_number=frame_no,
                                     time_seconds=t_sec, class_name=d.class_name, confidence=d.confidence,
                                     track_id=d.track_id, x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                                     frame_image_path=evidence)
                if person_db is not None:
                    person_db.find_or_create(frame_bgr=frame, person_bbox=d.box, confidence=d.confidence,
                                             source=video_path.name, reid_extractor=reid_extractor,
                                             frame_save_path=str(evidence))

            event_ids = {id(e) for e in events}
            for d in dets:
                if d.is_person:
                    logger.upsert_tracked_entity(
                        run_id=run_id, track_id=d.track_id, class_name=d.class_name, frame_number=frame_no,
                        time_seconds=t_sec, confidence=d.confidence, bbox=d.box,
                        is_violation=d.violation and not monitor.is_excluded(d),
                        violation_logged=id(d) in event_ids)

            cur_fps = 1.0 / max(time.perf_counter() - started, 1e-9)
            stats.register_processing_fps(cur_fps)
            stats.register_frame(bool(dets), any(d.violation for d in dets))
            detector.draw_overlay(vis, processed_frames=stats.stats.processed_frames,
                                  violation_events=stats.stats.violations_count, current_fps=cur_fps,
                                  source_fps=fps, tracking_enabled=use_tracking)
            if out is not None:
                out.write(vis)
            if display:
                cv2.imshow(window, detector.resize_for_display(vis, config.display_max_width,
                                                               config.display_max_height))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Остановлено пользователем.")
                    break
    except Exception as exc:
        logger.mark_run_failed(run_id, str(exc))
        raise
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        if display:
            cv2.destroyAllWindows()

    summary_path = config.reports_dir / f"run_{run_id}_summary.txt"
    chart_path = config.reports_dir / f"run_{run_id}_detections_chart.png"
    stats.save_summary_txt(summary_path, video_path=video_path, fps=fps,
                           confidence_threshold=confidence_threshold, scenario_name=scenario_name,
                           use_tracking=use_tracking, tracker=tracker, annotated_video_path=annotated_path)
    stats.save_chart_png(chart_path)
    logger.finish_run(run_id=run_id, stats=stats.stats, fps=fps, summary_path=summary_path,
                      chart_path=chart_path, annotated_video_path=annotated_path)

    s = stats.stats
    print(f"\nЗапуск #{run_id}: кадров {s.processed_frames}, кадров с нарушениями {s.frames_with_violations}, "
          f"событий {s.violations_count}, средний FPS {s.avg_processing_fps()}")
    print(f"База: {config.db_path}\nСводка: {summary_path}")
    if annotated_path:
        print(f"Видео с разметкой: {annotated_path}")
    return run_id


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Контроль касок по видео")
    p.add_argument("--video", type=Path, required=True, help="видеофайл")
    p.add_argument("--model", type=Path, default=AppConfig().model_path)
    p.add_argument("--scenario", default="", help="метка запуска, например «смена 1»")
    p.add_argument("--conf", type=float, default=0.45)
    p.add_argument("--show", action="store_true", help="окно с обработкой (q для выхода)")
    p.add_argument("--no-save-video", dest="save_video", action="store_false")
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml / botsort.yaml")
    p.add_argument("--no-track", dest="use_tracking", action="store_false")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv if argv is not None else sys.argv[1:])
    process_video(video_path=a.video, model_path=a.model, scenario_name=a.scenario,
                  confidence_threshold=a.conf, display=a.show, save_annotated_video=a.save_video,
                  use_tracking=a.use_tracking, tracker=a.tracker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
