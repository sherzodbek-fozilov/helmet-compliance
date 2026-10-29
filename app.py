# operator web ui: video processing, live cameras, exclusion zone, journals
# python app.py -> http://127.0.0.1:5000
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from app_helpers import discover_models
from config import AppConfig
from detector import SafetyDetector
from event_logger import EventLogger
from main import process_video
from person_db import PersonDB
from reid import ReIDExtractor
from streaming import LiveSession, parse_source

PROJECT_ROOT = Path(__file__).resolve().parent

app = Flask(__name__)
config = AppConfig(model_path=PROJECT_ROOT / "models" / "ppe_yolo11s.pt",
                   output_dir=PROJECT_ROOT / "output", input_dir=PROJECT_ROOT / "input_videos")
config.ensure_dirs()
logger = EventLogger(config.db_path)
person_db = PersonDB(config.persons_db_path, config.crops_dir)

_reid: ReIDExtractor | None = None
_reid_failed = False
_reid_lock = threading.Lock()
LIVE: dict[str, LiveSession | None] = {"session": None}
LIVE_LOCK = threading.Lock()


def get_reid() -> ReIDExtractor | None:
    """None if the model can't be built (e.g. no internet on first run): violations are
    still logged, the persons journal just won't group repeat violators"""
    global _reid, _reid_failed
    with _reid_lock:  # first call builds the model, keep it to one thread
        if _reid is None and not _reid_failed:
            try:
                _reid = ReIDExtractor()
            except Exception as exc:
                _reid_failed = True
                app.logger.warning("re-id disabled: %s", exc)
    return _reid


def _error_page(message: str, code: int = 500):
    return render_template("error.html", error_text=message), code


def _start_live(source, is_camera: bool, detector: SafetyDetector, display_mode: str) -> None:
    with LIVE_LOCK:
        if LIVE["session"]:
            LIVE["session"].stop()
        session = LiveSession(source, detector, config, is_camera=is_camera, person_db=person_db,
                              reid_extractor=get_reid(), event_logger=logger, display_mode=display_mode)
        session.start()
        LIVE["session"] = session


@app.route("/")
def index():
    runs = logger.get_runs(limit=20)
    for run in runs:
        run["video_name"] = Path(run["video_file"]).name if run.get("video_file") else "—"
    return render_template("index.html", runs=runs, stats=logger.get_dashboard_stats())


@app.route("/upload", methods=["GET", "POST"])
def upload():
    models = discover_models(PROJECT_ROOT)
    if request.method == "GET":
        return render_template("upload.html", models=models)

    if not models:
        return _error_page("Нет ни одной модели в папке models/. Скачайте веса из Releases.", 400)
    model_path = Path(request.form.get("model_path") or models[0]["path"])
    try:
        conf = float(request.form.get("confidence_threshold") or config.confidence_threshold)
    except ValueError:
        return _error_page("Порог уверенности должен быть числом.", 400)
    use_tracking = request.form.get("use_tracking") == "on"
    tracker = request.form.get("tracker") or "bytetrack.yaml"
    display_mode = request.form.get("display_mode") or "both"

    if request.form.get("mode") == "camera":
        try:
            detector = SafetyDetector(model_path, conf, use_tracking=use_tracking, tracker=tracker)
        except Exception as exc:
            return _error_page(str(exc))
        _start_live(parse_source(request.form.get("camera_source") or "0"), True, detector, display_mode)
        return redirect(url_for("live_view", kind="camera"))

    uploaded = request.files.get("video")
    if not (uploaded and uploaded.filename):
        return _error_page("Видеофайл не выбран.", 400)
    # secure_filename drops cyrillic letters entirely ("<cyrillic>_1.mp4" -> "1.mp4"), so prefix a timestamp
    stem = secure_filename(Path(uploaded.filename).stem) or "video"
    video_path = config.input_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{stem}{Path(uploaded.filename).suffix.lower() or '.mp4'}"
    uploaded.save(video_path)

    if request.form.get("live_preview") == "on":
        try:
            detector = SafetyDetector(model_path, conf, use_tracking=use_tracking, tracker=tracker)
        except Exception as exc:
            return _error_page(str(exc))
        _start_live(str(video_path), False, detector, display_mode)
        return redirect(url_for("live_view", kind="file"))

    try:
        run_id = process_video(video_path=video_path, model_path=model_path, confidence_threshold=conf,
                               save_annotated_video=request.form.get("save_annotated_video") == "on",
                               use_tracking=use_tracking, tracker=tracker, config=config,
                               person_db=person_db, reid_extractor=get_reid())
    except Exception as exc:
        return _error_page(str(exc))
    return redirect(url_for("run_detail", run_id=run_id))


@app.route("/live/<kind>")
def live_view(kind: str):
    return render_template("live.html", kind=kind, label="Камера" if kind == "camera" else "Видеофайл")


@app.route("/live_feed")
def live_feed():
    session = LIVE["session"]
    if session is None:
        return "Нет активного сеанса", 404
    return Response(session.mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/live_stop")
def live_stop():
    with LIVE_LOCK:
        if LIVE["session"]:
            LIVE["session"].stop()
            LIVE["session"] = None
    return redirect(url_for("upload"))


@app.route("/zone_editor")
def zone_editor():
    session = LIVE["session"]
    if session is None:
        return redirect(url_for("upload"))
    zone = session.zone_polygon
    return render_template("zone_editor.html", current_zone=zone, has_zone=len(zone) >= 3,
                           kind="camera" if session.is_camera else "file")


@app.route("/zone_set", methods=["POST"])
def zone_set():
    session = LIVE["session"]
    if session is None:
        return {"ok": False, "error": "Нет активного сеанса"}, 404
    try:
        poly = [(int(p[0]), int(p[1])) for p in (request.get_json(force=True) or {}).get("polygon", [])]
    except (TypeError, ValueError, IndexError) as exc:
        return {"ok": False, "error": str(exc)}, 400
    session.set_zone(poly)
    return {"ok": True, "points": len(poly)}


@app.route("/api/cameras")
def api_cameras():
    try:
        from camera_detect import detect_cameras
        return {"ok": True, "cameras": detect_cameras(max_index=6)}
    except Exception as exc:
        return {"ok": False, "cameras": [], "error": str(exc)}


@app.route("/persons")
def persons_journal():
    date_from, date_to = request.args.get("date_from", ""), request.args.get("date_to", "")
    source_f = request.args.get("source", "")
    persons = person_db.get_persons(date_from=date_from or None, date_to=date_to or None,
                                    source_filter=source_f or None)
    return render_template("persons.html", persons=persons, total=person_db.total_persons(),
                           date_from=date_from, date_to=date_to, source_f=source_f)


@app.route("/persons/<int:person_id>")
def person_detail(person_id: int):
    person = person_db.get_person(person_id)
    if not person:
        return _error_page("Персона не найдена.", 404)
    return render_template("person_detail.html", p=person, events=person_db.get_events(person_id),
                           vectors=person_db.vector_count(person_id))


@app.route("/persons/<int:person_id>/crop")
def person_crop(person_id: int):
    person = person_db.get_person(person_id)
    path = Path(person["crop_image_path"]) if person and person["crop_image_path"] else None
    if path is None or not path.exists():
        return "нет фото", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/runs/<int:run_id>")
def run_detail(run_id: int):
    run = logger.get_run(run_id)
    if not run:
        return _error_page("Запуск не найден.", 404)
    return render_template("run_detail.html", run=run, violations=logger.get_violations(run_id, limit=500),
                           tracked_entities=logger.get_tracked_entities(run_id, limit=1000))


@app.route("/export/<int:run_id>")
def export_csv(run_id: int):
    path = config.reports_dir / f"run_{run_id}_violations.csv"
    logger.export_violations_csv(run_id, path)
    return send_file(path, as_attachment=True, conditional=True)


@app.route("/export_tracks/<int:run_id>")
def export_tracks_csv(run_id: int):
    path = config.reports_dir / f"run_{run_id}_tracked_entities.csv"
    logger.export_tracked_entities_csv(run_id, path)
    return send_file(path, as_attachment=True, conditional=True)


@app.route("/files")
def download_file():
    path = Path(request.args.get("path", "")).resolve()
    if not path.is_relative_to(config.output_dir.resolve()):  # only files from output/
        return "Доступ запрещён", 403
    if not path.exists():
        return "Файл не найден", 404
    return send_file(path, as_attachment=request.args.get("download") == "1", conditional=True)


if __name__ == "__main__":
    # threaded: the mjpeg stream keeps one request open for the whole session
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)
