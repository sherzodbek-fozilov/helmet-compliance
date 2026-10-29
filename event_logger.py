# sqlite log of runs, violation events and per-run track summaries
from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from stats_manager import RunStats


class EventLogger:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {row[1] for row in rows}

    def _ensure_column(self, conn: sqlite3.Connection, table_name: str, column_name: str, definition: str) -> None:
        if column_name not in self._columns(conn, table_name):
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scenario_name TEXT,
                    video_file TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    confidence_threshold REAL NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    fps REAL,
                    avg_processing_fps REAL DEFAULT 0,
                    use_tracking INTEGER DEFAULT 1,
                    tracker TEXT,
                    processed_frames INTEGER DEFAULT 0,
                    frames_with_detections INTEGER DEFAULT 0,
                    frames_with_violations INTEGER DEFAULT 0,
                    violations_count INTEGER DEFAULT 0,
                    saved_violation_frames INTEGER DEFAULT 0,
                    detections_by_class_json TEXT DEFAULT '{}',
                    avg_confidence_by_class_json TEXT DEFAULT '{}',
                    summary_path TEXT,
                    chart_path TEXT,
                    annotated_video_path TEXT,
                    status TEXT DEFAULT 'running',
                    error_message TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    video_file TEXT NOT NULL,
                    frame_number INTEGER NOT NULL,
                    time_seconds REAL NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    track_id INTEGER,
                    x1 INTEGER NOT NULL,
                    y1 INTEGER NOT NULL,
                    x2 INTEGER NOT NULL,
                    y2 INTEGER NOT NULL,
                    frame_image_path TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tracked_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    first_seen_frame INTEGER NOT NULL,
                    last_seen_frame INTEGER NOT NULL,
                    first_seen_time REAL NOT NULL,
                    last_seen_time REAL NOT NULL,
                    detections_count INTEGER DEFAULT 1,
                    best_confidence REAL DEFAULT 0,
                    violation_detected INTEGER DEFAULT 0,
                    violation_logged INTEGER DEFAULT 0,
                    first_violation_frame INTEGER,
                    first_violation_time REAL,
                    evidence_image_path TEXT,
                    last_bbox_json TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id),
                    UNIQUE(run_id, track_id, class_name)
                )
                """
            )

            # databases created by older versions lack these columns
            self._ensure_column(conn, "runs", "avg_processing_fps", "REAL DEFAULT 0")
            self._ensure_column(conn, "runs", "use_tracking", "INTEGER DEFAULT 1")
            self._ensure_column(conn, "runs", "tracker", "TEXT")
            self._ensure_column(conn, "runs", "annotated_video_path", "TEXT")
            self._ensure_column(conn, "runs", "error_message", "TEXT")
            self._ensure_column(conn, "violations", "track_id", "INTEGER")
            conn.commit()

    def start_run(
        self,
        *,
        video_file: Path | str,
        model_path: Path | str,
        confidence_threshold: float,
        scenario_name: str = "",
        use_tracking: bool = True,
        tracker: str = "bytetrack.yaml",
    ) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO runs (
                    scenario_name, video_file, model_path, confidence_threshold,
                    started_at, use_tracking, tracker, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'running')
                """,
                (
                    scenario_name,
                    str(video_file),
                    str(model_path),
                    confidence_threshold,
                    datetime.now().isoformat(timespec="seconds"),
                    1 if use_tracking else 0,
                    tracker,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def upsert_tracked_entity(
        self,
        *,
        run_id: int,
        track_id: int | None,
        class_name: str,
        frame_number: int,
        time_seconds: float,
        confidence: float,
        bbox: tuple[int, int, int, int],
        is_violation: bool,
        violation_logged: bool = False,
        evidence_image_path: Path | str | None = None,
    ) -> None:
        """first/last appearance per track_id within one run. track ids are
        tracker ids for this video only, not identities across videos."""
        if track_id is None:
            return

        x1, y1, x2, y2 = bbox
        bbox_json = json.dumps({"x1": x1, "y1": y1, "x2": x2, "y2": y2}, ensure_ascii=False)
        now = datetime.now().isoformat(timespec="seconds")
        evidence = str(evidence_image_path) if evidence_image_path else ""

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM tracked_entities
                WHERE run_id = ? AND track_id = ? AND class_name = ?
                """,
                (run_id, track_id, class_name),
            ).fetchone()

            if row is None:
                conn.execute(
                    """
                    INSERT INTO tracked_entities (
                        run_id, track_id, class_name,
                        first_seen_frame, last_seen_frame,
                        first_seen_time, last_seen_time,
                        detections_count, best_confidence,
                        violation_detected, violation_logged,
                        first_violation_frame, first_violation_time,
                        evidence_image_path, last_bbox_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        track_id,
                        class_name,
                        frame_number,
                        frame_number,
                        round(time_seconds, 2),
                        round(time_seconds, 2),
                        confidence,
                        1 if is_violation else 0,
                        1 if violation_logged else 0,
                        frame_number if is_violation else None,
                        round(time_seconds, 2) if is_violation else None,
                        evidence,
                        bbox_json,
                        now,
                    ),
                )
            else:
                first_violation_frame = row["first_violation_frame"]
                first_violation_time = row["first_violation_time"]
                if is_violation and first_violation_frame is None:
                    first_violation_frame = frame_number
                    first_violation_time = round(time_seconds, 2)

                existing_evidence = row["evidence_image_path"] or ""
                if not existing_evidence and evidence:
                    existing_evidence = evidence

                conn.execute(
                    """
                    UPDATE tracked_entities SET
                        last_seen_frame = ?,
                        last_seen_time = ?,
                        detections_count = detections_count + 1,
                        best_confidence = CASE WHEN ? > best_confidence THEN ? ELSE best_confidence END,
                        violation_detected = CASE WHEN ? = 1 THEN 1 ELSE violation_detected END,
                        violation_logged = CASE WHEN ? = 1 THEN 1 ELSE violation_logged END,
                        first_violation_frame = ?,
                        first_violation_time = ?,
                        evidence_image_path = ?,
                        last_bbox_json = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        frame_number,
                        round(time_seconds, 2),
                        confidence,
                        confidence,
                        1 if is_violation else 0,
                        1 if violation_logged else 0,
                        first_violation_frame,
                        first_violation_time,
                        existing_evidence,
                        bbox_json,
                        now,
                        row["id"],
                    ),
                )
            conn.commit()

    def log_violation(
        self,
        *,
        run_id: int,
        video_file: Path | str,
        frame_number: int,
        time_seconds: float,
        class_name: str,
        confidence: float,
        track_id: int | None,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        frame_image_path: Path | str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO violations (
                    run_id, timestamp, video_file, frame_number, time_seconds,
                    class_name, confidence, track_id, x1, y1, x2, y2, frame_image_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now().isoformat(timespec="seconds"),
                    str(video_file),
                    frame_number,
                    round(time_seconds, 2),
                    class_name,
                    confidence,
                    track_id,
                    x1,
                    y1,
                    x2,
                    y2,
                    str(frame_image_path) if frame_image_path else "",
                ),
            )
            conn.commit()

    def finish_run(
        self,
        *,
        run_id: int,
        stats: RunStats,
        fps: float,
        summary_path: Path | str,
        chart_path: Path | str,
        annotated_video_path: Path | str | None = None,
        status: str = "finished",
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE runs SET
                    finished_at = ?,
                    fps = ?,
                    avg_processing_fps = ?,
                    processed_frames = ?,
                    frames_with_detections = ?,
                    frames_with_violations = ?,
                    violations_count = ?,
                    saved_violation_frames = ?,
                    detections_by_class_json = ?,
                    avg_confidence_by_class_json = ?,
                    summary_path = ?,
                    chart_path = ?,
                    annotated_video_path = ?,
                    status = ?
                WHERE id = ?
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    fps,
                    stats.avg_processing_fps(),
                    stats.processed_frames,
                    stats.frames_with_detections,
                    stats.frames_with_violations,
                    stats.violations_count,
                    stats.saved_violation_frames,
                    json.dumps(dict(stats.detections_by_class), ensure_ascii=False),
                    json.dumps(stats.avg_confidence_by_class(), ensure_ascii=False),
                    str(summary_path),
                    str(chart_path),
                    str(annotated_video_path) if annotated_video_path else "",
                    status,
                    run_id,
                ),
            )
            conn.commit()

    def mark_run_failed(self, run_id: int, error_message: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE runs SET finished_at = ?, status = 'failed', error_message = ?
                WHERE id = ?
                """,
                (datetime.now().isoformat(timespec="seconds"), error_message, run_id),
            )
            conn.commit()

    def get_dashboard_stats(self) -> dict[str, Any]:
        """numbers for the dashboard on the main page"""
        with self.connect() as conn:
            # totals
            total_runs = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE status = 'finished'"
            ).fetchone()[0]

            total_violations = conn.execute(
                "SELECT COALESCE(SUM(violations_count), 0) FROM runs WHERE status = 'finished'"
            ).fetchone()[0]

            total_frames = conn.execute(
                "SELECT COALESCE(SUM(processed_frames), 0) FROM runs WHERE status = 'finished'"
            ).fetchone()[0]

            # today, in local time
            today_violations = conn.execute(
                """SELECT COALESCE(SUM(violations_count), 0) FROM runs
                   WHERE status = 'finished'
                   AND date(started_at) = date('now', 'localtime')"""
            ).fetchone()[0]

            # last 7 days for the chart
            week_rows = conn.execute(
                """SELECT date(started_at) as day,
                          COALESCE(SUM(violations_count), 0) as viol,
                          COUNT(*) as runs_count
                   FROM runs
                   WHERE status = 'finished'
                   AND started_at >= date('now', 'localtime', '-6 days')
                   GROUP BY date(started_at)
                   ORDER BY day ASC"""
            ).fetchall()

            # latest events for the feed
            recent_violations = conn.execute(
                """SELECT v.timestamp, v.class_name, v.confidence,
                          v.frame_image_path, v.run_id, v.track_id
                   FROM violations v
                   ORDER BY v.id DESC LIMIT 5"""
            ).fetchall()

            # sessions still running
            live_count = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE status = 'running'"
            ).fetchone()[0]

        # days without runs get zeros
        from datetime import date, timedelta
        days_map = {row["day"]: row for row in week_rows}
        chart_labels = []
        chart_violations = []
        chart_runs = []
        for i in range(6, -1, -1):
            d = (date.today() - timedelta(days=i)).isoformat()
            short = d[5:]  # MM-DD
            chart_labels.append(short)
            row = days_map.get(d)
            chart_violations.append(int(row["viol"]) if row else 0)
            chart_runs.append(int(row["runs_count"]) if row else 0)

        return {
            "total_runs": total_runs,
            "total_violations": total_violations,
            "total_frames": total_frames,
            "today_violations": today_violations,
            "live_count": live_count,
            "chart_labels": chart_labels,
            "chart_violations": chart_violations,
            "chart_runs": chart_runs,
            "recent_violations": [dict(r) for r in recent_violations],
        }

    def get_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_violations(self, run_id: int, limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM violations
                WHERE run_id = ?
                ORDER BY frame_number ASC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_tracked_entities(self, run_id: int, limit: int = 1000) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tracked_entities
                WHERE run_id = ?
                ORDER BY violation_logged DESC, track_id ASC, class_name ASC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def export_violations_csv(self, run_id: int, csv_path: Path | str) -> Path:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.get_violations(run_id, limit=100000)
        fieldnames = [
            "id",
            "run_id",
            "timestamp",
            "video_file",
            "frame_number",
            "time_seconds",
            "class_name",
            "confidence",
            "track_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "frame_image_path",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return csv_path

    def export_tracked_entities_csv(self, run_id: int, csv_path: Path | str) -> Path:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.get_tracked_entities(run_id, limit=100000)
        fieldnames = [
            "id",
            "run_id",
            "track_id",
            "class_name",
            "first_seen_frame",
            "last_seen_frame",
            "first_seen_time",
            "last_seen_time",
            "detections_count",
            "best_confidence",
            "violation_detected",
            "violation_logged",
            "first_violation_frame",
            "first_violation_time",
            "evidence_image_path",
            "last_bbox_json",
            "updated_at",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return csv_path
