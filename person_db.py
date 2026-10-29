# repeat violator journal: each violation event is matched against people seen in the last
# SEARCH_DAYS days by their re-id embeddings (reid.py)
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

# picked on market-1501 train identities so that ~5% of new people get merged into an existing
# profile when the journal holds 50 people (tools/eval_reid.py). raise it to merge less often
MATCH_THRESHOLD = 0.80
VECTORS_PER_PERSON = 3  # last few embeddings, often from different cameras: +12 pp found vs keeping one
SEARCH_DAYS = 30
HEAD_SHOULDERS = 0.55   # photo shown in the journal; the embedding uses the full person box


class PersonDB:
    def __init__(self, db_path: Path, crops_dir: Path) -> None:
        self.db_path = db_path
        self.crops_dir = crops_dir
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS persons (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                first_seen      TEXT NOT NULL,
                last_seen       TEXT NOT NULL,
                source          TEXT NOT NULL,
                violation_count INTEGER DEFAULT 1,
                best_confidence REAL DEFAULT 0,
                crop_image_path TEXT
            );
            CREATE TABLE IF NOT EXISTS person_vectors (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id  INTEGER NOT NULL REFERENCES persons(id),
                vector     BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vectors_person ON person_vectors(person_id);
            CREATE TABLE IF NOT EXISTS person_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id   INTEGER NOT NULL REFERENCES persons(id),
                event_time  TEXT NOT NULL,
                source      TEXT NOT NULL,
                confidence  REAL,
                frame_path  TEXT,
                crop_path   TEXT
            );
            """)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def find_or_create(self, *, frame_bgr: np.ndarray, person_bbox: tuple[int, int, int, int],
                       confidence: float, source: str, reid_extractor=None,
                       event_time: datetime | None = None, frame_save_path: str = "") -> int:
        """frame_bgr must be the clean frame: boxes drawn on it would end up in the embedding"""
        when = event_time or datetime.now()
        ts = when.strftime("%Y-%m-%d %H:%M:%S")
        photo = _crop(frame_bgr, person_bbox, HEAD_SHOULDERS)
        body = _crop(frame_bgr, person_bbox, 1.0)
        vector = reid_extractor.extract(body) if (reid_extractor is not None and body is not None) else None
        crop_path = self._save_crop(photo, when) if photo is not None else ""
        person_id = self._search_similar(vector) if vector is not None else None

        with self._conn() as conn:
            if person_id is not None:
                # keep the photo from the most confident detection
                conn.execute("""
                    UPDATE persons SET
                        last_seen = ?,
                        violation_count = violation_count + 1,
                        crop_image_path = CASE WHEN ? > best_confidence THEN ? ELSE crop_image_path END,
                        best_confidence = MAX(best_confidence, ?)
                    WHERE id = ?""", (ts, confidence, crop_path, confidence, person_id))
            else:
                cur = conn.execute("""
                    INSERT INTO persons (first_seen, last_seen, source, violation_count,
                                         best_confidence, crop_image_path)
                    VALUES (?, ?, ?, 1, ?, ?)""", (ts, ts, source, confidence, crop_path))
                person_id = cur.lastrowid
            if vector is not None:
                conn.execute("INSERT INTO person_vectors (person_id, vector) VALUES (?, ?)",
                             (person_id, vector.astype(np.float32).tobytes()))
                conn.execute("""
                    DELETE FROM person_vectors WHERE person_id = ? AND id NOT IN (
                        SELECT id FROM person_vectors WHERE person_id = ? ORDER BY id DESC LIMIT ?)""",
                             (person_id, person_id, VECTORS_PER_PERSON))
            conn.execute("""
                INSERT INTO person_events (person_id, event_time, source, confidence, frame_path, crop_path)
                VALUES (?, ?, ?, ?, ?, ?)""", (person_id, ts, source, confidence, frame_save_path, crop_path))
        return person_id

    def _search_similar(self, vector: np.ndarray) -> int | None:
        """best match over all stored vectors of people seen recently, None below the threshold"""
        since = (datetime.now() - timedelta(days=SEARCH_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT v.person_id, v.vector FROM person_vectors v JOIN persons p ON p.id = v.person_id
                WHERE p.last_seen >= ?""", (since,)).fetchall()
        if not rows:
            return None
        mat = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
        if mat.shape[1] != vector.shape[0]:  # vectors from another model
            return None
        sims = mat @ vector  # both l2-normalised
        best = int(np.argmax(sims))
        return int(rows[best]["person_id"]) if sims[best] >= MATCH_THRESHOLD else None

    def get_persons(self, *, date_from: str | None = None, date_to: str | None = None,
                    source_filter: str | None = None, limit: int = 200) -> list[sqlite3.Row]:
        q, params = "SELECT * FROM persons WHERE 1=1", []
        if date_from:
            q += " AND last_seen >= ?"
            params.append(date_from + " 00:00:00")
        if date_to:
            q += " AND last_seen <= ?"
            params.append(date_to + " 23:59:59")
        if source_filter:
            q += " AND source LIKE ?"
            params.append(f"%{source_filter}%")
        q += " ORDER BY last_seen DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            return conn.execute(q, params).fetchall()

    def get_person(self, person_id: int) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()

    def get_events(self, person_id: int, limit: int = 50) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute("SELECT * FROM person_events WHERE person_id = ? ORDER BY event_time DESC LIMIT ?",
                                (person_id, limit)).fetchall()

    def vector_count(self, person_id: int) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM person_vectors WHERE person_id = ?", (person_id,)).fetchone()[0]

    def total_persons(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]

    def _save_crop(self, crop: np.ndarray, when: datetime) -> str:
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        # random suffix: two violators in the same second would otherwise share a file name
        path = self.crops_dir / f"person_{when:%Y-%m-%d_%H-%M-%S}_{uuid.uuid4().hex[:6]}.jpg"
        cv2.imwrite(str(path), crop)
        return str(path)


def _crop(frame: np.ndarray, bbox: tuple[int, int, int, int], top_part: float) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    y2 = min(frame.shape[0], y1 + int((y2 - y1) * top_part))
    crop = frame[max(0, y1):y2, max(0, x1):min(frame.shape[1], x2)]
    return crop if crop.size else None
