"""SQLite 事件状态存储。"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class EventStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    usgs_id TEXT,
                    source TEXT,
                    lon REAL,
                    lat REAL,
                    depth REAL,
                    mag REAL,
                    mag_type TEXT,
                    category TEXT,
                    place TEXT,
                    mainshock_utc TEXT,
                    detected_at TEXT,
                    status TEXT,
                    t1_t2_deadline TEXT,
                    t3_deadline TEXT,
                    output_dir TEXT,
                    reminded_t12_urgent INTEGER DEFAULT 0,
                    reminded_t3_prep INTEGER DEFAULT 0,
                    reminded_t3_urgent INTEGER DEFAULT 0,
                    refreshed_t2 INTEGER DEFAULT 0,
                    refreshed_t3 INTEGER DEFAULT 0,
                    competition_eligible INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT,
                    kind TEXT,
                    sent_at TEXT,
                    UNIQUE(event_id, kind)
                );
                """
            )
            cols = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
            if "competition_eligible" not in cols:
                conn.execute(
                    "ALTER TABLE events ADD COLUMN competition_eligible INTEGER DEFAULT 0"
                )

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_active_events(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM events
                WHERE status NOT IN ('done')
                ORDER BY mainshock_utc DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def insert_event(self, ev: Dict[str, Any]) -> bool:
        """插入新事件；已存在则返回 False。"""
        if self.get_event(ev["event_id"]):
            return False

        now = datetime.now(timezone.utc).isoformat()
        ms = ev["mainshock_utc"]
        if isinstance(ms, datetime):
            ms_iso = ms.astimezone(timezone.utc).isoformat()
        else:
            ms_iso = str(ms)

        ms_dt = ev["mainshock_utc"]
        if not isinstance(ms_dt, datetime):
            ms_dt = datetime.fromisoformat(ms_iso.replace("Z", "+00:00"))

        t12 = (ms_dt + timedelta(hours=24)).isoformat()
        t3 = (ms_dt + timedelta(hours=72)).isoformat()

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO events (
                    event_id, usgs_id, source, lon, lat, depth, mag, mag_type,
                    category, place, mainshock_utc, detected_at, status,
                    t1_t2_deadline, t3_deadline, output_dir, competition_eligible
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ev["event_id"],
                    ev.get("usgs_id"),
                    ev.get("source", "USGS"),
                    ev["lon"],
                    ev["lat"],
                    ev["depth"],
                    ev["mag"],
                    ev["mag_type"],
                    ev.get("category", "submit"),
                    ev.get("place", ""),
                    ms_iso,
                    now,
                    "new",
                    t12,
                    t3,
                    ev.get("output_dir", ""),
                    1 if ev.get("competition_eligible") else 0,
                ),
            )
        return True

    def update_status(self, event_id: str, status: str, output_dir: str = "") -> None:
        with self._conn() as conn:
            if output_dir:
                conn.execute(
                    "UPDATE events SET status = ?, output_dir = ? WHERE event_id = ?",
                    (status, output_dir, event_id),
                )
            else:
                conn.execute(
                    "UPDATE events SET status = ? WHERE event_id = ?",
                    (status, event_id),
                )

    def set_flag(self, event_id: str, column: str) -> None:
        allowed = {
            "reminded_t12_urgent",
            "reminded_t3_prep",
            "reminded_t3_urgent",
            "refreshed_t2",
            "refreshed_t3",
        }
        if column not in allowed:
            raise ValueError(column)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE events SET {column} = 1 WHERE event_id = ?",
                (event_id,),
            )

    def reminder_sent(self, event_id: str, kind: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM reminders WHERE event_id = ? AND kind = ?",
                (event_id, kind),
            ).fetchone()
        return row is not None

    def mark_reminder(self, event_id: str, kind: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO reminders (event_id, kind, sent_at)
                VALUES (?, ?, ?)
                """,
                (event_id, kind, now),
            )
