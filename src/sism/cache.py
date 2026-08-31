"""Content-addressed on-disk cache so re-runs and crashes cost nothing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class ResponseCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._db = self.root / "cache.sqlite3"
        self._lock = threading.Lock()
        with self._connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS entries "
                "(k TEXT PRIMARY KEY, v TEXT NOT NULL, ts REAL DEFAULT (unixepoch()))"
            )

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, k: str) -> dict | None:
        with self._lock, self._connect() as con:
            row = con.execute("SELECT v FROM entries WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, v: dict) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO entries (k, v) VALUES (?, ?)",
                (k, json.dumps(v, ensure_ascii=False)),
            )

    def __len__(self) -> int:
        with self._lock, self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
