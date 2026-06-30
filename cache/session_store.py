"""
session_store.py

SQLite-backed conversation session memory.

Each session holds a list of messages (role + content) forming the
conversation history. Sessions expire after SESSION_TTL_HOURS (default 24h).

The engine injects the session history into the LLM prompt so the model
can refer to earlier turns in the same conversation.

Storage: cache/sessions.db (SQLite, auto-created)
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

_DB_PATH = Path(__file__).parent / "sessions.db"
_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "24"))


@dataclass
class Message:
    role: str    # "user" | "assistant"
    content: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}


class SessionStore:
    """
    Manages conversation sessions with 24h expiry.

    Usage:
        store = SessionStore()
        sid = store.create_session()
        store.append(sid, "user", "My name is Kishore")
        store.append(sid, "assistant", "Nice to meet you, Kishore!")
        history = store.get_history(sid)
        # [{"role": "user", "content": "My name is Kishore"}, ...]
    """

    def __init__(self, db_path: Path = _DB_PATH, ttl_hours: int = _TTL_HOURS) -> None:
        self._conn = self._init_db(db_path)
        self._ttl = ttl_hours
        self._cleanup_expired()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        now = _now()
        expires_at = _future(self._ttl)
        self._conn.execute(
            "INSERT INTO sessions (id, messages, created_at, updated_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, "[]", now, now, expires_at),
        )
        self._conn.commit()
        return session_id

    def get_history(self, session_id: str) -> List[dict]:
        """Return list of message dicts for a session, or [] if not found/expired."""
        row = self._conn.execute(
            "SELECT messages, expires_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return []
        messages_json, expires_at = row
        if _is_expired(expires_at):
            self.delete(session_id)
            return []
        return json.loads(messages_json)

    def append(self, session_id: str, role: str, content: str) -> bool:
        """Append a message to a session. Returns False if session not found."""
        row = self._conn.execute(
            "SELECT messages FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            return False
        messages = json.loads(row[0])
        messages.append(Message(role=role, content=content).to_dict())
        now = _now()
        new_expires = _future(self._ttl)  # reset TTL on activity
        self._conn.execute(
            "UPDATE sessions SET messages = ?, updated_at = ?, expires_at = ? WHERE id = ?",
            (json.dumps(messages), now, new_expires, session_id),
        )
        self._conn.commit()
        return True

    def session_exists(self, session_id: str) -> bool:
        row = self._conn.execute(
            "SELECT expires_at FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            return False
        if _is_expired(row[0]):
            self.delete(session_id)
            return False
        return True

    def clear(self, session_id: str) -> bool:
        """Clear all messages from a session without deleting it."""
        cur = self._conn.execute(
            "UPDATE sessions SET messages = '[]', updated_at = ? WHERE id = ?",
            (_now(), session_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete(self, session_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def list_sessions(self) -> List[dict]:
        rows = self._conn.execute(
            "SELECT id, created_at, updated_at, expires_at, "
            "json_array_length(messages) FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [
            {
                "session_id": r[0],
                "created_at": r[1],
                "updated_at": r[2],
                "expires_at": r[3],
                "message_count": r[4] or 0,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cleanup_expired(self) -> None:
        self._conn.execute(
            "DELETE FROM sessions WHERE expires_at < ?",
            (datetime.now(timezone.utc).isoformat(),),
        )
        self._conn.commit()

    @staticmethod
    def _init_db(path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                messages    TEXT    NOT NULL DEFAULT '[]',
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL,
                expires_at  TEXT    NOT NULL
            )
        """)
        conn.commit()
        return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _future(hours: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _is_expired(expires_at: str) -> bool:
    try:
        exp = datetime.fromisoformat(expires_at)
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) > exp
    except ValueError:
        return True
