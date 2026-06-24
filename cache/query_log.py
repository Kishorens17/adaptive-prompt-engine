"""
query_log.py

Append-only audit log of every query processed by the engine.

Every entry records:
    - The original query text
    - Complexity tier (LOW / MEDIUM / HIGH)
    - Model actually used
    - Token counts (input + output)
    - Actual cost in USD
    - Cost saved vs. always-using-the-baseline-model
    - Latency in milliseconds
    - Whether the result came from the semantic cache
    - Confidence score
    - Timestamp (UTC ISO-8601)

Storage: cache/query_log.db (SQLite, created automatically)

Uses:
    1. Analytics dashboard  — charts, cost savings, cache hit rate
    2. Audit trail          — who asked what, when, at what cost
    3. Fine-tuning export   — high-confidence Q&A pairs as training data
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DB_PATH = Path(__file__).parent / "query_log.db"


@dataclass
class LogEntry:
    query: str
    complexity_tier: str          # "low" | "medium" | "high"
    model_used: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_saved_usd: float
    latency_ms: float
    cache_hit: bool
    confidence: float
    timestamp: str = ""           # filled automatically if empty

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class QueryLogger:
    """
    Thin write-once log for every engine response.

    Usage:
        logger = QueryLogger()
        logger.log(LogEntry(
            query="What is the capital of France?",
            complexity_tier="low",
            model_used="gemini-2.0-flash-lite",
            input_tokens=38,
            output_tokens=3,
            cost_usd=0.000003,
            cost_saved_usd=0.000147,
            latency_ms=210.0,
            cache_hit=False,
            confidence=0.98,
        ))

        rows = logger.recent(limit=20)
        stats = logger.aggregate_stats()
    """

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self._conn = self._init_db(db_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log(self, entry: LogEntry) -> None:
        self._conn.execute(
            """INSERT INTO query_log
               (query, complexity_tier, model_used, input_tokens, output_tokens,
                cost_usd, cost_saved_usd, latency_ms, cache_hit, confidence, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.query,
                entry.complexity_tier,
                entry.model_used,
                entry.input_tokens,
                entry.output_tokens,
                entry.cost_usd,
                entry.cost_saved_usd,
                entry.latency_ms,
                1 if entry.cache_hit else 0,
                entry.confidence,
                entry.timestamp,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def recent(self, limit: int = 50) -> list[dict]:
        """Return the most recent `limit` log entries as dicts."""
        rows = self._conn.execute(
            """SELECT query, complexity_tier, model_used, input_tokens, output_tokens,
                      cost_usd, cost_saved_usd, latency_ms, cache_hit, confidence, timestamp
               FROM query_log
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        keys = [
            "query", "complexity_tier", "model_used", "input_tokens", "output_tokens",
            "cost_usd", "cost_saved_usd", "latency_ms", "cache_hit", "confidence", "timestamp",
        ]
        return [dict(zip(keys, row)) for row in rows]

    def aggregate_stats(self) -> dict:
        """Return aggregate statistics for the analytics dashboard."""
        row = self._conn.execute(
            """SELECT
                COUNT(*)                         AS total_queries,
                SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) AS cache_hits,
                SUM(input_tokens + output_tokens) AS total_tokens,
                SUM(cost_usd)                    AS total_cost_usd,
                SUM(cost_saved_usd)              AS total_saved_usd,
                AVG(latency_ms)                  AS avg_latency_ms,
                AVG(confidence)                  AS avg_confidence
               FROM query_log"""
        ).fetchone()
        if not row or row[0] == 0:
            return {
                "total_queries": 0, "cache_hits": 0, "total_tokens": 0,
                "total_cost_usd": 0.0, "total_saved_usd": 0.0,
                "avg_latency_ms": 0.0, "avg_confidence": 0.0,
                "cache_hit_rate": 0.0,
            }
        total_queries = row[0] or 0
        cache_hits = row[1] or 0
        return {
            "total_queries": total_queries,
            "cache_hits": cache_hits,
            "total_tokens": row[2] or 0,
            "total_cost_usd": round(row[3] or 0.0, 6),
            "total_saved_usd": round(row[4] or 0.0, 6),
            "avg_latency_ms": round(row[5] or 0.0, 1),
            "avg_confidence": round(row[6] or 0.0, 3),
            "cache_hit_rate": round(cache_hits / max(total_queries, 1), 3),
        }

    def daily_token_usage(self, days: int = 7) -> list[dict]:
        """Token usage per day for the last N days (for the dashboard chart)."""
        rows = self._conn.execute(
            """SELECT DATE(timestamp) AS day,
                      SUM(input_tokens + output_tokens) AS tokens,
                      SUM(cost_usd) AS cost_usd
               FROM query_log
               WHERE timestamp >= DATE('now', ? || ' days')
               GROUP BY day
               ORDER BY day""",
            (f"-{days}",),
        ).fetchall()
        return [{"day": r[0], "tokens": r[1] or 0, "cost_usd": round(r[2] or 0.0, 6)} for r in rows]

    def model_distribution(self) -> list[dict]:
        """How many queries went to each model tier."""
        rows = self._conn.execute(
            """SELECT complexity_tier, COUNT(*) AS count
               FROM query_log
               GROUP BY complexity_tier"""
        ).fetchall()
        return [{"tier": r[0], "count": r[1]} for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _init_db(path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                query           TEXT    NOT NULL,
                complexity_tier TEXT    DEFAULT 'unknown',
                model_used      TEXT    DEFAULT 'unknown',
                input_tokens    INTEGER DEFAULT 0,
                output_tokens   INTEGER DEFAULT 0,
                cost_usd        REAL    DEFAULT 0.0,
                cost_saved_usd  REAL    DEFAULT 0.0,
                latency_ms      REAL    DEFAULT 0.0,
                cache_hit       INTEGER DEFAULT 0,
                confidence      REAL    DEFAULT 0.0,
                timestamp       TEXT    NOT NULL
            )
        """)
        conn.commit()
        return conn
