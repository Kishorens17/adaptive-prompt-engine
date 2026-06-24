"""
semantic_cache.py

Semantic response cache backed by SQLite.

How it works:
    1. Every incoming query is embedded with sentence-transformers.
    2. The embedding is compared (cosine similarity) against all stored
       query embeddings in cache.db.
    3. If the best match exceeds the similarity threshold (default 0.98),
       the stored answer is returned immediately — zero LLM tokens spent.
    4. On a cache miss, the result from the LLM is stored for future use.

Why 0.98 threshold?
    Queries that differ only in a key detail (e.g. year "2025" vs "2026",
    or a number) embed very close to each other (~0.96–0.97 cosine sim).
    0.98 ensures only true paraphrases/rewrites of the exact same question
    are served from cache, not near-duplicates with different answers.
    Configurable via CACHE_SIMILARITY_THRESHOLD in .env.

Storage:
    cache/cache.db  (SQLite, created automatically on first use)
    Embeddings stored as binary blobs (numpy array → bytes).
"""

from __future__ import annotations

import os
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Default threshold — queries with cosine similarity above this are
# considered "the same question" and served from cache.
DEFAULT_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.98"))

_DB_PATH = Path(__file__).parent / "cache.db"


def _serialize(arr) -> bytes:
    """Convert a 1-D numpy float32 array to bytes for SQLite storage."""
    return struct.pack(f"{len(arr)}f", *arr.tolist())


def _deserialize(blob: bytes):
    """Restore a numpy float32 array from bytes."""
    import numpy as np  # type: ignore
    count = len(blob) // 4
    return np.array(struct.unpack(f"{count}f", blob), dtype=np.float32)


def _cosine(a, b) -> float:
    import numpy as np  # type: ignore
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SemanticCache:
    """
    SQLite-backed semantic cache for LLM responses.

    Usage:
        cache = SemanticCache()
        hit = cache.get("What is the capital of France?")
        if hit:
            print(hit)  # "Paris"
        else:
            answer = llm.complete(query)
            cache.put(query, answer, model="gemini-2.0-flash-lite", cost_usd=0.000048)
    """

    def __init__(
        self,
        db_path: Path = _DB_PATH,
        threshold: float = DEFAULT_THRESHOLD,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.threshold = threshold
        self._db_path = db_path
        self._conn = self._init_db(db_path)
        # Lazily loaded — only import sentence-transformers if cache is used.
        self._model = None
        self._model_name = model_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> Optional[str]:
        """
        Return a cached answer if a semantically similar query was seen before,
        otherwise return None. Also increments hit_count for the matched entry.
        """
        q_emb = self._embed(query)
        rows = self._conn.execute(
            "SELECT id, query_embedding, answer FROM cache_entries"
        ).fetchall()
        best_id, best_score, best_answer = None, -1.0, None
        for row_id, blob, answer in rows:
            stored_emb = _deserialize(blob)
            score = _cosine(q_emb, stored_emb)
            if score > best_score:
                best_score, best_id, best_answer = score, row_id, answer

        if best_score >= self.threshold and best_id is not None:
            self._conn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE id = ?",
                (best_id,),
            )
            self._conn.commit()
            return best_answer
        return None

    def put(
        self,
        query: str,
        answer: str,
        model: str = "unknown",
        cost_usd: float = 0.0,
    ) -> None:
        """Store a query-answer pair with its embedding."""
        q_emb = self._embed(query)
        blob = _serialize(q_emb)
        self._conn.execute(
            """INSERT INTO cache_entries
               (query_text, query_embedding, answer, model_used, cost_usd, timestamp, hit_count)
               VALUES (?, ?, ?, ?, ?, ?, 0)""",
            (query, blob, answer, model, cost_usd, _now()),
        )
        self._conn.commit()

    def stats(self) -> dict:
        """Return basic cache statistics."""
        row = self._conn.execute(
            "SELECT COUNT(*), SUM(hit_count) FROM cache_entries"
        ).fetchone()
        total_entries = row[0] or 0
        total_hits = row[1] or 0
        return {"total_entries": total_entries, "total_hits": total_hits}

    def clear(self) -> None:
        """Remove all cached entries."""
        self._conn.execute("DELETE FROM cache_entries")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self._model_name)
        return self._model.encode([text], convert_to_numpy=True)[0].astype("float32")

    @staticmethod
    def _init_db(path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text     TEXT    NOT NULL,
                query_embedding BLOB   NOT NULL,
                answer         TEXT    NOT NULL,
                model_used     TEXT    DEFAULT 'unknown',
                cost_usd       REAL    DEFAULT 0.0,
                timestamp      TEXT    NOT NULL,
                hit_count      INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
