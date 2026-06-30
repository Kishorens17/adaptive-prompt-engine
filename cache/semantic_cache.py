"""
semantic_cache.py

Semantic response cache backed by SQLite with vectorized numpy lookups.

Improvement over v1:
    Instead of iterating rows one-by-one and computing cosine similarity
    sequentially (O(N) with high Python overhead), all stored embeddings
    are loaded into a numpy matrix and similarity is computed in a single
    batched matrix operation (O(N) but fully vectorized — ~100x faster).

    For production scale (>100K entries), replace numpy ops with FAISS:
        import faiss
        index = faiss.IndexFlatIP(384)
        index.add(embeddings_matrix)
        D, I = index.search(query_emb.reshape(1, -1), k=1)

How it works:
    1. Query is embedded with sentence-transformers (all-MiniLM-L6-v2).
    2. All stored embeddings loaded as numpy matrix on cache init.
    3. Single matrix-vector cosine similarity computed.
    4. Best match above threshold (default 0.92) → return cached answer.
    5. Miss → LLM call, then store new embedding.

Storage: cache/cache.db (SQLite, auto-created)
"""

from __future__ import annotations

import os
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92"))
_DB_PATH = Path(__file__).parent / "cache.db"


def _serialize(arr) -> bytes:
    return struct.pack(f"{len(arr)}f", *arr.tolist())


def _deserialize(blob: bytes):
    import numpy as np  # type: ignore
    count = len(blob) // 4
    return np.array(struct.unpack(f"{count}f", blob), dtype=np.float32)


class SemanticCache:
    """
    SQLite-backed semantic cache with vectorized numpy similarity search.

    Usage:
        cache = SemanticCache()
        hit = cache.get("What is the capital of France?")
        if hit:
            return hit
        answer = llm.complete(query)
        cache.put(query, answer, model="gemini-2.5-flash", cost_usd=0.000012)
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
        self._model = None
        self._model_name = model_name
        # In-memory numpy matrix: shape (N, dim) — rebuilt on init + updated on put()
        self._emb_matrix = None   # numpy array or None
        self._emb_ids: list[int] = []
        self._emb_answers: list[str] = []
        self._index_built = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> Optional[str]:
        """Return cached answer if a semantically similar query was seen before."""
        q_emb = self._embed(query)
        self._ensure_index()

        if self._emb_matrix is None or len(self._emb_ids) == 0:
            return None

        import numpy as np  # type: ignore
        # Vectorized cosine similarity: (N, dim) @ (dim,) = (N,)
        norms = np.linalg.norm(self._emb_matrix, axis=1)
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            return None
        sims = (self._emb_matrix @ q_emb) / (norms * q_norm + 1e-10)

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= self.threshold:
            row_id = self._emb_ids[best_idx]
            self._conn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE id = ?",
                (row_id,),
            )
            self._conn.commit()
            return self._emb_answers[best_idx]
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
        cur = self._conn.execute(
            """INSERT INTO cache_entries
               (query_text, query_embedding, answer, model_used, cost_usd, timestamp, hit_count)
               VALUES (?, ?, ?, ?, ?, ?, 0)""",
            (query, blob, answer, model, cost_usd, _now()),
        )
        self._conn.commit()

        # Update in-memory index incrementally (no full rebuild)
        import numpy as np  # type: ignore
        self._emb_ids.append(cur.lastrowid)
        self._emb_answers.append(answer)
        new_emb = q_emb.reshape(1, -1).astype(np.float32)
        if self._emb_matrix is None:
            self._emb_matrix = new_emb
        else:
            self._emb_matrix = np.vstack([self._emb_matrix, new_emb])

    def stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*), SUM(hit_count) FROM cache_entries"
        ).fetchone()
        return {
            "total_entries": row[0] or 0,
            "total_hits": row[1] or 0,
        }

    def clear(self) -> None:
        self._conn.execute("DELETE FROM cache_entries")
        self._conn.commit()
        self._emb_matrix = None
        self._emb_ids = []
        self._emb_answers = []
        self._index_built = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """Load all embeddings into numpy matrix on first use."""
        if self._index_built:
            return
        import numpy as np  # type: ignore
        rows = self._conn.execute(
            "SELECT id, query_embedding, answer FROM cache_entries"
        ).fetchall()
        if not rows:
            self._index_built = True
            return
        ids, blobs, answers = zip(*rows)
        embs = np.array([_deserialize(b) for b in blobs], dtype=np.float32)
        self._emb_matrix = embs
        self._emb_ids = list(ids)
        self._emb_answers = list(answers)
        self._index_built = True

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
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text      TEXT    NOT NULL,
                query_embedding BLOB    NOT NULL,
                answer          TEXT    NOT NULL,
                model_used      TEXT    DEFAULT 'unknown',
                cost_usd        REAL    DEFAULT 0.0,
                timestamp       TEXT    NOT NULL,
                hit_count       INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
