"""
knowledge_base.py

SQLite-backed knowledge base for RAG (Retrieval-Augmented Generation).

How it works:
    1. User uploads a document (text string) via POST /v1/knowledge-base/upload.
    2. The document is split into chunks (by paragraph or fixed size).
    3. Each chunk is embedded with sentence-transformers.
    4. Embeddings stored as binary blobs in SQLite.
    5. On query, top-k most similar chunks are retrieved (vectorized numpy search).
    6. Retrieved chunks are prepended as context in the RAG prompt.

Storage: cache/knowledge_base.db (SQLite, auto-created)
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

_DB_PATH = Path(__file__).parent / "knowledge_base.db"
_CHUNK_SIZE = 400        # words per chunk
_CHUNK_OVERLAP = 50      # words overlap between chunks
_EMBED_MODEL = "all-MiniLM-L6-v2"


@dataclass
class Chunk:
    chunk_id: int
    doc_id: int
    source: str
    text: str
    similarity: float = 0.0


@dataclass
class Document:
    doc_id: int
    source: str
    chunk_count: int
    created_at: str


def _serialize(arr) -> bytes:
    return struct.pack(f"{len(arr)}f", *arr.tolist())


def _deserialize(blob: bytes):
    import numpy as np
    count = len(blob) // 4
    return np.array(struct.unpack(f"{count}f", blob), dtype=np.float32)


class KnowledgeBase:
    """
    Stores and retrieves document chunks for RAG.

    Usage:
        kb = KnowledgeBase()
        doc_id = kb.add_document("The Eiffel Tower is 330m tall.", source="facts.txt")
        chunks = kb.search("How tall is the Eiffel Tower?", k=3)
        kb.delete_document(doc_id)
    """

    def __init__(
        self,
        db_path: Path = _DB_PATH,
        embed_model: str = _EMBED_MODEL,
    ) -> None:
        self._conn = self._init_db(db_path)
        self._embed_model_name = embed_model
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(self, text: str, source: str = "upload") -> int:
        """Chunk, embed, and store a document. Returns doc_id."""
        chunks = self._chunk_text(text)
        cur = self._conn.execute(
            "INSERT INTO documents (source, chunk_count, created_at) VALUES (?, ?, ?)",
            (source, len(chunks), _now()),
        )
        doc_id = cur.lastrowid

        for chunk_text in chunks:
            emb = self._embed(chunk_text)
            blob = _serialize(emb)
            self._conn.execute(
                "INSERT INTO chunks (doc_id, text, embedding) VALUES (?, ?, ?)",
                (doc_id, chunk_text, blob),
            )
        self._conn.commit()
        return doc_id

    def search(self, query: str, k: int = 3) -> List[Chunk]:
        """Return top-k most relevant chunks for a query."""
        import numpy as np

        q_emb = self._embed(query)
        rows = self._conn.execute(
            """SELECT c.id, c.doc_id, d.source, c.text, c.embedding
               FROM chunks c JOIN documents d ON c.doc_id = d.id"""
        ).fetchall()

        if not rows:
            return []

        ids, doc_ids, sources, texts, blobs = zip(*rows)
        matrix = np.array([_deserialize(b) for b in blobs], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            return []
        sims = (matrix @ q_emb) / (norms * q_norm + 1e-10)

        top_k_idx = np.argsort(sims)[::-1][:k]
        return [
            Chunk(
                chunk_id=ids[i],
                doc_id=doc_ids[i],
                source=sources[i],
                text=texts[i],
                similarity=float(sims[i]),
            )
            for i in top_k_idx
        ]

    def has_documents(self) -> bool:
        row = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return (row[0] or 0) > 0

    def list_documents(self) -> List[Document]:
        rows = self._conn.execute(
            "SELECT id, source, chunk_count, created_at FROM documents ORDER BY id DESC"
        ).fetchall()
        return [Document(doc_id=r[0], source=r[1], chunk_count=r[2], created_at=r[3]) for r in rows]

    def delete_document(self, doc_id: int) -> bool:
        self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        cur = self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def document_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] or 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str) -> List[str]:
        """Split text into overlapping word-level chunks."""
        words = text.split()
        if len(words) <= _CHUNK_SIZE:
            return [text]
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + _CHUNK_SIZE, len(words))
            chunks.append(" ".join(words[start:end]))
            start += _CHUNK_SIZE - _CHUNK_OVERLAP
        return chunks

    def _embed(self, text: str):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self._embed_model_name)
        return self._model.encode([text], convert_to_numpy=True)[0].astype("float32")

    @staticmethod
    def _init_db(path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT    NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                created_at  TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                text      TEXT    NOT NULL,
                embedding BLOB    NOT NULL
            )
        """)
        conn.commit()
        return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
