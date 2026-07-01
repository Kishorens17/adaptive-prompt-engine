"""
llm_judge.py

LLM-as-Judge confidence evaluator.

Replaces the rule-based ConfidenceEvaluator (hedge-word counting, word-length
checks) with a self-evaluation loop where the LLM reasons about the quality
of its own answer using a structured rubric.

How it works:
    1. After the primary LLM generates an answer, the judge is called with
       both the original query and the answer.
    2. The judge LLM returns a JSON object rating four dimensions (0–10):
          accuracy, completeness, relevance, clarity
       plus an overall score (0.0–1.0) and one-sentence reasoning.
    3. `overall` becomes the confidence score fed back into the escalation logic.

Judge caching:
    Results are cached in SQLite by hash(query + answer) so the same
    query+answer pair is never judged twice (saves API calls).

Fallback:
    If the judge LLM call fails or returns invalid JSON, the old
    rule-based ConfidenceEvaluator is used transparently — zero regressions.

Judge model:
    Always uses the fastest/cheapest model regardless of main query routing:
        Gemini  → gemini-2.5-flash
        OpenAI  → gpt-4o-mini
        Groq    → llama-3.3-70b-versatile
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from evaluator.confidence_evaluator import ConfidenceEvaluator

_DB_PATH = Path(__file__).parent / "judge_cache.db"

_JUDGE_FAST_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "groq":   "llama-3.3-70b-versatile",
    "mock":   "mock-model",
}

_JUDGE_PROMPT_TEMPLATE = """\
You are an expert answer quality evaluator. Evaluate the AI-generated answer below.

Query: {query}

Answer: {answer}

Rate the answer on these four dimensions (integer 0–10):
- accuracy:     Is the information factually correct?
- completeness: Does it fully answer what was asked?
- relevance:    Does it stay on topic without irrelevant content?
- clarity:      Is it well-structured, clear, and easy to understand?

Also provide:
- overall: A single quality score from 0.0 to 1.0 (float)
- reasoning: One concise sentence explaining the most important strength or weakness

Respond with ONLY valid JSON, no markdown, no extra text:
{{"accuracy": <int>, "completeness": <int>, "relevance": <int>, "clarity": <int>, "overall": <float>, "reasoning": "<string>"}}"""


@dataclass
class JudgeResult:
    accuracy: int
    completeness: int
    relevance: int
    clarity: int
    overall: float
    reasoning: str
    from_cache: bool = False

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "relevance": self.relevance,
            "clarity": self.clarity,
            "overall": round(self.overall, 3),
            "reasoning": self.reasoning,
        }


class LLMJudgeEvaluator:
    """
    LLM-based confidence evaluator with the same interface as ConfidenceEvaluator.

    Drop-in replacement — any code that calls .score(query, response_text)
    and reads .threshold will work unchanged.
    """

    def __init__(
        self,
        llm_client,
        threshold: float = 0.75,
        db_path: Path = _DB_PATH,
        enabled: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.threshold = threshold
        self.enabled = enabled
        self._fallback = ConfidenceEvaluator(threshold=threshold)
        self._conn = self._init_db(db_path)
        self._judge_model = _JUDGE_FAST_MODELS.get(llm_client.provider, "mock-model")

    # ------------------------------------------------------------------
    # Public API (same interface as ConfidenceEvaluator)
    # ------------------------------------------------------------------

    def score(self, query: str, response_text: str) -> float:
        """Return a confidence score in [0.0, 1.0]."""
        if not self.enabled or self.llm_client.provider == "mock":
            return self._fallback.score(query, response_text)

        result = self._judge(query, response_text)
        return result.overall

    def score_detailed(self, query: str, response_text: str) -> Optional[JudgeResult]:
        """Return full JudgeResult with all quality dimensions, or None on failure."""
        if not self.enabled or self.llm_client.provider == "mock":
            return None
        return self._judge(query, response_text)

    def passes_threshold(self, query: str, response_text: str) -> bool:
        return self.score(query, response_text) >= self.threshold

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _judge(self, query: str, response_text: str) -> JudgeResult:
        """Call judge LLM (or return from cache). Falls back to rule-based on error."""
        cache_key = _hash(query + response_text)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            prompt = _JUDGE_PROMPT_TEMPLATE.format(
                query=query.strip(),
                answer=response_text.strip()[:2000],  # cap to avoid huge prompts
            )
            response = self.llm_client.complete(
                prompt=prompt,
                temperature=0.0,  # deterministic judging
                model=self._judge_model,
            )
            result = self._parse_judge_response(response.text)
            self._store_cached(cache_key, result)
            return result
        except Exception:  # noqa: BLE001 — judge must never crash the engine
            fallback_score = self._fallback.score(query, response_text)
            return JudgeResult(
                accuracy=5, completeness=5, relevance=5, clarity=5,
                overall=fallback_score, reasoning="Fallback: rule-based score used."
            )

    @staticmethod
    def _parse_judge_response(text: str) -> JudgeResult:
        """Parse JSON from judge LLM response. Raises ValueError on bad JSON."""
        # Strip any markdown code fences
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        data = json.loads(text)
        return JudgeResult(
            accuracy=int(data.get("accuracy", 5)),
            completeness=int(data.get("completeness", 5)),
            relevance=int(data.get("relevance", 5)),
            clarity=int(data.get("clarity", 5)),
            overall=float(data.get("overall", 0.7)),
            reasoning=str(data.get("reasoning", "")),
        )

    def _get_cached(self, key: str) -> Optional[JudgeResult]:
        row = self._conn.execute(
            "SELECT accuracy, completeness, relevance, clarity, overall, reasoning "
            "FROM judge_cache WHERE key = ?",
            (key,),
        ).fetchone()
        if not row:
            return None
        return JudgeResult(
            accuracy=row[0], completeness=row[1],
            relevance=row[2], clarity=row[3],
            overall=row[4], reasoning=row[5], from_cache=True,
        )

    def _store_cached(self, key: str, result: JudgeResult) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO judge_cache
               (key, accuracy, completeness, relevance, clarity, overall, reasoning, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (key, result.accuracy, result.completeness, result.relevance,
             result.clarity, result.overall, result.reasoning,
             datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    @staticmethod
    def _init_db(path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS judge_cache (
                key          TEXT PRIMARY KEY,
                accuracy     INTEGER,
                completeness INTEGER,
                relevance    INTEGER,
                clarity      INTEGER,
                overall      REAL,
                reasoning    TEXT,
                timestamp    TEXT
            )
        """)
        conn.commit()
        return conn


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
