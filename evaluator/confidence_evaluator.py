"""
confidence_evaluator.py

The decision engine inside each Handler. Takes an LLM response (and the
original query, for length-relative scoring) and returns a float
confidence score in [0.0, 1.0]. If the score is below the configured
threshold (default 0.75), the owning Handler passes the query to the
next, stronger strategy in the chain.

Per the project design, this is intentionally rule-based rather than a
trained model: it combines three lightweight, fully explainable
signals so the scoring logic can be inspected and justified in the
accompanying research paper (no black-box reward model needed for a
project at this scope).

Signals combined:
    1. Uncertainty language detection — hedging phrases lower confidence.
    2. Response length relative to query complexity — a one-sentence
       answer to a clearly multi-step query is flagged as under-answered.
    3. Self-reported confidence tag — if the LLM was instructed to
       append [CONFIDENCE: x.xx], that value is parsed and blended in
       directly as the strongest signal (it's the model's own estimate).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

HEDGE_PHRASES: List[str] = [
    "i think",
    "i believe",
    "possibly",
    "perhaps",
    "i'm not sure",
    "i am not sure",
    "it might be",
    "it could be",
    "not certain",
    "i guess",
    "may be",
    "maybe",
    "unclear",
    "hard to say",
    "i'm uncertain",
    "to some extent",
    "as far as i know",
]

# Query patterns that suggest the question expects a longer, multi-step
# answer. If the response is suspiciously short relative to one of
# these, the length signal penalizes confidence.
COMPLEX_QUERY_PATTERNS = [
    r"\bprove\b",
    r"\bexplain why\b",
    r"\bcompare\b",
    r"\bcontrast\b",
    r"\banalyz", 
    r"\bstep[s]? by step\b",
    r"\bwalk me through\b",
    r"\bderive\b",
    r"\bwhat are the (?:differences|implications|consequences)\b",
]

CONFIDENCE_TAG_RE = re.compile(r"\[CONFIDENCE:\s*([0-9]*\.?[0-9]+)\s*\]", re.IGNORECASE)


@dataclass
class EvaluatorWeights:
    """Relative weight given to each signal when no self-reported tag is present."""

    hedge_weight: float = 0.5
    length_weight: float = 0.5


@dataclass
class ConfidenceEvaluator:
    """
    Rule-based confidence scorer.

    score() always returns a float clamped to [0.0, 1.0]. Each
    sub-signal is computed independently so they can be unit tested
    and reasoned about separately (see tests/test_evaluator.py-style
    coverage inside test_strategies.py).
    """

    threshold: float = 0.75
    weights: EvaluatorWeights = field(default_factory=EvaluatorWeights)
    min_expected_words_simple: int = 5
    min_expected_words_complex: int = 25

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, query: str, response_text: str) -> float:
        self_reported = self._parse_self_reported_confidence(response_text)
        if self_reported is not None:
            # The model's own stated confidence is the strongest signal
            # available (when present), but we still nudge it slightly
            # using the hedge-language signal so a response that says
            # "[CONFIDENCE: 0.9]" while also being full of hedging
            # phrases doesn't get a free pass.
            hedge_score = self._hedge_score(response_text)
            return self._clamp(0.85 * self_reported + 0.15 * hedge_score)

        hedge_score = self._hedge_score(response_text)
        length_score = self._length_score(query, response_text)
        blended = (
            self.weights.hedge_weight * hedge_score
            + self.weights.length_weight * length_score
        )
        return self._clamp(blended)

    def passes_threshold(self, query: str, response_text: str) -> bool:
        return self.score(query, response_text) >= self.threshold

    # ------------------------------------------------------------------
    # Individual signals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_self_reported_confidence(response_text: str) -> float | None:
        match = CONFIDENCE_TAG_RE.search(response_text)
        if not match:
            return None
        try:
            value = float(match.group(1))
        except ValueError:
            return None
        return max(0.0, min(1.0, value))

    @staticmethod
    def _hedge_score(response_text: str) -> float:
        """
        Returns a score in [0.0, 1.0] where 1.0 means "no hedging detected"
        and lower values mean more hedging language was found. Each hedge
        phrase found reduces the score, with diminishing impact per extra
        hedge so a single hedge doesn't tank the score disproportionately.
        """
        text_lower = response_text.lower()
        hedge_count = sum(1 for phrase in HEDGE_PHRASES if phrase in text_lower)
        if hedge_count == 0:
            return 1.0
        # Each additional hedge phrase costs less than the previous one.
        penalty = 1.0 - (1.0 / (1.0 + hedge_count))
        return max(0.0, 1.0 - penalty * 1.1)

    def _length_score(self, query: str, response_text: str) -> float:
        """
        Returns a score in [0.0, 1.0] reflecting whether the response
        length is adequate given the apparent complexity of the query.
        A query matching a "complex" pattern (prove, compare, explain
        why, ...) is expected to need a longer response; a short
        response to such a query is penalized.
        """
        word_count = len(response_text.split())
        is_complex_query = self._looks_complex(query)
        expected_min = (
            self.min_expected_words_complex if is_complex_query else self.min_expected_words_simple
        )

        if word_count >= expected_min:
            return 1.0
        if expected_min == 0:
            return 1.0
        return max(0.0, word_count / expected_min)

    @staticmethod
    def _looks_complex(query: str) -> bool:
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in COMPLEX_QUERY_PATTERNS)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))
