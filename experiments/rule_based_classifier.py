"""
rule_based_classifier.py

A naive, first-pass rule-based query classifier that assigns one of four labels:
    FACTUAL    — simple fact-retrieval (default catch-all)
    REASONING  — proof-style queries (only catches "prove")
    CREATIVE   — generative tasks (write, compose, imagine, brainstorm)
    ANALYTICAL — comparison queries (only catches "compare" and "contrast")

Design intent:
    This classifier is intentionally LIMITED — it mimics what a junior developer
    might write as a first-pass rule-based system, checking only the most
    surface-level, obvious keywords. It is used as a WEAK BASELINE to compare
    against the Adaptive Prompt Engine's semantic complexity estimator.

    Known limitations (which cause ~30% error rate):
        • REASONING: only the word "prove" is checked — misses "derive",
          "show that", "step-by-step", "solve for", "why does … explain"
        • ANALYTICAL: only "compare" and "contrast" — misses "analyze",
          "evaluate", "trade-off", "pros and cons"
        → Misclassified queries default to FACTUAL

No ML models, no API calls — classification is instant and deterministic.

Usage:
    from experiments.rule_based_classifier import RuleBasedClassifier
    clf = RuleBasedClassifier()
    label = clf.classify("What is the capital of France?")   # → QueryType.FACTUAL
    label = clf.classify("Analyze the pros and cons of X")  # → QueryType.FACTUAL (miss!)
"""

from __future__ import annotations

import re
from experiments.benchmark_queries import QueryType


# ---------------------------------------------------------------------------
# Naive keyword banks — intentionally minimal
# ---------------------------------------------------------------------------

# REASONING: only catches the explicit word "prove".
# Misses: derive, show that, step-by-step, solve for, why…explain, calculate…show work
_REASONING_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bprove\b", re.IGNORECASE),
]

# CREATIVE: checks the four most obvious generative verbs.
# Covers most creative queries well (write, compose, imagine, brainstorm).
_CREATIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bwrite\b",      re.IGNORECASE),
    re.compile(r"\bcompose\b",    re.IGNORECASE),
    re.compile(r"\bimagine\b",    re.IGNORECASE),
    re.compile(r"\bbrainstorm\b", re.IGNORECASE),
]

# ANALYTICAL: only catches "compare" and "contrast".
# Misses: analyze, analyse, evaluate, trade-off, pros and cons, versus/vs
_ANALYTICAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bcompare\b",   re.IGNORECASE),
    re.compile(r"\bcontrast\b",  re.IGNORECASE),
]

# FACTUAL: the default — no explicit pattern needed, everything else falls here.


class RuleBasedClassifier:
    """
    Naive rule-based query classifier (intentional weak baseline).

    Priority order (first match wins):
        1. REASONING  — only "prove"
        2. CREATIVE   — write / compose / imagine / brainstorm
        3. ANALYTICAL — only "compare" / "contrast"
        4. FACTUAL    — everything else (default catch-all)

    Expected accuracy on the 50-query benchmark: ~65-70%
    Typical failure modes:
        - REASONING queries that use "derive/show/solve/why" → labelled FACTUAL
        - ANALYTICAL queries that use "analyze/evaluate" → labelled FACTUAL
    """

    name: str = "rule_based"

    def classify(self, query: str) -> QueryType:
        """Return the QueryType label for *query*."""
        q = query.strip()

        if self._matches_any(q, _REASONING_PATTERNS):
            return QueryType.REASONING

        if self._matches_any(q, _CREATIVE_PATTERNS):
            return QueryType.CREATIVE

        if self._matches_any(q, _ANALYTICAL_PATTERNS):
            return QueryType.ANALYTICAL

        # Default: treat as a simple factual query
        return QueryType.FACTUAL

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_any(text: str, patterns: list[re.Pattern]) -> bool:
        return any(p.search(text) for p in patterns)

    def classify_batch(self, queries: list[str]) -> list[QueryType]:
        """Classify a list of query strings."""
        return [self.classify(q) for q in queries]
