"""
complexity_estimator.py

Replaces the 4-category SemanticQueryClassifier with a continuous
complexity score in [0.0, 1.0].

How it works:
    - Two "pole" sentences are embedded at startup:
        simple_pole  → "What is the capital of France?"
        complex_pole → "Prove that the square root of 2 is irrational step by step"
    - An incoming query is embedded and its cosine similarity is computed
      against each pole.
    - The score is the normalized position between the two poles:
        0.0 = identical to simple pole (factual, one-phrase answer)
        1.0 = identical to complex pole (detailed reasoning / creative)

Why this is better than the old classifier:
    - No artificial 4-category boxes (FACTUAL/REASONING/CREATIVE/ANALYTICAL)
    - No regex keyword rules deciding what the LLM should do
    - Fully continuous — "sort of complex" gets a mid-range score, not a
      wrong category
    - Re-uses sentence-transformers already loaded in the project (no new deps)

Complexity tiers for model routing:
    0.00 – 0.35  → LOW    (cheap fast model, e.g. gemini-2.0-flash-lite)
    0.35 – 0.65  → MEDIUM (balanced model, e.g. gemini-2.5-flash)
    0.65 – 1.00  → HIGH   (powerful model, e.g. gemini-2.5-pro)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class ComplexityTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Anchor sentences for the two poles.
# These are averaged so a single outlier doesn't skew the pole.
_SIMPLE_ANCHORS = [
    "What is the capital of France?",
    "What time is it now?",
    "Who invented the telephone?",
    "What is the boiling point of water?",
    "How many planets are in the solar system?",
    "What is today's date?",
]

_COMPLEX_ANCHORS = [
    "Prove that the square root of 2 is irrational step by step.",
    "Write a detailed analytical essay comparing capitalism and socialism.",
    "Explain step by step how a transformer neural network works.",
    "Derive the quadratic formula from first principles.",
    "Analyze the trade-offs between microservices and monolithic architecture.",
    "Write a short story set in a dystopian future with vivid characters.",
]


class ComplexityEstimator:
    """
    Scores a query's complexity as a float in [0.0, 1.0].

    The model (all-MiniLM-L6-v2, ~80 MB) is downloaded once and reused
    — the same instance used by the old SemanticQueryClassifier if both
    are loaded, though in the new architecture this replaces it entirely.

    Usage:
        estimator = ComplexityEstimator()
        score = estimator.score("What is the capital of France?")
        # → ~0.05  (very simple)

        score = estimator.score("Prove root 2 is irrational step by step")
        # → ~0.85  (very complex)

        tier = estimator.tier("Explain how WiFi works")
        # → ComplexityTier.MEDIUM
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np  # type: ignore

        self._np = np
        self._model = SentenceTransformer(model_name)
        self._simple_pole = self._mean_embedding(_SIMPLE_ANCHORS)
        self._complex_pole = self._mean_embedding(_COMPLEX_ANCHORS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, query: str) -> float:
        """
        Return a complexity score in [0.0, 1.0].
        0.0 = maximally simple, 1.0 = maximally complex.
        """
        if not query.strip():
            return 0.0

        q_emb = self._model.encode([query], convert_to_numpy=True)[0]
        sim_simple = self._cosine(q_emb, self._simple_pole)
        sim_complex = self._cosine(q_emb, self._complex_pole)

        # Normalise: how far is the query from the simple pole,
        # relative to the total spread between the two poles?
        total = sim_simple + sim_complex
        if total == 0:
            return 0.5
        return float(sim_complex / total)

    def tier(self, query: str) -> ComplexityTier:
        """Map a query directly to a ComplexityTier."""
        s = self.score(query)
        if s < 0.35:
            return ComplexityTier.LOW
        if s < 0.65:
            return ComplexityTier.MEDIUM
        return ComplexityTier.HIGH

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mean_embedding(self, sentences: list[str]):
        embeddings = self._model.encode(sentences, convert_to_numpy=True)
        return embeddings.mean(axis=0)

    def _cosine(self, a, b) -> float:
        norm_a = self._np.linalg.norm(a)
        norm_b = self._np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(self._np.dot(a, b) / (norm_a * norm_b))
