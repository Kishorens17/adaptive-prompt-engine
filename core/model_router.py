"""
model_router.py

Maps a ComplexityTier to the most cost-effective Gemini model for that tier,
and tracks the cost per 1K tokens for each model.

Why this matters:
    A simple "what time is it?" routed to gemini-2.5-pro wastes ~50× the
    tokens it needs to. This router ensures simple queries go to the cheapest
    fast model, and only genuinely complex queries reach the powerful (and
    expensive) models.

Configuration (via .env):
    ROUTER_LOW_MODEL_GEMINI    = gemini-2.0-flash-lite   (default)
    ROUTER_MEDIUM_MODEL_GEMINI = gemini-2.5-flash         (default)
    ROUTER_HIGH_MODEL_GEMINI   = gemini-2.5-pro           (default)

Budget override (CLI --budget flag):
    low      → always use LOW tier model regardless of complexity
    balanced → use LOW/MEDIUM only (cap at MEDIUM even for high complexity)
    quality  → always use HIGH tier model (ignore complexity score)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from core.complexity_estimator import ComplexityTier


# ---------------------------------------------------------------------------
# Cost table (USD per 1K tokens, input+output blended estimate)
# Updated: June 2025 — verify against provider pricing pages
# ---------------------------------------------------------------------------
_COST_PER_1K: dict[str, float] = {
    # Gemini models
    "gemini-2.0-flash-lite": 0.000_075,
    "gemini-2.5-flash":      0.000_375,
    "gemini-2.5-pro":        0.003_750,
    # Mock / local
    "mock-model":            0.0,
}

# Baseline cost for "cost saved" calculation — what you'd pay if you
# always used the most expensive model.
_BASELINE_MODEL: dict[str, str] = {
    "gemini": "gemini-2.5-pro",
    "mock":   "mock-model",
}


@dataclass
class RoutingDecision:
    model: str
    tier: ComplexityTier
    cost_per_1k: float
    baseline_model: str
    baseline_cost_per_1k: float


class ModelRouter:
    """
    Given a provider name and a ComplexityTier, returns the optimal model
    and its cost metadata.

    Usage:
        router = ModelRouter(provider="gemini")
        decision = router.route(ComplexityTier.LOW)
        # decision.model → "gemini-2.0-flash-lite"
        # decision.cost_per_1k → 0.000075
    """

    def __init__(self, provider: str = "gemini", budget: str = "balanced") -> None:
        """
        Args:
            provider: "gemini", "openai", or "mock"
            budget:   "low"      → always use LOW model
                      "balanced" → use LOW/MEDIUM, cap at MEDIUM (default)
                      "quality"  → always use HIGH model
        """
        self.provider = provider.lower()
        self.budget = budget.lower()
        self._tier_map = self._build_tier_map()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, tier: ComplexityTier) -> RoutingDecision:
        """Return the model and cost info for a given complexity tier."""
        effective_tier = self._apply_budget(tier)
        model = self._tier_map[effective_tier]
        baseline = _BASELINE_MODEL.get(self.provider, model)
        return RoutingDecision(
            model=model,
            tier=effective_tier,
            cost_per_1k=_COST_PER_1K.get(model, 0.0),
            baseline_model=baseline,
            baseline_cost_per_1k=_COST_PER_1K.get(baseline, 0.0),
        )

    def estimate_cost(self, model: str, total_tokens: int) -> float:
        """Compute actual cost in USD for a given model and token count."""
        rate = _COST_PER_1K.get(model, 0.0)
        return rate * total_tokens / 1000.0

    def estimate_saved(self, model: str, baseline_model: str, total_tokens: int) -> float:
        """Compute how much was saved vs. using the baseline model."""
        baseline_rate = _COST_PER_1K.get(baseline_model, 0.0)
        actual_rate = _COST_PER_1K.get(model, 0.0)
        return max(0.0, (baseline_rate - actual_rate) * total_tokens / 1000.0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_tier_map(self) -> dict[ComplexityTier, str]:
        p = self.provider
        if p == "mock":
            return {t: "mock-model" for t in ComplexityTier}

        defaults: dict[str, dict[str, str]] = {
            "gemini": {
                "low":    "gemini-2.5-flash",   # flash-lite has tight free-tier quotas
                "medium": "gemini-2.5-flash",
                "high":   "gemini-2.5-pro",
            },
        }
        d = defaults.get(p, defaults["gemini"])

        return {
            ComplexityTier.LOW:    os.getenv(f"ROUTER_LOW_MODEL_{p.upper()}",    d["low"]),
            ComplexityTier.MEDIUM: os.getenv(f"ROUTER_MEDIUM_MODEL_{p.upper()}", d["medium"]),
            ComplexityTier.HIGH:   os.getenv(f"ROUTER_HIGH_MODEL_{p.upper()}",   d["high"]),
        }

    def _apply_budget(self, tier: ComplexityTier) -> ComplexityTier:
        if self.budget == "low":
            return ComplexityTier.LOW
        if self.budget == "balanced" and tier == ComplexityTier.HIGH:
            return ComplexityTier.MEDIUM
        if self.budget == "quality":
            return ComplexityTier.HIGH
        return tier
