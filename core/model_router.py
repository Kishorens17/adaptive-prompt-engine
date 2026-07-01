"""
model_router.py

Maps a ComplexityTier to the most cost-effective model for that tier,
across all supported providers (Gemini, OpenAI, Groq).

Cost table is updated June 2025 — verify against provider pricing pages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from core.complexity_estimator import ComplexityTier


# ---------------------------------------------------------------------------
# Cost table (USD per 1K tokens, input+output blended estimate)
# ---------------------------------------------------------------------------
_COST_PER_1K: dict[str, float] = {
    # Gemini
    "gemini-2.0-flash-lite": 0.000_075,
    "gemini-2.5-flash":      0.000_375,
    "gemini-2.5-pro":        0.003_750,
    # OpenAI
    "gpt-4o-mini":           0.000_150,
    "gpt-4o":                0.002_500,
    # Groq (very cheap inference)
    "llama-3.3-70b-versatile": 0.000_059,
    "mixtral-8x7b-32768":    0.000_027,
    # NVIDIA
    "nvidia/nemotron-3-ultra-550b-a55b": 0.0,
    "deepseek-ai/deepseek-r1-0528": 0.0,
    "mistralai/mistral-medium-3-128k": 0.0,
    "qwen/qwen3.5-122b-a10b": 0.0,
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": 0.0,
    # Mock / local
    "mock-model":            0.0,
}

# Baseline model per provider for "cost saved" calculation
_BASELINE_MODEL: dict[str, str] = {
    "gemini": "gemini-2.5-pro",
    "openai": "gpt-4o",
    "groq":   "llama-3.3-70b-versatile",
    "nvidia": "mistralai/mistral-medium-3-128k",
    "mock":   "mock-model",
}

# Default tier → model mapping per provider
_TIER_DEFAULTS: dict[str, dict[str, str]] = {
    "gemini": {
        "low":    "gemini-2.5-flash",
        "medium": "gemini-2.5-flash",
        "high":   "gemini-2.5-pro",
    },
    "openai": {
        "low":    "gpt-4o-mini",
        "medium": "gpt-4o-mini",
        "high":   "gpt-4o",
    },
    "groq": {
        "low":    "llama-3.3-70b-versatile",
        "medium": "llama-3.3-70b-versatile",
        "high":   "llama-3.3-70b-versatile",
    },
    "nvidia": {
        "low":    "mistralai/mistral-medium-3-128k",
        "medium": "qwen/qwen3.5-122b-a10b",
        "high":   "deepseek-ai/deepseek-r1-0528",
    },
    "mock": {
        "low":    "mock-model",
        "medium": "mock-model",
        "high":   "mock-model",
    },
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
    Given a provider and ComplexityTier, returns the optimal model.

    Usage:
        router = ModelRouter(provider="groq", budget="balanced")
        decision = router.route(ComplexityTier.LOW)
        # decision.model → "llama-3.3-70b-versatile"
    """

    def __init__(self, provider: str = "gemini", budget: str = "balanced") -> None:
        self.provider = provider.lower()
        self.budget = budget.lower()
        self._tier_map = self._build_tier_map()

    def route(self, tier: ComplexityTier) -> RoutingDecision:
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
        return _COST_PER_1K.get(model, 0.0) * total_tokens / 1000.0

    def estimate_saved(
        self, model: str, baseline_model: str, total_tokens: int
    ) -> float:
        baseline_rate = _COST_PER_1K.get(baseline_model, 0.0)
        actual_rate = _COST_PER_1K.get(model, 0.0)
        return max(0.0, (baseline_rate - actual_rate) * total_tokens / 1000.0)

    def _build_tier_map(self) -> dict[ComplexityTier, str]:
        p = self.provider
        defaults = _TIER_DEFAULTS.get(p, _TIER_DEFAULTS["gemini"])
        env_prefix = p.upper()
        return {
            ComplexityTier.LOW:    os.getenv(f"ROUTER_LOW_MODEL_{env_prefix}",    defaults["low"]),
            ComplexityTier.MEDIUM: os.getenv(f"ROUTER_MEDIUM_MODEL_{env_prefix}", defaults["medium"]),
            ComplexityTier.HIGH:   os.getenv(f"ROUTER_HIGH_MODEL_{env_prefix}",   defaults["high"]),
        }

    def _apply_budget(self, tier: ComplexityTier) -> ComplexityTier:
        if self.budget == "low":
            return ComplexityTier.LOW
        if self.budget == "balanced" and tier == ComplexityTier.HIGH:
            return ComplexityTier.MEDIUM
        if self.budget == "quality":
            return ComplexityTier.HIGH
        return tier
