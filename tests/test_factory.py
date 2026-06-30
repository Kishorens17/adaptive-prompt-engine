"""
test_factory.py → renamed: test_engine.py (test_factory.py kept for compatibility)

Tests for the new AdaptivePromptEngine (replaces old factory/handler chain tests).
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from main import AdaptivePromptEngine, EngineResult


@pytest.fixture
def engine() -> AdaptivePromptEngine:
    return AdaptivePromptEngine(provider="mock", use_cache=False)


class TestAdaptivePromptEngine:
    def test_ask_returns_engine_result(self, engine):
        result = engine.ask("What is the capital of France?")
        assert isinstance(result, EngineResult)

    def test_answer_is_non_empty(self, engine):
        result = engine.ask("What is 2 + 2?")
        assert len(result.answer.strip()) > 0

    def test_confidence_in_range(self, engine):
        result = engine.ask("Explain recursion.")
        assert 0.0 <= result.confidence <= 1.0

    def test_model_used_is_mock(self, engine):
        result = engine.ask("What is the capital of France?")
        assert result.model_used == "mock-model"

    def test_cache_hit_is_false_when_disabled(self, engine):
        result = engine.ask("What time is it?")
        assert result.cache_hit is False

    def test_complexity_tier_set(self, engine):
        result = engine.ask("What is Python?")
        assert result.complexity_tier in ("low", "medium", "high")

    def test_ask_and_format_strips_confidence_tag(self, engine):
        out = engine.ask_and_format("What is Paris?")
        assert "[CONFIDENCE:" not in out

    def test_verbose_format_includes_metadata(self):
        e = AdaptivePromptEngine(provider="mock", verbose=True, use_cache=False)
        out = e.ask_and_format("What is 2+2?")
        assert "Model:" in out
        assert "Tokens:" in out

    def test_budget_low_uses_mock_model(self):
        e = AdaptivePromptEngine(provider="mock", budget="low", use_cache=False)
        result = e.ask("Complex query: prove something.")
        assert result.model_used == "mock-model"

    def test_empty_query_raises(self, engine):
        with pytest.raises(ValueError, match="must not be empty"):
            engine.ask("   ")
