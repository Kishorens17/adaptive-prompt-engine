"""
test_handlers.py

Tests for the handler layer (base handler + self-consistency handler).
All other old handlers (ZeroShot, FewShot, CoT) have been removed.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from evaluator.confidence_evaluator import ConfidenceEvaluator
from handlers.base_handler import HandlerResult
from handlers.self_consistency_handler import SelfConsistencyHandler
from llm.llm_client import LLMClient


@pytest.fixture
def mock_client() -> LLMClient:
    return LLMClient(provider="mock")

@pytest.fixture
def evaluator() -> ConfidenceEvaluator:
    return ConfidenceEvaluator()


class TestHandlerResult:
    def test_defaults(self):
        r = HandlerResult(final_answer="ok", final_confidence=0.9, handled_by="adaptive")
        assert r.escalations == []
        assert r.total_llm_calls == 0

    def test_fields(self):
        r = HandlerResult(
            final_answer="Paris", final_confidence=0.95,
            handled_by="adaptive", escalations=[], total_llm_calls=1,
        )
        assert r.final_answer == "Paris"
        assert r.final_confidence == 0.95
        assert r.handled_by == "adaptive"


class TestSelfConsistencyHandler:
    def test_handle_returns_result(self, mock_client, evaluator):
        h = SelfConsistencyHandler(mock_client, evaluator, threshold=0.75)
        result = h.handle("What is 2 + 2?")
        assert isinstance(result, HandlerResult)
        assert len(result.final_answer) > 0
        assert 0.0 <= result.final_confidence <= 1.0
        assert result.handled_by == "self_consistency"

    def test_is_terminal_when_no_next(self, mock_client, evaluator):
        h = SelfConsistencyHandler(mock_client, evaluator)
        assert h.is_terminal is True

    def test_llm_calls_counted(self, mock_client, evaluator):
        h = SelfConsistencyHandler(mock_client, evaluator, threshold=0.0)
        result = h.handle("test query")
        assert result.total_llm_calls >= 1
