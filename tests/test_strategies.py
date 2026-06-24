"""
test_strategies.py

Unit tests for the new AdaptivePromptStrategy and SelfConsistencyStrategy.
Uses LLMClient(provider="mock") throughout — runs fully offline, no API key needed.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from evaluator.confidence_evaluator import ConfidenceEvaluator
from llm.llm_client import LLMClient
from strategies.adaptive_prompt import AdaptivePromptStrategy, _SYSTEM_INSTRUCTION
from strategies.self_consistency import SelfConsistencyStrategy


@pytest.fixture
def mock_client() -> LLMClient:
    return LLMClient(provider="mock")

@pytest.fixture
def evaluator() -> ConfidenceEvaluator:
    return ConfidenceEvaluator()


class TestAdaptivePromptStrategy:
    def test_name(self, mock_client):
        assert AdaptivePromptStrategy(mock_client).name == "adaptive"

    def test_build_prompt_returns_raw_query(self, mock_client):
        s = AdaptivePromptStrategy(mock_client)
        assert s.build_prompt("What is the capital of France?") == "What is the capital of France?"

    def test_system_instruction_present(self):
        assert "calibrate" in _SYSTEM_INSTRUCTION.lower()
        assert "CONFIDENCE" in _SYSTEM_INSTRUCTION

    def test_execute_returns_text_and_confidence(self, mock_client):
        s = AdaptivePromptStrategy(mock_client)
        text, conf = s.execute("What is 2 + 2?")
        assert isinstance(text, str) and len(text) > 0
        assert 0.0 <= conf <= 1.0

    def test_execute_with_model_override(self, mock_client):
        s = AdaptivePromptStrategy(mock_client)
        text, conf = s.execute("Why is the sky blue?", model="mock-model", baseline_model="mock-model")
        assert isinstance(text, str)
        assert 0.0 <= conf <= 1.0


class TestSelfConsistencyStrategy:
    def test_name(self, mock_client):
        assert SelfConsistencyStrategy(mock_client).name == "self_consistency"

    def test_makes_multiple_llm_calls(self, mock_client):
        calls = []
        original = mock_client.complete
        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        mock_client.complete = counting
        SelfConsistencyStrategy(mock_client, num_samples=3).execute("What is 2+2?")
        assert len(calls) == 3

    def test_majority_vote(self):
        winner, ratio = SelfConsistencyStrategy._majority_vote(["paris", "paris", "london"])
        assert winner == "paris"
        assert ratio == pytest.approx(2 / 3)

    def test_extract_final_answer_with_tag(self):
        assert SelfConsistencyStrategy._extract_final_answer("...\nFinal Answer: 42") == "42"

    def test_confidence_in_range(self, mock_client):
        _, conf = SelfConsistencyStrategy(mock_client, num_samples=2).execute("Is earth round?")
        assert 0.0 <= conf <= 1.0

    def test_model_kwarg_accepted(self, mock_client):
        s = SelfConsistencyStrategy(mock_client, num_samples=2)
        text, conf = s.execute("Test query", model="mock-model", baseline_model="mock-model")
        assert isinstance(text, str)


class TestConfidenceEvaluator:
    def test_self_reported_tag_parsed(self, evaluator):
        assert evaluator.score("q", "Answer. [CONFIDENCE: 0.90]") > 0.7

    def test_hedging_lowers_score(self, evaluator):
        hedged = "I think it might possibly be correct, but I'm not sure."
        confident = "The answer is definitively 42."
        assert evaluator.score("q", hedged) < evaluator.score("q", confident)

    def test_short_answer_to_complex_query_penalized(self, evaluator):
        q = "Prove that the square root of 2 is irrational."
        assert evaluator.score(q, "Yes.") < evaluator.score(q, "We prove by contradiction. " * 5)

    def test_passes_threshold(self):
        ev = ConfidenceEvaluator(threshold=0.75)
        assert ev.passes_threshold("q", "Answer. [CONFIDENCE: 0.9]") is True
        assert ev.passes_threshold("q", "Answer. [CONFIDENCE: 0.3]") is False

    def test_score_always_in_bounds(self, evaluator):
        for text in ["", "maybe I'm not sure unclear", "[CONFIDENCE: 5.0]", "[CONFIDENCE: -1]", "Solid answer."]:
            s = evaluator.score("q", text)
            assert 0.0 <= s <= 1.0
