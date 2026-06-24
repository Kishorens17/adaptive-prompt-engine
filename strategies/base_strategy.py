"""
base_strategy.py

Strategy pattern — abstract base.

Every concrete prompting technique (zero-shot, few-shot, chain-of-thought,
self-consistency) implements this interface. Calling code never needs to
know which concrete strategy it holds; it only calls execute(query).

This is what lets the rest of the system add a brand-new prompting
technique by adding one new file here and registering it in
factory/strategy_factory.py — no existing code is touched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from evaluator.confidence_evaluator import ConfidenceEvaluator
from llm.llm_client import LLMClient


class PromptStrategy(ABC):
    """
    Abstract base class for all prompting strategies.

    Subclasses must implement `build_prompt` (how the query is turned
    into the actual text sent to the LLM) and `name` (a short label used
    in logs/experiments). The shared `execute` method handles the common
    work: build the prompt, call the LLM, score confidence.
    """

    def __init__(self, llm_client: LLMClient, evaluator: ConfidenceEvaluator | None = None):
        self.llm_client = llm_client
        self.evaluator = evaluator or ConfidenceEvaluator()

    @property
    @abstractmethod
    def name(self) -> str:
        """Short strategy name, e.g. 'zero_shot', 'chain_of_thought'."""
        raise NotImplementedError

    @abstractmethod
    def build_prompt(self, query: str) -> str:
        """Transform a raw user query into the strategy-specific prompt text."""
        raise NotImplementedError

    def execute(
        self,
        query: str,
        model: "str | None" = None,
        baseline_model: "str | None" = None,
    ) -> Tuple[str, float]:
        """
        Run the strategy end-to-end: build prompt -> call LLM -> score confidence.

        Args:
            model:          Model name override from ModelRouter.
            baseline_model: Baseline model for cost-saving calculation.

        Returns:
            (response_text, confidence_score) where confidence_score is in [0.0, 1.0].
        """
        prompt = self.build_prompt(query)
        llm_response = self.llm_client.complete(
            prompt, model=model, baseline_model=baseline_model
        )
        confidence = self.evaluator.score(query=query, response_text=llm_response.text)
        return llm_response.text, confidence

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} name={self.name!r}>"
