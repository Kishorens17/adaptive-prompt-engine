"""
self_consistency_handler.py

Final, strongest link in every escalation chain. Wraps
SelfConsistencyStrategy. Per the project design this handler always
returns its answer regardless of the confidence score, because there
is no stronger strategy left to escalate to — it is always the
terminal node (is_terminal is True whenever set_next() has not been
called on it, which StrategyFactory guarantees by construction).
"""

from __future__ import annotations

from handlers.base_handler import PromptHandler
from strategies.self_consistency import SelfConsistencyStrategy


class SelfConsistencyHandler(PromptHandler):
    """Chain link wrapping SelfConsistencyStrategy. Always terminal by design."""

    def __init__(
        self,
        llm_client,
        evaluator=None,
        threshold: float = 0.75,
        num_samples: int = 3,
        sample_temperature: float = 0.9,
    ):
        strategy = SelfConsistencyStrategy(
            llm_client,
            evaluator,
            num_samples=num_samples,
            sample_temperature=sample_temperature,
        )
        super().__init__(strategy, threshold)
