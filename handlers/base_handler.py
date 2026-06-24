"""
base_handler.py

Chain of Responsibility pattern — abstract base.

Each concrete Handler wraps exactly one PromptStrategy. A Handler's
job is narrow and decoupled: try its strategy, check the confidence
score against the threshold, and either return the answer or pass the
query along to whatever handler comes next. A handler never knows
*which* handler comes next, or how many more are left in the chain —
it only knows "next" as an opaque reference. This is what lets the
chain be reordered, extended, or shortened without touching any
handler's internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from strategies.base_strategy import PromptStrategy


@dataclass
class HandlerResult:
    """
    Full record of what happened when a query was handled, including
    the escalation trail. Returned by PromptHandler.handle() so callers
    (main.py, the experiment runner) can log token usage and escalation
    frequency without each handler needing to know about logging.
    """

    final_answer: str
    final_confidence: float
    handled_by: str
    escalations: List[str] = field(default_factory=list)
    total_llm_calls: int = 0


class PromptHandler(ABC):
    """
    Abstract base for all chain links. Concrete subclasses (one per
    strategy strength level) only need to supply `strategy_name` for
    logging — the escalation/passing logic itself lives here, shared
    by every handler in the chain.
    """

    def __init__(self, strategy: PromptStrategy, threshold: float = 0.75):
        self.strategy = strategy
        self.threshold = threshold
        self._next: Optional["PromptHandler"] = None

    def set_next(self, handler: "PromptHandler") -> "PromptHandler":
        """Link the next handler in the chain. Returns `handler` to allow chaining calls."""
        self._next = handler
        return handler

    @property
    def is_terminal(self) -> bool:
        """True if this handler has no successor (always returns its answer)."""
        return self._next is None

    def handle(self, query: str, _trail: Optional[List[str]] = None, _call_count: int = 0) -> HandlerResult:
        """
        Try this handler's strategy. If confidence clears the threshold
        (or this is the last handler in the chain), return the result.
        Otherwise, delegate to the next handler.
        """
        trail = _trail if _trail is not None else []
        response_text, confidence = self.strategy.execute(query)
        call_count = _call_count + self._llm_calls_for_this_strategy()

        if confidence >= self.threshold or self.is_terminal:
            return HandlerResult(
                final_answer=response_text,
                final_confidence=confidence,
                handled_by=self.strategy.name,
                escalations=trail,
                total_llm_calls=call_count,
            )

        trail.append(self.strategy.name)
        return self._next.handle(query, _trail=trail, _call_count=call_count)

    def _llm_calls_for_this_strategy(self) -> int:
        """
        Number of underlying LLM API calls this handler's strategy makes.
        Most strategies make exactly one call; self-consistency makes
        several. Subclasses for multi-call strategies override this.
        """
        return getattr(self.strategy, "num_samples", 1)
