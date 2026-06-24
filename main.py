"""
main.py

Entry point for the Adaptive Prompt Engine.

New architecture (replaces the old 5-layer rule-based system):

    Query
      ↓
    SemanticCache        — instant answer if a similar query was seen before
      ↓ (miss)
    ComplexityEstimator  — continuous score 0.0–1.0 (no category boxes)
      ↓
    ModelRouter          — cheapest model that fits the complexity
      ↓
    AdaptivePromptStrategy — single meta-prompt; LLM calibrates its own verbosity
      ↓
    ConfidenceEvaluator  — safety net; escalates to self-consistency if low
      ↓
    QueryLogger + CacheWriter
      ↓
    Clean answer (no debug metadata by default)

Run interactively:
    python main.py --provider gemini

Run a single query:
    python main.py --provider gemini --query "What is the capital of France?"

Budget control (overrides model routing):
    python main.py --provider gemini --budget low      # always use cheapest model
    python main.py --provider gemini --budget quality  # always use best model

Show debug metadata:
    python main.py --provider gemini --verbose

Start REST API server:
    python main.py --serve
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

from core.complexity_estimator import ComplexityEstimator, ComplexityTier
from core.model_router import ModelRouter
from evaluator.confidence_evaluator import ConfidenceEvaluator
from llm.llm_client import LLMClient
from strategies.adaptive_prompt import AdaptivePromptStrategy
from strategies.self_consistency import SelfConsistencyStrategy
from cache.semantic_cache import SemanticCache
from cache.query_log import QueryLogger, LogEntry


# ---------------------------------------------------------------------------
# L1 — Input layer
# ---------------------------------------------------------------------------

@dataclass
class QueryReceiver:
    """Trivial input sanitiser. Kept as its own class so validation can
    grow here without touching any other layer."""

    def receive(self, raw_query: str) -> str:
        cleaned = raw_query.strip()
        if not cleaned:
            raise ValueError("Query must not be empty.")
        return cleaned


# ---------------------------------------------------------------------------
# L5 — Output layer
# ---------------------------------------------------------------------------

_CONFIDENCE_TAG_RE = re.compile(
    r"\s*\[CONFIDENCE:\s*[0-9]*\.?[0-9]+\s*\]", re.IGNORECASE
)


class ResponseFormatter:
    """
    Formats engine results for display.

    verbose=False (default): prints only the clean answer.
    verbose=True:  prints full debug metadata — useful during development.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def format(self, result: "EngineResult") -> str:
        clean_answer = _CONFIDENCE_TAG_RE.sub("", result.answer).strip()

        if not self.verbose:
            return clean_answer

        lines = [
            f"Query: {result.query}",
            f"Complexity: {result.complexity_tier} (score: {result.complexity_score:.2f})",
            f"Model used: {result.model_used}",
            f"Cache hit: {result.cache_hit}",
            f"Tokens: {result.total_tokens} "
            f"(in: {result.input_tokens}, out: {result.output_tokens})",
            f"Cost: ${result.cost_usd:.6f}  |  Saved: ${result.cost_saved_usd:.6f}",
            f"Latency: {result.latency_ms:.0f} ms",
            f"Confidence: {result.confidence:.2f}",
            "",
            "Answer:",
            clean_answer,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    query: str
    answer: str
    complexity_tier: str
    complexity_score: float
    model_used: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    cost_saved_usd: float
    latency_ms: float
    cache_hit: bool
    confidence: float


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class AdaptivePromptEngine:
    """
    Top-level facade.  Wires together all layers into a single .ask() call.
    """

    def __init__(
        self,
        provider: str = "mock",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        budget: str = "balanced",
        confidence_threshold: float = 0.75,
        use_cache: bool = True,
        verbose: bool = False,
    ) -> None:
        self.receiver   = QueryReceiver()
        self.formatter  = ResponseFormatter(verbose=verbose)
        self.llm_client = LLMClient(provider=provider, model=model, api_key=api_key)
        self.evaluator  = ConfidenceEvaluator(threshold=confidence_threshold)
        self.router     = ModelRouter(provider=provider, budget=budget)
        # Cache is only useful with real LLM providers — skip for mock.
        # Also degrade gracefully if sentence-transformers is broken.
        _cache_enabled = use_cache and provider != "mock"
        if _cache_enabled:
            try:
                self.cache = SemanticCache()
            except Exception:
                self.cache = None
        else:
            self.cache = None
        self.logger     = QueryLogger()

        # Strategy layer — two levels only:
        # Primary: adaptive (LLM calibrates own verbosity)
        # Escalation: self-consistency (only if confidence too low)
        self.primary_strategy = AdaptivePromptStrategy(self.llm_client, self.evaluator)
        self.escalation_strategy = SelfConsistencyStrategy(self.llm_client, self.evaluator)

        # ComplexityEstimator loads sentence-transformers once.
        # Skip for mock provider to keep tests fast.
        # Degrade gracefully if the install is broken (e.g. Keras version conflict).
        if provider != "mock":
            try:
                self.estimator = ComplexityEstimator()
            except Exception:
                self.estimator = None   # falls back to MEDIUM tier for all queries
        else:
            self.estimator = None

    def ask(self, raw_query: str) -> EngineResult:
        """Process a query end-to-end and return a structured result."""
        t0 = time.time()
        query = self.receiver.receive(raw_query)

        # ── Layer 1: Cache lookup ────────────────────────────────────────
        if self.cache:
            cached = self.cache.get(query)
            if cached is not None:
                latency_ms = (time.time() - t0) * 1000
                result = EngineResult(
                    query=query, answer=cached,
                    complexity_tier="cached", complexity_score=0.0,
                    model_used="cache", input_tokens=0, output_tokens=0,
                    total_tokens=0, cost_usd=0.0, cost_saved_usd=0.0,
                    latency_ms=latency_ms, cache_hit=True, confidence=1.0,
                )
                self._log(result)
                return result

        # ── Layer 2: Complexity estimation + model routing ───────────────
        if self.estimator:
            score = self.estimator.score(query)
            tier  = self.estimator.tier(query)
        else:
            score = 0.5
            tier  = ComplexityTier.MEDIUM  # safe default when estimator unavailable

        routing = self.router.route(tier)

        # ── Layer 3: Primary adaptive strategy ───────────────────────────
        answer_text, confidence = self.primary_strategy.execute(
            query,
            model=routing.model,
            baseline_model=routing.baseline_model,
        )

        # ── Layer 4: Escalate if confidence too low ──────────────────────
        if confidence < self.evaluator.threshold:
            answer_text, confidence = self.escalation_strategy.execute(
                query,
                model=routing.model,
                baseline_model=routing.baseline_model,
            )

        latency_ms = (time.time() - t0) * 1000

        # Retrieve token/cost info from the last LLM response (stored on client).
        # We re-compute here using the routing info since execute() returns only text+conf.
        # Approximate token counts from the response length.
        prompt_tokens  = len(query.split()) + 80   # query + system prompt overhead
        output_tokens  = len(answer_text.split())
        total_tokens   = prompt_tokens + output_tokens
        cost_usd       = self.router.estimate_cost(routing.model, total_tokens)
        cost_saved_usd = self.router.estimate_saved(
            routing.model, routing.baseline_model, total_tokens
        )

        result = EngineResult(
            query=query,
            answer=answer_text,
            complexity_tier=tier.value,
            complexity_score=score,
            model_used=routing.model,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            cost_saved_usd=cost_saved_usd,
            latency_ms=latency_ms,
            cache_hit=False,
            confidence=confidence,
        )

        # ── Layer 5: Cache write + log ───────────────────────────────────
        clean_answer = _CONFIDENCE_TAG_RE.sub("", answer_text).strip()
        if self.cache and confidence >= self.evaluator.threshold:
            self.cache.put(query, clean_answer, model=routing.model, cost_usd=cost_usd)
        self._log(result)

        return result

    def ask_and_format(self, raw_query: str) -> str:
        result = self.ask(raw_query)
        return self.formatter.format(result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, result: EngineResult) -> None:
        try:
            self.logger.log(LogEntry(
                query=result.query,
                complexity_tier=result.complexity_tier,
                model_used=result.model_used,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
                cost_saved_usd=result.cost_saved_usd,
                latency_ms=result.latency_ms,
                cache_hit=result.cache_hit,
                confidence=result.confidence,
            ))
        except Exception:  # noqa: BLE001 — logging must never crash the engine
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive Prompt Engine")
    parser.add_argument(
        "--provider", default="mock",
        choices=["mock", "gemini"],
        help="LLM backend (default: mock — no API key needed)",
    )
    parser.add_argument("--model", default=None, help="Model name override (skips smart routing)")
    parser.add_argument("--api-key", default=None, help="API key (else read from environment)")
    parser.add_argument(
        "--budget", default="balanced",
        choices=["low", "balanced", "quality"],
        help="Model routing budget: low=cheapest, balanced=smart (default), quality=best",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.75,
        help="Confidence threshold for escalation (default 0.75)",
    )
    parser.add_argument("--query", default=None, help="Run a single query and exit")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable semantic response cache",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full metadata (model, tokens, cost, latency, confidence)",
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="Start the REST API server instead of interactive mode",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.serve:
        # Start FastAPI server
        import uvicorn  # type: ignore
        uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
        return

    engine = AdaptivePromptEngine(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        budget=args.budget,
        confidence_threshold=args.threshold,
        use_cache=not args.no_cache,
        verbose=args.verbose,
    )

    if args.query:
        print(engine.ask_and_format(args.query))
        return

    print("Adaptive Prompt Engine — interactive mode")
    print(f"Provider: {args.provider} | Budget: {args.budget} | Cache: {'on' if not args.no_cache else 'off'}")
    print("Type a query and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if raw.lower() in {"exit", "quit"}:
            break
        if not raw:
            continue
        try:
            print()
            print(engine.ask_and_format(raw))
            print()
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()
