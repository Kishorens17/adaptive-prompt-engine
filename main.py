"""
main.py — Entry point for the Adaptive Prompt Engine.

Architecture:
    Query
      ↓
    SessionStore      — inject conversation history for multi-turn
      ↓
    SemanticCache     — instant answer if similar query seen before
      ↓ (miss)
    ComplexityEstimator — continuous score 0.0–1.0
      ↓
    ModelRouter       — cheapest model that fits the complexity
      ↓
    Strategy selector:
        RAGStrategy       — if knowledge base has documents (auto)
        ToolUseStrategy   — if tools are registered (auto)
        AdaptivePromptStrategy — default
      ↓
    LLMJudgeEvaluator — LLM rates its own answer quality (real providers)
    ConfidenceEvaluator — rule-based fallback (mock / judge failure)
      ↓
    Escalation: SelfConsistencyStrategy if confidence < threshold
      ↓
    CacheWriter + SessionStore.append + QueryLogger
      ↓
    Clean answer

CLI usage:
    python main.py --provider gemini --query "What is 2+2?"
    python main.py --provider openai --budget quality
    python main.py --provider groq --verbose
    python main.py --serve
    python main.py --serve --port 8080
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
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
# L1 — Input
# ---------------------------------------------------------------------------

@dataclass
class QueryReceiver:
    def receive(self, raw_query: str) -> str:
        cleaned = raw_query.strip()
        if not cleaned:
            raise ValueError("Query must not be empty.")
        return cleaned


# ---------------------------------------------------------------------------
# L5 — Output
# ---------------------------------------------------------------------------

_CONFIDENCE_TAG_RE = re.compile(
    r"\s*\[CONFIDENCE:\s*[0-9]*\.?[0-9]+\s*\]", re.IGNORECASE
)


class ResponseFormatter:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def format(self, result: "EngineResult") -> str:
        clean_answer = _CONFIDENCE_TAG_RE.sub("", result.answer).strip()
        if not self.verbose:
            return clean_answer
        lines = [
            f"Query:        {result.query}",
            f"Complexity:   {result.complexity_tier} (score: {result.complexity_score:.2f})",
            f"Model:        {result.model_used}",
            f"Strategy:     {result.strategy_used}",
            f"Cache hit:    {result.cache_hit}",
            f"Tokens:       {result.total_tokens} (in: {result.input_tokens}, out: {result.output_tokens})",
            f"Cost:         ${result.cost_usd:.6f}  |  Saved: ${result.cost_saved_usd:.6f}",
            f"Latency:      {result.latency_ms:.0f} ms",
            f"Confidence:   {result.confidence:.2f}",
        ]
        if result.quality:
            q = result.quality
            lines.append(
                f"Quality:      accuracy={q.get('accuracy')}, "
                f"completeness={q.get('completeness')}, "
                f"relevance={q.get('relevance')}, "
                f"clarity={q.get('clarity')}"
            )
            lines.append(f"Judge note:   {q.get('reasoning', '')}")
        lines += ["", "Answer:", clean_answer]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine result
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
    strategy_used: str = "adaptive"
    quality: Optional[dict] = field(default=None)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class AdaptivePromptEngine:
    """
    Top-level facade. Wires all layers into a single .ask() call.
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
        self.provider = provider
        self.receiver = QueryReceiver()
        self.formatter = ResponseFormatter(verbose=verbose)
        self.llm_client = LLMClient(provider=provider, model=model, api_key=api_key)
        self.router = ModelRouter(provider=provider, budget=budget)

        # Cache: enabled for real providers; gracefully disabled on errors
        _cache_enabled = use_cache and provider != "mock"
        self.cache: Optional[SemanticCache] = None
        if _cache_enabled:
            try:
                self.cache = SemanticCache()
            except Exception:
                pass

        self.logger = QueryLogger()

        # Evaluator: LLM-as-Judge for real providers, rule-based for mock
        if provider != "mock":
            try:
                from evaluator.llm_judge import LLMJudgeEvaluator
                self.evaluator = LLMJudgeEvaluator(
                    llm_client=self.llm_client,
                    threshold=confidence_threshold,
                )
            except Exception:
                self.evaluator = ConfidenceEvaluator(threshold=confidence_threshold)
        else:
            self.evaluator = ConfidenceEvaluator(threshold=confidence_threshold)

        # Complexity estimator (lazy for mock to keep tests fast)
        self.estimator: Optional[ComplexityEstimator] = None
        if provider != "mock":
            try:
                self.estimator = ComplexityEstimator()
            except Exception:
                pass

        # Knowledge base (auto-routing)
        self.knowledge_base = None
        if provider != "mock":
            try:
                from cache.knowledge_base import KnowledgeBase
                self.knowledge_base = KnowledgeBase()
            except Exception:
                pass

        # Tool registry
        self.tool_registry = None
        if provider != "mock":
            try:
                from strategies.tool_use_strategy import ToolRegistry
                self.tool_registry = ToolRegistry()
            except Exception:
                pass

        # Strategy layer
        self._build_strategies()

    def _build_strategies(self) -> None:
        """Build strategy instances. Called once on init."""
        self.escalation_strategy = SelfConsistencyStrategy(
            self.llm_client, self.evaluator
        )

        # Primary: RAG if KB has docs, tool-use if tools registered, else adaptive
        if self.knowledge_base is not None:
            from strategies.rag_strategy import RAGStrategy
            self.primary_strategy = RAGStrategy(
                self.llm_client, self.evaluator, self.knowledge_base
            )
            self._primary_name = "rag"
        elif self.tool_registry is not None:
            from strategies.tool_use_strategy import ToolUseStrategy
            self.primary_strategy = ToolUseStrategy(
                self.llm_client, self.evaluator, self.tool_registry
            )
            self._primary_name = "tool_use"
        else:
            self.primary_strategy = AdaptivePromptStrategy(
                self.llm_client, self.evaluator
            )
            self._primary_name = "adaptive"

    def refresh_strategy(self) -> None:
        """Re-select strategy (call after uploading docs or registering tools)."""
        self._build_strategies()

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def ask(self, raw_query: str) -> EngineResult:
        return self._ask_internal(raw_query, history=None)

    def ask_with_history(
        self, raw_query: str, history: list
    ) -> EngineResult:
        return self._ask_internal(raw_query, history=history)

    def ask_and_format(self, raw_query: str) -> str:
        return self.formatter.format(self.ask(raw_query))

    # ------------------------------------------------------------------
    # Internal processing
    # ------------------------------------------------------------------

    def _ask_internal(
        self, raw_query: str, history: Optional[list]
    ) -> EngineResult:
        t0 = time.time()
        query = self.receiver.receive(raw_query)

        # ── Cache lookup (skip for session queries — context matters) ──
        if self.cache and not history:
            cached = self.cache.get(query)
            if cached is not None:
                result = EngineResult(
                    query=query, answer=cached,
                    complexity_tier="cached", complexity_score=0.0,
                    model_used="cache", input_tokens=0, output_tokens=0,
                    total_tokens=0, cost_usd=0.0, cost_saved_usd=0.0,
                    latency_ms=(time.time() - t0) * 1000,
                    cache_hit=True, confidence=1.0, strategy_used="cache",
                )
                self._log(result)
                return result

        # ── Complexity + routing ──────────────────────────────────────
        if self.estimator:
            score = self.estimator.score(query)
            tier = self.estimator.tier(query)
        else:
            score, tier = 0.5, ComplexityTier.MEDIUM

        routing = self.router.route(tier)

        # ── Primary strategy ──────────────────────────────────────────
        # Pass history to LLM client if session-aware
        if history and hasattr(self.primary_strategy, "llm_client"):
            # Temporarily inject history via direct client call
            answer_text, confidence = self._ask_with_history_strategy(
                query, history, routing.model, routing.baseline_model
            )
            strategy_used = "adaptive+history"
        else:
            answer_text, confidence = self.primary_strategy.execute(
                query,
                model=routing.model,
                baseline_model=routing.baseline_model,
            )
            strategy_used = getattr(self.primary_strategy, "name", self._primary_name)

        # ── Escalate if confidence too low ───────────────────────────
        if confidence < self.evaluator.threshold:
            answer_text, confidence = self.escalation_strategy.execute(
                query,
                model=routing.model,
                baseline_model=routing.baseline_model,
            )
            strategy_used += "+self_consistency"

        latency_ms = (time.time() - t0) * 1000

        # ── Token counting ───────────────────────────────────────────
        # Use LLM-reported counts when available (Gemini/OpenAI track them)
        # For now estimate from text length as a safe fallback
        prompt_tokens = max(1, len(query.split()) + 80)
        output_tokens = max(1, len(answer_text.split()))
        total_tokens = prompt_tokens + output_tokens

        cost_usd = self.router.estimate_cost(routing.model, total_tokens)
        cost_saved_usd = self.router.estimate_saved(
            routing.model, routing.baseline_model, total_tokens
        )

        # ── Judge quality metadata ───────────────────────────────────
        quality = None
        if hasattr(self.evaluator, "score_detailed"):
            judge_result = self.evaluator.score_detailed(query, answer_text)
            if judge_result:
                quality = judge_result.to_dict()

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
            strategy_used=strategy_used,
            quality=quality,
        )

        # ── Cache write + log ────────────────────────────────────────
        clean_answer = _CONFIDENCE_TAG_RE.sub("", answer_text).strip()
        if self.cache and not history and confidence >= self.evaluator.threshold:
            self.cache.put(query, clean_answer, model=routing.model, cost_usd=cost_usd)
        self._log(result)

        return result

    def _ask_with_history_strategy(
        self, query: str, history: list,
        model: Optional[str], baseline_model: Optional[str],
    ):
        """Execute adaptive strategy with conversation history injected."""
        from strategies.adaptive_prompt import _SYSTEM_INSTRUCTION  # type: ignore
        llm_response = self.llm_client.complete(
            prompt=query,
            system=_SYSTEM_INSTRUCTION,
            model=model,
            baseline_model=baseline_model,
            history=history,
        )
        confidence = self.evaluator.score(query, llm_response.text)
        return llm_response.text, confidence

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
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive Prompt Engine")
    parser.add_argument(
        "--provider", default="mock",
        choices=["mock", "gemini", "openai", "groq", "nvidia"],
        help="LLM backend (default: mock — no API key needed)",
    )
    parser.add_argument("--model", default=None, help="Model override (skips smart routing)")
    parser.add_argument("--api-key", default=None, help="API key (else read from .env)")
    parser.add_argument(
        "--budget", default="balanced",
        choices=["low", "balanced", "quality"],
    )
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--query", default=None, help="Run a single query and exit")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--serve", action="store_true", help="Start REST API server")
    parser.add_argument("--port", type=int, default=8081, help="Server port (default: 8081)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.serve:
        import uvicorn  # type: ignore
        port = args.port
        host = args.host
        print("\n" + "-" * 55)
        print("  >   Adaptive Prompt Engine - API Server")
        print("-" * 55)
        print(f"  Local:      http://localhost:{port}")
        print(f"  Network:    http://{host}:{port}")
        print(f"  API docs:   http://localhost:{port}/docs")
        print(f"  Dashboard:  http://localhost:{port}/dashboard")
        print("-" * 55 + "\n")
        uvicorn.run("api.server:app", host=host, port=port, reload=True)
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

    provider_label = args.provider
    print(f"\nAdaptive Prompt Engine — interactive mode")
    print(f"Provider: {provider_label} | Budget: {args.budget} | Cache: {'on' if not args.no_cache else 'off'}")
    print("Type a query and press Enter. Type 'exit' to stop.\n")

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
