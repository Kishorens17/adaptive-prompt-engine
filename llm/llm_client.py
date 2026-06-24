"""
llm_client.py

Thin wrapper around Google Gemini API.

This module deliberately knows nothing about prompting strategies,
confidence scoring, or query classification — its only job is:
"given a prompt string (and a sampling temperature), return raw text
from the model, plus a token-usage estimate."

Keeping this wrapper thin is intentional: every Strategy class
(strategies/*.py) depends on this single class rather than on the
Gemini SDK directly. If you add a new provider in future, only this
file changes.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Normalized response shape returned by every provider backend."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    model_used: str = "unknown"
    cost_usd: float = 0.0
    cost_saved_usd: float = 0.0
    raw: Optional[dict] = None

    @property
    def tokens(self) -> int:
        return self.total_tokens


class LLMClientError(RuntimeError):
    """Raised when the underlying provider call fails after retries."""


class LLMClient:
    """
    Provider-agnostic LLM client.

    Usage:
        client = LLMClient(provider="openai", model="gpt-3.5-turbo")
        response = client.complete("Explain recursion in one sentence.")
        print(response.text, response.tokens)

    Supported providers: "openai", "gemini", "mock".
    The "mock" provider requires no API key and no network access — it
    is used by the test suite and is also useful while you're wiring
    up the rest of the system before you have API keys configured.
    """

    def __init__(
        self,
        provider: str = "mock",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 2,
        timeout: float = 30.0,
    ) -> None:
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

        if self.provider == "gemini":
            self.model = model or "gemini-2.5-flash"
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
            self._client = self._init_gemini()
        elif self.provider == "mock":
            self.model = model or "mock-model"
            self.api_key = None
            self._client = None
        else:
            raise ValueError(f"Unknown provider: {provider!r}. Supported: 'gemini', 'mock'.")

    # ------------------------------------------------------------------
    # Provider initialization
    # ------------------------------------------------------------------

    def _init_gemini(self):
        try:
            from google import genai  # type: ignore  # pip install google-genai
        except ImportError as exc:
            raise LLMClientError(
                "The 'google-genai' package is required for "
                "provider='gemini'. Install it with: "
                "pip install google-genai"
            ) from exc
        if not self.api_key:
            raise LLMClientError(
                "No Gemini API key found. Set GEMINI_API_KEY in your "
                "environment or pass api_key= explicitly."
            )
        return genai.Client(api_key=self.api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        model: Optional[str] = None,
        baseline_model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send `prompt` to the configured provider and return a normalized
        LLMResponse. Retries on transient failures up to max_retries times.

        Args:
            model:          Override the default model for this call.
                            Used by ModelRouter to select the cheapest
                            appropriate model per query.
            baseline_model: The model we would have used without smart routing.
                            Used to compute cost_saved_usd in the response.
        """
        temp = self.temperature if temperature is None else temperature
        # Use per-call model override if provided, else fall back to instance default.
        effective_model = model or self.model

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                start = time.time()
                if self.provider == "gemini":
                    result = self._complete_gemini(prompt, temp, system, effective_model)
                else:
                    result = self._complete_mock(prompt, temp, system, effective_model)
                result.latency_seconds = time.time() - start
                result.model_used = effective_model
                # Attach cost metadata using the routing cost table.
                result.cost_usd, result.cost_saved_usd = self._compute_cost(
                    effective_model, baseline_model, result.total_tokens
                )
                return result
            except Exception as exc:  # noqa: BLE001 - we re-raise as our own type
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
        raise LLMClientError(
            f"LLM call failed after {self.max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    @staticmethod
    def _compute_cost(
        model: str, baseline_model: Optional[str], total_tokens: int
    ) -> tuple[float, float]:
        """Return (cost_usd, cost_saved_usd) for a completed call."""
        try:
            from core.model_router import _COST_PER_1K  # type: ignore
        except ImportError:
            return 0.0, 0.0
        rate = _COST_PER_1K.get(model, 0.0)
        cost = rate * total_tokens / 1000.0
        if baseline_model and baseline_model != model:
            baseline_rate = _COST_PER_1K.get(baseline_model, 0.0)
            saved = max(0.0, (baseline_rate - rate) * total_tokens / 1000.0)
        else:
            saved = 0.0
        return cost, saved

    # ------------------------------------------------------------------
    # Provider-specific implementations
    # ------------------------------------------------------------------

    def _complete_gemini(
        self, prompt: str, temperature: float, system: Optional[str],
        model: Optional[str] = None,
    ) -> LLMResponse:
        from google import genai  # type: ignore

        effective_model = model or self.model
        # Build config. Gemini 2.5 Pro is a "thinking" model and requires
        # thinking_budget to be set explicitly; without it the API may reject
        # the call or burn hidden thinking tokens.
        config_kwargs: dict = {"temperature": temperature}
        if system:
            config_kwargs["system_instruction"] = system
        if "2.5-pro" in effective_model:
            # thinking_budget=0 disables the reasoning chain so the model
            # behaves like a standard generation model (faster + cheaper).
            config_kwargs["thinking_config"] = genai.types.ThinkingConfig(
                thinking_budget=0
            )
        response = self._client.models.generate_content(
            model=effective_model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(**config_kwargs),
        )
        text = response.text or ""
        usage_meta = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_seconds=0.0,
        )

    def _complete_mock(
        self, prompt: str, temperature: float, system: Optional[str],
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Deterministic, offline stand-in for a real LLM. Produces a
        plausible-looking answer so the rest of the pipeline (confidence
        scoring, escalation, tests) can be exercised without network
        access or an API key. Word count scales loosely with prompt
        length so longer/strategy-augmented prompts "answer more".
        """
        prompt_lower = prompt.lower()
        is_cot = "step by step" in prompt_lower or "think through" in prompt_lower
        is_few_shot = "example" in prompt_lower

        if is_cot:
            text = (
                "Let's work through this step by step. First, we identify the "
                "core claim. Second, we reason about each component in turn. "
                "Third, we combine the intermediate results. Therefore, the "
                "conclusion follows directly from the steps above. "
                "[CONFIDENCE: 0.81]"
            )
        elif is_few_shot:
            text = (
                "Following the style shown in the examples, here is a response "
                "that mirrors the requested tone and structure, adapted to the "
                "new input. [CONFIDENCE: 0.78]"
            )
        else:
            text = "This is a direct answer to the query. [CONFIDENCE: 0.62]"

        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(text.split()))
        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_seconds=0.0,
        )
