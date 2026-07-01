"""
llm_client.py

Provider-agnostic LLM wrapper supporting Gemini, OpenAI, Groq, and Mock.

Providers:
    "gemini" — Google Gemini (google-genai SDK)
    "openai" — OpenAI GPT models (openai SDK)
    "groq"   — Groq ultra-fast inference (groq SDK, same API as openai)
    "mock"   — Offline deterministic stub for tests

Token counting:
    - Gemini: real counts from usage_metadata (API-reported)
    - OpenAI/Groq: real counts from tiktoken BPE encoder
    - Mock: word-split approximation

Streaming:
    complete_stream() yields text chunks; complete() returns full LLMResponse.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Generator, Iterator, Optional


@dataclass
class LLMResponse:
    """Normalized response returned by every provider."""
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


def _count_tokens_tiktoken(text: str, model: str) -> int:
    """Count tokens via tiktoken BPE. Falls back to word-split on failure."""
    try:
        import tiktoken  # type: ignore
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text.split()))


class LLMClient:
    """
    Provider-agnostic LLM client.

    Usage:
        client = LLMClient(provider="gemini")
        response = client.complete("Explain recursion.", system="Be concise.")
        for chunk in client.complete_stream("Tell me a story"):
            print(chunk, end="", flush=True)
    """

    SUPPORTED_PROVIDERS = ("gemini", "openai", "groq", "nvidia", "mock")

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

        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider!r}. "
                f"Supported: {self.SUPPORTED_PROVIDERS}"
            )

        if self.provider == "gemini":
            self.model = model or "gemini-2.5-flash"
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
            self._client = self._init_gemini()
        elif self.provider == "openai":
            self.model = model or "gpt-4o-mini"
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self._client = self._init_openai()
        elif self.provider == "groq":
            self.model = model or "llama-3.3-70b-versatile"
            self.api_key = api_key or os.environ.get("GROQ_API_KEY")
            self._client = self._init_groq()
        elif self.provider == "nvidia":
            self.model = model or "mistralai/mistral-medium-3-128k"
            self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
            self._client = self._init_nvidia()
        else:  # mock
            self.model = model or "mock-model"
            self.api_key = None
            self._client = None

    # ------------------------------------------------------------------
    # Provider initialization
    # ------------------------------------------------------------------

    def _init_gemini(self):
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise LLMClientError(
                "Install google-genai: pip install google-genai"
            ) from exc
        if not self.api_key:
            raise LLMClientError(
                "No Gemini API key. Set GEMINI_API_KEY in .env"
            )
        return genai.Client(api_key=self.api_key)

    def _init_openai(self):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise LLMClientError(
                "Install openai: pip install openai"
            ) from exc
        if not self.api_key:
            raise LLMClientError(
                "No OpenAI API key. Set OPENAI_API_KEY in .env"
            )
        return OpenAI(api_key=self.api_key, timeout=self.timeout)

    def _init_groq(self):
        try:
            from groq import Groq  # type: ignore
        except ImportError as exc:
            raise LLMClientError(
                "Install groq: pip install groq"
            ) from exc
        if not self.api_key:
            raise LLMClientError(
                "No Groq API key. Set GROQ_API_KEY in .env"
            )
        return Groq(api_key=self.api_key, timeout=self.timeout)

    def _init_nvidia(self):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise LLMClientError(
                "Install openai: pip install openai"
            ) from exc
        if not self.api_key:
            raise LLMClientError(
                "No NVIDIA API key. Set NVIDIA_API_KEY in .env"
            )
        return OpenAI(api_key=self.api_key, base_url="https://integrate.api.nvidia.com/v1", timeout=self.timeout)

    # ------------------------------------------------------------------
    # Public API — full response
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        model: Optional[str] = None,
        baseline_model: Optional[str] = None,
        history: Optional[list] = None,
    ) -> LLMResponse:
        """
        Send prompt to provider, return normalized LLMResponse.
        history: list of {"role": "user"|"assistant", "content": str} for multi-turn.
        """
        temp = self.temperature if temperature is None else temperature
        effective_model = model or self.model
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                start = time.time()
                if self.provider == "gemini":
                    result = self._complete_gemini(
                        prompt, temp, system, effective_model, history
                    )
                elif self.provider in ("openai", "groq", "nvidia"):
                    result = self._complete_openai_compat(
                        prompt, temp, system, effective_model, history
                    )
                else:
                    result = self._complete_mock(
                        prompt, temp, system, effective_model
                    )
                result.latency_seconds = time.time() - start
                result.model_used = effective_model
                result.cost_usd, result.cost_saved_usd = self._compute_cost(
                    effective_model, baseline_model, result.total_tokens
                )
                return result
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise LLMClientError(
            f"LLM call failed after {self.max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Public API — streaming
    # ------------------------------------------------------------------

    def complete_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        model: Optional[str] = None,
        history: Optional[list] = None,
    ) -> Generator[str, None, None]:
        """
        Generator that yields text chunks as they stream from the provider.
        Each yielded value is a raw text chunk (no metadata).
        """
        temp = self.temperature if temperature is None else temperature
        effective_model = model or self.model

        if self.provider == "gemini":
            yield from self._stream_gemini(
                prompt, temp, system, effective_model, history
            )
        elif self.provider in ("openai", "groq", "nvidia"):
            yield from self._stream_openai_compat(
                prompt, temp, system, effective_model, history
            )
        else:
            yield from self._stream_mock(prompt, temp, system, effective_model)

    # ------------------------------------------------------------------
    # Cost calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cost(
        model: str, baseline_model: Optional[str], total_tokens: int
    ) -> tuple[float, float]:
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
    # Gemini — complete + stream
    # ------------------------------------------------------------------

    def _complete_gemini(
        self, prompt: str, temperature: float,
        system: Optional[str], model: str,
        history: Optional[list] = None,
    ) -> LLMResponse:
        from google import genai  # type: ignore

        config_kwargs: dict = {"temperature": temperature}
        if system:
            config_kwargs["system_instruction"] = system
        if "2.5-pro" in model:
            config_kwargs["thinking_config"] = genai.types.ThinkingConfig(
                thinking_budget=0
            )

        contents = self._build_gemini_contents(prompt, history)

        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=genai.types.GenerateContentConfig(**config_kwargs),
        )
        text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_seconds=0.0,
        )

    def _stream_gemini(
        self, prompt: str, temperature: float,
        system: Optional[str], model: str,
        history: Optional[list] = None,
    ) -> Iterator[str]:
        from google import genai  # type: ignore

        config_kwargs: dict = {"temperature": temperature}
        if system:
            config_kwargs["system_instruction"] = system
        if "2.5-pro" in model:
            config_kwargs["thinking_config"] = genai.types.ThinkingConfig(
                thinking_budget=0
            )

        contents = self._build_gemini_contents(prompt, history)

        for chunk in self._client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=genai.types.GenerateContentConfig(**config_kwargs),
        ):
            if chunk.text:
                yield chunk.text

    @staticmethod
    def _build_gemini_contents(prompt: str, history: Optional[list]) -> list:
        """Build Gemini contents list from history + current prompt."""
        if not history:
            return [prompt]
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        return contents

    # ------------------------------------------------------------------
    # OpenAI / Groq — complete + stream (identical API)
    # ------------------------------------------------------------------

    def _complete_openai_compat(
        self, prompt: str, temperature: float,
        system: Optional[str], model: str,
        history: Optional[list] = None,
    ) -> LLMResponse:
        messages = self._build_openai_messages(prompt, system, history)
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else _count_tokens_tiktoken(prompt, model)
        completion_tokens = usage.completion_tokens if usage else _count_tokens_tiktoken(text, model)
        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=(usage.total_tokens if usage else prompt_tokens + completion_tokens),
            latency_seconds=0.0,
        )

    def _stream_openai_compat(
        self, prompt: str, temperature: float,
        system: Optional[str], model: str,
        history: Optional[list] = None,
    ) -> Iterator[str]:
        messages = self._build_openai_messages(prompt, system, history)
        stream = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @staticmethod
    def _build_openai_messages(
        prompt: str, system: Optional[str], history: Optional[list]
    ) -> list:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    # ------------------------------------------------------------------
    # Mock — complete + stream
    # ------------------------------------------------------------------

    def _complete_mock(
        self, prompt: str, temperature: float,
        system: Optional[str], model: Optional[str] = None,
    ) -> LLMResponse:
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

    def _stream_mock(
        self, prompt: str, temperature: float,
        system: Optional[str], model: Optional[str] = None,
    ) -> Iterator[str]:
        import time as _time
        full = self._complete_mock(prompt, temperature, system, model).text
        for word in full.split():
            yield word + " "
            _time.sleep(0.02)
