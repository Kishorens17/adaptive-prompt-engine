"""
adaptive_prompt.py

The single, unified prompting strategy that replaces zero_shot, few_shot,
and chain_of_thought.

Core idea:
    Instead of using regex rules to decide which prompt template to apply,
    we write ONE meta-prompt that instructs the LLM to read the question
    and calibrate its own answer depth. The model is intelligent enough to
    know when a question needs a one-word answer vs. a step-by-step proof
    — we just need to tell it to use that intelligence.

This satisfies the project's core goal:
    "The model must understand the question and give a clear answer using
     the minimum amount of tokens" — without the engine making that decision
     through rules.

The prompt template (used for all queries):
    System: You are a precise assistant. Read the question carefully and
            calibrate your answer depth to exactly what the question needs.
            [rules for calibrating depth...]
    User:   {query}

Optional escalation:
    If the confidence score returned is below the threshold, the engine
    re-queries with an explicit "think step by step" instruction as a
    last resort (handled by SelfConsistencyStrategy, unchanged).
"""

from __future__ import annotations

from strategies.base_strategy import PromptStrategy

# The meta-prompt system instruction.
# This is the single source of truth for how the engine communicates
# with the LLM. No regex, no keyword matching — the LLM reads this
# and decides the appropriate depth itself.
_SYSTEM_INSTRUCTION = """\
You are a precise, efficient assistant. Answer using the minimum words the question actually requires.

Calibrate your answer depth:
- Simple fact or lookup  → one word or very short phrase only (e.g. "Paris", "Hans Lippershey"). Nothing more.
- Needs a little context → 2–3 sentences max
- Needs explanation      → explain clearly and directly, no padding
- Needs step-by-step     → numbered steps, then a one-line summary

Rules:
1. Never restate or echo the question back.
2. No filler phrases like "Great question!", "Certainly!", "Of course!".
3. No unnecessary background or context the user didn't ask for.
4. Stop the instant the question is fully answered.
5. At the very end, append: [CONFIDENCE: x.xx]  (0.00–1.00).\
"""


class AdaptivePromptStrategy(PromptStrategy):
    """
    Single adaptive prompting strategy. The LLM calibrates its own
    verbosity based on the question — no rule-based routing needed.

    This replaces ZeroShotStrategy, FewShotStrategy, and
    ChainOfThoughtStrategy with a single, cleaner approach.
    """

    @property
    def name(self) -> str:
        return "adaptive"

    def build_prompt(self, query: str) -> str:
        # The system instruction does all the work.
        # The user turn is just the raw question — clean and direct.
        return query

    def execute(
        self,
        query: str,
        model: "str | None" = None,
        baseline_model: "str | None" = None,
    ):
        """
        Override execute() to pass the system instruction separately,
        and thread the model routing decision through to the LLM client.
        """
        prompt = self.build_prompt(query)
        llm_response = self.llm_client.complete(
            prompt=prompt,
            system=_SYSTEM_INSTRUCTION,
            model=model,
            baseline_model=baseline_model,
        )
        confidence = self.evaluator.score(
            query=query,
            response_text=llm_response.text,
        )
        return llm_response.text, confidence
