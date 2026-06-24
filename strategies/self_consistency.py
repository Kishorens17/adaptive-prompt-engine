"""
self_consistency.py

Concrete Strategy: Self-Consistency prompting.

The strongest (and most expensive) rung of the escalation ladder.
Calls the LLM multiple times (default: 3) with chain-of-thought-style
prompting and a non-zero temperature so the samples can disagree, then
takes a majority vote over the final answers. This trades token cost
for reliability and is reserved for queries that have escalated past
zero-shot, few-shot, and chain-of-thought without reaching the
confidence threshold.

Because this is the last handler in the chain (see
handlers/self_consistency_handler.py), it is also configured to always
"handle" — there is nowhere further to escalate to.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from strategies.base_strategy import PromptStrategy


class SelfConsistencyStrategy(PromptStrategy):
    """
    Samples the LLM `num_samples` times and returns the majority-vote answer.

    Confidence is derived from the agreement ratio: if 3/3 samples agree,
    confidence is high; if samples are split, confidence is lower. This
    is combined with (not replaced by) the underlying ConfidenceEvaluator
    score on the chosen response, per the project's confidence-scoring
    design (see evaluator/confidence_evaluator.py).
    """

    def __init__(self, llm_client, evaluator=None, num_samples: int = 3, sample_temperature: float = 0.9):
        super().__init__(llm_client, evaluator)
        self.num_samples = num_samples
        self.sample_temperature = sample_temperature

    @property
    def name(self) -> str:
        return "self_consistency"

    def build_prompt(self, query: str) -> str:
        return (
            "Think through this problem step by step, then give a clearly "
            "labeled final answer on its own line in the form "
            "'Final Answer: <answer>'.\n\n"
            f"Question: {query}\n\nLet's think step by step:"
        )

    def execute(
        self,
        query: str,
        model: "str | None" = None,
        baseline_model: "str | None" = None,
    ) -> Tuple[str, float]:
        prompt = self.build_prompt(query)
        samples: List[str] = []
        final_answers: List[str] = []

        for _ in range(self.num_samples):
            llm_response = self.llm_client.complete(
                prompt, temperature=self.sample_temperature,
                model=model, baseline_model=baseline_model,
            )
            samples.append(llm_response.text)
            final_answers.append(self._extract_final_answer(llm_response.text))

        majority_answer, agreement_ratio = self._majority_vote(final_answers)

        representative = next(
            (s for s, a in zip(samples, final_answers) if a == majority_answer),
            samples[0],
        )

        base_confidence = self.evaluator.score(query=query, response_text=representative)
        combined_confidence = (0.6 * agreement_ratio) + (0.4 * base_confidence)
        combined_confidence = max(0.0, min(1.0, combined_confidence))

        return representative, combined_confidence

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        """Pull out the 'Final Answer: ...' line, falling back to the last line."""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("final answer:"):
                return stripped.split(":", 1)[1].strip().lower()
        non_empty = [l.strip() for l in text.splitlines() if l.strip()]
        return non_empty[-1].lower() if non_empty else text.strip().lower()

    @staticmethod
    def _majority_vote(answers: List[str]) -> Tuple[str, float]:
        """Return (most_common_answer, agreement_ratio)."""
        if not answers:
            return "", 0.0
        counts = Counter(answers)
        winner, win_count = counts.most_common(1)[0]
        return winner, win_count / len(answers)
