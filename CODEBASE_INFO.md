# CODEBASE INFO — Adaptive Prompt Chain Engine (APCE)

This document explains what every file in this codebase does, how the
three GoF design patterns are implemented, and how to extend the
system. It's a companion to `HOW_TO_RUN.md` (which tells you how to
*run* things) and to the original project context document (which
explains the *why* behind the architecture and the research framing).

---

## 1. What this project is

APCE is a Python middleware system that sits between a user's query
and an LLM API. It:

1. Classifies the query type (FACTUAL / REASONING / CREATIVE / ANALYTICAL)
2. Selects an appropriate starting prompting strategy
3. Calls the LLM and scores the response's confidence
4. Escalates to a stronger strategy if confidence is too low
5. Returns the final answer once a strategy clears the threshold (or
   the chain runs out of stronger options)

Three classic design patterns form the architectural backbone:
**Strategy**, **Chain of Responsibility**, and **Factory Method**.

---

## 2. Full file structure

```
adaptive_prompt_engine/
├── main.py                          Entry point / CLI, wires all 5 layers together
├── requirements.txt                 Python dependencies
├── pytest.ini                       Test runner configuration
├── .env.example                     Template for API key environment variables
├── .gitignore
├── HOW_TO_RUN.md                    Setup and usage instructions
├── CODEBASE_INFO.md                 This file
│
├── classifier/
│   ├── __init__.py
│   └── query_classifier.py          Detects FACTUAL/REASONING/CREATIVE/ANALYTICAL
│
├── strategies/                      STRATEGY PATTERN
│   ├── __init__.py
│   ├── base_strategy.py             Abstract PromptStrategy
│   ├── zero_shot.py                 ZeroShotStrategy
│   ├── few_shot.py                  FewShotStrategy
│   ├── chain_of_thought.py          ChainOfThoughtStrategy
│   └── self_consistency.py          SelfConsistencyStrategy (3x LLM calls + vote)
│
├── handlers/                        CHAIN OF RESPONSIBILITY PATTERN
│   ├── __init__.py
│   ├── base_handler.py              Abstract PromptHandler + HandlerResult
│   ├── zero_shot_handler.py
│   ├── few_shot_handler.py
│   ├── cot_handler.py
│   └── self_consistency_handler.py  Always terminal (last resort)
│
├── factory/                         FACTORY METHOD PATTERN
│   ├── __init__.py
│   └── strategy_factory.py          StrategyFactory + HandlerChainBuilder
│
├── evaluator/
│   ├── __init__.py
│   └── confidence_evaluator.py      Rule-based 0.0–1.0 confidence scorer
│
├── llm/
│   ├── __init__.py
│   └── llm_client.py                Provider-agnostic LLM wrapper (mock/OpenAI/Gemini)
│
├── experiments/
│   ├── __init__.py
│   ├── benchmark_queries.py         50-query benchmark dataset (4 types)
│   ├── run_benchmark.py             Runs adaptive vs zero-shot vs CoT baselines
│   └── results/
│       └── benchmark_*.csv          Sample benchmark output (already generated)
│
└── tests/
    ├── __init__.py
    ├── conftest.py                  Path setup so pytest finds the package
    ├── test_strategies.py           Tests for Strategy classes + evaluator
    ├── test_handlers.py             Tests for Chain of Responsibility
    └── test_factory.py              Tests for Factory Method + classifier
```

---

## 3. The five-layer architecture

| Layer | Components | File(s) | Responsibility |
|---|---|---|---|
| L1 — Input | `QueryReceiver` | `main.py` | Accept and validate raw user query |
| L2 — Classification | `QueryClassifier` | `classifier/query_classifier.py` | Detect query type |
| L3 — Creation | `StrategyFactory`, `HandlerChainBuilder` | `factory/strategy_factory.py` | Build the right handler chain |
| L4 — Execution | `PromptHandler`s, `PromptStrategy`s | `handlers/`, `strategies/` | Format prompt, call LLM, score confidence |
| L5 — Output | `ResponseFormatter` | `main.py` | Clean and return the final answer |

Each layer only talks to the layer next to it. `main.py`'s
`AdaptivePromptChainEngine` class is the facade that wires all five
together — most external integrations (a web API, a different CLI)
should depend on that class rather than reaching into individual
layers.

---

## 4. Pattern 1 — Strategy (`strategies/`)

**Problem it solves:** without Strategy, you'd have one function full
of `if query_type == 'factual': ... elif query_type == 'reasoning': ...`
branches. Every new prompting technique means editing that function
and risking breaking the existing branches.

**How it's implemented here:** `base_strategy.py` defines the abstract
`PromptStrategy` with a single contract:

```python
execute(query: str) -> tuple[str, float]   # (response_text, confidence_score)
```

Four concrete strategies implement it:

- **`ZeroShotStrategy`** — sends the query directly, minimal wrapping. Cheapest.
- **`FewShotStrategy`** — prepends worked examples to steer style/format.
- **`ChainOfThoughtStrategy`** — asks the model to reason step by step.
- **`SelfConsistencyStrategy`** — calls the LLM 3x (configurable via
  `num_samples`) at a higher temperature, takes a majority vote over
  extracted final answers, and blends the agreement ratio with the
  rule-based evaluator score. This is the strongest and most expensive
  strategy, always used as the last resort in every chain.

**To add a new strategy:** create a new file in `strategies/`
subclassing `PromptStrategy`, implement `name` and `build_prompt`.
That's it — nothing in `handlers/` or `factory/` needs to change
unless you also want a dedicated Handler/chain-position for it (see
Pattern 2 and 3 below).

---

## 5. Pattern 2 — Chain of Responsibility (`handlers/`)

**Problem it solves:** escalation logic ("if this strategy's answer
isn't good enough, try a stronger one") needs to be decoupled from any
specific ordering of strategies, so the chain can be reordered/extended
without touching existing handler code.

**How it's implemented here:** `base_handler.py` defines the abstract
`PromptHandler`. Each handler wraps exactly one `PromptStrategy` and
implements the shared `handle()` method (you don't need to override
this in subclasses — it's implemented once in the base class):

```python
def handle(self, query):
    response, confidence = self.strategy.execute(query)
    if confidence >= self.threshold or self.is_terminal:
        return HandlerResult(...)          # done
    return self._next.handle(query, ...)   # escalate
```

A handler only knows about `self._next` as an opaque reference — it
never knows what kind of handler comes after it, or how many more
links remain. `set_next()` wires handlers together; `is_terminal`
(true when `set_next` was never called) means "always return,
regardless of confidence" — this is how `SelfConsistencyHandler` is
guaranteed to be the final answer-giver in every chain.

Four concrete handlers exist, one per strategy:
`ZeroShotHandler`, `FewShotHandler`, `CoTHandler`, `SelfConsistencyHandler`.

`HandlerResult` (also in `base_handler.py`) carries the final answer,
confidence, which strategy actually answered, the full escalation
trail (list of strategy names that were tried and rejected), and the
total number of underlying LLM calls made (useful for cost analysis in
the benchmark).

---

## 6. Pattern 3 — Factory Method (`factory/strategy_factory.py`)

**Problem it solves:** something has to decide which Handler chain to
build for a given query type, and centralizing that decision means the
rest of the codebase never needs to import concrete Handler classes
directly — reducing coupling.

**How it's implemented here:** `StrategyFactory.create_chain(query_type)`
builds and wires the appropriate chain, returning the entry-point
handler. The escalation paths implemented match the project's Table 4:

| Query Type | Chain (entry → escalation → ... → terminal) |
|---|---|
| FACTUAL | Zero-Shot → Few-Shot → Self-Consistency |
| REASONING | CoT → Self-Consistency |
| CREATIVE | Few-Shot → CoT → Self-Consistency |
| ANALYTICAL | CoT → Few-Shot → Self-Consistency |

Every chain terminates at `SelfConsistencyHandler` so there's always a
final answer, no matter how a query escalates.

`HandlerChainBuilder` is a small convenience class combining
`QueryClassifier.classify()` and `StrategyFactory.create_chain()` into
one call — `build_chain_for_query(query)` returns
`(entry_handler, detected_query_type)`. `main.py` uses this rather than
calling the classifier and factory separately.

**To change an escalation path:** edit the relevant `_build_*_chain()`
private method in `StrategyFactory`. No other file needs to change.

**To add a fifth query type/strategy combination:** add the enum value
to `QueryType` in `classifier/query_classifier.py`, add a new
`_build_..._chain()` method in `StrategyFactory`, and register it in
the `builders` dict inside `create_chain()`.

---

## 7. The Confidence Evaluator (`evaluator/confidence_evaluator.py`)

Not a GoF pattern, but the decision engine that makes the Chain of
Responsibility's escalation choices meaningful. Returns a float in
`[0.0, 1.0]` by combining three signals:

1. **Self-reported confidence tag** — if the LLM was prompted to
   append `[CONFIDENCE: 0.82]`, that's parsed directly via regex and
   used as the dominant signal (weighted 85%), lightly adjusted (15%)
   by the hedge-language signal below so a confidently-tagged-but-
   visibly-hedging response doesn't get a free pass.
2. **Uncertainty/hedge language detection** — counts hedging phrases
   ("I think", "possibly", "I'm not sure", etc.) with diminishing
   penalty per additional hedge found.
3. **Response length relative to query complexity** — queries matching
   "complex" patterns (prove, compare, explain why, derive, ...) are
   expected to need longer answers; a short answer to such a query is
   penalized proportionally.

When no self-reported tag is present, signals 2 and 3 are blended
50/50 by default (`EvaluatorWeights`).

This is deliberately rule-based and fully explainable rather than a
trained model — per the project's design rationale, this keeps the
system inspectable for the accompanying research paper, with a
trained reward model noted as a future-work direction.

---

## 8. The LLM Client (`llm/llm_client.py`)

A thin, provider-agnostic wrapper. Every Strategy depends on this
class, never on the OpenAI or Gemini SDKs directly — so switching
providers, or adding a new one, only touches this one file.

Three providers are supported:

- **`mock`** (default) — deterministic, offline, no API key or network
  needed. Produces plausible-looking text that scales loosely with
  prompt complexity, so the rest of the pipeline (confidence scoring,
  escalation) can be exercised meaningfully without spending API
  budget. This is what the test suite uses exclusively.
- **`openai`** — wraps the official `openai` Python SDK
  (`pip install openai`, requires `OPENAI_API_KEY`).
- **`gemini`** — wraps `google-generativeai`
  (`pip install google-generativeai`, requires `GEMINI_API_KEY`).

`LLMResponse` is the normalized return shape (`text`, `prompt_tokens`,
`completion_tokens`, `total_tokens`, `latency_seconds`) so the rest of
the system never has to branch on which provider answered.

---

## 9. The Query Classifier (`classifier/query_classifier.py`)

Lightweight keyword/regex heuristic classifier — no GPU, no embeddings
required, per the project's tech-stack rationale (a heavier
sentence-transformers-based semantic classifier could be swapped in
later behind the same `classify(query) -> QueryType` interface without
touching any other module).

Pattern banks exist for each of the four `QueryType` values
(`FACTUAL`, `REASONING`, `CREATIVE`, `ANALYTICAL`). When a query
matches multiple categories, a fixed precedence order
(`CREATIVE > REASONING > ANALYTICAL > FACTUAL`) breaks the tie, since
creative/reasoning intent phrasing tends to be more deliberate/specific
than a generic factual-looking question opener.

---

## 10. The benchmark harness (`experiments/`)

- **`benchmark_queries.py`** — 50 queries, evenly spread across the
  four query types (12-13 each), each labeled with its expected type.
- **`run_benchmark.py`** — runs every query through three
  configurations (`adaptive`, `always_zero_shot`, `always_cot`),
  records the strategy that answered, confidence score, escalation
  trail, and total LLM call count for each, writes everything to a CSV
  under `experiments/results/`, and prints a console summary table.

This directly produces the data needed for Section 5 ("Experiments
and Results") of the research paper structure outlined in the
project's original context document — comparing the adaptive system
against naive single-strategy baselines on quality (via confidence as
a proxy, or your own manual/BLEU scoring on the saved CSV) and token
cost (via the LLM-call count, or actual token counts if you switch to
a real provider, since `LLMResponse` already tracks `total_tokens`).

---

## 11. The test suite (`tests/`)

57 tests, all running offline against the mock LLM provider — no API
key needed to verify the system works:

- **`test_strategies.py`** — each Strategy's `name`, `build_prompt`,
  and `execute` behavior; `SelfConsistencyStrategy`'s majority-vote and
  multi-call logic; every signal inside `ConfidenceEvaluator`.
- **`test_handlers.py`** — escalation behavior using a `StubStrategy`/
  `StubHandler` test double for precise confidence control, plus
  integration tests against the real concrete handlers with the mock
  LLM, including a full zero-shot → self-consistency escalation trail
  and LLM-call counting.
- **`test_factory.py`** — `QueryClassifier` routing for representative
  queries of each type, `StrategyFactory`'s chain construction (entry
  point matches Table 4, every chain terminates at
  `SelfConsistencyHandler`), and `HandlerChainBuilder`'s end-to-end
  convenience path.

---

## 12. Extending the system — common scenarios

**Add a new prompting strategy (e.g. "Tree of Thought"):**
1. Create `strategies/tree_of_thought.py` subclassing `PromptStrategy`.
2. Create `handlers/tot_handler.py` subclassing `PromptHandler`,
   wrapping the new strategy.
3. Wire it into one or more `_build_*_chain()` methods in
   `factory/strategy_factory.py`.
4. Add tests mirroring the existing pattern in `tests/test_strategies.py`
   and `tests/test_handlers.py`.

No existing strategy, handler, or factory code needs to be modified —
only added to. This is the Open/Closed Principle benefit the project's
research paper argues for.

**Add a new query type:**
1. Add the value to `QueryType` in `classifier/query_classifier.py`
   and a pattern bank for it.
2. Add a `_build_<type>_chain()` method in `StrategyFactory` and
   register it in the `builders` dict.

**Swap in a different confidence-scoring approach (e.g. a trained
reward model):**
Implement a class with the same `score(query, response_text) -> float`
interface as `ConfidenceEvaluator` and pass an instance of it wherever
`evaluator=` is accepted (strategies, handlers, the factory). Nothing
else needs to change.

**Add a new LLM provider:**
Add a new branch in `LLMClient.__init__` and a `_complete_<provider>()`
method in `llm/llm_client.py`. Every Strategy/Handler/Factory keeps
working unmodified since they only depend on `LLMClient.complete()`'s
normalized `LLMResponse` return shape.
