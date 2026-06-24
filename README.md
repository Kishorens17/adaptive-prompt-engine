# Adaptive Prompt Engine

> Intelligent LLM middleware that routes every query to the cheapest appropriate model,
> caches semantically similar results, and lets the LLM calibrate its own answer depth —
> all with zero rule-based routing.

---

## What This Project Is

The Adaptive Prompt Engine is a Python-based LLM middleware layer. You put it between
your application and any LLM provider (Gemini, OpenAI). It automatically:

- **Scores the complexity** of every incoming query (continuous 0–1 score, not rigid categories)
- **Routes to the cheapest model** that can handle that complexity
- **Caches semantically similar queries** so identical or paraphrased questions cost zero tokens
- **Uses a single adaptive meta-prompt** so the LLM itself decides how detailed to be
- **Logs every call** with full cost, latency, and token metadata
- **Exposes a REST API + live dashboard** for integration and monitoring

---

## Journey: What Was Built (Session by Session)

### Phase 0 — Initial Prototype (Before This Session)
The project started as a rule-based multi-strategy engine:
- **SemanticQueryClassifier** — classified queries into 4 boxes: FACTUAL / REASONING / CREATIVE / ANALYTICAL
- **StrategyFactory** — routed each box to a different handler chain
- **Handler Chain** (Chain of Responsibility) — ZeroShotHandler → FewShotHandler → CoTHandler → SelfConsistencyHandler
- **ChainOfThoughtStrategy** — used regex keyword matching (`_REASONING_KEYWORDS`, `_SIMPLE_FACTUAL_KEYWORDS`) to pick prompt templates
- **ConfidenceEvaluator** — rule-based scorer using hedge-phrase detection + word-count heuristics

**Problem identified:** The engine was contradicting its own goal. It was using rules to decide what the LLM should do, when the LLM itself is intelligent enough to calibrate its own verbosity.

---

### Phase 1 — Token Efficiency Fix (Early This Session)
- Added `verbose` flag to `ResponseFormatter` — debug metadata (query type, handler name, escalations, LLM call count) now hidden by default
- Stripped `[CONFIDENCE: x.xx]` tag from user-facing output (it's internal signal, not user text)
- Added `--verbose` CLI flag to `main.py`
- Added 3-tier prompt selection to `ChainOfThoughtStrategy` (later superseded)

---

### Phase 2 — Core Architecture Refactor
**Removed** (rule-based, no longer needed):
- `classifier/query_classifier.py` — the entire 4-category semantic classifier
- `factory/strategy_factory.py` — the routing-by-category factory
- `strategies/zero_shot.py` — replaced by single adaptive strategy
- `strategies/few_shot.py` — replaced by single adaptive strategy
- `strategies/chain_of_thought.py` — replaced by single adaptive strategy
- `handlers/zero_shot_handler.py`, `few_shot_handler.py`, `cot_handler.py`

**Created:**

#### `core/complexity_estimator.py`
Replaces the 4-category classifier with a **continuous complexity score** (0.0–1.0).
- Uses the same `sentence-transformers` model already in the project (no new imports)
- Embeds the query and measures its cosine distance between a "simple pole" and a "complex pole"
- 0.0 = maximally simple (factual lookup), 1.0 = maximally complex (detailed proof/essay)
- No artificial category boxes. "Sort of complex" gets a mid-range score.

#### `core/model_router.py`
Maps complexity score to the cheapest model for that tier:

| Tier | Score | Gemini | OpenAI |
|------|-------|--------|--------|
| LOW | 0.00–0.35 | `gemini-2.5-flash` | `gpt-3.5-turbo` |
| MEDIUM | 0.35–0.65 | `gemini-2.5-flash` | `gpt-4o-mini` |
| HIGH | 0.65–1.00 | `gemini-2.5-pro` | `gpt-4o` |

Also tracks cost per 1K tokens and computes `cost_saved_usd` vs always using the premium model.

#### `strategies/adaptive_prompt.py`
A **single meta-prompt** replacing all three old strategies.
The system instruction tells the LLM to calibrate its own depth:
- Simple fact → one phrase or sentence
- Needs context → 2–3 sentences
- Needs explanation → clear and direct, no padding
- Needs step-by-step → numbered steps, then a one-line summary

No regex. No keyword matching. The LLM reads and decides.

#### `llm/llm_client.py` (modified)
- Added `model` and `baseline_model` override parameters to `complete()`
- Added `cost_usd` and `cost_saved_usd` to `LLMResponse`
- Gemini now uses `system_instruction` natively in `GenerateContentConfig`

---

### Phase 3 — Semantic Caching

#### `cache/semantic_cache.py`
SQLite-backed cache with embedding-based similarity matching:
1. Every query is embedded with `sentence-transformers`
2. Cosine similarity compared against all stored embeddings
3. If best match > 0.92 → return cached answer (0 LLM tokens, instant)
4. On miss → call LLM, store result for next time

Configurable via `CACHE_SIMILARITY_THRESHOLD` in `.env`.

#### `cache/query_log.py`
Append-only audit log of every engine call:
- Query, complexity tier, model used, input/output tokens
- Cost in USD, cost saved vs baseline
- Latency in ms, cache hit flag, confidence score, timestamp
- Powers the analytics dashboard and future fine-tuning data export

---

### Phase 4 — REST API

#### `api/server.py` (FastAPI)
| Endpoint | Method | Description |
|---|---|---|
| `/v1/query` | POST | Process a query, return answer + full metadata |
| `/v1/stats` | GET | Aggregate stats (cost, tokens, cache hit rate) |
| `/v1/logs` | GET | Recent query log entries |
| `/v1/daily-usage` | GET | Token + cost per day (last N days) |
| `/v1/model-dist` | GET | Query count per complexity tier |
| `/v1/cache/stats` | GET | Cache size and hit count |
| `/v1/cache` | DELETE | Clear the semantic cache |
| `/dashboard` | GET | Web analytics dashboard |

---

### Phase 5 — Web Analytics Dashboard

#### `dashboard/index.html`
A single-page dark-mode dashboard with:
- **Stats bar** — Total queries, cost saved, total spend, avg latency (auto-refreshes every 15s)
- **Token usage chart** — Daily token usage for the last 7 days (Chart.js line graph)
- **Model tier distribution** — Donut chart showing how often each tier was used
- **Try It Live panel** — Type a query, choose budget + provider, see answer + metadata instantly
- **Recent query log** — Scrollable table of the last 30 queries with all metadata

---

### Phase 6 — Cleanup
- Deleted all 8 obsolete files (`classifier/`, `factory/`, old strategy files, old handlers)
- Rewrote all 3 test files to match the new architecture
- Rewrote the benchmark runner to use the new engine
- **31/31 tests pass**

---

## Final Project Structure

```
adaptive_prompt_engine/
│
├── main.py                         ← Engine entry point (CLI + serve mode)
│
├── core/
│   ├── complexity_estimator.py     ← Continuous query complexity score (0.0–1.0)
│   └── model_router.py             ← Cheapest-model routing + cost tracking
│
├── strategies/
│   ├── base_strategy.py            ← Abstract base
│   ├── adaptive_prompt.py          ← Single meta-prompt (THE core strategy)
│   └── self_consistency.py         ← Escalation fallback only
│
├── handlers/
│   ├── base_handler.py             ← Chain of Responsibility base
│   └── self_consistency_handler.py ← Last-resort escalation handler
│
├── cache/
│   ├── semantic_cache.py           ← SQLite + embedding cache
│   └── query_log.py                ← Full audit log
│
├── evaluator/
│   └── confidence_evaluator.py     ← Rule-based confidence scorer (safety net)
│
├── llm/
│   └── llm_client.py               ← Provider-agnostic LLM wrapper (Gemini/OpenAI/Mock)
│
├── api/
│   ├── server.py                   ← FastAPI REST server
│   └── models.py                   ← Pydantic request/response models
│
├── dashboard/
│   └── index.html                  ← Web analytics dashboard
│
├── experiments/
│   ├── benchmark_queries.py        ← 50-query test set
│   └── run_benchmark.py            ← Benchmark runner (CSV output)
│
├── tests/
│   ├── test_factory.py             ← Engine integration tests
│   ├── test_handlers.py            ← Handler layer tests
│   └── test_strategies.py          ← Strategy + evaluator tests
│
├── .env                            ← API keys + config (never commit this)
├── requirements.txt
└── README.md                       ← This file
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> If you see a Keras version warning, also run:
> ```bash
> pip install tf-keras
> ```

### 2. Set Your API Key

Open `.env` and fill in your key:

```env
GEMINI_API_KEY=your_key_here
# or
OPENAI_API_KEY=your_key_here
```

### 3. Run Interactively (CLI)

```bash
# Default: mock provider (no API key needed, for testing)
python main.py

# With Gemini (smart model routing)
python main.py --provider gemini

# With budget control
python main.py --provider gemini --budget low       # always cheapest model
python main.py --provider gemini --budget quality   # always best model

# Show full metadata (tokens, cost, model, latency)
python main.py --provider gemini --verbose

# Disable semantic cache
python main.py --provider gemini --no-cache
```

### 4. Run a Single Query (non-interactive)

```bash
python main.py --provider gemini --query "what is the capital of france?"
python main.py --provider gemini --query "why does ice float on water?" --verbose
```

### 5. Start the REST API + Dashboard

```bash
python main.py --serve
```

Then open:
- **API docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8000/dashboard

Or run uvicorn directly:
```bash
uvicorn api.server:app --reload --port 8000
```

### 6. Call the API

```bash
# Ask a question
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "budget": "balanced", "provider": "gemini"}'

# Get stats
curl http://localhost:8000/v1/stats

# View recent queries
curl http://localhost:8000/v1/logs?limit=10

# Clear cache
curl -X DELETE http://localhost:8000/v1/cache
```

### 7. Run Tests

```bash
python -m pytest tests/ -v
```

Expected output: **31 passed**

### 8. Run the Benchmark

```bash
# Offline (mock provider, instant)
python -m experiments.run_benchmark

# With real LLM
python -m experiments.run_benchmark --provider gemini --budget balanced
```

Results saved to `experiments/results/benchmark_<timestamp>.csv`.

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--provider` | `mock` | `mock` \| `gemini` \| `openai` |
| `--budget` | `balanced` | `low` \| `balanced` \| `quality` |
| `--model` | auto | Override model (skips smart routing) |
| `--api-key` | from `.env` | API key override |
| `--threshold` | `0.75` | Confidence threshold for escalation |
| `--query` | — | Single query, non-interactive |
| `--no-cache` | off | Disable semantic cache |
| `--verbose` | off | Show full metadata in output |
| `--serve` | off | Start REST API server |

---

## Environment Variables (`.env`)

```env
# LLM keys
GEMINI_API_KEY=...
OPENAI_API_KEY=...

# Model routing overrides (optional)
ROUTER_LOW_MODEL_GEMINI=gemini-2.5-flash
ROUTER_MEDIUM_MODEL_GEMINI=gemini-2.5-flash
ROUTER_HIGH_MODEL_GEMINI=gemini-2.5-pro

# Cache sensitivity (0.0–1.0, higher = stricter matching)
CACHE_SIMILARITY_THRESHOLD=0.92
```
