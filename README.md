# Adaptive Prompt Engine

> Intelligent LLM middleware that routes every query to the cheapest appropriate model,
> answers from your knowledge base automatically, remembers conversation history,
> lets the LLM judge its own answer quality, and caches semantically similar results —
> all without a single hardcoded rule.

[![Tests](https://img.shields.io/badge/tests-31%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Providers](https://img.shields.io/badge/providers-Gemini%20%7C%20OpenAI%20%7C%20Groq-orange)]()

---

## What This Does

The Adaptive Prompt Engine sits between your application and any LLM provider. It automatically:

| Capability | How |
|---|---|
| **Smart model routing** | Continuous complexity score (0–1) routes cheap queries to Flash/mini, hard ones to Pro/GPT-4o |
| **Multi-provider** | Gemini · OpenAI · Groq — switch with `--provider` |
| **Semantic cache** | Vectorized numpy similarity search; same question = 0 tokens spent |
| **RAG (auto)** | Upload docs → every query automatically uses them as context |
| **Tool use** | LLM calls your registered webhook tools via native function calling |
| **Session memory** | Multi-turn conversations with 24h expiry |
| **LLM-as-Judge** | LLM rates its own answer (accuracy, completeness, relevance, clarity) — no word-counting rules |
| **Streaming** | Token-by-token SSE stream endpoint |
| **REST API + Dashboard** | FastAPI with 20+ endpoints and a live analytics dashboard |

---

## Project Structure

```
adaptive_prompt_engine/
│
├── main.py                          ← Engine entry point (CLI + serve mode)
│
├── core/
│   ├── complexity_estimator.py      ← Continuous complexity score 0.0–1.0 (embeddings)
│   └── model_router.py              ← Routes tier → cheapest model + cost tracking
│
├── strategies/
│   ├── base_strategy.py             ← Abstract base class
│   ├── adaptive_prompt.py           ← Single meta-prompt (LLM self-calibrates depth)
│   ├── rag_strategy.py              ← RAG — auto-retrieves KB context before answering
│   ├── tool_use_strategy.py         ← Function calling + webhook tool execution
│   └── self_consistency.py          ← Escalation fallback (3× votes)
│
├── evaluator/
│   ├── confidence_evaluator.py      ← Rule-based scorer (fallback / mock provider)
│   └── llm_judge.py                 ← LLM-as-Judge: accuracy/completeness/relevance/clarity
│
├── cache/
│   ├── semantic_cache.py            ← Vectorized numpy similarity cache (SQLite)
│   ├── knowledge_base.py            ← RAG document store — chunk/embed/search
│   ├── session_store.py             ← Conversation memory with 24h TTL
│   └── query_log.py                 ← Full audit log (tokens, cost, latency)
│
├── llm/
│   └── llm_client.py                ← Gemini / OpenAI / Groq / Mock — unified interface
│
├── api/
│   ├── server.py                    ← FastAPI REST server (20+ endpoints)
│   └── models.py                    ← Pydantic request/response models
│
├── dashboard/
│   └── index.html                   ← Live analytics dashboard (dark mode)
│
├── handlers/
│   ├── base_handler.py              ← Chain of Responsibility base
│   └── self_consistency_handler.py  ← Last-resort escalation handler
│
├── experiments/
│   ├── benchmark_queries.py         ← 50-query test dataset
│   └── run_benchmark.py             ← Benchmark runner (CSV output)
│
├── tests/
│   ├── test_factory.py              ← Engine integration tests
│   ├── test_handlers.py             ← Handler layer tests
│   └── test_strategies.py           ← Strategy + evaluator tests
│
├── .env.example                     ← Copy to .env and fill in your keys
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & Create a Virtual Environment

```bash
git clone https://github.com/Kishorens17/adaptive-prompt-engine.git
cd adaptive-prompt-engine

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> If you see a Keras/tf-keras warning from sentence-transformers, run:
> ```bash
> pip install tf-keras
> ```

### 3. Configure API Keys

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required — pick at least one provider
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
```

> **Groq is free** with generous limits — great for testing. Get a key at [console.groq.com](https://console.groq.com).

---

## Running the Engine

### Interactive CLI (no server needed)

```bash
# Offline mode — no API key needed (for testing the pipeline)
python main.py

# With Gemini
python main.py --provider gemini

# With OpenAI
python main.py --provider openai

# With Groq (free + ultra-fast)
python main.py --provider groq

# Force cheapest model for all queries
python main.py --provider gemini --budget low

# Force best model for all queries
python main.py --provider gemini --budget quality

# Show full metadata: model used, tokens, cost, latency, confidence, judge rating
python main.py --provider gemini --verbose

# Disable semantic cache
python main.py --provider gemini --no-cache
```

### Single Query (non-interactive)

```bash
python main.py --provider gemini --query "What is the capital of France?"
python main.py --provider groq   --query "Explain how transformers work" --verbose
python main.py --provider openai --query "Write a haiku about recursion" --budget quality
```

### Start the REST API + Dashboard

```bash
python main.py --serve
```

The terminal will print:

```
───────────────────────────────────────────────────
  🚀  Adaptive Prompt Engine — API Server
───────────────────────────────────────────────────
  Local:      http://localhost:8000
  Network:    http://0.0.0.0:8000
  API docs:   http://localhost:8000/docs
  Dashboard:  http://localhost:8000/dashboard
───────────────────────────────────────────────────
```

Custom port:

```bash
python main.py --serve --port 9000
```

Or run uvicorn directly:

```bash
uvicorn api.server:app --reload --port 8000
```

### Run Tests

```bash
python -m pytest tests/ -v
```

Expected: **31 passed** (all offline — no API key needed)

### Run the Benchmark

```bash
# Offline / mock (instant)
python -m experiments.run_benchmark

# With a real provider
python -m experiments.run_benchmark --provider gemini --budget balanced
```

Results are saved to `experiments/results/benchmark_<timestamp>.csv`.

---

## API Reference

### Core Query

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/query` | Process a query, return answer + full metadata |
| `POST` | `/v1/query/stream` | Stream answer as Server-Sent Events |

**Example — regular query:**
```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "provider": "gemini",
    "budget": "balanced"
  }'
```

**Response:**
```json
{
  "answer": "Paris",
  "complexity_tier": "low",
  "complexity_score": 0.08,
  "model_used": "gemini-2.5-flash",
  "total_tokens": 42,
  "cost_usd": 0.000016,
  "cost_saved_usd": 0.000141,
  "latency_ms": 312.4,
  "cache_hit": false,
  "confidence": 0.94,
  "strategy_used": "adaptive",
  "quality": {
    "accuracy": 10,
    "completeness": 10,
    "relevance": 10,
    "clarity": 10,
    "overall": 0.97,
    "reasoning": "Direct, correct single-word answer to a simple factual question."
  }
}
```

**Example — streaming:**
```bash
curl -X POST http://localhost:8000/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain gravity", "provider": "gemini"}'
```

Events received:
```
data: {"chunk": "Gravity ", "done": false}
data: {"chunk": "is ", "done": false}
...
data: {"done": true, "answer": "Gravity is...", "model_used": "gemini-2.5-flash", ...}
```

---

### Session Memory (Multi-Turn)

```bash
# Create a session
SESSION=$(curl -s -X POST http://localhost:8000/v1/sessions | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# Ask within the session
curl -X POST http://localhost:8000/v1/sessions/$SESSION/query \
  -H "Content-Type: application/json" \
  -d '{"query": "My name is Kishore", "provider": "gemini"}'

# Follow-up — engine remembers the name
curl -X POST http://localhost:8000/v1/sessions/$SESSION/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is my name?", "provider": "gemini"}'

# View history
curl http://localhost:8000/v1/sessions/$SESSION/history

# Delete session
curl -X DELETE http://localhost:8000/v1/sessions/$SESSION
```

---

### Knowledge Base (RAG)

Once a document is uploaded, **all queries automatically use it as context**.

```bash
# Upload a document
curl -X POST http://localhost:8000/v1/knowledge-base/upload \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The Eiffel Tower is 330 metres tall and was completed in 1889.",
    "source": "facts.txt"
  }'

# Query — engine auto-retrieves relevant context
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How tall is the Eiffel Tower?", "provider": "gemini"}'

# List documents
curl http://localhost:8000/v1/knowledge-base

# Delete a document
curl -X DELETE http://localhost:8000/v1/knowledge-base/1
```

---

### Tool Registration

Register a webhook tool — the LLM will call it automatically when relevant.

```bash
curl -X POST http://localhost:8000/v1/tools/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters_schema": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "City name"}
      },
      "required": ["city"]
    },
    "webhook_url": "https://your-service.com/weather"
  }'

# List tools
curl http://localhost:8000/v1/tools

# Remove a tool
curl -X DELETE http://localhost:8000/v1/tools/get_weather
```

Built-in tools (always available, no webhook needed):
- `get_current_datetime` — returns current UTC time
- `calculate` — evaluates math expressions safely

---

### Analytics & Cache

```bash
# Aggregate statistics
curl http://localhost:8000/v1/stats

# Recent queries (last 50)
curl http://localhost:8000/v1/logs

# Daily token usage (last 7 days)
curl http://localhost:8000/v1/daily-usage

# Model tier distribution
curl http://localhost:8000/v1/model-dist

# Cache stats
curl http://localhost:8000/v1/cache/stats

# Clear cache
curl -X DELETE http://localhost:8000/v1/cache
```

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--provider` | `mock` | `mock` \| `gemini` \| `openai` \| `groq` |
| `--budget` | `balanced` | `low` \| `balanced` \| `quality` |
| `--model` | auto | Override model (skips smart routing) |
| `--api-key` | from `.env` | API key override |
| `--threshold` | `0.75` | Confidence threshold for escalation |
| `--query` | — | Single query, non-interactive mode |
| `--no-cache` | off | Disable semantic cache for this session |
| `--verbose` | off | Show full metadata (model, tokens, cost, judge score) |
| `--serve` | off | Start REST API server |
| `--port` | `8000` | Server port (used with `--serve`) |
| `--host` | `0.0.0.0` | Server host (used with `--serve`) |

---

## Model Routing Table

| Tier | Score | Gemini | OpenAI | Groq | Cost/1K tokens |
|---|---|---|---|---|---|
| LOW | 0.00–0.35 | gemini-2.5-flash | gpt-4o-mini | llama-3.3-70b | ~$0.00006–0.00015 |
| MEDIUM | 0.35–0.65 | gemini-2.5-flash | gpt-4o-mini | llama-3.3-70b | ~$0.00015–0.00038 |
| HIGH | 0.65–1.00 | gemini-2.5-pro | gpt-4o | llama-3.3-70b | ~$0.00250–0.00375 |

Override per tier via `.env`:
```env
ROUTER_LOW_MODEL_GEMINI=gemini-2.0-flash-lite
ROUTER_HIGH_MODEL_OPENAI=gpt-4o
```

---

## Environment Variables

```env
# ── API Keys ──────────────────────────────────────────────────────────────
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
GROQ_API_KEY=your_key

# ── Model Routing Overrides (optional) ───────────────────────────────────
ROUTER_LOW_MODEL_GEMINI=gemini-2.5-flash
ROUTER_HIGH_MODEL_GEMINI=gemini-2.5-pro
ROUTER_LOW_MODEL_OPENAI=gpt-4o-mini
ROUTER_HIGH_MODEL_OPENAI=gpt-4o

# ── Semantic Cache ────────────────────────────────────────────────────────
CACHE_SIMILARITY_THRESHOLD=0.92   # 0.0–1.0, higher = stricter match required

# ── Session Memory ────────────────────────────────────────────────────────
SESSION_TTL_HOURS=24              # Session expiry (resets on each query)
```

---

## How It Works — Architecture

```
User Query
    │
    ▼
SessionStore ──────── inject conversation history (if session query)
    │
    ▼
SemanticCache ─────── return instantly if similar query seen before (0 tokens)
    │ (miss)
    ▼
ComplexityEstimator ── embed query → cosine distance between simple/complex poles → 0.0–1.0 score
    │
    ▼
ModelRouter ─────────── score → tier (LOW/MEDIUM/HIGH) → cheapest model for that tier
    │
    ▼
Strategy Selector:
    ├── RAGStrategy       if KnowledgeBase has documents (auto)
    ├── ToolUseStrategy   if tools are registered (auto)
    └── AdaptivePromptStrategy  (default)
    │
    ▼
LLM Call (Gemini / OpenAI / Groq)
    │
    ▼
LLMJudgeEvaluator ── LLM rates its own answer: accuracy/completeness/relevance/clarity
    │  (fallback: rule-based ConfidenceEvaluator on error or mock provider)
    │
    ├── confidence ≥ 0.75 → done
    └── confidence < 0.75 → SelfConsistencyStrategy (3× calls + majority vote)
    │
    ▼
CacheStore + SessionStore.append + QueryLogger
    │
    ▼
Clean answer + metadata
```

---

## Design Patterns

| Pattern | Where | Why |
|---|---|---|
| **Strategy** | `strategies/` | Swap prompting techniques without touching routing logic |
| **Chain of Responsibility** | `handlers/` | Escalation chain: primary → self-consistency |
| **Factory Method** | `core/model_router.py` | Centralized model selection logic |

---

*Built as part of Software Design Patterns (SDP) coursework — Amrita Vishwa Vidyapeetham.*
