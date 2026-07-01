# CODEBASE INFO — Adaptive Prompt Engine v3.0

This document explains what every file does, how the architecture works,
and how to extend the system. Companion to `README.md`.

---

## 1. What This Project Is

A Python middleware system that sits between a user's query and an LLM API.

**v3.0 pipeline:**

```
Query → SessionStore → SemanticCache → ComplexityEstimator → ModelRouter
      → Strategy (RAG / ToolUse / Adaptive) → LLM
      → LLMJudgeEvaluator → [escalate?] → SelfConsistency
      → CacheWrite + SessionAppend + QueryLog → Answer
```

**Design patterns used:**
- **Strategy** — `strategies/` (swap prompting approach without touching routing)
- **Chain of Responsibility** — `handlers/` (escalation: primary → self-consistency)
- **Factory Method** — `core/model_router.py` (centralized model selection)

---

## 2. Full File Structure

```
adaptive_prompt_engine/
│
├── main.py                          Entry point — CLI + serve mode + engine facade
├── requirements.txt                 Python dependencies
├── pytest.ini                       Test runner config
├── .env.example                     API key template (copy to .env)
├── .gitignore
├── README.md
├── CODEBASE_INFO.md                 This file
│
├── core/
│   ├── complexity_estimator.py      Continuous complexity score 0.0–1.0
│   └── model_router.py              Tier → cheapest model + cost tracking
│
├── strategies/
│   ├── base_strategy.py             Abstract PromptStrategy base class
│   ├── adaptive_prompt.py           Single meta-prompt (LLM self-calibrates)  ← PRIMARY
│   ├── rag_strategy.py              RAG — retrieves KB context before answering ← NEW
│   ├── tool_use_strategy.py         Function calling + webhook tool execution   ← NEW
│   └── self_consistency.py          Escalation: 3× calls + majority vote
│
├── evaluator/
│   ├── confidence_evaluator.py      Rule-based scorer (fallback + mock)
│   └── llm_judge.py                 LLM-as-Judge: structured quality rubric     ← NEW
│
├── cache/
│   ├── semantic_cache.py            Vectorized numpy similarity cache (SQLite)
│   ├── knowledge_base.py            RAG document store — chunk/embed/search      ← NEW
│   ├── session_store.py             Conversation memory with 24h TTL             ← NEW
│   └── query_log.py                 Audit log (tokens, cost, latency per query)
│
├── llm/
│   └── llm_client.py                Gemini / OpenAI / Groq / Mock — unified API
│
├── api/
│   ├── server.py                    FastAPI REST server (20+ endpoints)
│   └── models.py                    Pydantic request/response models
│
├── handlers/
│   ├── base_handler.py              Chain of Responsibility abstract base
│   └── self_consistency_handler.py  Terminal escalation handler
│
├── dashboard/
│   └── index.html                   Dark-mode analytics dashboard
│
├── experiments/
│   ├── benchmark_queries.py         50-query benchmark dataset
│   └── run_benchmark.py             Benchmark runner → CSV output
│
└── tests/
    ├── conftest.py                  Path setup for pytest
    ├── test_factory.py              Engine integration tests
    ├── test_handlers.py             Handler escalation tests
    └── test_strategies.py           Strategy + evaluator tests
```

---

## 3. Layer Architecture

| Layer | Component | File | Responsibility |
|---|---|---|---|
| L1 — Input | `QueryReceiver` | `main.py` | Validate and sanitize raw query |
| L2 — Memory | `SessionStore` | `cache/session_store.py` | Inject conversation history |
| L3 — Cache | `SemanticCache` | `cache/semantic_cache.py` | Return instantly if seen before |
| L4 — Complexity | `ComplexityEstimator` | `core/complexity_estimator.py` | Score query difficulty 0–1 |
| L5 — Routing | `ModelRouter` | `core/model_router.py` | Map score → cheapest model |
| L6 — Strategy | `RAG/ToolUse/Adaptive` | `strategies/` | Build and send the right prompt |
| L7 — Evaluation | `LLMJudgeEvaluator` | `evaluator/llm_judge.py` | LLM rates its own answer |
| L8 — Escalation | `SelfConsistencyStrategy` | `strategies/self_consistency.py` | Retry if confidence too low |
| L9 — Output | `ResponseFormatter` | `main.py` | Clean and present the final answer |

---

## 4. ComplexityEstimator (`core/complexity_estimator.py`)

Scores a query's complexity as a continuous float in `[0.0, 1.0]`:

- Two "pole" embeddings averaged from anchor sentences:
  - **Simple pole**: "What is the capital of France?", "How many planets?", etc.
  - **Complex pole**: "Prove sqrt(2) is irrational step by step", "Write a detailed essay...", etc.
- Incoming query embedded → cosine similarity to each pole computed
- Score = `sim_complex / (sim_simple + sim_complex)` → normalized position between poles

No category boxes. "Somewhat complex" gets `0.5`, not forced into MEDIUM.

**Tier mapping:**

| Score | Tier |
|---|---|
| 0.00 – 0.35 | `LOW` |
| 0.35 – 0.65 | `MEDIUM` |
| 0.65 – 1.00 | `HIGH` |

---

## 5. ModelRouter (`core/model_router.py`)

Maps a `ComplexityTier` to the most cost-effective model per provider:

| Tier | Gemini | OpenAI | Groq |
|---|---|---|---|
| LOW | gemini-2.5-flash | gpt-4o-mini | llama-3.3-70b-versatile |
| MEDIUM | gemini-2.5-flash | gpt-4o-mini | llama-3.3-70b-versatile |
| HIGH | gemini-2.5-pro | gpt-4o | llama-3.3-70b-versatile |

Budget override:
- `low` → always use LOW tier model
- `balanced` → cap at MEDIUM even for HIGH complexity
- `quality` → always use HIGH tier model

All model prices tracked in `_COST_PER_1K` dict for cost saving calculations.

---

## 6. LLM Client (`llm/llm_client.py`)

Provider-agnostic wrapper. Supports:

| Provider | SDK | Key env var |
|---|---|---|
| `gemini` | `google-genai` | `GEMINI_API_KEY` |
| `openai` | `openai` | `OPENAI_API_KEY` |
| `groq` | `groq` | `GROQ_API_KEY` |
| `mock` | built-in | none |

Key methods:
- `complete(prompt, system, model, history)` → `LLMResponse`
- `complete_stream(prompt, system, model, history)` → generator of text chunks

Token counting:
- Gemini: real counts from `usage_metadata` (API-reported)
- OpenAI/Groq: exact BPE counts via `tiktoken`
- Mock: word-split approximation

Conversation history is passed through the `history` parameter — a list of
`{"role": "user"|"assistant", "content": str}` dicts. Each provider formats
this natively (Gemini uses `contents` list, OpenAI/Groq use `messages`).

**To add a new provider:** add `_init_<name>()` and `_complete_<name>()` methods.
The rest of the system is unaffected — they only call `client.complete()`.

---

## 7. Strategy Pattern (`strategies/`)

All strategies implement `PromptStrategy` with two methods:
- `name: str` — short identifier
- `build_prompt(query) -> str` — transform query into prompt text
- `execute(query, model, baseline_model) -> (text, confidence)` — run end-to-end

### AdaptivePromptStrategy (`adaptive_prompt.py`)
Single meta-prompt. System instruction tells LLM to calibrate its own verbosity:
- Simple fact → one word
- Needs context → 2–3 sentences
- Needs explanation → clear and direct
- Needs step-by-step → numbered steps

No regex, no keyword matching. The LLM reads the system instruction and decides.

### RAGStrategy (`rag_strategy.py`) ← NEW
Auto-activates when `KnowledgeBase.has_documents()` is True.
- Retrieves top-3 most relevant chunks via cosine similarity
- Prepends them as context with source labels
- System instruction tells LLM to answer ONLY from provided context
- Falls back to adaptive behavior when KB is empty

### ToolUseStrategy (`tool_use_strategy.py`) ← NEW
- Loads registered tools from `ToolRegistry`
- Uses provider's native function-calling API (OpenAI/Groq `tools=`, Gemini `FunctionDeclaration`)
- If LLM returns a tool call: executes it (builtin or webhook POST) → feeds result back → gets final answer
- Falls back to adaptive if no tools registered or provider is mock

Built-in tools (no webhook needed):
- `get_current_datetime` → current UTC time
- `calculate` → safe math expression evaluator

### SelfConsistencyStrategy (`self_consistency.py`)
Escalation only — called when primary strategy confidence < threshold.
Calls LLM 3× at higher temperature, takes majority vote on final answers.

---

## 8. LLM Judge (`evaluator/llm_judge.py`) ← NEW

Replaces rule-based confidence scoring with LLM self-evaluation.

**Judge prompt** asks the LLM to rate the answer on:
- `accuracy` (0–10): Is the information correct?
- `completeness` (0–10): Does it fully answer the question?
- `relevance` (0–10): Is it on topic?
- `clarity` (0–10): Is it well-structured?
- `overall` (0.0–1.0): Single quality score
- `reasoning`: One-sentence explanation

Returns JSON. `overall` → confidence score used for escalation decisions.

**Caching:** Results cached by `SHA256(query + answer)` in `cache/judge_cache.db`.
Same query+answer pair is never judged twice.

**Fallback:** If judge call fails or returns invalid JSON → silently uses old
`ConfidenceEvaluator` score. Zero regressions, never crashes the engine.

**Judge model:** Always uses the fastest/cheapest model (not the routed model):
- Gemini → `gemini-2.5-flash`
- OpenAI → `gpt-4o-mini`
- Groq → `llama-3.3-70b-versatile`

---

## 9. Semantic Cache (`cache/semantic_cache.py`)

SQLite-backed with vectorized numpy similarity search.

**v3.0 improvement:** All stored embeddings loaded into a numpy matrix on init.
Similarity computed as a single matrix-vector multiplication — ~100× faster
than the previous row-by-row Python loop.

```python
# O(N) but fully vectorized — one numpy call for all N entries
sims = (emb_matrix @ q_emb) / (norms * q_norm + 1e-10)
best_idx = np.argmax(sims)
```

Matrix updated incrementally on each `put()` — no full rebuild needed.

**Threshold:** Default 0.92 (configurable via `CACHE_SIMILARITY_THRESHOLD` in `.env`).
At 0.92: "What is the capital of France?" and "Tell me France's capital city?" → cache hit.
"What is the capital of Germany?" → cache miss (different answer).

---

## 10. Knowledge Base (`cache/knowledge_base.py`) ← NEW

SQLite-backed document store for RAG.

**Flow:**
1. `add_document(text, source)` → chunk into 400-word overlapping segments → embed each → store
2. `search(query, k=3)` → embed query → vectorized cosine similarity → return top-k chunks
3. `has_documents()` → triggers auto-RAG in the engine

**Chunking:** 400 words per chunk with 50-word overlap to preserve context at boundaries.

**Storage:** `cache/knowledge_base.db` (two tables: `documents`, `chunks`)

---

## 11. Session Store (`cache/session_store.py`) ← NEW

Conversation memory backed by SQLite.

- Sessions created via `create_session()` → returns UUID
- Messages appended via `append(session_id, role, content)`
- History retrieved as list of `{"role", "content", "timestamp"}` dicts
- **24h TTL** — expires after 24 hours of inactivity; reset on each query
- `_cleanup_expired()` runs on init to remove stale sessions
- **Graceful expiry:** expired session → 404 response, not a crash

---

## 12. REST API (`api/server.py`)

### Core Endpoints
| Method | Path | Description |
|---|---|---|
| POST | `/v1/query` | Process query, return answer + metadata |
| POST | `/v1/query/stream` | Streaming SSE — word-by-word then final metadata event |

### Sessions
| Method | Path | Description |
|---|---|---|
| POST | `/v1/sessions` | Create session |
| POST | `/v1/sessions/{id}/query` | Query within session |
| GET | `/v1/sessions/{id}/history` | Get message history |
| DELETE | `/v1/sessions/{id}` | Delete session |
| DELETE | `/v1/sessions/{id}/messages` | Clear messages only |

### Knowledge Base
| Method | Path | Description |
|---|---|---|
| POST | `/v1/knowledge-base/upload` | Upload and index a document |
| GET | `/v1/knowledge-base` | List all documents |
| DELETE | `/v1/knowledge-base/{doc_id}` | Delete a document |

### Tools
| Method | Path | Description |
|---|---|---|
| POST | `/v1/tools/register` | Register a webhook tool |
| GET | `/v1/tools` | List all tools |
| DELETE | `/v1/tools/{name}` | Remove a tool |

### Analytics
| Method | Path | Description |
|---|---|---|
| GET | `/v1/stats` | Aggregate stats |
| GET | `/v1/logs` | Recent query log |
| GET | `/v1/daily-usage` | Token/cost per day |
| GET | `/v1/model-dist` | Queries per complexity tier |
| GET | `/v1/cache/stats` | Cache size and hits |
| DELETE | `/v1/cache` | Clear cache |
| GET | `/dashboard` | Analytics dashboard UI |

---

## 13. Extending the System

### Add a new LLM provider
1. Add `_init_<name>()` and `_complete_<name>()` to `llm/llm_client.py`
2. Add cost table entry to `core/model_router.py._COST_PER_1K`
3. Add tier defaults to `core/model_router.py._TIER_DEFAULTS`
4. Add to `SUPPORTED_PROVIDERS` and `choices` in CLI args

### Add a new prompting strategy
1. Create `strategies/<name>.py` subclassing `PromptStrategy`
2. Implement `name` property and `build_prompt()` (or override `execute()`)
3. Wire it into `AdaptivePromptEngine._build_strategies()` in `main.py`

### Add a new built-in tool
Add to `_BUILTIN_TOOLS` dict and `_run_builtin()` in `strategies/tool_use_strategy.py`.
No other file needs to change.

### Swap confidence scoring approach
Implement a class with `.score(query, response_text) -> float` and `.threshold: float`.
Pass it as `evaluator=` in `AdaptivePromptEngine.__init__()`. Nothing else changes.

### Add a new API endpoint
Add a FastAPI route to `api/server.py` and a Pydantic model to `api/models.py`.
The engine facade in `main.py` is the only integration point needed.
