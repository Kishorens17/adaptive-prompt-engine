"""
server.py — FastAPI REST API for the Adaptive Prompt Engine.

Endpoints:
    POST /v1/query        — Process a query, return answer + metadata
    GET  /v1/stats        — Aggregate statistics (cost, tokens, cache rate)
    GET  /v1/logs         — Recent query log entries
    GET  /v1/daily-usage  — Token + cost usage per day (last 7 days)
    GET  /v1/model-dist   — Query count by complexity tier
    DELETE /v1/cache      — Clear the semantic cache

Dashboard:
    GET /dashboard        — Web analytics UI (served from dashboard/)

Run:
    uvicorn api.server:app --reload
    # or: python main.py --serve
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as a module
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.models import (
    QueryRequest, QueryResponse, StatsResponse,
    DailyUsage, ModelDistribution,
)
from cache.query_log import QueryLogger
from cache.semantic_cache import SemanticCache
from main import AdaptivePromptEngine


app = FastAPI(
    title="Adaptive Prompt Engine API",
    description=(
        "Intelligent LLM middleware that routes queries to the cheapest "
        "appropriate model and caches semantically similar results."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons (created once at startup)
_logger = QueryLogger()
_cache  = SemanticCache()

# Engine instances keyed by (provider, budget) — lazy init on first use
_engines: dict[str, AdaptivePromptEngine] = {}


def _get_engine(provider: str, budget: str) -> AdaptivePromptEngine:
    key = f"{provider}:{budget}"
    if key not in _engines:
        _engines[key] = AdaptivePromptEngine(
            provider=provider,
            budget=budget,
            use_cache=True,
            verbose=False,
        )
    return _engines[key]


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.post("/v1/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Process a query and return the answer with full cost and routing metadata.
    """
    try:
        engine = _get_engine(request.provider, request.budget)
        result = engine.ask(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    import re
    clean = re.sub(r"\s*\[CONFIDENCE:\s*[0-9]*\.?[0-9]+\s*\]", "", result.answer).strip()

    return QueryResponse(
        answer=clean,
        complexity_tier=result.complexity_tier,
        complexity_score=round(result.complexity_score, 3),
        model_used=result.model_used,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        cost_usd=round(result.cost_usd, 6),
        cost_saved_usd=round(result.cost_saved_usd, 6),
        latency_ms=round(result.latency_ms, 1),
        cache_hit=result.cache_hit,
        confidence=round(result.confidence, 3),
    )


@app.get("/v1/stats", response_model=StatsResponse)
async def stats_endpoint():
    """Return aggregate statistics across all processed queries."""
    data = _logger.aggregate_stats()
    return StatsResponse(**data)


@app.get("/v1/logs")
async def logs_endpoint(limit: int = 50):
    """Return the most recent query log entries."""
    return _logger.recent(limit=min(limit, 200))


@app.get("/v1/daily-usage")
async def daily_usage_endpoint(days: int = 7):
    """Return token usage and cost per day for the last N days."""
    return _logger.daily_token_usage(days=days)


@app.get("/v1/model-dist")
async def model_distribution_endpoint():
    """Return query count by complexity tier."""
    return _logger.model_distribution()


@app.delete("/v1/cache")
async def clear_cache_endpoint():
    """Clear the semantic response cache."""
    _cache.clear()
    return {"message": "Cache cleared successfully."}


@app.get("/v1/cache/stats")
async def cache_stats_endpoint():
    """Return cache size and hit statistics."""
    return _cache.stats()


# ---------------------------------------------------------------------------
# Dashboard — static file serving
# ---------------------------------------------------------------------------

_DASHBOARD_DIR = _PROJECT_ROOT / "dashboard"

if _DASHBOARD_DIR.exists():
    app.mount(
        "/dashboard/static",
        StaticFiles(directory=str(_DASHBOARD_DIR)),
        name="dashboard_static",
    )

    @app.get("/dashboard")
    async def dashboard():
        return FileResponse(str(_DASHBOARD_DIR / "index.html"))
