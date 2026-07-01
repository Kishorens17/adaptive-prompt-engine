"""
server.py — FastAPI REST API for the Adaptive Prompt Engine.

Endpoints:
    POST /v1/query              — Process query, return answer + metadata
    POST /v1/query/stream       — Streaming SSE response (word-by-word)

    GET  /v1/stats              — Aggregate statistics
    GET  /v1/logs               — Recent query log entries
    GET  /v1/daily-usage        — Token + cost usage per day
    GET  /v1/model-dist         — Query count by complexity tier

    DELETE /v1/cache            — Clear semantic cache
    GET    /v1/cache/stats      — Cache size and hit stats

    POST   /v1/sessions                 — Create a conversation session
    POST   /v1/sessions/{id}/query      — Query within a session (history-aware)
    GET    /v1/sessions/{id}/history    — Get session messages
    DELETE /v1/sessions/{id}            — Delete a session
    DELETE /v1/sessions/{id}/messages   — Clear messages (keep session)

    POST   /v1/knowledge-base/upload    — Upload and index a document
    GET    /v1/knowledge-base           — List indexed documents
    DELETE /v1/knowledge-base/{doc_id}  — Delete a document

    POST   /v1/tools/register   — Register a webhook tool
    GET    /v1/tools             — List all registered tools
    DELETE /v1/tools/{name}      — Remove a tool

    GET  /dashboard             — Analytics dashboard UI
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.models import (
    QueryRequest, QueryResponse, QualityDimensions,
    StatsResponse, DailyUsage, ModelDistribution,
    SessionCreateResponse, SessionQueryRequest,
    SessionHistoryResponse, MessageModel,
    DocumentUploadRequest, DocumentUploadResponse,
    DocumentInfo, KnowledgeBaseStats,
    ToolRegisterRequest, ToolInfo, ToolListResponse,
)
from cache.query_log import QueryLogger
from cache.semantic_cache import SemanticCache
from cache.session_store import SessionStore
from cache.knowledge_base import KnowledgeBase
from strategies.tool_use_strategy import ToolRegistry
from main import AdaptivePromptEngine

_CONFIDENCE_TAG_RE = re.compile(
    r"\s*\[CONFIDENCE:\s*[0-9]*\.?[0-9]+\s*\]", re.IGNORECASE
)

app = FastAPI(
    title="Adaptive Prompt Engine API",
    description=(
        "Intelligent LLM middleware with semantic caching, multi-provider routing, "
        "RAG, tool use, and conversation memory."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons
_logger = QueryLogger()
_cache = SemanticCache()
_sessions = SessionStore()
_knowledge_base = KnowledgeBase()
_tool_registry = ToolRegistry()

# Engine cache keyed by (provider, budget)
_engines: dict[str, AdaptivePromptEngine] = {}

SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "24"))


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


def _clean_answer(text: str) -> str:
    return _CONFIDENCE_TAG_RE.sub("", text).strip()


def _build_query_response(result) -> QueryResponse:
    clean = _clean_answer(result.answer)
    quality = None
    if result.quality:
        quality = QualityDimensions(**result.quality)
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
        quality=quality,
        strategy_used=result.strategy_used,
    )


# ===========================================================================
# Core query endpoints
# ===========================================================================

@app.post("/v1/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a query and return the answer with full metadata."""
    try:
        engine = _get_engine(request.provider, request.budget)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, engine.ask, request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _build_query_response(result)


@app.post("/v1/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """
    Stream the answer as Server-Sent Events.

    Events:
        data: {"chunk": "word ", "done": false}
        data: {"done": true, "answer": "...", "model_used": "...", ...}
    """
    engine = _get_engine(request.provider, request.budget)

    async def event_generator():
        loop = asyncio.get_event_loop()
        # Run full inference in thread pool (non-blocking for FastAPI)
        result = await loop.run_in_executor(None, engine.ask, request.query)
        clean = _clean_answer(result.answer)

        # Stream the answer word-by-word
        words = clean.split()
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
            await asyncio.sleep(0.015)

        # Final event with full metadata
        meta = {
            "done": True,
            "answer": clean,
            "model_used": result.model_used,
            "complexity_tier": result.complexity_tier,
            "complexity_score": round(result.complexity_score, 3),
            "total_tokens": result.total_tokens,
            "cost_usd": round(result.cost_usd, 6),
            "cost_saved_usd": round(result.cost_saved_usd, 6),
            "latency_ms": round(result.latency_ms, 1),
            "cache_hit": result.cache_hit,
            "confidence": round(result.confidence, 3),
            "strategy_used": result.strategy_used,
        }
        if result.quality:
            meta["quality"] = result.quality
        yield f"data: {json.dumps(meta)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ===========================================================================
# Analytics endpoints
# ===========================================================================

@app.get("/v1/stats", response_model=StatsResponse)
async def stats_endpoint():
    return StatsResponse(**_logger.aggregate_stats())


@app.get("/v1/logs")
async def logs_endpoint(limit: int = 50):
    return _logger.recent(limit=min(limit, 200))


@app.get("/v1/daily-usage")
async def daily_usage_endpoint(days: int = 7):
    return _logger.daily_token_usage(days=days)


@app.get("/v1/model-dist")
async def model_distribution_endpoint():
    return _logger.model_distribution()


@app.delete("/v1/cache")
async def clear_cache_endpoint():
    _cache.clear()
    return {"message": "Cache cleared."}


@app.get("/v1/cache/stats")
async def cache_stats_endpoint():
    return _cache.stats()


# ===========================================================================
# Session endpoints
# ===========================================================================

@app.post("/v1/sessions", response_model=SessionCreateResponse)
async def create_session():
    """Create a new conversation session."""
    sid = _sessions.create_session()
    return SessionCreateResponse(session_id=sid, expires_in_hours=SESSION_TTL_HOURS)


@app.post("/v1/sessions/{session_id}/query", response_model=QueryResponse)
async def session_query_endpoint(session_id: str, request: SessionQueryRequest):
    """Process a query within a session (history-aware)."""
    if not _sessions.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    history = _sessions.get_history(session_id)
    # Convert to LLM message format (strip timestamps)
    llm_history = [{"role": m["role"], "content": m["content"]} for m in history]

    try:
        engine = _get_engine(request.provider, request.budget)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, engine.ask_with_history, request.query, llm_history
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Persist turn to session
    clean = _clean_answer(result.answer)
    _sessions.append(session_id, "user", request.query)
    _sessions.append(session_id, "assistant", clean)

    return _build_query_response(result)


@app.get("/v1/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def session_history_endpoint(session_id: str):
    """Get the message history of a session."""
    if not _sessions.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    history = _sessions.get_history(session_id)
    messages = [MessageModel(**m) for m in history]
    return SessionHistoryResponse(
        session_id=session_id,
        messages=messages,
        message_count=len(messages),
    )


@app.delete("/v1/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    deleted = _sessions.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": "Session deleted."}


@app.delete("/v1/sessions/{session_id}/messages")
async def clear_session_messages_endpoint(session_id: str):
    cleared = _sessions.clear(session_id)
    if not cleared:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": "Session messages cleared."}


# ===========================================================================
# Knowledge base endpoints
# ===========================================================================

@app.post("/v1/knowledge-base/upload", response_model=DocumentUploadResponse)
async def upload_document_endpoint(request: DocumentUploadRequest):
    """Upload and index a document for RAG."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Document text must not be empty.")
    try:
        loop = asyncio.get_event_loop()
        doc_id = await loop.run_in_executor(
            None, _knowledge_base.add_document, request.text, request.source
        )
        # Refresh engines so RAG strategy is picked up
        for engine in _engines.values():
            engine.refresh_strategy()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    docs = _knowledge_base.list_documents()
    chunk_count = next((d.chunk_count for d in docs if d.doc_id == doc_id), 0)
    return DocumentUploadResponse(
        doc_id=doc_id,
        source=request.source,
        chunk_count=chunk_count,
        message=f"Document indexed successfully with {chunk_count} chunk(s).",
    )


@app.get("/v1/knowledge-base", response_model=KnowledgeBaseStats)
async def list_documents_endpoint():
    docs = _knowledge_base.list_documents()
    return KnowledgeBaseStats(
        document_count=len(docs),
        documents=[
            DocumentInfo(
                doc_id=d.doc_id, source=d.source,
                chunk_count=d.chunk_count, created_at=d.created_at,
            )
            for d in docs
        ],
    )


@app.delete("/v1/knowledge-base/{doc_id}")
async def delete_document_endpoint(doc_id: int):
    deleted = _knowledge_base.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")
    for engine in _engines.values():
        engine.refresh_strategy()
    return {"message": f"Document {doc_id} deleted."}


# ===========================================================================
# Tool endpoints
# ===========================================================================

@app.post("/v1/tools/register")
async def register_tool_endpoint(request: ToolRegisterRequest):
    """Register a webhook-based tool for LLM function calling."""
    _tool_registry.register(
        name=request.name,
        description=request.description,
        parameters_schema=request.parameters_schema,
        webhook_url=request.webhook_url,
    )
    return {"message": f"Tool '{request.name}' registered successfully."}


@app.get("/v1/tools", response_model=ToolListResponse)
async def list_tools_endpoint():
    tools = _tool_registry.list_tools()
    return ToolListResponse(
        tools=[ToolInfo(**t) for t in tools],
        count=len(tools),
    )


@app.delete("/v1/tools/{name}")
async def delete_tool_endpoint(name: str):
    deleted = _tool_registry.delete(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found.")
    return {"message": f"Tool '{name}' removed."}


# ===========================================================================
# Dashboard
# ===========================================================================

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
