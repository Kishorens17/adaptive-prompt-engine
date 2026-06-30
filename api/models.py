"""
models.py — Pydantic request/response models for the REST API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core query models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., description="The question or prompt to process")
    budget: Literal["low", "balanced", "quality"] = Field(
        "balanced",
        description="Model routing budget: low=cheapest, balanced=smart, quality=best",
    )
    provider: Literal["gemini", "openai", "groq", "mock"] = Field(
        "gemini",
        description="LLM provider to use",
    )


class QualityDimensions(BaseModel):
    accuracy: int
    completeness: int
    relevance: int
    clarity: int
    overall: float
    reasoning: str


class QueryResponse(BaseModel):
    answer: str
    complexity_tier: str
    complexity_score: float
    model_used: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    cost_saved_usd: float
    latency_ms: float
    cache_hit: bool
    confidence: float
    quality: Optional[QualityDimensions] = None
    strategy_used: Optional[str] = None


# ---------------------------------------------------------------------------
# Stats / analytics models
# ---------------------------------------------------------------------------

class StatsResponse(BaseModel):
    total_queries: int
    cache_hits: int
    total_tokens: int
    total_cost_usd: float
    total_saved_usd: float
    avg_latency_ms: float
    avg_confidence: float
    cache_hit_rate: float


class DailyUsage(BaseModel):
    day: str
    tokens: int
    cost_usd: float


class ModelDistribution(BaseModel):
    tier: str
    count: int


# ---------------------------------------------------------------------------
# Session models
# ---------------------------------------------------------------------------

class SessionCreateResponse(BaseModel):
    session_id: str
    expires_in_hours: int


class SessionQueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask in this session")
    budget: Literal["low", "balanced", "quality"] = "balanced"
    provider: Literal["gemini", "openai", "groq", "mock"] = "gemini"


class MessageModel(BaseModel):
    role: str
    content: str
    timestamp: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[MessageModel]
    message_count: int


# ---------------------------------------------------------------------------
# Knowledge base models
# ---------------------------------------------------------------------------

class DocumentUploadRequest(BaseModel):
    text: str = Field(..., description="Raw text content to index")
    source: str = Field("upload", description="Source identifier (filename, URL, etc.)")


class DocumentUploadResponse(BaseModel):
    doc_id: int
    source: str
    chunk_count: int
    message: str


class DocumentInfo(BaseModel):
    doc_id: int
    source: str
    chunk_count: int
    created_at: str


class KnowledgeBaseStats(BaseModel):
    document_count: int
    documents: List[DocumentInfo]


# ---------------------------------------------------------------------------
# Tool models
# ---------------------------------------------------------------------------

class ToolRegisterRequest(BaseModel):
    name: str = Field(..., description="Unique tool name (snake_case)")
    description: str = Field(..., description="What the tool does — shown to the LLM")
    parameters_schema: Dict[str, Any] = Field(
        ...,
        description="JSON Schema describing the tool's parameters",
    )
    webhook_url: str = Field(
        ...,
        description="URL to POST to when the LLM calls this tool",
    )


class ToolInfo(BaseModel):
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    webhook_url: str
    created_at: str


class ToolListResponse(BaseModel):
    tools: List[ToolInfo]
    count: int
