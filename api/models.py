"""
models.py — Pydantic request/response models for the REST API.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="The question or prompt to process")
    budget: Literal["low", "balanced", "quality"] = Field(
        "balanced",
        description="Model routing budget: low=cheapest, balanced=smart, quality=best",
    )
    provider: Literal["gemini", "mock"] = Field(
        "gemini",
        description="LLM provider to use",
    )


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
