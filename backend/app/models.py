"""DataPilot — Pydantic models for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────

class Complexity(str, Enum):
    SIMPLE = "simple"
    NORMAL = "normal"
    COMPLEX = "complex"


class AgentName(str, Enum):
    PLANNER = "planner"
    DATA = "data"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    VERIFIER = "verifier"


class CacheStatus(str, Enum):
    HIT = "HIT"
    MISS = "MISS"
    DISABLED = "DISABLED"


# ── Request Models ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """User query request."""
    question: str = Field(..., min_length=1, max_length=2000)
    dataset_id: str = Field(..., description="Dataset filename or ID")
    use_cache: bool = Field(default=True)


class UploadResponse(BaseModel):
    """Response after dataset upload."""
    dataset_id: str
    filename: str
    rows: int
    columns: int
    column_names: list[str]


# ── Agent Step Tracking ───────────────────────────────────────────────

class AgentStep(BaseModel):
    """Tracks a single agent's execution within the pipeline."""
    agent: AgentName
    model_used: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    latency_ms: Optional[float] = None
    tool_calls: list[str] = Field(default_factory=list)
    status: str = "pending"
    summary: str = ""


class ToolCallRecord(BaseModel):
    """Record of a single MCP tool call."""
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    cache_status: CacheStatus = CacheStatus.MISS
    latency_ms: float = 0.0
    success: bool = True


# ── Response Models ───────────────────────────────────────────────────

class QueryResponse(BaseModel):
    """Full response to a user query."""
    question: str
    answer: str
    dataset_id: str
    visualization: Optional[str] = Field(None, description="Base64 PNG chart")
    chart_type: Optional[str] = None

    # Pipeline metadata
    complexity: Complexity = Complexity.NORMAL
    model_used: str = ""
    agent_steps: list[AgentStep] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)

    # Cache info
    cache_status: CacheStatus = CacheStatus.MISS
    semantic_cache_status: CacheStatus = CacheStatus.MISS

    # Performance
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    # Verification
    verified: bool = False
    verification_notes: str = ""


class DatasetInfo(BaseModel):
    """Information about an available dataset."""
    dataset_id: str
    filename: str
    rows: int
    columns: int
    column_names: list[str]
    size_bytes: int = 0


class MetricsResponse(BaseModel):
    """Observability metrics snapshot."""
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    semantic_cache_hit_rate: float = 0.0
    total_cache_hits: int = 0
    total_cache_misses: int = 0
    total_tool_calls: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    model_usage: dict[str, int] = Field(default_factory=dict)
    avg_tokens_per_request: float = 0.0
    total_failures: int = 0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    redis_connected: bool = False
    datasets_available: int = 0
    version: str = "1.0.0"
