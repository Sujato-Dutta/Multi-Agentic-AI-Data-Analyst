"""DataPilot — Observability and metrics tracking.

Lightweight application-level metrics without external monitoring infrastructure.
Tracks request counts, latencies, model usage, cache performance, and failures.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

from app.models import MetricsResponse

logger = logging.getLogger("datapilot.metrics")


@dataclass
class RequestRecord:
    """Record for a single request's metrics."""
    request_id: str
    question: str
    dataset_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_latency_ms: float = 0.0
    agent_latencies: dict[str, float] = field(default_factory=dict)
    model_selected: str = ""
    complexity: str = ""
    cache_hit: bool = False
    semantic_cache_hit: bool = False
    tool_calls: list[str] = field(default_factory=list)
    llm_calls: int = 0
    tokens_used: int = 0
    estimated_cost: float = 0.0
    success: bool = True
    error: Optional[str] = None


class MetricsCollector:
    """Collects and reports application-level metrics."""

    def __init__(self):
        self.records: list[RequestRecord] = []
        self.model_usage: dict[str, int] = defaultdict(int)
        self.total_failures: int = 0
        self._current_request: Optional[RequestRecord] = None

    def start_request(self, request_id: str, question: str, dataset_id: str) -> RequestRecord:
        """Begin tracking a new request."""
        record = RequestRecord(
            request_id=request_id,
            question=question,
            dataset_id=dataset_id,
        )
        self._current_request = record
        return record

    def end_request(self, record: RequestRecord) -> None:
        """Finalize and store a request record."""
        self.records.append(record)
        if record.model_selected:
            self.model_usage[record.model_selected] += 1
        if not record.success:
            self.total_failures += 1
        logger.info(
            "Request %s completed: latency=%.0fms model=%s cache=%s tokens=%d",
            record.request_id,
            record.total_latency_ms,
            record.model_selected,
            "HIT" if record.cache_hit else "MISS",
            record.tokens_used,
        )

    def record_agent_latency(self, record: RequestRecord, agent_name: str, latency_ms: float) -> None:
        """Record the latency for a specific agent."""
        record.agent_latencies[agent_name] = latency_ms

    def get_summary(self) -> MetricsResponse:
        """Generate a summary of all collected metrics."""
        if not self.records:
            return MetricsResponse()

        latencies = [r.total_latency_ms for r in self.records]
        cache_hits = sum(1 for r in self.records if r.cache_hit)
        semantic_hits = sum(1 for r in self.records if r.semantic_cache_hit)
        total = len(self.records)

        all_tool_calls = sum(len(r.tool_calls) for r in self.records)
        all_llm_calls = sum(r.llm_calls for r in self.records)
        all_tokens = sum(r.tokens_used for r in self.records)

        return MetricsResponse(
            total_requests=total,
            avg_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
            cache_hit_rate=cache_hits / total if total > 0 else 0.0,
            semantic_cache_hit_rate=semantic_hits / total if total > 0 else 0.0,
            total_cache_hits=cache_hits,
            total_cache_misses=total - cache_hits,
            total_tool_calls=all_tool_calls,
            total_llm_calls=all_llm_calls,
            total_tokens=all_tokens,
            model_usage=dict(self.model_usage),
            avg_tokens_per_request=all_tokens / total if total > 0 else 0.0,
            total_failures=self.total_failures,
        )

    def get_recent_records(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the most recent N request records as dicts."""
        recent = self.records[-n:]
        return [
            {
                "request_id": r.request_id,
                "question": r.question[:80],
                "dataset_id": r.dataset_id,
                "timestamp": r.timestamp.isoformat(),
                "total_latency_ms": r.total_latency_ms,
                "model_selected": r.model_selected,
                "complexity": r.complexity,
                "cache_hit": r.cache_hit,
                "semantic_cache_hit": r.semantic_cache_hit,
                "tool_calls_count": len(r.tool_calls),
                "llm_calls": r.llm_calls,
                "tokens_used": r.tokens_used,
                "estimated_cost": r.estimated_cost,
                "success": r.success,
            }
            for r in recent
        ]

    def reset(self) -> None:
        """Reset all metrics."""
        self.records.clear()
        self.model_usage.clear()
        self.total_failures = 0


# ── Structured Logging Setup ─────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    """Configure structured JSON-like logging for the application."""
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)


# ── Global metrics instance ──────────────────────────────────────────
metrics_collector = MetricsCollector()
