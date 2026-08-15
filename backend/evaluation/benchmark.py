"""DataPilot — Evaluation benchmark runner.

Runs the pipeline against benchmark questions in three configurations:
- Baseline: Always Gemini 3.5 Flash Lite, no caching
- Experiment A: Intelligent routing, no caching
- Experiment B: Intelligent routing + Redis caching
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.agents.graph import run_pipeline
from app.cache.redis_cache import RedisCache
from app.cache.semantic_cache import SemanticCache
from app.config import get_settings
from app.mcp.server import load_dataset
from app.observability.metrics import MetricsCollector
from app.router.model_router import BaselineRouter, ModelRouter
from evaluation.scoring import score_result

logger = logging.getLogger("datapilot.eval")


@dataclass
class EvalResult:
    """Result for a single evaluation question."""
    question_id: int
    question: str
    expected_answer: str
    actual_answer: str
    complexity: str
    category: str
    config: str  # baseline, experiment_a, experiment_b

    # Metrics
    correct: bool = False
    score: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    cache_hit: bool = False
    semantic_cache_hit: bool = False
    model_used: str = ""
    estimated_cost: float = 0.0
    verified: bool = False
    error: Optional[str] = None


@dataclass
class EvalSummary:
    """Summary statistics for an evaluation configuration."""
    config: str
    total_questions: int = 0
    correct: int = 0
    accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_tokens: float = 0.0
    total_tokens: int = 0
    avg_llm_calls: float = 0.0
    avg_tool_calls: float = 0.0
    total_cost: float = 0.0
    avg_cost: float = 0.0
    cache_hit_rate: float = 0.0


async def run_evaluation(
    questions: list[dict[str, Any]],
    config: str,
    dry_run: bool = False,
    max_questions: Optional[int] = None,
) -> tuple[list[EvalResult], EvalSummary]:
    """Run evaluation for a single configuration.
    
    Args:
        questions: List of benchmark question dicts
        config: 'baseline', 'experiment_a', or 'experiment_b'
        dry_run: If True, skip actual LLM calls
        max_questions: Limit number of questions
    
    Returns:
        (results, summary)
    """
    settings = get_settings()

    # Configure router and caching based on config
    if config == "baseline":
        router = BaselineRouter()
        cache = None
        semantic_cache = None
    elif config == "experiment_a":
        router = ModelRouter()
        cache = None
        semantic_cache = None
    elif config == "experiment_b":
        router = ModelRouter()
        cache = RedisCache(
            redis_url=settings.upstash_redis_rest_url,
            redis_token=settings.upstash_redis_rest_token,
            ttl=settings.cache_ttl_seconds,
        )
        semantic_cache = SemanticCache(threshold=settings.semantic_cache_threshold)
    else:
        raise ValueError(f"Unknown config: {config}")

    metrics = MetricsCollector()
    results: list[EvalResult] = []

    # Limit questions
    eval_questions = questions[:max_questions] if max_questions else questions

    logger.info("Starting evaluation: config=%s questions=%d dry_run=%s", config, len(eval_questions), dry_run)

    for i, q in enumerate(eval_questions):
        logger.info("[%d/%d] Q: %s", i + 1, len(eval_questions), q["question"][:60])

        result = EvalResult(
            question_id=q["id"],
            question=q["question"],
            expected_answer=q.get("expected_answer", ""),
            actual_answer="",
            complexity=q.get("complexity", "unknown"),
            category=q.get("category", "unknown"),
            config=config,
        )

        if dry_run:
            result.actual_answer = "[DRY RUN] Skipped"
            result.latency_ms = 0
            result.score = 0
            results.append(result)
            continue

        try:
            start = time.time()
            response = await run_pipeline(
                question=q["question"],
                dataset_id=q.get("dataset_id", "sample_sales"),
                router=router,
                cache=cache,
                semantic_cache=semantic_cache,
                metrics=metrics,
                use_cache=(config == "experiment_b"),
            )
            elapsed = (time.time() - start) * 1000

            result.actual_answer = response.answer
            result.latency_ms = elapsed
            result.tokens_used = response.total_tokens
            result.llm_calls = response.llm_calls if hasattr(response, 'llm_calls') else len(response.agent_steps)
            result.tool_calls = len(response.tool_calls)
            result.cache_hit = response.cache_status.value == "HIT"
            result.semantic_cache_hit = response.semantic_cache_status.value == "HIT"
            result.model_used = response.model_used
            result.verified = response.verified
            result.estimated_cost = response.estimated_cost_usd

            # Score the result
            result.score, result.correct = score_result(
                q["question"], q.get("expected_answer", ""), response.answer
            )

        except Exception as e:
            logger.error("Eval error on Q%d: %s", q["id"], e)
            result.error = str(e)
            result.actual_answer = f"ERROR: {e}"

        results.append(result)

        # Small delay between requests to avoid rate limiting
        if not dry_run:
            await asyncio.sleep(0.5)

    # Compute summary
    summary = _compute_summary(results, config)
    return results, summary


def _compute_summary(results: list[EvalResult], config: str) -> EvalSummary:
    """Compute aggregate statistics from evaluation results."""
    valid = [r for r in results if r.error is None]
    total = len(results)

    if not valid:
        return EvalSummary(config=config, total_questions=total)

    latencies = [r.latency_ms for r in valid]
    tokens = [r.tokens_used for r in valid]

    import numpy as np
    
    summary = EvalSummary(
        config=config,
        total_questions=total,
        correct=sum(1 for r in valid if r.correct),
        accuracy=sum(1 for r in valid if r.correct) / len(valid) if valid else 0,
        avg_latency_ms=float(np.mean(latencies)) if latencies else 0,
        p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0,
        avg_tokens=float(np.mean(tokens)) if tokens else 0,
        total_tokens=sum(tokens),
        avg_llm_calls=float(np.mean([r.llm_calls for r in valid])) if valid else 0,
        avg_tool_calls=float(np.mean([r.tool_calls for r in valid])) if valid else 0,
        total_cost=sum(r.estimated_cost for r in valid),
        avg_cost=float(np.mean([r.estimated_cost for r in valid])) if valid else 0,
        cache_hit_rate=sum(1 for r in valid if r.cache_hit or r.semantic_cache_hit) / len(valid) if valid else 0,
    )
    return summary


def save_results(
    results: list[EvalResult],
    summaries: list[EvalSummary],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save evaluation results to CSV and JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "evaluation_results.csv"
    json_path = output_dir / "evaluation_results.json"

    # Save CSV
    fieldnames = [
        "question_id", "question", "expected_answer", "actual_answer",
        "complexity", "category", "config", "correct", "score",
        "latency_ms", "tokens_used", "llm_calls", "tool_calls",
        "cache_hit", "semantic_cache_hit", "model_used",
        "estimated_cost", "verified", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k, "") for k in fieldnames})

    # Save JSON with summaries
    data = {
        "results": [{k: getattr(r, k, "") for k in fieldnames} for r in results],
        "summaries": [
            {
                "config": s.config,
                "total_questions": s.total_questions,
                "correct": s.correct,
                "accuracy": s.accuracy,
                "avg_latency_ms": s.avg_latency_ms,
                "p95_latency_ms": s.p95_latency_ms,
                "avg_tokens": s.avg_tokens,
                "total_tokens": s.total_tokens,
                "avg_llm_calls": s.avg_llm_calls,
                "avg_tool_calls": s.avg_tool_calls,
                "total_cost": s.total_cost,
                "avg_cost": s.avg_cost,
                "cache_hit_rate": s.cache_hit_rate,
            }
            for s in summaries
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return csv_path, json_path
