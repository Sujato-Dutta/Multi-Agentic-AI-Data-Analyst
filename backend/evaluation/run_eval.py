"""DataPilot — Evaluation CLI entry point.

Usage:
    python -m evaluation.run_eval --config baseline --dry-run
    python -m evaluation.run_eval --config experiment_a --max-questions 10
    python -m evaluation.run_eval --config all
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.mcp.server import load_dataset
from app.observability.metrics import setup_logging
from evaluation.benchmark import run_evaluation, save_results
from evaluation.dataset_prep import prepare_benchmark


def main():
    parser = argparse.ArgumentParser(description="DataPilot Evaluation Pipeline")
    parser.add_argument(
        "--config",
        choices=["baseline", "experiment_a", "experiment_b", "all"],
        default="all",
        help="Evaluation configuration to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making actual LLM calls",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    args = parser.parse_args()

    setup_logging("INFO")
    logger = logging.getLogger("datapilot.eval.cli")

    # Prepare data
    data_dir = Path(__file__).parent.parent / "data"
    dataset_path = data_dir / "sample_sales.csv"

    if not dataset_path.exists():
        logger.error("Sample dataset not found at %s", dataset_path)
        sys.exit(1)

    # Load dataset
    load_dataset("sample_sales", dataset_path)

    # Generate benchmark questions
    logger.info("Preparing benchmark questions...")
    questions, csv_path, json_path = prepare_benchmark(data_dir)
    logger.info("Generated %d questions", len(questions))

    # Determine configs to run
    configs = []
    if args.config == "all":
        configs = ["baseline", "experiment_a", "experiment_b"]
    else:
        configs = [args.config]

    # Run evaluations
    all_results = []
    all_summaries = []

    for config in configs:
        logger.info("=" * 60)
        logger.info("Running evaluation: %s", config)
        logger.info("=" * 60)

        results, summary = asyncio.run(
            run_evaluation(
                questions=questions,
                config=config,
                dry_run=args.dry_run,
                max_questions=args.max_questions,
            )
        )

        all_results.extend(results)
        all_summaries.append(summary)

        # Print summary
        logger.info("Results for %s:", config)
        logger.info("  Accuracy:     %.1f%% (%d/%d)", summary.accuracy * 100, summary.correct, summary.total_questions)
        logger.info("  Avg Latency:  %.0f ms", summary.avg_latency_ms)
        logger.info("  P95 Latency:  %.0f ms", summary.p95_latency_ms)
        logger.info("  Avg Tokens:   %.0f", summary.avg_tokens)
        logger.info("  Total Cost:   $%.4f", summary.total_cost)
        logger.info("  Cache Rate:   %.1f%%", summary.cache_hit_rate * 100)

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent.parent.parent
    csv_out, json_out = save_results(all_results, all_summaries, output_dir)
    logger.info("Results saved to:")
    logger.info("  CSV:  %s", csv_out)
    logger.info("  JSON: %s", json_out)

    # Generate METRICS.md
    if not args.dry_run and len(all_summaries) > 1:
        _generate_metrics_md(all_summaries, output_dir)

    logger.info("Evaluation complete!")


def _generate_metrics_md(summaries: list, output_dir: Path):
    """Generate METRICS.md from evaluation summaries."""
    lines = ["# DataPilot — Evaluation Metrics\n"]
    lines.append("> **Note**: All values below are experimentally measured, not fabricated.\n")
    lines.append("## Configuration Comparison\n")
    lines.append("| Metric | Baseline | Experiment A | Experiment B |")
    lines.append("|--------|----------|--------------|--------------|")

    by_config = {s.config: s for s in summaries}
    b = by_config.get("baseline")
    a = by_config.get("experiment_a")
    e = by_config.get("experiment_b")

    def _val(s, attr, fmt=".1f"):
        if s is None:
            return "N/A"
        v = getattr(s, attr, 0)
        return f"{v:{fmt}}"

    lines.append(f"| Accuracy | {_val(b, 'accuracy', '.1%')} | {_val(a, 'accuracy', '.1%')} | {_val(e, 'accuracy', '.1%')} |")
    lines.append(f"| Avg Latency (ms) | {_val(b, 'avg_latency_ms', '.0f')} | {_val(a, 'avg_latency_ms', '.0f')} | {_val(e, 'avg_latency_ms', '.0f')} |")
    lines.append(f"| P95 Latency (ms) | {_val(b, 'p95_latency_ms', '.0f')} | {_val(a, 'p95_latency_ms', '.0f')} | {_val(e, 'p95_latency_ms', '.0f')} |")
    lines.append(f"| Avg Tokens/Question | {_val(b, 'avg_tokens', '.0f')} | {_val(a, 'avg_tokens', '.0f')} | {_val(e, 'avg_tokens', '.0f')} |")
    lines.append(f"| Total Tokens | {_val(b, 'total_tokens', ',')} | {_val(a, 'total_tokens', ',')} | {_val(e, 'total_tokens', ',')} |")
    lines.append(f"| Avg LLM Calls | {_val(b, 'avg_llm_calls')} | {_val(a, 'avg_llm_calls')} | {_val(e, 'avg_llm_calls')} |")
    lines.append(f"| Avg Cost/Question ($) | {_val(b, 'avg_cost', '.4f')} | {_val(a, 'avg_cost', '.4f')} | {_val(e, 'avg_cost', '.4f')} |")
    lines.append(f"| Cache Hit Rate | N/A | N/A | {_val(e, 'cache_hit_rate', '.1%')} |")

    # Reductions
    if b and a:
        lat_reduction = ((b.avg_latency_ms - a.avg_latency_ms) / b.avg_latency_ms * 100) if b.avg_latency_ms > 0 else 0
        token_reduction = ((b.avg_tokens - a.avg_tokens) / b.avg_tokens * 100) if b.avg_tokens > 0 else 0
        cost_reduction = ((b.avg_cost - a.avg_cost) / b.avg_cost * 100) if b.avg_cost > 0 else 0

        lines.append("\n## Optimization Impact (Experiment A vs Baseline)\n")
        lines.append(f"- Latency reduction: **{lat_reduction:.1f}%**")
        lines.append(f"- Token reduction: **{token_reduction:.1f}%**")
        lines.append(f"- Cost reduction: **{cost_reduction:.1f}%**")

    if b and e:
        lat_reduction = ((b.avg_latency_ms - e.avg_latency_ms) / b.avg_latency_ms * 100) if b.avg_latency_ms > 0 else 0
        token_reduction = ((b.avg_tokens - e.avg_tokens) / b.avg_tokens * 100) if b.avg_tokens > 0 else 0
        cost_reduction = ((b.avg_cost - e.avg_cost) / b.avg_cost * 100) if b.avg_cost > 0 else 0

        lines.append(f"\n## Optimization Impact (Experiment B vs Baseline)\n")
        lines.append(f"- Latency reduction: **{lat_reduction:.1f}%**")
        lines.append(f"- Token reduction: **{token_reduction:.1f}%**")
        lines.append(f"- Cost reduction: **{cost_reduction:.1f}%**")
        lines.append(f"- Cache hit rate: **{e.cache_hit_rate:.1%}**")

    metrics_path = output_dir / "METRICS.md"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"METRICS.md written to {metrics_path}")


if __name__ == "__main__":
    main()
