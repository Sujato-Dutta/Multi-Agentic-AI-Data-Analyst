"""DataPilot — Agent 1: Planner Agent.

Understands the user's question, determines complexity,
breaks the task into executable steps, decides which agents/tools
are required, and selects an appropriate model via the router.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.utils import extract_json, extract_text
from app.models import Complexity

logger = logging.getLogger("datapilot.agents.planner")

PLANNER_SYSTEM_PROMPT = """You are the Planner Agent of DataPilot, a data analysis system.

Your job is to analyze the user's question and create an execution plan.

Given:
- The user's question
- The dataset schema (columns, types, statistics)

You must output a JSON object with exactly these fields:
{
  "understanding": "Brief restatement of what the user is asking",
  "complexity": "simple|normal|complex",
  "needs_visualization": true/false,
  "visualization_type": "bar|line|pie|scatter|horizontal_bar|none",
  "steps": [
    {
      "agent": "data|analysis|visualization",
      "action": "description of what this step should do",
      "tool": "suggested MCP tool name or null",
      "tool_params": {} or null
    }
  ],
  "data_columns_needed": ["col1", "col2"]
}

Rules:
- Keep plans minimal. Simple questions need fewer steps.
- Only include visualization if it genuinely adds value.
- Prefer deterministic MCP tools (filter_data, aggregate_data) over run_python_analysis.
- For simple lookups, a single data agent step may suffice.
- Output ONLY valid JSON, no markdown fences, no explanation.
"""


async def run_planner(
    question: str,
    schema: dict[str, Any],
    llm: Any,
) -> dict[str, Any]:
    """Run the planner agent to create an execution plan.
    
    Args:
        question: User's natural-language question
        schema: Dataset schema from get_schema()
        llm: LangChain LLM instance
    
    Returns:
        Execution plan dict with steps, complexity, viz needs
    """
    # Build a concise schema summary
    schema_summary = _format_schema(schema)

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Dataset Schema:\n{schema_summary}\n\nUser Question: {question}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        plan = _parse_plan(response.content)
        logger.info(
            "Plan created: complexity=%s steps=%d needs_viz=%s",
            plan.get("complexity", "unknown"),
            len(plan.get("steps", [])),
            plan.get("needs_visualization", False),
        )
        return plan
    except Exception as e:
        logger.error("Planner failed: %s", e)
        # Return a sensible default plan
        return {
            "understanding": question,
            "complexity": "normal",
            "needs_visualization": False,
            "visualization_type": "none",
            "steps": [
                {
                    "agent": "data",
                    "action": "Retrieve relevant data for the question",
                    "tool": "execute_query",
                    "tool_params": None,
                },
                {
                    "agent": "analysis",
                    "action": "Analyze the data to answer the question",
                    "tool": "run_python_analysis",
                    "tool_params": None,
                },
            ],
            "data_columns_needed": [],
        }


def _format_schema(schema: dict[str, Any]) -> str:
    """Format schema dict into a concise text summary."""
    lines = [f"Dataset: {schema.get('dataset_id', 'unknown')} ({schema.get('row_count', '?')} rows)"]
    lines.append("Columns:")
    for col_name, col_info in schema.get("columns", {}).items():
        dtype = col_info.get("dtype", "unknown")
        parts = [f"  - {col_name} ({dtype})"]
        if "sample_values" in col_info:
            samples = col_info["sample_values"][:3]
            parts.append(f" samples: {samples}")
        if "min" in col_info and col_info["min"] is not None:
            parts.append(f" range: [{col_info['min']:.1f}, {col_info['max']:.1f}]")
        lines.append("".join(parts))
    return "\n".join(lines)


def _parse_plan(content: Any) -> dict[str, Any]:
    """Parse LLM response into a plan dict. Handles markdown fences and thinking text."""
    try:
        plan = extract_json(content)
    except Exception as e:
        raise ValueError(f"Could not parse plan from LLM response: {e}")

    if not isinstance(plan, dict):
        raise ValueError(f"Expected dict for plan, got {type(plan)}")

    # Validate required fields
    plan.setdefault("complexity", "normal")
    plan.setdefault("needs_visualization", False)
    plan.setdefault("visualization_type", "none")
    plan.setdefault("steps", [])
    plan.setdefault("data_columns_needed", [])
    plan.setdefault("understanding", "")

    return plan
