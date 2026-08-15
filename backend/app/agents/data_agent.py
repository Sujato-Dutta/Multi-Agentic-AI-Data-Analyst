"""DataPilot — Agent 2: Data Agent.

Inspects schema, identifies relevant columns, and retrieves/filters/aggregates
data through MCP tools. Prefers deterministic tools over LLM reasoning.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.utils import extract_json, extract_text
from app.mcp.server import call_tool

logger = logging.getLogger("datapilot.agents.data")

DATA_AGENT_SYSTEM_PROMPT = """You are the Data Agent of DataPilot, a data analysis system.

Your job is to retrieve the right data from the dataset to answer the user's question.

You have access to these MCP tools:
- get_schema(dataset_id): Get column names, types, statistics
- sample_data(dataset_id, n): Get first N rows
- filter_data(dataset_id, column, operator, value, limit): Filter rows
- aggregate_data(dataset_id, group_by, agg_column, agg_func): Group-by aggregation
- execute_query(dataset_id, query_str): Pandas query string

Given the plan and schema, decide which tool(s) to call and output a JSON array of tool calls:
[
  {
    "tool": "tool_name",
    "params": {"param1": "value1", ...}
  }
]

Rules:
- Always include dataset_id in params.
- Prefer aggregate_data and filter_data over execute_query when possible.
- Minimize data retrieval — only get what's needed.
- Output ONLY valid JSON, no markdown fences, no explanation.
"""


async def run_data_agent(
    question: str,
    plan: dict[str, Any],
    schema: dict[str, Any],
    dataset_id: str,
    llm: Any,
    cache: Any = None,
) -> dict[str, Any]:
    """Run the data agent to retrieve relevant data.
    
    Args:
        question: User's question
        plan: Execution plan from planner
        schema: Dataset schema
        dataset_id: ID of the target dataset
        llm: LangChain LLM instance
        cache: Optional RedisCache for tool result caching
    
    Returns:
        Dict with retrieved data and tool call records
    """
    schema_summary = _format_schema_brief(schema)
    plan_summary = json.dumps(plan.get("steps", []), indent=2)

    messages = [
        SystemMessage(content=DATA_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Dataset: {dataset_id}\n"
            f"Schema:\n{schema_summary}\n\n"
            f"Plan steps:\n{plan_summary}\n\n"
            f"User question: {question}\n\n"
            f"Output the tool calls as a JSON array."
        )),
    ]

    tool_calls = []
    retrieved_data = {}
    tool_records = []

    try:
        response = await llm.ainvoke(messages)
        calls = _parse_tool_calls(response.content)

        for call_spec in calls:
            tool_name = call_spec["tool"]
            params = call_spec.get("params", {})
            params["dataset_id"] = dataset_id

            # Check cache first
            cached_result = None
            if cache:
                cached_result = cache.get(tool_name, params)

            if cached_result is not None:
                result = cached_result
                cache_status = "HIT"
                logger.info("Data agent: cache HIT for %s", tool_name)
            else:
                result = call_tool(tool_name, **params)
                cache_status = "MISS"
                if cache:
                    cache.set(tool_name, params, result)

            tool_calls.append(tool_name)
            tool_records.append({
                "tool_name": tool_name,
                "params": params,
                "cache_status": cache_status,
                "success": "error" not in result,
            })
            retrieved_data[f"{tool_name}_{len(tool_records)}"] = result

    except Exception as e:
        logger.error("Data agent LLM call failed: %s — falling back to sample", e)
        # Fallback: just get sample data
        result = call_tool("sample_data", dataset_id=dataset_id, n=10)
        retrieved_data["sample_data_fallback"] = result
        tool_calls.append("sample_data")
        tool_records.append({
            "tool_name": "sample_data",
            "params": {"dataset_id": dataset_id, "n": 10},
            "cache_status": "MISS",
            "success": True,
        })

    return {
        "retrieved_data": retrieved_data,
        "tool_calls": tool_calls,
        "tool_records": tool_records,
    }


def _format_schema_brief(schema: dict[str, Any]) -> str:
    """Brief schema for data agent context."""
    lines = []
    for col_name, col_info in schema.get("columns", {}).items():
        dtype = col_info.get("dtype", "?")
        lines.append(f"  {col_name} ({dtype})")
    return "\n".join(lines)


def _parse_tool_calls(content: Any) -> list[dict[str, Any]]:
    """Parse tool call array from LLM response."""
    try:
        calls = extract_json(content)
        if isinstance(calls, dict):
            calls = [calls]
        if isinstance(calls, list):
            return calls
        raise ValueError(f"Expected list or dict, got {type(calls)}")
    except Exception as e:
        raise ValueError(f"Could not parse tool calls: {e}")
