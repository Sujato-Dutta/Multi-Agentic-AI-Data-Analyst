"""DataPilot — Agent 3: Analysis Agent.

Performs calculations, multi-step analysis, trend detection,
comparisons, and ranking operations on retrieved data.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.utils import extract_json, extract_text, extract_tokens
from app.mcp.server import call_tool

logger = logging.getLogger("datapilot.agents.analysis")

ANALYSIS_SYSTEM_PROMPT = """You are the Analysis Agent of DataPilot, a data analysis system.

Your job is to analyze retrieved data and answer the user's question.

Given:
- The user's question
- Retrieved data from the Data Agent
- The execution plan

You can either:
1. Analyze the data directly and provide an answer.
2. Call run_python_analysis to perform complex calculations.

If you need to run Python code, output a JSON object:
{
  "needs_code": true,
  "code": "Python code here. Dataset is available as 'df'. Assign result to 'result'.",
  "preliminary_analysis": "Brief notes on what you're computing"
}

If you can answer directly from the data, output:
{
  "needs_code": false,
  "answer": "Your analytical answer here",
  "key_findings": ["finding 1", "finding 2"],
  "data_summary": {"key metrics or values"}
}

Rules:
- Be precise with numbers — use exact values from the data.
- Include relevant metrics, percentages, and comparisons.
- Keep answers concise but complete.
- If the data is insufficient, say so.
- Output ONLY valid JSON, no markdown fences.
"""


async def run_analysis_agent(
    question: str,
    plan: dict[str, Any],
    retrieved_data: dict[str, Any],
    dataset_id: str,
    llm: Any,
    cache: Any = None,
) -> dict[str, Any]:
    """Run the analysis agent on retrieved data.
    
    Args:
        question: User's question
        plan: Execution plan
        retrieved_data: Data from the data agent
        dataset_id: Dataset ID
        llm: LangChain LLM
        cache: Optional cache
    
    Returns:
        Analysis results dict
    """
    # Prepare data summary (limit size for LLM context)
    data_summary = _summarize_data(retrieved_data)

    messages = [
        SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"User question: {question}\n\n"
            f"Retrieved data:\n{data_summary}\n\n"
            f"Analyze this data and answer the question."
        )),
    ]

    tool_calls = []
    tool_records = []
    tokens = 0

    try:
        response = await llm.ainvoke(messages)
        tokens += extract_tokens(response)
        result = _parse_analysis(response.content)

        if result.get("needs_code", False):
            code = result["code"]
            logger.info("Analysis agent running Python code")

            # Check cache
            cached_result = None
            if cache:
                cached_result = cache.get("run_python_analysis", {"code": code, "dataset_id": dataset_id})

            if cached_result is not None:
                py_result = cached_result
                cache_status = "HIT"
            else:
                py_result = call_tool("run_python_analysis", dataset_id=dataset_id, code=code)
                cache_status = "MISS"
                if cache:
                    cache.set("run_python_analysis", {"code": code, "dataset_id": dataset_id}, py_result)

            tool_calls.append("run_python_analysis")
            tool_records.append({
                "tool_name": "run_python_analysis",
                "params": {"code": code[:100]},
                "cache_status": cache_status,
                "success": "error" not in py_result,
            })

            # If code ran, do a final interpretation call
            if "error" not in py_result:
                interp_messages = [
                    SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
                    HumanMessage(content=(
                        f"User question: {question}\n\n"
                        f"Python execution result:\n{json.dumps(py_result, indent=2)}\n\n"
                        f"Provide the final answer based on these computed results."
                    )),
                ]
                interp_response = await llm.ainvoke(interp_messages)
                tokens += extract_tokens(interp_response)
                interp_result = _parse_analysis(interp_response.content)
                result["answer"] = interp_result.get("answer", result.get("answer", ""))
                result["key_findings"] = interp_result.get("key_findings", result.get("key_findings", []))

    except Exception as e:
        logger.error("Analysis agent failed: %s", e)
        result = {
            "answer": f"Analysis could not be completed: {str(e)}",
            "key_findings": [],
            "data_summary": {},
        }

    result["tool_calls"] = tool_calls
    result["tool_records"] = tool_records
    result["tokens"] = tokens
    return result


def _summarize_data(retrieved_data: dict[str, Any]) -> str:
    """Create a concise text summary of retrieved data for LLM context."""
    parts = []
    for key, data in retrieved_data.items():
        if isinstance(data, dict):
            # Limit data records shown
            if "data" in data and isinstance(data["data"], list):
                records = data["data"][:15]  # Limit to 15 records
                summary = {k: v for k, v in data.items() if k != "data"}
                summary["data"] = records
                summary["_note"] = f"Showing {len(records)} of {data.get('matched_rows', data.get('returned_rows', len(data.get('data', []))))} rows"
                parts.append(f"{key}: {json.dumps(summary, default=str)}")
            else:
                parts.append(f"{key}: {json.dumps(data, default=str)[:2000]}")
        else:
            parts.append(f"{key}: {str(data)[:1000]}")
    return "\n\n".join(parts)


def _parse_analysis(content: Any) -> dict[str, Any]:
    """Parse analysis result from LLM response."""
    try:
        data = extract_json(content)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    text = extract_text(content).strip()
    return {
        "needs_code": False,
        "answer": text,
        "key_findings": [],
        "data_summary": {},
    }
