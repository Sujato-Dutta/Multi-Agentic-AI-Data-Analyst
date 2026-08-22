"""DataPilot — Agent 4: Visualization Agent.

Determines whether a chart adds value, selects chart type,
and generates visualization through MCP tools.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.utils import extract_json, extract_text, extract_tokens
from app.mcp.server import call_tool

logger = logging.getLogger("datapilot.agents.visualization")

VIZ_SYSTEM_PROMPT = """You are the Visualization Agent of DataPilot.

Your job is to generate a chart that helps visualize the analysis results.

Given:
- The user's question
- Analysis results and data
- The planned chart type

Generate chart configuration as JSON:
{
  "should_visualize": true/false,
  "chart_type": "bar|line|pie|scatter|horizontal_bar",
  "x_data": ["label1", "label2", ...],
  "y_data": [value1, value2, ...],
  "title": "Chart Title",
  "x_label": "X Axis Label",
  "y_label": "Y Axis Label"
}

Rules:
- Only visualize if it genuinely adds insight. Set should_visualize=false otherwise.
- Bar charts for comparisons/rankings.
- Line charts for trends over time.
- Pie charts for composition/proportions (max 6 segments).
- Scatter plots for relationships between two numeric variables.
- Keep data to ≤12 data points for readability.
- Use clear, descriptive labels.
- Output ONLY valid JSON.
"""


async def run_visualization_agent(
    question: str,
    plan: dict[str, Any],
    analysis_result: dict[str, Any],
    retrieved_data: dict[str, Any],
    dataset_id: str,
    llm: Any,
) -> dict[str, Any]:
    """Run the visualization agent to generate a chart if useful.
    
    Returns:
        Dict with visualization (base64 PNG) or None if skipped
    """
    # Check plan — skip if not needed
    if not plan.get("needs_visualization", False):
        logger.info("Visualization skipped per plan")
        return {"skipped": True, "reason": "Plan indicated no visualization needed"}

    # Prepare context
    data_context = _prepare_viz_context(analysis_result, retrieved_data)

    messages = [
        SystemMessage(content=VIZ_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Question: {question}\n\n"
            f"Planned chart type: {plan.get('visualization_type', 'auto')}\n\n"
            f"Analysis results:\n{json.dumps(analysis_result.get('data_summary', {}), default=str)[:2000]}\n\n"
            f"Available data:\n{data_context}\n\n"
            f"Generate the chart configuration."
        )),
    ]

    try:
        response = await llm.ainvoke(messages)
        tokens = extract_tokens(response)
        viz_config = _parse_viz_config(response.content)

        if not viz_config.get("should_visualize", True):
            logger.info("Visualization agent decided chart not needed")
            return {"skipped": True, "reason": "Agent determined visualization not valuable", "tokens": tokens}

        # Generate the chart via MCP
        chart_result = call_tool(
            "create_visualization",
            dataset_id=dataset_id,
            chart_type=viz_config["chart_type"],
            x_data=viz_config["x_data"],
            y_data=viz_config["y_data"],
            title=viz_config.get("title", ""),
            x_label=viz_config.get("x_label", ""),
            y_label=viz_config.get("y_label", ""),
        )

        logger.info("Visualization generated: type=%s", viz_config["chart_type"])
        return {
            "skipped": False,
            "image_base64": chart_result["image_base64"],
            "chart_type": viz_config["chart_type"],
            "title": viz_config.get("title", ""),
            "tokens": tokens,
        }

    except Exception as e:
        logger.error("Visualization agent failed: %s", e)
        return {"skipped": True, "failed": True, "reason": f"Visualization generation failed: {str(e)}", "tokens": 0}


def _prepare_viz_context(analysis: dict, retrieved_data: dict) -> str:
    """Extract visualizable data from analysis and retrieved data."""
    parts = []

    # Check analysis for data_summary
    if "data_summary" in analysis:
        parts.append(f"Analysis data: {json.dumps(analysis['data_summary'], default=str)[:1500]}")

    if "code_result" in analysis:
        parts.append(f"Code result: {json.dumps(analysis['code_result'], default=str)[:1500]}")

    # Check retrieved data for aggregation results
    for key, data in retrieved_data.items():
        if isinstance(data, dict) and "data" in data:
            records = data["data"][:12]
            parts.append(f"{key} ({len(records)} rows): {json.dumps(records, default=str)[:1500]}")

    return "\n".join(parts) if parts else "No structured data available"


def _parse_viz_config(content: Any) -> dict[str, Any]:
    """Parse visualization config from LLM response."""
    try:
        data = extract_json(content)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning("Failed to parse viz config: %s", e)
    return {"should_visualize": False, "reason": "Could not parse config"}
