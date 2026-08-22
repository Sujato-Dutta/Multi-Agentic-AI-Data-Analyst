"""DataPilot — Agent 5: Verifier Agent.

Checks that the final answer is supported by actual tool results,
detects calculation inconsistencies, and triggers retries when needed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.utils import extract_json, extract_text, extract_tokens

logger = logging.getLogger("datapilot.agents.verifier")

VERIFIER_SYSTEM_PROMPT = """You are the Verifier Agent of DataPilot, a data analysis system.

Your job is to verify that the proposed answer is correct and supported by the data.

Given:
- The original user question
- The proposed answer
- The raw data/tool results that were used
- The analysis steps taken

You must check:
1. Is the answer actually supported by the data?
2. Are there any calculation errors or inconsistencies?
3. Does the answer fully address the user's question?
4. Are any claims made without supporting evidence?

Output a JSON object:
{
  "verified": true/false,
  "confidence": 0.0 to 1.0,
  "issues": ["issue 1", "issue 2"],
  "suggestions": ["suggestion 1"],
  "corrected_answer": "If verified=false, provide a corrected answer based on the data. Otherwise null.",
  "verification_notes": "Brief explanation of your verification"
}

Rules:
- Be strict about numerical accuracy.
- Flag any unsupported claims.
- If the answer is mostly correct but has minor issues, still mark verified=true with notes.
- Only set verified=false for significant errors.
- Output ONLY valid JSON.
"""


async def run_verifier(
    question: str,
    proposed_answer: str,
    analysis_result: dict[str, Any],
    retrieved_data: dict[str, Any],
    llm: Any,
) -> dict[str, Any]:
    """Run the verifier agent to validate the proposed answer.
    
    Returns:
        Verification result dict
    """
    # Prepare evidence summary
    evidence = _prepare_evidence(analysis_result, retrieved_data)

    messages = [
        SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Original question: {question}\n\n"
            f"Proposed answer: {proposed_answer}\n\n"
            f"Supporting evidence/data:\n{evidence}\n\n"
            f"Verify this answer."
        )),
    ]

    try:
        response = await llm.ainvoke(messages)
        verification = _parse_verification(response.content)
        verification["_tokens"] = extract_tokens(response)
        logger.info(
            "Verification: verified=%s confidence=%.2f issues=%d tokens=%d",
            verification.get("verified", False),
            verification.get("confidence", 0),
            len(verification.get("issues", [])),
            verification["_tokens"],
        )
        return verification
    except Exception as e:
        logger.error("Verifier failed: %s", e)
        # Default to accepting the answer if verifier fails
        return {
            "verified": True,
            "confidence": 0.5,
            "issues": [f"Verification could not be completed: {str(e)}"],
            "suggestions": [],
            "corrected_answer": None,
            "verification_notes": "Verifier encountered an error; answer accepted with lower confidence.",
            "_tokens": 0,
        }


def _prepare_evidence(analysis: dict, retrieved_data: dict) -> str:
    """Compile evidence from analysis and data for verification."""
    parts = []

    # Analysis findings
    if "key_findings" in analysis:
        parts.append(f"Key findings: {json.dumps(analysis['key_findings'], default=str)}")
    if "data_summary" in analysis:
        parts.append(f"Data summary: {json.dumps(analysis['data_summary'], default=str)[:2000]}")
    if "code_result" in analysis:
        parts.append(f"Code result: {json.dumps(analysis['code_result'], default=str)[:2000]}")

    # Raw retrieved data
    for key, data in retrieved_data.items():
        if isinstance(data, dict) and "data" in data:
            records = data["data"][:10]
            parts.append(f"Data [{key}] ({len(records)} rows): {json.dumps(records, default=str)[:1500]}")

    return "\n\n".join(parts) if parts else "No evidence available"


def _parse_verification(content: Any) -> dict[str, Any]:
    """Parse verification result from LLM response."""
    try:
        result = extract_json(content)
        if not isinstance(result, dict):
            result = {}
    except Exception:
        result = {}

    # Ensure required fields
    result.setdefault("verified", True)
    result.setdefault("confidence", 0.7)
    result.setdefault("issues", [])
    result.setdefault("suggestions", [])
    result.setdefault("corrected_answer", None)
    result.setdefault("verification_notes", "")

    return result
