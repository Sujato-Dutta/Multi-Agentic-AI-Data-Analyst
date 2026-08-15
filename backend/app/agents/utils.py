"""DataPilot — Agent utility functions for robust content & JSON extraction."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("datapilot.agents.utils")


def extract_text(content: Any) -> str:
    """Extract plain text from LLM response content.
    
    Handles str, list of strings, or list of dict/parts from LangChain / Google GenAI.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
            elif hasattr(part, "text"):
                parts.append(str(part.text))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content) if content is not None else ""


def extract_json(content: Any) -> Any:
    """Extract JSON object or array from response content.
    
    Handles thinking/reasoning prefixes, markdown code fences, and loose text.
    """
    text = extract_text(content).strip()
    if not text:
        raise ValueError("Empty response content")

    # 1. Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code blocks ```json ... ``` or ``` ... ```
    code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    for block in reversed(code_blocks):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            pass

    # 3. Search for outermost matching brackets { ... } or [ ... ]
    # Check objects { ... }
    last_brace = text.rfind("}")
    if last_brace != -1:
        first_brace = text.find("{")
        if first_brace != -1 and last_brace > first_brace:
            try:
                return json.loads(text[first_brace:last_brace + 1])
            except json.JSONDecodeError:
                # Try progressively later starting braces (in case thinking contains a brace)
                for start_idx in range(len(text) - 1, -1, -1):
                    if text[start_idx] == '{' and start_idx < last_brace:
                        try:
                            return json.loads(text[start_idx:last_brace + 1])
                        except json.JSONDecodeError:
                            continue

    # Check arrays [ ... ]
    last_bracket = text.rfind("]")
    if last_bracket != -1:
        first_bracket = text.find("[")
        if first_bracket != -1 and last_bracket > first_bracket:
            try:
                return json.loads(text[first_bracket:last_bracket + 1])
            except json.JSONDecodeError:
                for start_idx in range(len(text) - 1, -1, -1):
                    if text[start_idx] == '[' and start_idx < last_bracket:
                        try:
                            return json.loads(text[start_idx:last_bracket + 1])
                        except json.JSONDecodeError:
                            continue

    raise ValueError(f"Could not extract valid JSON from model response: {text[:200]}")
