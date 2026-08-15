"""DataPilot — Automated scoring for evaluation results.

Scores answers using fuzzy matching and keyword overlap
since exact match is too strict for free-form analytical answers.
"""

from __future__ import annotations

import re
from typing import Any


def score_result(
    question: str,
    expected_answer: str,
    actual_answer: str,
) -> tuple[float, bool]:
    """Score an actual answer against the expected answer.
    
    Uses a combination of:
    - Keyword overlap (main signal)
    - Number extraction and matching
    - Structural similarity
    
    Returns:
        (score: 0.0-1.0, is_correct: bool)
    """
    if not actual_answer or actual_answer.startswith("ERROR"):
        return 0.0, False

    expected_lower = expected_answer.lower().strip()
    actual_lower = actual_answer.lower().strip()

    # Extract numbers from both
    expected_numbers = set(_extract_numbers(expected_lower))
    actual_numbers = set(_extract_numbers(actual_lower))

    # Extract key words (non-stopword, non-numeric tokens)
    expected_keywords = _extract_keywords(expected_lower)
    actual_keywords = _extract_keywords(actual_lower)

    scores = []

    # 1. Number matching (weight: 0.4)
    if expected_numbers:
        number_overlap = len(expected_numbers & actual_numbers) / len(expected_numbers)
        scores.append(("numbers", number_overlap, 0.4))
    else:
        scores.append(("numbers", 0.5, 0.1))  # Neutral if no numbers expected

    # 2. Keyword overlap (weight: 0.4)
    if expected_keywords:
        keyword_overlap = len(expected_keywords & actual_keywords) / len(expected_keywords)
        scores.append(("keywords", keyword_overlap, 0.4))
    else:
        scores.append(("keywords", 0.5, 0.2))

    # 3. Response quality (weight: 0.2) — has substance, not just an error
    quality = _assess_quality(actual_answer)
    scores.append(("quality", quality, 0.2))

    # Weighted average
    total_weight = sum(w for _, _, w in scores)
    weighted_score = sum(s * w for _, s, w in scores) / total_weight if total_weight > 0 else 0

    # Threshold for "correct"
    is_correct = weighted_score >= 0.4

    return round(weighted_score, 3), is_correct


def _extract_numbers(text: str) -> list[str]:
    """Extract numeric values from text."""
    # Match integers and decimals
    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    # Normalize: remove trailing zeros
    normalized = []
    for n in numbers:
        try:
            val = float(n)
            if val == int(val):
                normalized.append(str(int(val)))
            else:
                normalized.append(f"{val:.2f}")
        except ValueError:
            normalized.append(n)
    return normalized


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "of", "in", "to", "for", "with", "on", "at", "from", "by",
        "about", "as", "into", "through", "during", "before", "after",
        "and", "but", "or", "nor", "not", "so", "yet", "both",
        "each", "few", "more", "most", "other", "some", "such",
        "no", "only", "own", "same", "than", "too", "very",
        "this", "that", "these", "those", "it", "its",
        "all", "any", "every", "per", "column", "row", "data",
        "sum", "count", "avg", "mean", "total", "value", "values",
    }
    words = re.findall(r'[a-z]+', text)
    return {w for w in words if w not in stopwords and len(w) > 2}


def _assess_quality(answer: str) -> float:
    """Assess the quality/substance of an answer."""
    if not answer:
        return 0.0
    if answer.startswith("ERROR") or answer.startswith("An error"):
        return 0.0
    if len(answer) < 10:
        return 0.2

    # Has numbers (analytical substance)
    has_numbers = bool(re.search(r'\d', answer))
    # Has reasonable length
    word_count = len(answer.split())
    length_score = min(word_count / 20, 1.0)

    return (0.5 if has_numbers else 0.2) + (length_score * 0.5)
