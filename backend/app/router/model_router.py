"""DataPilot — Complexity-aware model router.

Routes requests to the appropriate Gemini/Gemma model
based on question complexity analysis.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.models import Complexity

logger = logging.getLogger("datapilot.router")

# ── Complexity Signals ────────────────────────────────────────────────

# Keywords and patterns that indicate higher complexity
COMPLEX_SIGNALS = [
    r"\bcorrelat",
    r"\btrend\b",
    r"\bforecast",
    r"\bpredict",
    r"\bregress",
    r"\bstatistic",
    r"\bsignifican",
    r"\boutlier",
    r"\banomaly",
    r"\bcompare.*across",
    r"\bgrowth.*rate",
    r"\byear.over.year",
    r"\bmonth.over.month",
    r"\bpercentage.*change",
    r"\bif.*then.*else",
    r"\bwhich.*most.*and.*least",
    r"\bmultiple.*factor",
    r"\bweighted",
    r"\bmedian",
    r"\bstandard.deviation",
    r"\bpercentile",
    r"\bdistribution",
    r"\brelationship.between",
]

SIMPLE_SIGNALS = [
    r"\bhow many\b",
    r"\btotal\b",
    r"\bcount\b",
    r"\blist\b",
    r"\bshow\b",
    r"\bwhat is\b",
    r"\bmaximum\b",
    r"\bminimum\b",
    r"\baverage\b",
    r"\bsum\b",
    r"\btop \d+\b",
    r"\bbottom \d+\b",
    r"\blargest\b",
    r"\bsmallest\b",
    r"\bhighest\b",
    r"\blowest\b",
]

# Multiple sub-questions or analytical steps
MULTI_STEP_PATTERN = re.compile(
    r"\band\b.*\b(compare|also|additionally|furthermore|then|did)\b",
    re.IGNORECASE,
)


@dataclass
class RoutingDecision:
    """Records a model routing decision with reasoning."""
    question: str
    complexity: Complexity
    model_name: str
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    override: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question[:100],
            "complexity": self.complexity.value,
            "model": self.model_name,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "override": self.override,
        }


class ModelRouter:
    """Routes questions to the appropriate model based on complexity."""

    def __init__(self, override_model: Optional[str] = None):
        self.settings = get_settings()
        self.override_model = override_model
        self.decisions: list[RoutingDecision] = []

        low_model = self.settings.model_low
        if low_model == "gemma-4-31b":
            low_model = "gemma-4-31b-it"

        self._model_map = {
            Complexity.COMPLEX: self.settings.model_high,   # gemini-3.5-flash-lite
            Complexity.NORMAL: self.settings.model_medium,   # gemini-3.1-flash-lite
            Complexity.SIMPLE: low_model,                    # gemma-4-31b-it
        }

    def analyze_complexity(self, question: str) -> tuple[Complexity, str]:
        """Determine question complexity with reasoning."""
        q_lower = question.lower().strip()
        reasons = []

        # Count complexity signals
        complex_hits = sum(1 for p in COMPLEX_SIGNALS if re.search(p, q_lower))
        simple_hits = sum(1 for p in SIMPLE_SIGNALS if re.search(p, q_lower))

        # Check for multi-step questions
        has_multi_step = bool(MULTI_STEP_PATTERN.search(q_lower))
        question_marks = q_lower.count("?")
        comma_clauses = len([c for c in q_lower.split(",") if len(c.strip()) > 10])

        # Word count as a rough proxy for question complexity
        word_count = len(q_lower.split())

        # Decision logic
        if complex_hits >= 2 or (complex_hits >= 1 and has_multi_step):
            complexity = Complexity.COMPLEX
            reasons.append(f"{complex_hits} complex signal(s) detected")
            if has_multi_step:
                reasons.append("multi-step question detected")
        elif has_multi_step or question_marks > 1 or (complex_hits >= 1 and word_count > 20):
            complexity = Complexity.NORMAL
            if has_multi_step:
                reasons.append("multi-step question")
            if question_marks > 1:
                reasons.append(f"{question_marks} sub-questions")
            if complex_hits >= 1:
                reasons.append(f"{complex_hits} complexity signal(s) with long question")
        elif simple_hits >= 1 and complex_hits == 0 and word_count <= 15:
            complexity = Complexity.SIMPLE
            reasons.append(f"{simple_hits} simple signal(s), short question")
        elif word_count <= 8 and complex_hits == 0:
            complexity = Complexity.SIMPLE
            reasons.append("very short, straightforward question")
        else:
            complexity = Complexity.NORMAL
            reasons.append("default routing — moderate complexity")

        reason = "; ".join(reasons) if reasons else "default"
        return complexity, reason

    def select_model(self, question: str) -> tuple[str, Complexity, RoutingDecision]:
        """Select the best model for a question.
        
        Returns (model_name, complexity, decision_record).
        """
        complexity, reason = self.analyze_complexity(question)

        if self.override_model:
            model_name = self.override_model
            decision = RoutingDecision(
                question=question,
                complexity=complexity,
                model_name=model_name,
                reason=f"OVERRIDE: {reason}",
                override=True,
            )
        else:
            model_name = self._model_map[complexity]
            decision = RoutingDecision(
                question=question,
                complexity=complexity,
                model_name=model_name,
                reason=reason,
            )

        self.decisions.append(decision)
        logger.info(
            "Routing decision: complexity=%s model=%s reason='%s'",
            complexity.value, model_name, reason,
        )
        return model_name, complexity, decision

    def get_llm(self, question: str) -> tuple[ChatGoogleGenerativeAI, Complexity, RoutingDecision]:
        """Get a LangChain LLM instance for the question.
        
        Returns (llm, complexity, decision).
        """
        model_name, complexity, decision = self.select_model(question)
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.settings.google_api_key,
            temperature=0.1,
            max_output_tokens=2048,
        )
        return llm, complexity, decision

    def get_llm_by_complexity(self, complexity: Complexity) -> ChatGoogleGenerativeAI:
        """Get LLM for a specific complexity level (no question analysis)."""
        model_name = self.override_model or self._model_map[complexity]
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.settings.google_api_key,
            temperature=0.1,
            max_output_tokens=2048,
        )

    def get_routing_log(self) -> list[dict]:
        """Return all routing decisions as dicts."""
        return [d.to_dict() for d in self.decisions]


# ── Baseline Router (always uses highest model) ──────────────────────

class BaselineRouter(ModelRouter):
    """Always routes to the highest-capability model (for evaluation baseline)."""

    def __init__(self):
        super().__init__()
        settings = get_settings()
        self.override_model = settings.model_high
