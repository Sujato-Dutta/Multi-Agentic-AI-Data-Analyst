"""Tests for the complexity-aware model router."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import Complexity
from app.router.model_router import ModelRouter, BaselineRouter


class TestModelRouter:
    """Tests for complexity analysis and model routing."""

    def setup_method(self):
        self.router = ModelRouter()

    def test_simple_question_routing(self):
        complexity, _ = self.router.analyze_complexity("How many rows are there?")
        assert complexity == Complexity.SIMPLE

    def test_simple_count_question(self):
        complexity, _ = self.router.analyze_complexity("What is the total revenue?")
        assert complexity == Complexity.SIMPLE

    def test_simple_top_n(self):
        complexity, _ = self.router.analyze_complexity("Show the top 5 products")
        assert complexity == Complexity.SIMPLE

    def test_normal_comparison(self):
        complexity, _ = self.router.analyze_complexity(
            "Which category generates the most revenue and how does it compare to others?"
        )
        assert complexity in (Complexity.NORMAL, Complexity.COMPLEX)

    def test_complex_trend(self):
        complexity, _ = self.router.analyze_complexity(
            "What is the year-over-year growth trend for each category and is there a correlation with quantity?"
        )
        assert complexity == Complexity.COMPLEX

    def test_complex_statistical(self):
        complexity, _ = self.router.analyze_complexity(
            "Calculate the standard deviation and percentile distribution of revenue"
        )
        assert complexity == Complexity.COMPLEX

    def test_complex_multi_step(self):
        complexity, _ = self.router.analyze_complexity(
            "What are the top 5 customers by revenue, and did their purchase frequency increase over time?"
        )
        assert complexity in (Complexity.NORMAL, Complexity.COMPLEX)

    def test_model_selection(self):
        model_name, complexity, decision = self.router.select_model("How many rows?")
        assert model_name  # Should return a non-empty model name
        assert decision.model_name == model_name

    def test_routing_log(self):
        self.router.select_model("Test question 1")
        self.router.select_model("Test question 2")
        log = self.router.get_routing_log()
        assert len(log) == 2
        assert log[0]["question"] == "Test question 1"

    def test_override_model(self):
        router = ModelRouter(override_model="custom-model")
        model_name, _, decision = router.select_model("Any question")
        assert model_name == "custom-model"
        assert decision.override is True

    def test_complexity_reasons_populated(self):
        _, reason = self.router.analyze_complexity("What is the average revenue?")
        assert reason  # Should have a non-empty reason string

    def test_short_question_simple(self):
        complexity, _ = self.router.analyze_complexity("Total profit?")
        assert complexity == Complexity.SIMPLE


class TestBaselineRouter:
    """Tests for the baseline router that always uses the highest model."""

    def test_always_uses_high_model(self):
        router = BaselineRouter()
        # Simple question still gets high model
        model, _, decision = router.select_model("How many rows?")
        assert "gemini-3.5-flash-lite" in model or decision.override is True

    def test_complex_also_uses_high(self):
        router = BaselineRouter()
        model1, _, _ = router.select_model("How many rows?")
        model2, _, _ = router.select_model("Analyze the correlation between price and quantity")
        assert model1 == model2  # Same model for everything
