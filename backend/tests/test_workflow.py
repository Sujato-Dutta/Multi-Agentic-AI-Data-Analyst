"""Tests for the core workflow / LangGraph pipeline.

Tests the graph structure and state flow using mock LLMs
to avoid real API calls during testing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import Complexity, QueryResponse
from app.mcp.server import load_dataset, get_schema, call_tool
from app.router.model_router import ModelRouter
from app.cache.redis_cache import RedisCache
from app.cache.semantic_cache import SemanticCache
from evaluation.scoring import score_result, _extract_numbers, _extract_keywords


@pytest.fixture(autouse=True)
def load_test_dataset():
    """Load sample dataset for workflow tests."""
    data_path = Path(__file__).parent.parent / "data" / "sample_sales.csv"
    if data_path.exists():
        load_dataset("test_sales", data_path)
    else:
        pytest.skip("Sample dataset not found")


class TestWorkflowComponents:
    """Test individual workflow components without LLM calls."""

    def test_schema_retrieval(self):
        schema = call_tool("get_schema", dataset_id="test_sales")
        assert schema["row_count"] == 50
        assert "category" in schema["columns"]

    def test_data_retrieval_pipeline(self):
        """Test that data tools chain correctly."""
        # Step 1: Get schema
        schema = call_tool("get_schema", dataset_id="test_sales")
        assert "columns" in schema

        # Step 2: Filter data
        filtered = call_tool("filter_data", dataset_id="test_sales",
                            column="category", operator="==", value="Electronics")
        assert filtered["matched_rows"] > 0

        # Step 3: Aggregate
        agg = call_tool("aggregate_data", dataset_id="test_sales",
                        group_by="category", agg_column="revenue", agg_func="sum")
        assert agg["group_count"] == 3

    def test_analysis_via_python(self):
        """Test Python analysis tool execution."""
        code = """
result = {
    'total_revenue': df['revenue'].sum(),
    'avg_profit': df['profit'].mean(),
    'top_category': df.groupby('category')['revenue'].sum().idxmax()
}
"""
        result = call_tool("run_python_analysis", dataset_id="test_sales", code=code)
        assert result["success"] is True
        assert "total_revenue" in result["result"]
        assert "top_category" in result["result"]

    def test_visualization_generation(self):
        """Test chart generation."""
        agg = call_tool("aggregate_data", dataset_id="test_sales",
                        group_by="category", agg_column="revenue", agg_func="sum")
        
        x_data = [d["category"] for d in agg["data"]]
        y_data = [d["sum_revenue"] for d in agg["data"]]

        chart = call_tool("create_visualization", dataset_id="test_sales",
                         chart_type="bar", x_data=x_data, y_data=y_data,
                         title="Revenue by Category")
        assert "image_base64" in chart
        assert len(chart["image_base64"]) > 100

    def test_full_tool_chain(self):
        """Test a realistic tool chain: schema → filter → aggregate → visualize."""
        # 1. Schema
        schema = call_tool("get_schema", dataset_id="test_sales")
        
        # 2. Aggregate
        agg = call_tool("aggregate_data", dataset_id="test_sales",
                        group_by="product", agg_column="revenue", agg_func="sum",
                        sort_by_result=True, limit=5)
        
        # 3. Visualize top 5
        x = [d["product"] for d in agg["data"]]
        y = [d["sum_revenue"] for d in agg["data"]]
        chart = call_tool("create_visualization", dataset_id="test_sales",
                         chart_type="horizontal_bar", x_data=x, y_data=y,
                         title="Top 5 Products by Revenue")
        
        assert len(x) == 5
        assert chart["chart_type"] == "horizontal_bar"


class TestCacheIntegration:
    """Test cache integration with tools."""

    def test_tool_result_caching(self):
        cache = RedisCache(redis_url=None, redis_token=None, ttl=60)
        cache.clear()

        params = {"dataset_id": "test_sales"}
        
        # Miss
        assert cache.get("get_schema", params) is None
        
        # Compute and cache
        result = call_tool("get_schema", dataset_id="test_sales")
        cache.set("get_schema", params, result)
        
        # Hit
        cached = cache.get("get_schema", params)
        assert cached is not None
        assert cached["row_count"] == 50

    def test_semantic_cache_workflow(self):
        sem_cache = SemanticCache(threshold=0.85)
        sem_cache.clear()

        # Store an answer
        sem_cache.set("What is the total revenue?", "test_sales", "Total revenue is $930K")

        # Similar question should hit
        hit = sem_cache.get("What is total revenue?", "test_sales")
        assert hit is not None


class TestScoring:
    """Test the evaluation scoring module."""

    def test_exact_match_scores_high(self):
        score, correct = score_result(
            "What is the total?",
            "50 rows",
            "There are 50 rows in the dataset"
        )
        assert score > 0.5
        assert correct is True

    def test_error_scores_zero(self):
        score, correct = score_result(
            "What is X?",
            "42",
            "ERROR: Something went wrong"
        )
        assert score == 0.0
        assert correct is False

    def test_number_extraction(self):
        nums = _extract_numbers("The revenue is 12345.67 and there are 50 rows")
        assert "12345.67" in nums
        assert "50" in nums

    def test_keyword_extraction(self):
        kw = _extract_keywords("electronics category highest revenue growth north region")
        assert "electronics" in kw
        assert "highest" in kw
        assert "growth" in kw
