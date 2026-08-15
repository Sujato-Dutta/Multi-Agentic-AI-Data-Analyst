"""Tests for MCP server tools."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.mcp.server import (
    aggregate_data,
    call_tool,
    create_visualization,
    execute_query,
    filter_data,
    get_schema,
    load_dataset,
    sample_data,
    run_python_analysis,
)


@pytest.fixture(autouse=True)
def load_test_dataset():
    """Load the sample sales dataset for all tests."""
    data_path = Path(__file__).parent.parent / "data" / "sample_sales.csv"
    if data_path.exists():
        load_dataset("test_sales", data_path)
    else:
        pytest.skip("Sample dataset not found")


class TestGetSchema:
    def test_returns_schema(self):
        result = get_schema("test_sales")
        assert result["dataset_id"] == "test_sales"
        assert result["row_count"] == 50
        assert "columns" in result
        assert "revenue" in result["columns"]

    def test_column_types(self):
        result = get_schema("test_sales")
        assert result["columns"]["revenue"]["dtype"] in ("float64", "int64")
        assert result["columns"]["category"]["dtype"] == "object"


class TestSampleData:
    def test_default_sample(self):
        result = sample_data("test_sales")
        assert result["sample_size"] == 5
        assert len(result["data"]) == 5

    def test_custom_sample_size(self):
        result = sample_data("test_sales", n=3)
        assert result["sample_size"] == 3
        assert len(result["data"]) == 3

    def test_capped_at_20(self):
        result = sample_data("test_sales", n=100)
        assert result["sample_size"] <= 20


class TestFilterData:
    def test_equality_filter(self):
        result = filter_data("test_sales", "category", "==", "Electronics")
        assert result["matched_rows"] > 0
        assert all(r["category"] == "Electronics" for r in result["data"])

    def test_gt_filter(self):
        result = filter_data("test_sales", "revenue", ">", 30000)
        assert result["matched_rows"] > 0
        assert all(r["revenue"] > 30000 for r in result["data"])

    def test_contains_filter(self):
        result = filter_data("test_sales", "product", "contains", "Laptop")
        assert result["matched_rows"] > 0

    def test_invalid_column(self):
        with pytest.raises(ValueError, match="not found"):
            filter_data("test_sales", "nonexistent", "==", "x")

    def test_invalid_operator(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            filter_data("test_sales", "category", "~=", "x")


class TestAggregateData:
    def test_sum_aggregation(self):
        result = aggregate_data("test_sales", "category", "revenue", "sum")
        assert result["group_count"] == 3
        assert len(result["data"]) == 3

    def test_mean_aggregation(self):
        result = aggregate_data("test_sales", "region", "profit", "mean")
        assert result["group_count"] == 4

    def test_multi_group_by(self):
        result = aggregate_data("test_sales", ["category", "region"], "revenue", "sum")
        assert result["group_count"] > 0

    def test_invalid_agg_func(self):
        with pytest.raises(ValueError, match="Unknown agg_func"):
            aggregate_data("test_sales", "category", "revenue", "invalid_func")


class TestExecuteQuery:
    def test_basic_query(self):
        result = execute_query("test_sales", "category == 'Electronics'")
        assert result["matched_rows"] > 0

    def test_numeric_query(self):
        result = execute_query("test_sales", "revenue > 25000")
        assert result["matched_rows"] > 0

    def test_invalid_query(self):
        result = execute_query("test_sales", "invalid_syntax!!!")
        assert "error" in result


class TestRunPythonAnalysis:
    def test_simple_calculation(self):
        result = run_python_analysis("test_sales", "result = df['revenue'].sum()")
        assert result["success"] is True
        assert result["result"] > 0

    def test_complex_analysis(self):
        code = """
result = df.groupby('category')['revenue'].sum().to_dict()
"""
        result = run_python_analysis("test_sales", code)
        assert result["success"] is True
        assert isinstance(result["result"], dict)

    def test_error_handling(self):
        result = run_python_analysis("test_sales", "result = 1/0")
        assert result["success"] is False
        assert "error" in result


class TestCreateVisualization:
    def test_bar_chart(self):
        result = create_visualization(
            "test_sales",
            chart_type="bar",
            x_data=["A", "B", "C"],
            y_data=[10, 20, 30],
            title="Test Chart",
        )
        assert "image_base64" in result
        assert len(result["image_base64"]) > 100

    def test_pie_chart(self):
        result = create_visualization(
            "test_sales",
            chart_type="pie",
            x_data=["Cat1", "Cat2", "Cat3"],
            y_data=[40, 35, 25],
        )
        assert "image_base64" in result

    def test_invalid_chart_type(self):
        with pytest.raises(ValueError, match="Unknown chart_type"):
            create_visualization("test_sales", "radar", ["A"], [1])


class TestCallTool:
    def test_dispatch(self):
        result = call_tool("get_schema", dataset_id="test_sales")
        assert result["row_count"] == 50

    def test_unknown_tool(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            call_tool("nonexistent_tool")
