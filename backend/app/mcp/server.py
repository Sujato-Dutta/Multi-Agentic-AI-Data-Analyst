"""DataPilot — MCP Server for dataset operations.

Exposes focused tools for schema inspection, data retrieval,
filtering, aggregation, query execution, Python analysis, and visualization.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("datapilot.mcp")

# ── Global dataset store ──────────────────────────────────────────────
# Maps dataset_id → DataFrame. Loaded on upload/selection.
_datasets: dict[str, pd.DataFrame] = {}


def load_dataset(dataset_id: str, path: str | Path) -> dict[str, Any]:
    """Load a CSV dataset into the store."""
    df = pd.read_csv(path)
    _datasets[dataset_id] = df
    logger.info("Loaded dataset %s: %d rows × %d cols", dataset_id, len(df), len(df.columns))
    return {
        "dataset_id": dataset_id,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
    }


def get_dataset(dataset_id: str) -> pd.DataFrame:
    """Retrieve a loaded dataset or raise."""
    if dataset_id not in _datasets:
        raise ValueError(f"Dataset '{dataset_id}' is not loaded. Available: {list(_datasets.keys())}")
    return _datasets[dataset_id]


def list_datasets() -> list[str]:
    """Return list of loaded dataset IDs."""
    return list(_datasets.keys())


# ── MCP Tool Functions ────────────────────────────────────────────────

def get_schema(dataset_id: str) -> dict[str, Any]:
    """Return column names, data types, row count, and basic statistics."""
    df = get_dataset(dataset_id)
    schema = {
        "dataset_id": dataset_id,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": {},
    }
    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = float(df[col].min()) if not df[col].isnull().all() else None
            col_info["max"] = float(df[col].max()) if not df[col].isnull().all() else None
            col_info["mean"] = float(df[col].mean()) if not df[col].isnull().all() else None
        elif pd.api.types.is_string_dtype(df[col]):
            col_info["sample_values"] = df[col].dropna().unique()[:5].tolist()
        schema["columns"][col] = col_info
    return schema


def sample_data(dataset_id: str, n: int = 5) -> dict[str, Any]:
    """Return the first N rows of the dataset."""
    df = get_dataset(dataset_id)
    n = min(n, len(df), 20)  # Cap at 20 rows
    sample = df.head(n)
    return {
        "dataset_id": dataset_id,
        "row_count": len(df),
        "sample_size": n,
        "data": sample.to_dict(orient="records"),
        "columns": list(df.columns),
    }


def filter_data(
    dataset_id: str,
    column: str,
    operator: str,
    value: Any,
    limit: int = 100,
) -> dict[str, Any]:
    """Filter rows by a condition on a single column.
    
    Operators: ==, !=, >, <, >=, <=, contains, in
    """
    df = get_dataset(dataset_id)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")

    ops = {
        "==": lambda s, v: s == v,
        "!=": lambda s, v: s != v,
        ">": lambda s, v: s > v,
        "<": lambda s, v: s < v,
        ">=": lambda s, v: s >= v,
        "<=": lambda s, v: s <= v,
        "contains": lambda s, v: s.astype(str).str.contains(str(v), case=False, na=False),
        "in": lambda s, v: s.isin(v if isinstance(v, list) else [v]),
    }
    if operator not in ops:
        raise ValueError(f"Unknown operator '{operator}'. Supported: {list(ops.keys())}")

    mask = ops[operator](df[column], value)
    result = df[mask].head(limit)
    return {
        "dataset_id": dataset_id,
        "filter": {"column": column, "operator": operator, "value": value},
        "matched_rows": int(mask.sum()),
        "returned_rows": len(result),
        "data": result.to_dict(orient="records"),
    }


def aggregate_data(
    dataset_id: str,
    group_by: str | list[str],
    agg_column: str,
    agg_func: str,
    sort_by_result: bool = True,
    limit: int = 50,
) -> dict[str, Any]:
    """Group-by aggregation on a column.
    
    agg_func: sum, mean, count, min, max, median, std
    """
    df = get_dataset(dataset_id)
    if group_by is None or group_by == "" or group_by == []:
        group_by = []
    elif isinstance(group_by, str):
        group_by = [group_by]

    for col in list(group_by) + [agg_column]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    valid_funcs = ["sum", "mean", "count", "min", "max", "median", "std"]
    if agg_func not in valid_funcs:
        raise ValueError(f"Unknown agg_func '{agg_func}'. Supported: {valid_funcs}")

    if not group_by:
        # Global aggregation
        val = df[agg_column].agg(agg_func)
        res_val = float(val) if isinstance(val, (int, float, np.number)) else val
        return {
            "dataset_id": dataset_id,
            "group_by": [],
            "agg_column": agg_column,
            "agg_func": agg_func,
            "group_count": 1,
            "result": res_val,
            "data": [{f"{agg_func}_{agg_column}": res_val}],
        }

    grouped = df.groupby(group_by)[agg_column].agg(agg_func).reset_index()
    grouped.columns = list(group_by) + [f"{agg_func}_{agg_column}"]

    if sort_by_result:
        result_col = f"{agg_func}_{agg_column}"
        grouped = grouped.sort_values(result_col, ascending=False)

    result = grouped.head(limit)
    return {
        "dataset_id": dataset_id,
        "group_by": group_by,
        "agg_column": agg_column,
        "agg_func": agg_func,
        "group_count": len(grouped),
        "data": result.to_dict(orient="records"),
    }


def execute_query(dataset_id: str, query_str: str) -> dict[str, Any]:
    """Execute a pandas query string on the dataset.
    
    Uses DataFrame.query() for filtering and eval() for column expressions.
    """
    df = get_dataset(dataset_id)
    try:
        result = df.query(query_str)
        result = result.head(100)  # Cap output
        return {
            "dataset_id": dataset_id,
            "query": query_str,
            "matched_rows": len(result),
            "data": result.to_dict(orient="records"),
            "columns": list(result.columns),
        }
    except Exception as e:
        return {
            "dataset_id": dataset_id,
            "query": query_str,
            "error": str(e),
            "matched_rows": 0,
            "data": [],
        }


def run_python_analysis(dataset_id: str, code: str) -> dict[str, Any]:
    """Execute Python analysis code against the dataset.
    
    The dataset is available as 'df' in the execution context.
    The code must assign its result to a variable called 'result'.
    """
    df = get_dataset(dataset_id)
    
    # Restricted execution environment
    allowed_builtins = {
        "len": len, "range": range, "int": int, "float": float,
        "str": str, "list": list, "dict": dict, "tuple": tuple,
        "sum": sum, "min": min, "max": max, "round": round,
        "sorted": sorted, "enumerate": enumerate, "zip": zip,
        "abs": abs, "bool": bool, "set": set, "print": print,
        "True": True, "False": False, "None": None,
        "isinstance": isinstance, "type": type,
    }
    
    exec_globals = {
        "__builtins__": allowed_builtins,
        "pd": pd,
        "np": np,
        "df": df.copy(),
    }
    exec_locals: dict[str, Any] = {}

    try:
        exec(code, exec_globals, exec_locals)  # noqa: S102
        
        result = exec_locals.get("result", "No 'result' variable set in the code.")
        
        # Serialize result
        if isinstance(result, pd.DataFrame):
            serialized = result.head(100).to_dict(orient="records")
        elif isinstance(result, pd.Series):
            serialized = result.head(100).to_dict()
        elif isinstance(result, np.ndarray):
            serialized = result.tolist()
        elif isinstance(result, (np.integer, np.floating)):
            serialized = float(result)
        else:
            serialized = result

        return {
            "dataset_id": dataset_id,
            "success": True,
            "result": serialized,
        }
    except Exception as e:
        return {
            "dataset_id": dataset_id,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def create_visualization(
    dataset_id: str,
    chart_type: str,
    x_data: list[Any],
    y_data: list[Any],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    color: Optional[str] = None,
    figsize: tuple[int, int] = (10, 6),
) -> dict[str, Any]:
    """Generate a chart and return as base64-encoded PNG.
    
    chart_type: bar, line, pie, scatter, horizontal_bar
    """
    valid_types = ["bar", "line", "pie", "scatter", "horizontal_bar"]
    if chart_type not in valid_types:
        raise ValueError(f"Unknown chart_type '{chart_type}'. Supported: {valid_types}")

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=figsize)
    
    chart_color = color or "#6366f1"
    
    # Build a nice color palette for multi-bar/pie
    colors = [
        "#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd",
        "#818cf8", "#4f46e5", "#7c3aed", "#5b21b6",
        "#e879f9", "#f472b6", "#fb7185", "#f87171",
    ]

    if chart_type == "bar":
        bars = ax.bar(range(len(x_data)), y_data, color=colors[:len(x_data)], edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(x_data)))
        ax.set_xticklabels(x_data, rotation=45, ha="right", fontsize=9)
        # Add value labels
        for bar, val in zip(bars, y_data):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:,.1f}" if isinstance(val, float) else str(val),
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
    elif chart_type == "horizontal_bar":
        bars = ax.barh(range(len(x_data)), y_data, color=colors[:len(x_data)], edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(x_data)))
        ax.set_yticklabels(x_data, fontsize=9)
    elif chart_type == "line":
        ax.plot(x_data, y_data, color=chart_color, linewidth=2.5, marker="o", markersize=6)
        ax.fill_between(x_data, y_data, alpha=0.1, color=chart_color)
    elif chart_type == "pie":
        wedges, texts, autotexts = ax.pie(
            y_data, labels=x_data, autopct="%1.1f%%",
            colors=colors[:len(x_data)], startangle=140,
            pctdistance=0.85, wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for t in autotexts:
            t.set_fontsize(9)
            t.set_fontweight("bold")
    elif chart_type == "scatter":
        ax.scatter(x_data, y_data, c=chart_color, s=80, alpha=0.7, edgecolors="white", linewidth=0.5)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    if x_label and chart_type != "pie":
        ax.set_xlabel(x_label, fontsize=11)
    if y_label and chart_type != "pie":
        ax.set_ylabel(y_label, fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    # Encode as base64 PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "dataset_id": dataset_id,
        "chart_type": chart_type,
        "image_base64": b64,
        "title": title,
    }


# ── Tool Registry (for agent access) ─────────────────────────────────

TOOL_REGISTRY = {
    "get_schema": {
        "function": get_schema,
        "description": "Get the schema of a dataset including column names, types, and statistics.",
        "parameters": {"dataset_id": "str"},
    },
    "sample_data": {
        "function": sample_data,
        "description": "Get a sample of the first N rows from a dataset.",
        "parameters": {"dataset_id": "str", "n": "int (default 5)"},
    },
    "filter_data": {
        "function": filter_data,
        "description": "Filter dataset rows by a condition on a column.",
        "parameters": {
            "dataset_id": "str", "column": "str",
            "operator": "str (==,!=,>,<,>=,<=,contains,in)",
            "value": "any", "limit": "int (default 100)",
        },
    },
    "aggregate_data": {
        "function": aggregate_data,
        "description": "Perform group-by aggregation on a column.",
        "parameters": {
            "dataset_id": "str", "group_by": "str or list[str]",
            "agg_column": "str", "agg_func": "str (sum,mean,count,min,max,median,std)",
        },
    },
    "execute_query": {
        "function": execute_query,
        "description": "Execute a pandas query string to filter the dataset.",
        "parameters": {"dataset_id": "str", "query_str": "str"},
    },
    "run_python_analysis": {
        "function": run_python_analysis,
        "description": "Execute Python code for complex analysis. Dataset available as 'df'. Assign result to 'result'.",
        "parameters": {"dataset_id": "str", "code": "str"},
    },
    "create_visualization": {
        "function": create_visualization,
        "description": "Generate a chart (bar, line, pie, scatter, horizontal_bar) as base64 PNG.",
        "parameters": {
            "dataset_id": "str", "chart_type": "str",
            "x_data": "list", "y_data": "list",
            "title": "str", "x_label": "str", "y_label": "str",
        },
    },
}


def call_tool(tool_name: str, **kwargs) -> dict[str, Any]:
    """Dispatch a tool call by name."""
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool '{tool_name}'. Available: {list(TOOL_REGISTRY.keys())}")
    func = TOOL_REGISTRY[tool_name]["function"]
    logger.info("MCP tool call: %s(%s)", tool_name, ", ".join(f"{k}={v!r}" for k, v in kwargs.items()))
    return func(**kwargs)
