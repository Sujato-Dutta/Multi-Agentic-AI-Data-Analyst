"""DataPilot — Evaluation benchmark dataset preparation.

Downloads/generates approximately 100 analytical QA examples
for benchmarking the multi-agent pipeline.
"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

# Fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def generate_benchmark_questions(data_dir: Path) -> list[dict[str, Any]]:
    """Generate ~100 analytical questions about the sample sales dataset.
    
    These questions span various complexity levels:
    - Simple: direct lookups, counts, sums
    - Normal: aggregations, rankings, multi-column analysis
    - Complex: trends, comparisons, multi-step reasoning
    
    Returns list of dicts with: question, expected_answer, complexity, category
    """
    questions = []
    
    # ── Simple Questions (30) ─────────────────────────────────────
    simple = [
        {"question": "How many rows are in the dataset?", "expected_answer": "50", "category": "count"},
        {"question": "What is the total revenue?", "expected_answer": "sum of revenue column", "category": "aggregation"},
        {"question": "How many unique categories are there?", "expected_answer": "3 (Electronics, Furniture, Clothing)", "category": "count"},
        {"question": "What is the highest single revenue transaction?", "expected_answer": "38499.45", "category": "lookup"},
        {"question": "List all unique products", "expected_answer": "Laptop, Smartphone, Office Chair, Tablet, Winter Jacket, Standing Desk, Running Shoes, Headphones, Bookshelf, Dress Shirt", "category": "list"},
        {"question": "How many unique customers are there?", "expected_answer": "5", "category": "count"},
        {"question": "What is the average unit price?", "expected_answer": "mean of unit_price", "category": "aggregation"},
        {"question": "What is the minimum quantity sold in a single transaction?", "expected_answer": "12", "category": "lookup"},
        {"question": "How many transactions are from the North region?", "expected_answer": "count of North", "category": "count"},
        {"question": "What is the total profit?", "expected_answer": "sum of profit column", "category": "aggregation"},
        {"question": "Show the top 5 transactions by revenue", "expected_answer": "top 5 rows sorted by revenue desc", "category": "ranking"},
        {"question": "What is the maximum quantity in any transaction?", "expected_answer": "250", "category": "lookup"},
        {"question": "How many transactions involved Laptops?", "expected_answer": "count of Laptop rows", "category": "count"},
        {"question": "What is the total cost across all transactions?", "expected_answer": "sum of cost column", "category": "aggregation"},
        {"question": "List all unique regions", "expected_answer": "North, South, East, West", "category": "list"},
        {"question": "What is the average revenue per transaction?", "expected_answer": "mean of revenue", "category": "aggregation"},
        {"question": "How many transactions are in the Electronics category?", "expected_answer": "count of Electronics", "category": "count"},
        {"question": "What is the lowest revenue transaction?", "expected_answer": "min revenue", "category": "lookup"},
        {"question": "What is the total quantity sold?", "expected_answer": "sum of quantity", "category": "aggregation"},
        {"question": "How many transactions occurred in January 2024?", "expected_answer": "count for Jan 2024", "category": "count"},
        {"question": "What is the average profit per transaction?", "expected_answer": "mean of profit", "category": "aggregation"},
        {"question": "Which product has the highest unit price?", "expected_answer": "Laptop at 999.99", "category": "lookup"},
        {"question": "How many transactions are from Acme Corp?", "expected_answer": "count of Acme Corp", "category": "count"},
        {"question": "What is the sum of revenue for the Furniture category?", "expected_answer": "sum of Furniture revenue", "category": "aggregation"},
        {"question": "What is the earliest transaction date?", "expected_answer": "2024-01-05", "category": "lookup"},
        {"question": "What is the latest transaction date?", "expected_answer": "2024-09-05", "category": "lookup"},
        {"question": "How many unique products are in the Electronics category?", "expected_answer": "4", "category": "count"},
        {"question": "What is the median revenue?", "expected_answer": "median of revenue column", "category": "aggregation"},
        {"question": "How many transactions have revenue greater than 20000?", "expected_answer": "count where revenue > 20000", "category": "count"},
        {"question": "What is the total revenue from Smartphones?", "expected_answer": "sum of Smartphone revenue", "category": "aggregation"},
    ]
    for q in simple:
        q["complexity"] = "simple"
    questions.extend(simple)

    # ── Normal Questions (40) ─────────────────────────────────────
    normal = [
        {"question": "Which category generates the most total revenue?", "expected_answer": "category with highest revenue sum", "category": "aggregation"},
        {"question": "What are the top 3 products by total revenue?", "expected_answer": "top 3 products sorted by revenue sum", "category": "ranking"},
        {"question": "What is the average revenue per category?", "expected_answer": "avg revenue grouped by category", "category": "aggregation"},
        {"question": "Which region has the highest total profit?", "expected_answer": "region with max profit sum", "category": "aggregation"},
        {"question": "What is the total revenue by customer, sorted descending?", "expected_answer": "revenue sum per customer sorted", "category": "ranking"},
        {"question": "Which product has the highest average profit margin?", "expected_answer": "product with best profit/revenue ratio", "category": "aggregation"},
        {"question": "How does revenue compare across regions?", "expected_answer": "revenue sum per region comparison", "category": "comparison"},
        {"question": "What is the total quantity sold per category?", "expected_answer": "quantity sum per category", "category": "aggregation"},
        {"question": "Which customer has the most transactions?", "expected_answer": "customer with highest transaction count", "category": "ranking"},
        {"question": "What is the average cost per product?", "expected_answer": "avg cost grouped by product", "category": "aggregation"},
        {"question": "Which month had the highest total revenue?", "expected_answer": "month with max revenue sum", "category": "aggregation"},
        {"question": "What are the bottom 3 products by total profit?", "expected_answer": "3 products with lowest profit sum", "category": "ranking"},
        {"question": "What is the revenue breakdown by region in percentage?", "expected_answer": "region revenue as % of total", "category": "aggregation"},
        {"question": "Which product-region combination generates the most revenue?", "expected_answer": "top product+region pair", "category": "aggregation"},
        {"question": "What is the average quantity sold per product?", "expected_answer": "avg quantity per product", "category": "aggregation"},
        {"question": "How many transactions per month are there?", "expected_answer": "transaction count by month", "category": "aggregation"},
        {"question": "Which customer spends the most on Electronics?", "expected_answer": "top customer for Electronics", "category": "aggregation"},
        {"question": "What is the total profit margin (profit/revenue) for each category?", "expected_answer": "margin per category", "category": "aggregation"},
        {"question": "Which region has the most diverse product mix?", "expected_answer": "region with most unique products", "category": "aggregation"},
        {"question": "What is the standard deviation of revenue?", "expected_answer": "std of revenue column", "category": "statistics"},
        {"question": "Rank all customers by their average order value", "expected_answer": "customers ranked by avg revenue per transaction", "category": "ranking"},
        {"question": "What percentage of total revenue comes from Electronics?", "expected_answer": "Electronics revenue / total revenue * 100", "category": "aggregation"},
        {"question": "Which product has the most consistent (lowest variance) pricing?", "expected_answer": "product with lowest price std", "category": "statistics"},
        {"question": "How many transactions per customer per region?", "expected_answer": "cross-tabulation customer × region", "category": "aggregation"},
        {"question": "What is the 90th percentile of revenue?", "expected_answer": "90th percentile of revenue", "category": "statistics"},
        {"question": "Which category has the highest cost-to-revenue ratio?", "expected_answer": "category with highest cost/revenue", "category": "aggregation"},
        {"question": "What is the total revenue from the South region for Electronics?", "expected_answer": "filtered sum", "category": "aggregation"},
        {"question": "Which month had the most transactions?", "expected_answer": "month with highest count", "category": "aggregation"},
        {"question": "What are the top 5 customers by total profit?", "expected_answer": "top 5 customers by profit sum", "category": "ranking"},
        {"question": "What is the average profit per unit across all products?", "expected_answer": "avg profit/quantity per product", "category": "aggregation"},
        {"question": "Which product has the widest geographic reach?", "expected_answer": "product sold in most regions", "category": "aggregation"},
        {"question": "Compare the total revenue of Clothing vs Furniture", "expected_answer": "revenue comparison between categories", "category": "comparison"},
        {"question": "What is the ratio of Electronics revenue to total revenue?", "expected_answer": "Electronics proportion", "category": "aggregation"},
        {"question": "Which customer-product pair has the highest total quantity?", "expected_answer": "top customer+product pair by quantity", "category": "aggregation"},
        {"question": "What is the average order size in terms of quantity?", "expected_answer": "mean of quantity column", "category": "aggregation"},
        {"question": "Which region contributes the least to total profit?", "expected_answer": "region with lowest profit sum", "category": "aggregation"},
        {"question": "What percentage of transactions are above the median revenue?", "expected_answer": "50% approx", "category": "statistics"},
        {"question": "How does quantity sold compare between the North and South regions?", "expected_answer": "quantity comparison", "category": "comparison"},
        {"question": "What is the average profit margin by product?", "expected_answer": "profit/revenue by product", "category": "aggregation"},
        {"question": "Which category has seen the most transactions?", "expected_answer": "category with highest count", "category": "aggregation"},
    ]
    for q in normal:
        q["complexity"] = "normal"
    questions.extend(normal)

    # ── Complex Questions (30) ─────────────────────────────────────
    complex_qs = [
        {"question": "Which category had the highest revenue growth from the first half to the second half of the dataset?", "expected_answer": "category comparison across time periods", "category": "trend"},
        {"question": "What are the top 5 customers by revenue, and did their purchase frequency increase over time?", "expected_answer": "top customers + frequency trend analysis", "category": "trend"},
        {"question": "Is there a correlation between quantity sold and profit margin across products?", "expected_answer": "correlation analysis", "category": "correlation"},
        {"question": "Which product shows the most consistent month-over-month revenue growth?", "expected_answer": "month-over-month growth by product", "category": "trend"},
        {"question": "Compare the revenue distribution across regions and identify any significant outliers", "expected_answer": "distribution analysis with outlier detection", "category": "distribution"},
        {"question": "What would be the estimated impact on total revenue if the bottom-performing product was discontinued?", "expected_answer": "what-if analysis", "category": "analysis"},
        {"question": "Analyze the relationship between unit price and quantity sold — do higher-priced items sell less?", "expected_answer": "price-quantity relationship", "category": "correlation"},
        {"question": "Which customer-region combinations are most profitable, and what drives their profitability?", "expected_answer": "multi-dimensional profitability analysis", "category": "analysis"},
        {"question": "Is there a seasonal pattern in revenue? Which months consistently perform best?", "expected_answer": "seasonality analysis", "category": "trend"},
        {"question": "Calculate the Pareto distribution — do 20% of products generate 80% of revenue?", "expected_answer": "Pareto analysis", "category": "distribution"},
        {"question": "What is the year-over-year growth trend for each product category?", "expected_answer": "YoY growth by category", "category": "trend"},
        {"question": "Which products have improving profit margins over time, and which are declining?", "expected_answer": "margin trend analysis", "category": "trend"},
        {"question": "Identify any anomalous transactions in terms of revenue or profit", "expected_answer": "anomaly detection", "category": "anomaly"},
        {"question": "What is the weighted average cost across all categories, weighted by transaction count?", "expected_answer": "weighted average calculation", "category": "statistics"},
        {"question": "Compare the top 3 and bottom 3 products across all metrics (revenue, profit, quantity, margin)", "expected_answer": "multi-metric comparison", "category": "comparison"},
        {"question": "If we increased the price of Running Shoes by 10%, what would be the estimated impact on total revenue assuming constant quantity?", "expected_answer": "price sensitivity analysis", "category": "analysis"},
        {"question": "Which region is growing the fastest in terms of number of transactions and total revenue?", "expected_answer": "growth rate by region", "category": "trend"},
        {"question": "Analyze the customer concentration risk — how much revenue comes from the top 2 customers?", "expected_answer": "concentration analysis", "category": "distribution"},
        {"question": "What percentage change in revenue occurred from Q1 to Q3 2024?", "expected_answer": "quarterly change", "category": "trend"},
        {"question": "Which product has the best revenue-to-cost efficiency, and how does it compare across regions?", "expected_answer": "efficiency analysis cross-tabulated", "category": "analysis"},
        {"question": "Predict which product category is likely to have the highest revenue next month based on the trend", "expected_answer": "trend-based prediction", "category": "forecast"},
        {"question": "What is the month-over-month percentage change in total revenue?", "expected_answer": "MoM change series", "category": "trend"},
        {"question": "Are there any products that are becoming less profitable over time despite increasing revenue?", "expected_answer": "divergent trend analysis", "category": "trend"},
        {"question": "Calculate the coefficient of variation for revenue across all product categories", "expected_answer": "CV per category", "category": "statistics"},
        {"question": "Which customer segment (by purchase frequency) generates the most profit per transaction?", "expected_answer": "segmentation analysis", "category": "analysis"},
        {"question": "Identify the top-performing month for each product category", "expected_answer": "category × month max", "category": "aggregation"},
        {"question": "What is the cumulative revenue over time, broken down by category?", "expected_answer": "cumulative sum by category", "category": "trend"},
        {"question": "Compare the profit margins of products in different regions — are there regional pricing differences?", "expected_answer": "margin variation by region", "category": "comparison"},
        {"question": "What is the average days between transactions for each customer?", "expected_answer": "inter-transaction timing", "category": "analysis"},
        {"question": "Analyze the basket size (total revenue per transaction) distribution and identify patterns", "expected_answer": "distribution analysis", "category": "distribution"},
    ]
    for q in complex_qs:
        q["complexity"] = "complex"
    questions.extend(complex_qs)

    # Shuffle with fixed seed
    random.shuffle(questions)

    # Add IDs
    for i, q in enumerate(questions):
        q["id"] = i + 1
        q["dataset_id"] = "sample_sales"

    return questions


def save_benchmark(questions: list[dict], output_dir: Path) -> tuple[Path, Path]:
    """Save benchmark questions to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "benchmark_questions.csv"
    json_path = output_dir / "benchmark_questions.json"

    # Save CSV
    fieldnames = ["id", "question", "expected_answer", "complexity", "category", "dataset_id"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for q in questions:
            writer.writerow({k: q[k] for k in fieldnames})

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


def prepare_benchmark(data_dir: Path) -> tuple[list[dict], Path, Path]:
    """Full preparation pipeline: generate questions and save to files."""
    questions = generate_benchmark_questions(data_dir)
    eval_dir = data_dir.parent / "evaluation" / "data"
    csv_path, json_path = save_benchmark(questions, eval_dir)
    return questions, csv_path, json_path


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    questions, csv_path, json_path = prepare_benchmark(data_dir)
    print(f"Generated {len(questions)} benchmark questions")
    print(f"  Simple:  {sum(1 for q in questions if q['complexity'] == 'simple')}")
    print(f"  Normal:  {sum(1 for q in questions if q['complexity'] == 'normal')}")
    print(f"  Complex: {sum(1 for q in questions if q['complexity'] == 'complex')}")
    print(f"Saved to: {csv_path} and {json_path}")
