"""DataPilot — LangGraph workflow orchestration.

Implements the multi-agent pipeline as a LangGraph StateGraph with
conditional edges for visualization skipping and verification retries.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.planner import run_planner
from app.agents.data_agent import run_data_agent
from app.agents.analysis import run_analysis_agent
from app.agents.visualization import run_visualization_agent
from app.agents.verifier import run_verifier
from app.cache.redis_cache import RedisCache
from app.cache.semantic_cache import SemanticCache
from app.mcp.server import call_tool, get_dataset
from app.models import AgentStep, AgentName, CacheStatus, Complexity, QueryResponse, ToolCallRecord
from app.observability.metrics import MetricsCollector, RequestRecord, calculate_cost
from app.router.model_router import ModelRouter

logger = logging.getLogger("datapilot.graph")

MAX_RETRIES = 2


# ── Graph State ───────────────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    """State that flows through the agent graph."""
    # Input
    question: str
    dataset_id: str
    use_cache: bool
    request_id: str

    # Planner output
    plan: dict[str, Any]
    complexity: str
    model_name: str

    # Data agent output
    schema: dict[str, Any]
    retrieved_data: dict[str, Any]
    data_tool_calls: list[str]
    data_tool_records: list[dict]

    # Analysis output
    analysis_result: dict[str, Any]
    answer: str

    # Visualization output
    visualization: Optional[str]  # base64 PNG
    chart_type: Optional[str]

    # Verifier output
    verified: bool
    verification_notes: str
    retry_count: int

    # Metadata
    agent_steps: list[dict]
    all_tool_records: list[dict]
    total_tokens: int
    llm_calls: int
    cache_hit: bool
    semantic_cache_hit: bool
    error: Optional[str]


# ── Node Functions ────────────────────────────────────────────────────

async def cache_check_node(state: GraphState, **kwargs) -> dict:
    """Check semantic cache before running the pipeline."""
    semantic_cache: SemanticCache = kwargs.get("semantic_cache")
    if not state.get("use_cache", True) or not semantic_cache:
        return {"semantic_cache_hit": False}

    cached = semantic_cache.get(state["question"], state["dataset_id"])
    if cached:
        logger.info("Semantic cache hit for question")
        return {
            "semantic_cache_hit": True,
            "cache_hit": True,
            "answer": cached["answer"],
            "visualization": cached.get("visualization"),
            "chart_type": cached.get("chart_type"),
            "verified": True,
            "verification_notes": f"Cached result (similarity: {cached.get('similarity_score', 0):.3f})",
        }
    return {"semantic_cache_hit": False}


async def planner_node(state: GraphState, **kwargs) -> dict:
    """Run the planner agent."""
    start = time.time()
    router: ModelRouter = kwargs.get("router")
    cache: RedisCache = kwargs.get("cache")
    on_step = kwargs.get("on_step")

    if on_step:
        try:
            await on_step({"type": "agent_start", "agent": "planner", "summary": "Analyzing question & data schema..."})
        except Exception:
            pass

    dataset_id = state["dataset_id"]

    # Get schema (with caching)
    schema_params = {"dataset_id": dataset_id}
    schema = cache.get("get_schema", schema_params) if cache and state.get("use_cache", True) else None
    if schema is None:
        schema = call_tool("get_schema", dataset_id=dataset_id)
        if cache:
            cache.set("get_schema", schema_params, schema)

    # Route model
    llm, complexity, decision = router.get_llm(state["question"])

    # Run planner
    plan = await run_planner(state["question"], schema, llm)

    latency = (time.time() - start) * 1000
    step = {
        "agent": AgentName.PLANNER.value,
        "model_used": decision.model_name,
        "latency_ms": latency,
        "status": "completed",
        "summary": f"Plan: {len(plan.get('steps', []))} steps, viz={plan.get('needs_visualization')}",
    }

    if on_step:
        try:
            await on_step({"type": "agent_complete", "agent": "planner", "summary": step["summary"]})
        except Exception:
            pass

    return {
        "plan": plan,
        "schema": schema,
        "complexity": complexity.value,
        "model_name": decision.model_name,
        "agent_steps": state.get("agent_steps", []) + [step],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "total_tokens": state.get("total_tokens", 0) + plan.get("_tokens", 0),
    }


async def data_node(state: GraphState, **kwargs) -> dict:
    """Run the data agent."""
    start = time.time()
    router: ModelRouter = kwargs.get("router")
    cache: RedisCache = kwargs.get("cache")
    on_step = kwargs.get("on_step")

    if on_step:
        try:
            await on_step({"type": "agent_start", "agent": "data", "summary": "Retrieving data via MCP tools..."})
        except Exception:
            pass

    llm = router.get_llm_by_complexity(Complexity(state.get("complexity", "normal")))

    result = await run_data_agent(
        question=state["question"],
        plan=state["plan"],
        schema=state["schema"],
        dataset_id=state["dataset_id"],
        llm=llm,
        cache=cache if state.get("use_cache", True) else None,
    )

    latency = (time.time() - start) * 1000
    step = {
        "agent": AgentName.DATA.value,
        "model_used": state.get("model_name", ""),
        "latency_ms": latency,
        "status": "completed",
        "summary": f"Retrieved data via {len(result['tool_calls'])} tool call(s)",
        "tool_calls": result["tool_calls"],
    }

    if on_step:
        try:
            await on_step({"type": "agent_complete", "agent": "data", "summary": step["summary"]})
        except Exception:
            pass

    return {
        "retrieved_data": result["retrieved_data"],
        "data_tool_calls": result["tool_calls"],
        "data_tool_records": result["tool_records"],
        "agent_steps": state.get("agent_steps", []) + [step],
        "all_tool_records": state.get("all_tool_records", []) + result["tool_records"],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "total_tokens": state.get("total_tokens", 0) + result.get("tokens", 0),
    }


async def analysis_node(state: GraphState, **kwargs) -> dict:
    """Run the analysis agent."""
    start = time.time()
    router: ModelRouter = kwargs.get("router")
    cache: RedisCache = kwargs.get("cache")
    on_step = kwargs.get("on_step")

    if on_step:
        try:
            await on_step({"type": "agent_start", "agent": "analysis", "summary": "Performing analytical computation..."})
        except Exception:
            pass

    # Use higher-capability model for analysis
    complexity = Complexity(state.get("complexity", "normal"))
    if complexity == Complexity.SIMPLE:
        complexity = Complexity.NORMAL  # Bump simple to normal for analysis
    llm = router.get_llm_by_complexity(complexity)

    result = await run_analysis_agent(
        question=state["question"],
        plan=state["plan"],
        retrieved_data=state["retrieved_data"],
        dataset_id=state["dataset_id"],
        llm=llm,
        cache=cache if state.get("use_cache", True) else None,
    )

    answer = result.get("answer", "Could not generate an answer.")
    latency = (time.time() - start) * 1000

    step = {
        "agent": AgentName.ANALYSIS.value,
        "model_used": state.get("model_name", ""),
        "latency_ms": latency,
        "status": "completed",
        "summary": f"Analysis complete: {len(result.get('key_findings', []))} findings",
        "tool_calls": result.get("tool_calls", []),
    }

    llm_bump = 1
    if result.get("tool_calls"):
        llm_bump = 2  # Extra call for code interpretation

    if on_step:
        try:
            await on_step({"type": "agent_complete", "agent": "analysis", "summary": step["summary"]})
        except Exception:
            pass

    return {
        "analysis_result": result,
        "answer": answer,
        "agent_steps": state.get("agent_steps", []) + [step],
        "all_tool_records": state.get("all_tool_records", []) + result.get("tool_records", []),
        "llm_calls": state.get("llm_calls", 0) + llm_bump,
        "total_tokens": state.get("total_tokens", 0) + result.get("tokens", 0),
    }


async def visualization_node(state: GraphState, **kwargs) -> dict:
    """Run the visualization agent (conditional)."""
    start = time.time()
    router: ModelRouter = kwargs.get("router")
    on_step = kwargs.get("on_step")

    if on_step:
        try:
            await on_step({"type": "agent_start", "agent": "visualization", "summary": "Generating chart..."})
        except Exception:
            pass

    llm = router.get_llm_by_complexity(Complexity.SIMPLE)  # Viz is straightforward

    result = await run_visualization_agent(
        question=state["question"],
        plan=state["plan"],
        analysis_result=state.get("analysis_result", {}),
        retrieved_data=state.get("retrieved_data", {}),
        dataset_id=state["dataset_id"],
        llm=llm,
    )

    latency = (time.time() - start) * 1000
    skipped = result.get("skipped", True)
    is_failed = result.get("failed", False)

    step = {
        "agent": AgentName.VISUALIZATION.value,
        "model_used": state.get("model_name", ""),
        "latency_ms": latency,
        "status": "failed" if is_failed else ("skipped" if skipped else "completed"),
        "summary": result.get("reason", f"Chart: {result.get('chart_type', 'none')}") if not skipped else result.get("reason", "Skipped"),
    }

    if on_step:
        try:
            evt_type = "agent_failed" if is_failed else "agent_complete"
            await on_step({"type": evt_type, "agent": "visualization", "summary": step["summary"]})
        except Exception:
            pass

    return {
        "visualization": result.get("image_base64") if not skipped else None,
        "chart_type": result.get("chart_type") if not skipped else None,
        "agent_steps": state.get("agent_steps", []) + [step],
        "llm_calls": state.get("llm_calls", 0) + (0 if skipped else 1),
        "total_tokens": state.get("total_tokens", 0) + result.get("tokens", 0),
    }


async def verifier_node(state: GraphState, **kwargs) -> dict:
    """Run the verifier agent."""
    start = time.time()
    router: ModelRouter = kwargs.get("router")
    on_step = kwargs.get("on_step")

    if on_step:
        try:
            await on_step({"type": "agent_start", "agent": "verifier", "summary": "Verifying calculation accuracy..."})
        except Exception:
            pass

    # Use high-capability model for verification
    llm = router.get_llm_by_complexity(Complexity.NORMAL)

    result = await run_verifier(
        question=state["question"],
        proposed_answer=state.get("answer", ""),
        analysis_result=state.get("analysis_result", {}),
        retrieved_data=state.get("retrieved_data", {}),
        llm=llm,
    )

    latency = (time.time() - start) * 1000
    verified = result.get("verified", True)

    # If not verified and we have a corrected answer, use it
    answer = state.get("answer", "")
    if not verified and result.get("corrected_answer"):
        answer = result["corrected_answer"]

    step = {
        "agent": AgentName.VERIFIER.value,
        "model_used": state.get("model_name", ""),
        "latency_ms": latency,
        "status": "completed",
        "summary": f"{'Verified' if verified else 'Failed'} (confidence: {result.get('confidence', 0):.2f})",
    }

    retry_count = state.get("retry_count", 0)

    if on_step:
        try:
            await on_step({"type": "agent_complete", "agent": "verifier", "summary": step["summary"]})
        except Exception:
            pass

    return {
        "verified": verified,
        "verification_notes": result.get("verification_notes", ""),
        "answer": answer,
        "retry_count": retry_count + (0 if verified else 1),
        "agent_steps": state.get("agent_steps", []) + [step],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "total_tokens": state.get("total_tokens", 0) + result.get("_tokens", 0),
    }


# ── Conditional Edges ────────────────────────────────────────────────

def should_skip_if_cached(state: GraphState) -> str:
    """Skip entire pipeline if semantic cache hit."""
    if state.get("semantic_cache_hit", False):
        return "cached"
    return "continue"


def should_visualize(state: GraphState) -> str:
    """Check if visualization is needed."""
    plan = state.get("plan", {})
    if plan.get("needs_visualization", False):
        return "visualize"
    return "skip_viz"


def should_retry(state: GraphState) -> str:
    """Check if verification failed and we should retry."""
    if not state.get("verified", True) and state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "finish"


# ── Build Graph ───────────────────────────────────────────────────────

def build_graph(
    router: ModelRouter,
    cache: Optional[RedisCache] = None,
    semantic_cache: Optional[SemanticCache] = None,
    on_step: Optional[Any] = None,
) -> Any:
    """Build and compile the LangGraph workflow.
    
    Returns a compiled graph that can be invoked with a GraphState dict.
    """
    graph = StateGraph(GraphState)

    # Bind dependencies to nodes via closures
    async def _cache_check(state):
        return await cache_check_node(state, semantic_cache=semantic_cache)

    async def _planner(state):
        return await planner_node(state, router=router, cache=cache, on_step=on_step)

    async def _data(state):
        return await data_node(state, router=router, cache=cache, on_step=on_step)

    async def _analysis(state):
        return await analysis_node(state, router=router, cache=cache, on_step=on_step)

    async def _visualization(state):
        return await visualization_node(state, router=router, on_step=on_step)

    async def _verifier(state):
        return await verifier_node(state, router=router, on_step=on_step)

    # Add nodes
    graph.add_node("cache_check", _cache_check)
    graph.add_node("planner", _planner)
    graph.add_node("data_agent", _data)
    graph.add_node("analysis_agent", _analysis)
    graph.add_node("visualization_agent", _visualization)
    graph.add_node("verifier", _verifier)

    # Set entry point
    graph.set_entry_point("cache_check")

    # Edges
    graph.add_conditional_edges(
        "cache_check",
        should_skip_if_cached,
        {"cached": END, "continue": "planner"},
    )
    graph.add_edge("planner", "data_agent")
    graph.add_edge("data_agent", "analysis_agent")
    graph.add_conditional_edges(
        "analysis_agent",
        should_visualize,
        {"visualize": "visualization_agent", "skip_viz": "verifier"},
    )
    graph.add_edge("visualization_agent", "verifier")
    graph.add_conditional_edges(
        "verifier",
        should_retry,
        {"retry": "analysis_agent", "finish": END},
    )

    return graph.compile()


# ── Pipeline Runner ───────────────────────────────────────────────────

async def run_pipeline(
    question: str,
    dataset_id: str,
    router: ModelRouter,
    cache: Optional[RedisCache] = None,
    semantic_cache: Optional[SemanticCache] = None,
    metrics: Optional[MetricsCollector] = None,
    use_cache: bool = True,
    on_step: Optional[Any] = None,
) -> QueryResponse:
    """Run the full DataPilot pipeline and return a structured response."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Track metrics
    record = None
    if metrics:
        record = metrics.start_request(request_id, question, dataset_id)

    try:
        # Build and run graph
        compiled = build_graph(router, cache, semantic_cache, on_step=on_step)
        initial_state: GraphState = {
            "question": question,
            "dataset_id": dataset_id,
            "use_cache": use_cache,
            "request_id": request_id,
            "retry_count": 0,
            "agent_steps": [],
            "all_tool_records": [],
            "total_tokens": 0,
            "llm_calls": 0,
            "cache_hit": False,
            "semantic_cache_hit": False,
            "error": None,
        }

        result = await compiled.ainvoke(initial_state)
        total_latency = (time.time() - start_time) * 1000

        # Build agent steps
        agent_steps = [
            AgentStep(
                agent=AgentName(s["agent"]),
                model_used=s.get("model_used"),
                latency_ms=s.get("latency_ms", 0),
                tool_calls=s.get("tool_calls", []),
                status=s.get("status", "completed"),
                summary=s.get("summary", ""),
            )
            for s in result.get("agent_steps", [])
        ]

        # Build tool call records
        tool_records = [
            ToolCallRecord(
                tool_name=t.get("tool_name", ""),
                arguments=t.get("params", {}),
                cache_status=CacheStatus(t.get("cache_status", "MISS")),
                success=t.get("success", True),
            )
            for t in result.get("all_tool_records", [])
        ]

        # Store in semantic cache
        answer = result.get("answer", "No answer generated.")
        total_tokens = result.get("total_tokens", 0)
        model_used = result.get("model_name", "")
        cost = calculate_cost(model_used, total_tokens)

        if semantic_cache and use_cache and not result.get("semantic_cache_hit"):
            semantic_cache.set(
                question=question,
                dataset_id=dataset_id,
                answer=answer,
                visualization=result.get("visualization"),
                chart_type=result.get("chart_type"),
                tokens_used=total_tokens,
            )

        response = QueryResponse(
            question=question,
            answer=answer,
            dataset_id=dataset_id,
            visualization=result.get("visualization"),
            chart_type=result.get("chart_type"),
            complexity=Complexity(result.get("complexity", "normal")),
            model_used=model_used,
            agent_steps=agent_steps,
            tool_calls=tool_records,
            cache_status=CacheStatus.HIT if result.get("cache_hit") else CacheStatus.MISS,
            semantic_cache_status=CacheStatus.HIT if result.get("semantic_cache_hit") else CacheStatus.MISS,
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            llm_calls=result.get("llm_calls", 0),
            verified=result.get("verified", False),
            verification_notes=result.get("verification_notes", ""),
        )

        # Update metrics
        if record and metrics:
            record.total_latency_ms = total_latency
            record.model_selected = model_used
            record.complexity = result.get("complexity", "")
            record.cache_hit = result.get("cache_hit", False)
            record.semantic_cache_hit = result.get("semantic_cache_hit", False)
            record.tool_calls = [t.tool_name for t in tool_records]
            record.llm_calls = result.get("llm_calls", 0)
            record.tokens_used = total_tokens
            record.estimated_cost = cost
            record.success = True
            metrics.end_request(record)

        return response

    except Exception as e:
        total_latency = (time.time() - start_time) * 1000
        logger.error("Pipeline failed: %s", e, exc_info=True)

        if record and metrics:
            record.total_latency_ms = total_latency
            record.success = False
            record.error = str(e)
            metrics.end_request(record)

        return QueryResponse(
            question=question,
            answer=f"An error occurred while processing your question: {str(e)}",
            dataset_id=dataset_id,
            total_latency_ms=total_latency,
            verified=False,
            verification_notes=f"Pipeline error: {str(e)}",
        )
