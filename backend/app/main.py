"""DataPilot — FastAPI backend application.

Main entry point with REST API endpoints for queries, dataset management,
metrics, and health checks. Includes WebSocket for live agent status.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

# Ensure backend directory is in sys.path so 'app' package is always importable
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agents.graph import run_pipeline
from app.cache.redis_cache import RedisCache
from app.cache.semantic_cache import SemanticCache
from app.config import get_settings
from app.mcp.server import get_dataset, list_datasets, load_dataset
from app.models import (
    DatasetInfo,
    HealthResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
)
from app.observability.metrics import MetricsCollector, metrics_collector, setup_logging
from app.router.model_router import ModelRouter

# ── App Setup ─────────────────────────────────────────────────────────

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("datapilot.api")

app = FastAPI(
    title="DataPilot API",
    description="Multi-Agent AI Data Analyst",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared State ──────────────────────────────────────────────────────

router = ModelRouter()
cache = RedisCache(
    redis_url=settings.upstash_redis_rest_url,
    redis_token=settings.upstash_redis_rest_token,
    ttl=settings.cache_ttl_seconds,
)
semantic_cache = SemanticCache(
    threshold=settings.semantic_cache_threshold,
)

# Active WebSocket connections for live updates
ws_connections: list[WebSocket] = []


# ── Startup ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Load sample datasets on startup."""
    data_dir = settings.data_path
    if data_dir.exists():
        for csv_file in data_dir.glob("*.csv"):
            try:
                load_dataset(csv_file.stem, csv_file)
                logger.info("Loaded dataset: %s", csv_file.stem)
            except Exception as e:
                logger.warning("Failed to load %s: %s", csv_file.name, e)

    # Ensure upload directory exists
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    logger.info("DataPilot API started | Redis: %s", "connected" if cache.is_redis else "in-memory")


# ── API Endpoints ─────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Submit an analytical question about a dataset."""
    # Validate dataset exists
    available = list_datasets()
    if request.dataset_id not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{request.dataset_id}' not found. Available: {available}",
        )

    logger.info("Query: dataset=%s question='%s'", request.dataset_id, request.question[:80])

    # Broadcast status to WebSocket clients
    await _broadcast({
        "type": "pipeline_start",
        "question": request.question,
        "dataset_id": request.dataset_id,
    })

    try:
        response = await run_pipeline(
            question=request.question,
            dataset_id=request.dataset_id,
            router=router,
            cache=cache if request.use_cache else None,
            semantic_cache=semantic_cache if request.use_cache else None,
            metrics=metrics_collector,
            use_cache=request.use_cache,
            on_step=_broadcast,
        )

        await _broadcast({
            "type": "pipeline_complete",
            "answer": response.answer[:200],
            "latency_ms": response.total_latency_ms,
            "model": response.model_used,
            "verified": response.verified,
        })

        return response

    except Exception as e:
        logger.error("Query failed: %s", e, exc_info=True)
        await _broadcast({"type": "pipeline_error", "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    dataset_id = Path(file.filename).stem
    save_path = settings.upload_path / file.filename

    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        info = load_dataset(dataset_id, save_path)
        logger.info("Uploaded dataset: %s (%d rows)", dataset_id, info["rows"])

        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "rows": info["rows"],
            "columns": info["columns"],
            "column_names": info["column_names"],
        }
    except Exception as e:
        logger.error("Upload failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.get("/api/datasets")
async def get_datasets():
    """List all available datasets."""
    datasets = []
    for dataset_id in list_datasets():
        try:
            df = get_dataset(dataset_id)
            datasets.append(DatasetInfo(
                dataset_id=dataset_id,
                filename=f"{dataset_id}.csv",
                rows=len(df),
                columns=len(df.columns),
                column_names=list(df.columns),
                size_bytes=df.memory_usage(deep=True).sum(),
            ))
        except Exception:
            pass
    return datasets


@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get application metrics summary."""
    summary = metrics_collector.get_summary()

    # Add cache stats
    cache_stats = cache.stats
    semantic_stats = semantic_cache.stats

    return summary


@app.get("/api/metrics/recent")
async def get_recent_metrics():
    """Get recent request records with detailed metrics."""
    return {
        "records": metrics_collector.get_recent_records(50),
        "cache_stats": cache.stats,
        "semantic_cache_stats": semantic_cache.stats,
        "routing_log": router.get_routing_log()[-20:],
    }


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        redis_connected=cache.is_redis,
        datasets_available=len(list_datasets()),
        version="1.0.0",
    )


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear all caches."""
    cache.clear()
    semantic_cache.clear()
    cache.reset_stats()
    semantic_cache.reset_stats()
    return {"status": "cleared"}


# ── WebSocket ─────────────────────────────────────────────────────────

@app.websocket("/ws/query")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live pipeline status updates."""
    await websocket.accept()
    ws_connections.append(websocket)
    logger.info("WebSocket client connected (%d total)", len(ws_connections))
    try:
        while True:
            data = await websocket.receive_text()
            # Client can send ping/heartbeat
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        ws_connections.remove(websocket)
        logger.info("WebSocket client disconnected (%d remaining)", len(ws_connections))


async def _broadcast(message: dict):
    """Send a message to all connected WebSocket clients."""
    if not ws_connections:
        return
    text = json.dumps(message, default=str)
    disconnected = []
    for ws in ws_connections:
        try:
            await ws.send_text(text)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        ws_connections.remove(ws)


# ── Static Files (Frontend) ───────────────────────────────────────────

from fastapi.staticfiles import StaticFiles

frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


# ── Run ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
