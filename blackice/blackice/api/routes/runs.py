"""Run management endpoints for BLACKICE API."""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect

from blackice.api.deps import ConfigDep
from blackice.api.schemas import (
    Edition,
    ErrorResponse,
    PhaseResult,
    ProviderType,
    RunCreateRequest,
    RunCreateResponse,
    RunDetail,
    RunListResponse,
    RunResultResponse,
    RunResumeRequest,
    RunStatus,
    RunSummary,
    StreamEvent,
)
from blackice.core.providers import create_execution_provider, create_memory_provider, create_model_provider
from blackice.flywheel import FlywheelConfig, UnifiedFlywheel

router = APIRouter(prefix="/runs", tags=["runs"])

# In-memory run storage (would be replaced with persistent storage in production)
_runs: dict[str, dict[str, Any]] = {}
_active_connections: dict[str, list[WebSocket]] = {}


def _generate_run_id() -> str:
    """Generate a unique run ID."""
    return f"run-{uuid.uuid4().hex[:12]}"


@router.post("", response_model=RunCreateResponse, status_code=201)
async def create_run(
    request: RunCreateRequest,
    background_tasks: BackgroundTasks,
    config: ConfigDep,
) -> RunCreateResponse:
    """Create and start a new BLACKICE run.

    The run executes asynchronously. Use GET /runs/{run_id} to check status
    or connect to WebSocket for real-time updates.
    """
    run_id = _generate_run_id()

    # Create workspace
    if request.workspace:
        workspace = Path(request.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
    else:
        workspace = Path(tempfile.mkdtemp(prefix=f"blackice-{run_id}-"))

    # Store run metadata
    _runs[run_id] = {
        "run_id": run_id,
        "vision": request.vision,
        "status": RunStatus.CREATED,
        "edition": request.edition,
        "provider": request.provider,
        "model": request.model,
        "workspace": str(workspace),
        "created_at": datetime.utcnow(),
        "updated_at": None,
        "completed_at": None,
        "phases": [],
        "artifacts": [],
        "current_phase": None,
        "error": None,
        "duration_ms": None,
        "context": request.context,
    }

    # Start run in background
    background_tasks.add_task(
        _execute_run,
        run_id=run_id,
        vision=request.vision,
        edition=request.edition,
        provider=request.provider,
        model=request.model,
        workspace=workspace,
        context=request.context,
    )

    return RunCreateResponse(
        run_id=run_id,
        status=RunStatus.CREATED,
        message="Run created and starting",
        workspace=str(workspace),
    )


async def _execute_run(
    run_id: str,
    vision: str,
    edition: Edition,
    provider: ProviderType,
    model: str | None,
    workspace: Path,
    context: dict[str, Any],
) -> None:
    """Execute a run in the background."""
    try:
        # Update status
        _runs[run_id]["status"] = RunStatus.RESEARCH
        _runs[run_id]["current_phase"] = "research"
        await _broadcast_event(run_id, "phase_start", {"phase": "research"})

        # Create providers
        model_provider = create_model_provider(
            provider_type=provider.value,
            model=model,
        )
        memory_provider = create_memory_provider()
        execution_provider = create_execution_provider(str(workspace))

        # Configure flywheel
        flywheel_config = FlywheelConfig(
            workspace_root=workspace,
            plan_timeout=300.0,
            implement_timeout=600.0,
            test_timeout=120.0,
            verify_timeout=120.0,
        )

        flywheel = UnifiedFlywheel(
            config=flywheel_config,
            model_provider=model_provider,
            execution_provider=execution_provider,
            memory_provider=memory_provider,
        )

        # Run the flywheel
        start_time = datetime.utcnow()
        result = await flywheel.run(
            run_id=run_id,
            vision=vision,
            context={"edition": edition.value, **context},
        )

        # Update run with results
        _runs[run_id]["status"] = RunStatus.COMPLETED if result.success else RunStatus.FAILED
        _runs[run_id]["completed_at"] = datetime.utcnow()
        _runs[run_id]["duration_ms"] = (datetime.utcnow() - start_time).total_seconds() * 1000
        _runs[run_id]["artifacts"] = result.artifacts or []
        _runs[run_id]["phases"] = ["plan", "implement", "test", "verify"]
        _runs[run_id]["current_phase"] = None

        await _broadcast_event(run_id, "run_complete", {
            "success": result.success,
            "artifacts": result.artifacts,
        })

        # Cleanup
        await model_provider.close()
        await memory_provider.close()

    except Exception as e:
        _runs[run_id]["status"] = RunStatus.FAILED
        _runs[run_id]["error"] = str(e)
        _runs[run_id]["current_phase"] = None
        await _broadcast_event(run_id, "run_error", {"error": str(e)})


async def _broadcast_event(run_id: str, event: str, data: dict[str, Any]) -> None:
    """Broadcast event to all connected WebSocket clients."""
    if run_id not in _active_connections:
        return

    event_data = StreamEvent(
        event=event,
        run_id=run_id,
        data=data,
    )

    disconnected = []
    for ws in _active_connections[run_id]:
        try:
            await ws.send_json(event_data.model_dump(mode="json"))
        except Exception:
            disconnected.append(ws)

    # Remove disconnected clients
    for ws in disconnected:
        _active_connections[run_id].remove(ws)


@router.get("", response_model=RunListResponse)
async def list_runs(
    limit: int = 20,
    offset: int = 0,
    status: RunStatus | None = None,
) -> RunListResponse:
    """List all runs with optional filtering."""
    runs = list(_runs.values())

    # Filter by status if provided
    if status:
        runs = [r for r in runs if r["status"] == status]

    # Sort by created_at descending
    runs.sort(key=lambda r: r["created_at"], reverse=True)

    # Paginate
    total = len(runs)
    runs = runs[offset : offset + limit]

    return RunListResponse(
        runs=[
            RunSummary(
                run_id=r["run_id"],
                vision=r["vision"][:100] + "..." if len(r["vision"]) > 100 else r["vision"],
                status=r["status"],
                edition=r["edition"],
                provider=r["provider"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                phase_count=len(r["phases"]),
                artifact_count=len(r["artifacts"]),
            )
            for r in runs
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{run_id}", response_model=RunDetail)
async def get_run(run_id: str) -> RunDetail:
    """Get detailed information about a specific run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    r = _runs[run_id]
    return RunDetail(
        run_id=r["run_id"],
        vision=r["vision"],
        status=r["status"],
        edition=r["edition"],
        provider=r["provider"],
        model=r["model"],
        workspace=r["workspace"],
        created_at=r["created_at"],
        updated_at=r["updated_at"],
        completed_at=r["completed_at"],
        phases=r["phases"],
        artifacts=r["artifacts"],
        current_phase=r["current_phase"],
        error=r["error"],
        duration_ms=r["duration_ms"],
        context=r["context"],
    )


@router.post("/{run_id}/resume", response_model=RunCreateResponse)
async def resume_run(
    run_id: str,
    request: RunResumeRequest,
    background_tasks: BackgroundTasks,
) -> RunCreateResponse:
    """Resume a failed or stopped run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run = _runs[run_id]
    if run["status"] not in [RunStatus.FAILED, RunStatus.CREATED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume run in status {run['status']}",
        )

    # Reset status and restart
    run["status"] = RunStatus.CREATED
    run["error"] = None
    run["updated_at"] = datetime.utcnow()

    background_tasks.add_task(
        _execute_run,
        run_id=run_id,
        vision=run["vision"],
        edition=run["edition"],
        provider=run["provider"],
        model=run["model"],
        workspace=Path(run["workspace"]),
        context=run["context"],
    )

    return RunCreateResponse(
        run_id=run_id,
        status=RunStatus.CREATED,
        message="Run resumed",
        workspace=run["workspace"],
    )


@router.delete("/{run_id}")
async def cancel_run(run_id: str) -> dict[str, str]:
    """Cancel a running run or delete a completed run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run = _runs[run_id]

    # If running, mark as failed
    if run["status"] in [RunStatus.RESEARCH, RunStatus.PLAN, RunStatus.IMPLEMENT, RunStatus.TEST, RunStatus.VERIFY]:
        run["status"] = RunStatus.FAILED
        run["error"] = "Cancelled by user"

    # Remove from storage
    del _runs[run_id]

    return {"message": f"Run {run_id} deleted"}


@router.websocket("/{run_id}/stream")
async def stream_run(websocket: WebSocket, run_id: str) -> None:
    """WebSocket endpoint for streaming run events."""
    if run_id not in _runs:
        await websocket.close(code=4004, reason="Run not found")
        return

    await websocket.accept()

    # Register connection
    if run_id not in _active_connections:
        _active_connections[run_id] = []
    _active_connections[run_id].append(websocket)

    try:
        # Send current status
        run = _runs[run_id]
        await websocket.send_json({
            "event": "connected",
            "run_id": run_id,
            "status": run["status"].value,
            "current_phase": run["current_phase"],
        })

        # Keep connection alive until run completes or client disconnects
        while True:
            try:
                # Wait for messages (ping/pong or commands)
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)

                if data.get("command") == "ping":
                    await websocket.send_json({"event": "pong"})

            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"event": "keepalive"})

            # Check if run completed
            if _runs.get(run_id, {}).get("status") in [RunStatus.COMPLETED, RunStatus.FAILED]:
                await websocket.send_json({
                    "event": "run_finished",
                    "run_id": run_id,
                    "status": _runs[run_id]["status"].value,
                })
                break

    except WebSocketDisconnect:
        pass
    finally:
        # Unregister connection
        if run_id in _active_connections and websocket in _active_connections[run_id]:
            _active_connections[run_id].remove(websocket)
