"""Run management endpoints for BLACKICE API.

P0 Security Fixes (Phase 8.1):
- Task tracking for proper cancellation
- Split cancel vs delete semantics
- Queue-based WebSocket sending (no concurrent sends)
- Server-controlled workspace paths
- Proper status checks before mutations

P1 Security Fixes (Phase 8.2):
- WebSocket single-sender guarantee (all sends through queue)
- Delete now purges workspace on disk with symlink attack prevention
"""

from __future__ import annotations

import asyncio
import shutil
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

# Server-controlled workspace root (security: no client path injection)
WORKSPACE_ROOT = Path(tempfile.gettempdir()) / "blackice-workspaces"
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

# In-memory run storage (would be replaced with persistent storage in production)
_runs: dict[str, dict[str, Any]] = {}
_tasks: dict[str, asyncio.Task] = {}  # Track async tasks for cancellation

# WebSocket message queues (one per connection for safe concurrent sending)
_ws_queues: dict[str, dict[int, asyncio.Queue]] = {}  # run_id -> {ws_id -> queue}
_ws_counter: int = 0


def _generate_run_id() -> str:
    """Generate a unique run ID."""
    return f"run-{uuid.uuid4().hex[:12]}"


def _is_terminal_status(status: RunStatus) -> bool:
    """Check if status is terminal (run finished)."""
    return status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]


def _is_running_status(status: RunStatus) -> bool:
    """Check if status indicates active execution."""
    return status in [
        RunStatus.CREATED,
        RunStatus.RESEARCH,
        RunStatus.PLAN,
        RunStatus.IMPLEMENT,
        RunStatus.TEST,
        RunStatus.VERIFY,
        RunStatus.DELIVER,
    ]


@router.post("", response_model=RunCreateResponse, status_code=201)
async def create_run(
    request: RunCreateRequest,
    config: ConfigDep,
) -> RunCreateResponse:
    """Create and start a new BLACKICE run.

    The run executes asynchronously. Use GET /runs/{run_id} to check status
    or connect to WebSocket for real-time updates.
    """
    run_id = _generate_run_id()

    # Server-controlled workspace (security: no client path injection)
    workspace = WORKSPACE_ROOT / run_id
    workspace.mkdir(parents=True, exist_ok=True)

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
        "phase_results": [],  # Track actual phase execution
        "artifacts": [],
        "current_phase": None,
        "error": None,
        "duration_ms": None,
        "context": request.context,
    }

    # Start run as tracked async task (enables cancellation)
    task = asyncio.create_task(
        _execute_run(
            run_id=run_id,
            vision=request.vision,
            edition=request.edition,
            provider=request.provider,
            model=request.model,
            workspace=workspace,
            context=request.context,
        )
    )
    _tasks[run_id] = task

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
    start_time = datetime.utcnow()

    try:
        # Check if cancelled before starting
        if run_id not in _runs:
            return
        if _runs[run_id]["status"] == RunStatus.CANCELLED:
            return

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

        # Run the flywheel (check for cancellation periodically)
        result = await flywheel.run(
            run_id=run_id,
            vision=vision,
            context={"edition": edition.value, **context},
        )

        # Check if cancelled during execution
        if run_id not in _runs or _runs[run_id]["status"] == RunStatus.CANCELLED:
            await model_provider.close()
            await memory_provider.close()
            return

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

    except asyncio.CancelledError:
        # Task was cancelled - update status if record still exists
        if run_id in _runs and _runs[run_id]["status"] != RunStatus.CANCELLED:
            _runs[run_id]["status"] = RunStatus.CANCELLED
            _runs[run_id]["error"] = "Cancelled by user"
            _runs[run_id]["current_phase"] = None
            _runs[run_id]["completed_at"] = datetime.utcnow()
            _runs[run_id]["duration_ms"] = (datetime.utcnow() - start_time).total_seconds() * 1000
        await _broadcast_event(run_id, "run_cancelled", {"reason": "User requested cancellation"})
        raise  # Re-raise to properly mark task as cancelled

    except Exception as e:
        # Only update if record still exists
        if run_id in _runs:
            _runs[run_id]["status"] = RunStatus.FAILED
            _runs[run_id]["error"] = str(e)
            _runs[run_id]["current_phase"] = None
            _runs[run_id]["completed_at"] = datetime.utcnow()
            _runs[run_id]["duration_ms"] = (datetime.utcnow() - start_time).total_seconds() * 1000
        await _broadcast_event(run_id, "run_error", {"error": str(e)})

    finally:
        # Cleanup task reference
        _tasks.pop(run_id, None)


async def _broadcast_event(run_id: str, event: str, data: dict[str, Any]) -> None:
    """Broadcast event to all connected WebSocket clients via queues.

    Uses per-connection queues to avoid concurrent sends on the same socket.
    """
    if run_id not in _ws_queues:
        return

    event_data = StreamEvent(
        event=event,
        run_id=run_id,
        data=data,
    ).model_dump(mode="json")

    # Put event in all connection queues for this run
    for queue in _ws_queues[run_id].values():
        try:
            queue.put_nowait(event_data)
        except asyncio.QueueFull:
            pass  # Drop event if queue is full (client too slow)


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


@router.get("/{run_id}/result", response_model=RunResultResponse)
async def get_run_result(run_id: str) -> RunResultResponse:
    """Get the final result of a completed run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    r = _runs[run_id]

    if not _is_terminal_status(r["status"]):
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} is still in progress (status: {r['status'].value})",
        )

    return RunResultResponse(
        run_id=r["run_id"],
        success=r["status"] == RunStatus.COMPLETED,
        status=r["status"],
        phases=r.get("phase_results", []),
        artifacts=r["artifacts"],
        total_duration_ms=r["duration_ms"] or 0,
        message=r["error"] if r["error"] else "Run completed successfully",
    )


@router.post("/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict[str, str]:
    """Cancel a running run.

    Stops execution gracefully. The run record is preserved for audit.
    Use DELETE /runs/{run_id} to purge the record entirely.
    """
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run = _runs[run_id]

    if _is_terminal_status(run["status"]):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel run in terminal status {run['status'].value}",
        )

    # Mark as cancelled first (so background task sees it)
    run["status"] = RunStatus.CANCELLED
    run["error"] = "Cancelled by user"
    run["updated_at"] = datetime.utcnow()

    # Cancel the async task if it exists
    if run_id in _tasks:
        task = _tasks[run_id]
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    await _broadcast_event(run_id, "run_cancelled", {"reason": "User requested cancellation"})

    return {"message": f"Run {run_id} cancelled", "status": "cancelled"}


@router.post("/{run_id}/resume", response_model=RunCreateResponse)
async def resume_run(
    run_id: str,
    request: RunResumeRequest,
) -> RunCreateResponse:
    """Resume a failed or cancelled run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run = _runs[run_id]

    if run["status"] not in [RunStatus.FAILED, RunStatus.CANCELLED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume run in status {run['status'].value}. Only failed or cancelled runs can be resumed.",
        )

    # Reset status and restart
    run["status"] = RunStatus.CREATED
    run["error"] = None
    run["updated_at"] = datetime.utcnow()
    run["completed_at"] = None

    # Start as tracked async task
    task = asyncio.create_task(
        _execute_run(
            run_id=run_id,
            vision=run["vision"],
            edition=run["edition"],
            provider=run["provider"],
            model=run["model"],
            workspace=Path(run["workspace"]),
            context=run["context"],
        )
    )
    _tasks[run_id] = task

    return RunCreateResponse(
        run_id=run_id,
        status=RunStatus.CREATED,
        message="Run resumed",
        workspace=run["workspace"],
    )


@router.delete("/{run_id}")
async def delete_run(run_id: str, force: bool = False) -> dict[str, str]:
    """Delete/purge a run record.

    Only allowed for terminal runs (completed/failed/cancelled) unless force=true.
    Cancels any running task before deletion if force=true.
    """
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run = _runs[run_id]

    # Check if running
    if _is_running_status(run["status"]):
        if not force:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete running run (status: {run['status'].value}). "
                       f"Use POST /runs/{run_id}/cancel first, or pass force=true.",
            )
        # Force cancel first
        run["status"] = RunStatus.CANCELLED
        run["error"] = "Force deleted by user"
        if run_id in _tasks:
            task = _tasks[run_id]
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

    # Cleanup task reference
    _tasks.pop(run_id, None)

    # Cleanup WebSocket queues
    _ws_queues.pop(run_id, None)

    # P1 Fix: Purge workspace directory on disk
    workspace = Path(run.get("workspace", ""))
    if workspace.exists():
        # Security: Verify workspace is under WORKSPACE_ROOT (prevent symlink attacks)
        try:
            resolved = workspace.resolve(strict=True)
            if resolved.is_relative_to(WORKSPACE_ROOT.resolve()):
                # Safe to delete - it's under our workspace root
                shutil.rmtree(workspace, ignore_errors=True)
        except (OSError, ValueError):
            # Path resolution failed or not relative to root - skip deletion
            pass

    # Remove from storage
    del _runs[run_id]

    return {"message": f"Run {run_id} deleted (workspace purged)"}


@router.websocket("/{run_id}/stream")
async def stream_run(websocket: WebSocket, run_id: str) -> None:
    """WebSocket endpoint for streaming run events.

    Uses per-connection message queue to avoid concurrent send issues.
    """
    global _ws_counter

    if run_id not in _runs:
        await websocket.close(code=4004, reason="Run not found")
        return

    await websocket.accept()

    # Create unique connection ID and message queue
    _ws_counter += 1
    ws_id = _ws_counter
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)  # Bounded to prevent memory issues

    # Register connection
    if run_id not in _ws_queues:
        _ws_queues[run_id] = {}
    _ws_queues[run_id][ws_id] = queue

    async def sender():
        """Send messages from queue to WebSocket."""
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"event": "keepalive"})
            except Exception:
                break

    async def receiver():
        """Receive messages from WebSocket."""
        while True:
            try:
                data = await websocket.receive_json()
                if data.get("command") == "ping":
                    await queue.put({"event": "pong"})
            except Exception:
                break

    try:
        # P1 Fix: ALL sends go through the queue (single-sender guarantee)
        # Run sender and receiver concurrently FIRST
        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())

        # Send current status through queue
        run = _runs.get(run_id)
        if run:
            await queue.put({
                "event": "connected",
                "run_id": run_id,
                "status": run["status"].value,
                "current_phase": run["current_phase"],
            })

        # Wait until run completes or connection closes
        while True:
            await asyncio.sleep(1.0)

            run = _runs.get(run_id)
            if not run:
                # Run was deleted - send through queue
                await queue.put({
                    "event": "run_deleted",
                    "run_id": run_id,
                })
                await asyncio.sleep(0.1)  # Give sender time to send
                break

            if _is_terminal_status(run["status"]):
                await queue.put({
                    "event": "run_finished",
                    "run_id": run_id,
                    "status": run["status"].value,
                })
                await asyncio.sleep(0.1)  # Give sender time to send
                break

        # Cancel tasks
        sender_task.cancel()
        receiver_task.cancel()

    except WebSocketDisconnect:
        pass
    finally:
        # Unregister connection
        if run_id in _ws_queues and ws_id in _ws_queues[run_id]:
            del _ws_queues[run_id][ws_id]
            if not _ws_queues[run_id]:
                del _ws_queues[run_id]
