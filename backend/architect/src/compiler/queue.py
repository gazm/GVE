from enum import Enum
from typing import Optional
from bson import ObjectId
import uuid
import asyncio
from datetime import datetime

class CompilePriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    IMMEDIATE = 3

# In-memory job store
_jobs = {}

# Sequential execution queue
_job_queue: asyncio.Queue = None  # type: ignore
_worker_task: Optional[asyncio.Task] = None

async def _queue_worker() -> None:
    """Process compile jobs sequentially from the queue."""
    global _job_queue
    
    print("[Queue Worker] Started", flush=True)
    
    while True:
        try:
            job_id = await _job_queue.get()
            print(f"[Queue Worker] Processing job {job_id}", flush=True)
            await _process_job(job_id)
            _job_queue.task_done()
        except Exception as e:
            print(f"[Queue Worker] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()

async def _process_job(job_id: str) -> None:
    """Actually run the compilation."""
    import sys
    
    job = _jobs.get(job_id)
    if not job:
        print(f"[!!] Job {job_id} not found in queue", flush=True)
        return
    
    # Check for cancellation
    if job.get("status") == "cancelled":
        print(f"[*] Job {job_id} was cancelled, skipping", flush=True)
        return
    
    job["status"] = "running"
    job["started_at"] = datetime.utcnow().isoformat()
    print(f"[*] Starting compile job {job_id} for asset {job['asset_id']}", flush=True)
    
    # Broadcast job started
    try:
        from ..api.websocket import broadcast_event
        await broadcast_event("queue:job_started", {
            "job_id": job_id,
            "asset_id": job["asset_id"],
            "status": "running"
        })
    except Exception as e:
        print(f"[!] WebSocket broadcast failed: {e}", flush=True)
    
    try:
        from .pipeline import compile_asset, CompileRequest
        
        request = CompileRequest(
            asset_id=job["asset_id"],
            priority=job.get("priority", CompilePriority.NORMAL),
            force_recompile=job.get("force_recompile", False),
            options=job.get("options"),
        )
        
        print(f"[*] Running compile_asset for {job['asset_id']}...", flush=True)
        result = await compile_asset(request)
        
        if result.success:
            job["status"] = "completed"
            job["binary_path"] = str(result.binary_path) if result.binary_path else None
            job["compile_time"] = result.compile_time_sec
            job["completed_at"] = datetime.utcnow().isoformat()
            job["compiler_output"] = f"✓ Compiled successfully to {result.binary_path}"
            print(f"[OK] Compiled {job['asset_id']} -> {result.binary_path}", flush=True)
            
            # Broadcast completion
            try:
                await broadcast_event("queue:job_completed", {
                    "job_id": job_id,
                    "asset_id": job["asset_id"],
                    "status": "completed",
                    "compile_time": result.compile_time_sec
                })
            except Exception:
                pass
        else:
            job["status"] = "failed"
            job["error"] = result.error
            job["compiler_output"] = f"✗ Compilation failed: {result.error}"
            job["completed_at"] = datetime.utcnow().isoformat()
            print(f"[!!] Compile failed for {job['asset_id']}: {result.error}", flush=True)
            
            # Broadcast failure
            try:
                await broadcast_event("queue:job_failed", {
                    "job_id": job_id,
                    "asset_id": job["asset_id"],
                    "status": "failed",
                    "error": result.error
                })
            except Exception:
                pass
            
    except Exception as e:
        import traceback
        job["status"] = "failed"
        job["error"] = str(e)
        job["compiler_output"] = f"✗ Exception: {str(e)[:150]}"
        job["completed_at"] = datetime.utcnow().isoformat()
        print(f"[!!] Compile exception for {job['asset_id']}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        
        # Broadcast failure
        try:
            from ..api.websocket import broadcast_event
            await broadcast_event("queue:job_failed", {
                "job_id": job_id,
                "asset_id": job["asset_id"],
                "status": "failed",
                "error": str(e)
            })
        except Exception:
            pass

async def enqueue_compile(
    asset_id: str | ObjectId, 
    priority: CompilePriority = CompilePriority.NORMAL,
    force_recompile: bool = False,
    compile_options: Optional[dict] = None
) -> str:
    """
    Queue an asset for compilation.
    Returns a job ID. If this asset already has a queued or running job, returns that job_id
    instead of starting a second one (avoids duplicate compiles from e.g. property save + recompile).
    """
    global _job_queue, _worker_task
    
    # Initialize queue if needed
    if _job_queue is None:
        _job_queue = asyncio.Queue()
    
    aid = str(asset_id)
    
    # Check for existing job
    for jid, job in list(_jobs.items()):
        if job.get("status") in ("queued", "running") and job.get("asset_id") == aid:
            return jid
    
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "job_type": "compile",
        "asset_id": aid,
        "status": "queued",
        "priority": priority,
        "force_recompile": force_recompile,
        "options": compile_options,
        "created_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "compile_time": None,
    }
    
    # Add to queue
    await _job_queue.put(job_id)
    
    # Start worker if not running
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_queue_worker())
        print("[*] Queue worker started", flush=True)
    
    # Broadcast job added
    try:
        from ..api.websocket import broadcast_event
        await broadcast_event("queue:job_added", {
            "job_id": job_id,
            "asset_id": aid,
            "status": "queued"
        })
    except Exception as e:
        print(f"[!] WebSocket broadcast failed: {e}", flush=True)
    
    return job_id

async def get_compile_status(job_id: str) -> dict:
    """Get status of a compile job."""
    return _jobs.get(job_id, {"status": "unknown"})

async def get_all_jobs() -> list[dict]:
    """Get all compile jobs."""
    return list(_jobs.values())

async def cancel_job(job_id: str) -> bool:
    """Cancel a compile job. Returns True if cancelled, False if not found or already completed."""
    job = _jobs.get(job_id)
    if not job:
        return False
    
    if job["status"] in ("completed", "failed", "cancelled"):
        return False
    
    job["status"] = "cancelled"
    job["completed_at"] = datetime.utcnow().isoformat()
    
    # Broadcast cancellation
    try:
        from ..api.websocket import broadcast_event
        await broadcast_event("queue:job_cancelled", {
            "job_id": job_id,
            "asset_id": job["asset_id"],
            "status": "cancelled"
        })
    except Exception:
        pass
    
    return True

async def clear_completed_jobs(max_age_seconds: int = 3600) -> int:
    """Remove completed/failed/cancelled jobs older than max_age_seconds. Returns count removed."""
    from datetime import datetime, timedelta
    
    now = datetime.utcnow()
    to_remove = []
    
    for job_id, job in _jobs.items():
        if job["status"] in ("completed", "failed", "cancelled"):
            completed_at = job.get("completed_at")
            if completed_at:
                completed_time = datetime.fromisoformat(completed_at)
                if (now - completed_time).total_seconds() > max_age_seconds:
                    to_remove.append(job_id)
    
    for job_id in to_remove:
        del _jobs[job_id]
    
    return len(to_remove)

