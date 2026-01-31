from enum import Enum
from typing import Optional
from bson import ObjectId
import uuid
import asyncio

class CompilePriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    IMMEDIATE = 3

# Stubbed in-memory job store
_jobs = {}

async def _process_job(job_id: str) -> None:
    """Actually run the compilation."""
    import sys
    
    job = _jobs.get(job_id)
    if not job:
        print(f"[!!] Job {job_id} not found in queue", flush=True)
        return
    
    job["status"] = "running"
    print(f"[*] Starting compile job {job_id} for asset {job['asset_id']}", flush=True)
    
    try:
        from .pipeline import compile_asset, CompileRequest
        
        request = CompileRequest(
            asset_id=job["asset_id"],
            priority=job.get("priority", CompilePriority.NORMAL),
            force_recompile=job.get("force_recompile", False),
        )
        
        print(f"[*] Running compile_asset for {job['asset_id']}...", flush=True)
        result = await compile_asset(request)
        
        if result.success:
            job["status"] = "completed"
            job["binary_path"] = str(result.binary_path) if result.binary_path else None
            job["compile_time"] = result.compile_time_sec
            print(f"[OK] Compiled {job['asset_id']} -> {result.binary_path}", flush=True)
        else:
            job["status"] = "failed"
            job["error"] = result.error
            print(f"[!!] Compile failed for {job['asset_id']}: {result.error}", flush=True)
            
    except Exception as e:
        import traceback
        job["status"] = "failed"
        job["error"] = str(e)
        print(f"[!!] Compile exception for {job['asset_id']}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()

async def enqueue_compile(
    asset_id: str | ObjectId, 
    priority: CompilePriority = CompilePriority.NORMAL,
    force_recompile: bool = False
) -> str:
    """
    Queue an asset for compilation.
    Returns a job ID.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "asset_id": str(asset_id),
        "status": "queued",
        "priority": priority,
        "force_recompile": force_recompile,
    }
    
    # Fire and forget - run compilation in background
    asyncio.create_task(_process_job(job_id))
    
    return job_id

async def get_compile_status(job_id: str) -> dict:
    """Get status of a compile job."""
    return _jobs.get(job_id, {"status": "unknown"})
