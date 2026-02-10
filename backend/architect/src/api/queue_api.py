# backend/architect/src/api/queue_api.py
"""
Unified Job Queue API - aggregates compile and AI generation jobs.

Provides a single interface to view and manage all background jobs:
- Compile jobs (from src.compiler.queue)
- AI generation jobs (from generate_tasks)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.templating import Jinja2Templates

from ..paths import get_template_dir
from ..compiler.queue import get_all_jobs as get_compile_jobs, cancel_job as cancel_compile_job
from .generate_tasks import jobs as generate_jobs, concept_jobs
from .websocket import broadcast_event

router = APIRouter()

# Setup templates using centralized path configuration
templates = Jinja2Templates(directory=get_template_dir())


async def get_unified_queue() -> dict:
    """
    Get all jobs from both compile and generate queues.
    
    Returns:
        {
            "jobs": [
                {
                    "job_id": str,
                    "job_type": "compile" | "generate",
                    "asset_id": str | None,
                    "prompt": str | None,
                    "status": str,
                    "priority": str | None,
                    "created_at": str,
                    "started_at": str | None,
                    "completed_at": str | None,
                    "error": str | None,
                    "duration": float | None,
                    "current_stage": str | None,  # For AI jobs
                    "streaming_text": str | None,  # For AI jobs (if available)
                }
            ]
        }
    """
    # Get compile jobs
    compile_jobs_list = await get_compile_jobs()
    
    # Convert generate jobs to unified format
    gen_jobs = []
    for jid, job in generate_jobs.items():
        gen_jobs.append({
            "job_id": jid,
            "job_type": "generate",
            "asset_id": job.get("asset_id"),
            "prompt": job.get("request", {}).get("prompt", "")[:50] if job.get("request") else "",
            "status": job.get("status"),
            "priority": None,  # AI jobs don't have priority
            "created_at": None,  # Not tracked yet
            "started_at": None,
            "completed_at": None,
            "error": job.get("error"),
            "duration": job.get("result", {}).get("generation_time_sec") if job.get("result") else None,
            "current_stage": job.get("current_stage"),
            "streaming_text": job.get("streaming_text"),
        })
    
    # Convert concept jobs to unified format
    for jid, job in concept_jobs.items():
        gen_jobs.append({
            "job_id": jid,
            "job_type": "generate",
            "asset_id": None,  # Concept jobs don't have asset_id yet
            "prompt": job.get("prompt", "")[:50],
            "status": job.get("status"),
            "priority": None,
            "created_at": None,
            "started_at": None,
            "completed_at": None,
            "error": job.get("error"),
            "duration": None,
            "current_stage": "Concept",
            "streaming_text": None,
        })
    
    # Combine all jobs
    all_jobs = compile_jobs_list + gen_jobs
    
    # Sort by created_at descending (newest first), with None values last
    all_jobs.sort(
        key=lambda j: j.get("created_at") or "0000-00-00T00:00:00",
        reverse=True
    )
    
    return {"jobs": all_jobs}


@router.get("/queue")
async def list_queue():
    """Get unified list of all jobs (compile + generate)."""
    return await get_unified_queue()


@router.get("/queue/summary")
async def get_queue_summary():
    """
    Get queue summary with counts by status and type.
    
    Returns:
        {
            "queued": int,
            "running": int,
            "completed": int,
            "failed": int,
            "cancelled": int,
            "total": int,
            "by_type": {
                "compile": {"queued": int, "running": int, "completed": int},
                "generate": {"queued": int, "running": int, "completed": int}
            }
        }
    """
    queue_data = await get_unified_queue()
    jobs = queue_data["jobs"]
    
    summary = {
        "queued": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
        "total": len(jobs),
        "by_type": {
            "compile": {"queued": 0, "running": 0, "completed": 0, "failed": 0},
            "generate": {"queued": 0, "running": 0, "completed": 0, "failed": 0},
        }
    }
    
    for job in jobs:
        status = job.get("status", "unknown")
        job_type = job.get("job_type", "unknown")
        
        # Overall counts
        if status in summary:
            summary[status] += 1
        
        # Per-type counts
        if job_type in summary["by_type"] and status in summary["by_type"][job_type]:
            summary["by_type"][job_type][status] += 1
    
    return summary


@router.delete("/queue/{job_id}")
async def cancel_queue_job(job_id: str):
    """
    Cancel a job (compile or generate).
    
    Returns 200 if cancelled, 404 if not found, 400 if already completed.
    """
    # Try compile jobs first
    cancelled = await cancel_compile_job(job_id)
    if cancelled:
        return {"status": "cancelled", "job_id": job_id, "job_type": "compile"}
    
    # Try generate jobs
    if job_id in generate_jobs:
        job = generate_jobs[job_id]
        if job["status"] in ("completed", "failed", "cancelled"):
            raise HTTPException(status_code=400, detail="Job already completed")
        
        job["status"] = "cancelled"
        
        # Broadcast cancellation
        try:
            await broadcast_event("queue:job_cancelled", {
                "job_id": job_id,
                "status": "cancelled",
                "job_type": "generate"
            })
        except Exception:
            pass
        
        return {"status": "cancelled", "job_id": job_id, "job_type": "generate"}
    
    # Try concept jobs
    if job_id in concept_jobs:
        job = concept_jobs[job_id]
        if job["status"] in ("ready", "failed", "cancelled", "approved"):
            raise HTTPException(status_code=400, detail="Job already completed")
        
        job["status"] = "cancelled"
        
        # Broadcast cancellation
        try:
            await broadcast_event("queue:job_cancelled", {
                "job_id": job_id,
                "status": "cancelled",
                "job_type": "generate"
            })
        except Exception:
            pass
        
        return {"status": "cancelled", "job_id": job_id, "job_type": "generate"}
    
    raise HTTPException(status_code=404, detail="Job not found")


@router.get("/queue/partial")
async def get_queue_partial(request: Request):
    """
    Render queue table partial for HTMX.
    
    This endpoint returns just the table body HTML for dynamic updates.
    """
    queue_data = await get_unified_queue()
    jobs = queue_data["jobs"]
    
    # Add queue position for queued compile jobs
    compile_queue_position = 0
    for job in jobs:
        if job["job_type"] == "compile" and job["status"] == "queued":
            compile_queue_position += 1
            job["queue_position"] = compile_queue_position
    
    return templates.TemplateResponse("queue_table.html", {
        "request": request,
        "jobs": jobs
    })
