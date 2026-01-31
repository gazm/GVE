from fastapi import APIRouter, HTTPException
from src.compiler.queue import enqueue_compile, get_compile_status, CompilePriority
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class CompileTriggerRequest(BaseModel):
    priority: Optional[CompilePriority] = CompilePriority.NORMAL
    force_recompile: bool = False

@router.post("/{asset_id}")
async def trigger_compile(asset_id: str, request: CompileTriggerRequest):
    job_id = await enqueue_compile(
        asset_id, 
        priority=request.priority, 
        force_recompile=request.force_recompile
    )
    return {"job_id": job_id, "status": "queued"}

@router.get("/status/{job_id}")
async def check_status(job_id: str):
    status = await get_compile_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status
