# backend/architect/src/api/generate.py
"""
AI Generation API endpoints.

Handles asset generation requests via the AI pipeline.
All LLM calls run as background tasks (per module anti-patterns).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter()

# In-memory job tracking (use Redis in production)
_jobs: dict[str, dict[str, Any]] = {}


class GenerateRequestAPI(BaseModel):
    """API request to generate an asset."""
    
    prompt: str
    category: str | None = None
    style_reference: str | None = None
    track_override: str | None = None  # "matter", "landscape", "audio"

class GenerateJobResponse(BaseModel):
    """Response with job ID for tracking."""
    
    job_id: str
    status: str


class GenerateStatusResponse(BaseModel):
    """Status of a generation job."""
    
    job_id: str
    status: str  # "queued", "running", "completed", "failed"
    asset_id: str | None = None
    result: dict | None = None
    error: str | None = None


class EstimateRequest(BaseModel):
    """Request for cost/time estimate."""
    
    category: str
    styles: list[str] = []
    prompt_length: int = 0


class EstimateResponse(BaseModel):
    """Cost and time estimate for generation."""
    
    cost_usd: float
    estimated_time_sec: int
    category: str


class MaterialSuggestion(BaseModel):
    """Material suggestion response."""
    
    materials: list[str]


async def _run_generation(job_id: str, request: GenerateRequestAPI) -> None:
    """
    Background task to run AI generation.
    
    Updates job status in _jobs dict.
    """
    import traceback
    
    from ..ai_pipeline import generate_asset, GenerateRequest
    from ..ai_pipeline.orchestrator import GenerationTrack
    
    print(f"[*] Starting generation job {job_id}...")
    _jobs[job_id]["status"] = "running"
    
    try:
        # Convert API request to internal request
        track = None
        if request.track_override:
            track = GenerationTrack(request.track_override)
        
        internal_request = GenerateRequest(
            prompt=request.prompt,
            category=request.category,
            style_reference=request.style_reference,
            track_override=track,
        )
        
        print(f"[*] Job {job_id}: Calling generate_asset...")
        # Run generation
        result = await generate_asset(internal_request)
        
        # Store result
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["asset_id"] = result.asset_id
        _jobs[job_id]["result"] = {
            "asset_id": result.asset_id,
            "confidence": result.confidence,
            "generation_time_sec": result.generation_time_sec,
            "track_used": result.track_used.value,
        }
        
        print(f"[+] Generation job {job_id} completed: {result.asset_id}")
        
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
        error_trace = traceback.format_exc()
        print(f"[-] Generation job {job_id} failed: {e}")
        print(f"[-] Traceback:\n{error_trace}")


@router.post("/", response_model=GenerateJobResponse)
async def create_generation(
    request: GenerateRequestAPI,
    background_tasks: BackgroundTasks,
):
    """
    Start an asset generation job.
    
    Generation runs in background - poll /status/{job_id} for result.
    """
    # Create job
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "queued",
        "request": request.model_dump(),
        "asset_id": None,
        "result": None,
        "error": None,
    }
    
    # FastAPI BackgroundTasks handles async functions directly
    # It will run them in the background after the response is sent
    background_tasks.add_task(_run_generation, job_id, request)
    
    print(f"[*] Queued generation job {job_id}: {request.prompt[:50]}...")
    
    return GenerateJobResponse(job_id=job_id, status="queued")


@router.get("/status/{job_id}", response_model=GenerateStatusResponse)
async def get_generation_status(job_id: str):
    """Get the status of a generation job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _jobs[job_id]
    return GenerateStatusResponse(
        job_id=job_id,
        status=job["status"],
        asset_id=job.get("asset_id"),
        result=job.get("result"),
        error=job.get("error"),
    )


@router.get("/jobs")
async def list_jobs():
    """List all generation jobs (for debugging)."""
    return {
        job_id: {
            "status": job["status"],
            "prompt": job["request"]["prompt"][:50] + "...",
            "error": job.get("error"),
        }
        for job_id, job in _jobs.items()
    }


@router.get("/debug/{job_id}")
async def debug_job(job_id: str):
    """Debug endpoint to see full job state."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _jobs[job_id].copy()
    # Remove non-serializable task object
    job.pop("_task", None)
    return job


@router.post("/estimate", response_model=EstimateResponse)
async def estimate_generation_cost(request: EstimateRequest):
    """
    Estimate cost and time for asset generation.
    
    Pricing logic based on category complexity and style count.
    """
    # Base cost by category
    base_costs = {
        "prop": 0.02,
        "weapon": 0.04,
        "vehicle": 0.06,
        "character": 0.08,
        "environment": 0.10,
    }
    
    base_cost = base_costs.get(request.category.lower(), 0.02)
    
    # Add cost for styles (each style adds complexity)
    style_cost = len(request.styles) * 0.01
    
    # Add cost for long prompts (more detail = more processing)
    prompt_cost = 0.01 if request.prompt_length > 50 else 0.0
    
    total_cost = base_cost + style_cost + prompt_cost
    
    # Estimate time (roughly 50 seconds per $1 of cost, minimum 2 seconds)
    estimated_time = max(2, int(2 + (total_cost * 50)))
    
    return EstimateResponse(
        cost_usd=round(total_cost, 2),
        estimated_time_sec=estimated_time,
        category=request.category,
    )


@router.get("/suggest/materials", response_model=MaterialSuggestion)
async def suggest_materials(prompt: str):
    """
    Suggest materials based on prompt keywords.
    
    Uses keyword matching to suggest appropriate materials.
    In production, this could use an LLM or material database.
    """
    prompt_lower = prompt.lower()
    materials = []
    
    # Keyword-based material suggestions
    if any(word in prompt_lower for word in ['metal', 'steel', 'iron', 'gun', 'weapon']):
        materials.extend(['steel', 'aluminum'])
    
    if any(word in prompt_lower for word in ['wood', 'wooden', 'stock', 'handle']):
        materials.extend(['oak', 'walnut'])
    
    if any(word in prompt_lower for word in ['rust', 'worn', 'old', 'weathered']):
        materials.append('rusted steel')
    
    if any(word in prompt_lower for word in ['plastic', 'polymer', 'synthetic']):
        materials.append('ABS plastic')
    
    if any(word in prompt_lower for word in ['stone', 'concrete', 'rock']):
        materials.append('concrete')
    
    if any(word in prompt_lower for word in ['glass', 'transparent', 'clear']):
        materials.append('glass')
    
    if any(word in prompt_lower for word in ['fabric', 'cloth', 'textile']):
        materials.append('cotton')
    
    if any(word in prompt_lower for word in ['leather', 'hide']):
        materials.append('leather')
    
    # Remove duplicates while preserving order
    seen = set()
    unique_materials = []
    for m in materials:
        if m not in seen:
            seen.add(m)
            unique_materials.append(m)
    
    return MaterialSuggestion(materials=unique_materials)
