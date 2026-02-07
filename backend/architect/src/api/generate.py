# backend/architect/src/api/generate.py
"""
AI Generation API route handlers.

Two-Phase Generation Flow:
1. POST /concept - Generate concept image (fast)
2. User reviews concept image
3. POST /concept/{job_id}/approve - Proceed to 3D generation
   OR POST /concept/{job_id}/regenerate - Try again with feedback

Stage Previews:
- During 3D generation, previews are sent via WebSocket after each stage
- Previews available at GET /api/generate/preview/{preview_id}

Schemas live in generate_schemas.py, background tasks in generate_tasks.py.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response

from .generate_schemas import (
    GenerateRequestAPI, GenerateJobResponse, GenerateStatusResponse,
    EstimateRequest, EstimateResponse, MaterialSuggestion,
    ConceptRequestAPI, ConceptJobResponse, ConceptStatusResponse,
    ConceptApproveRequest, ConceptApproveResponse,
    ConceptRegenerateRequest,
)
from .generate_tasks import (
    jobs, concept_jobs, stage_previews,
    run_concept_generation, run_generation, run_generation_with_concept,
)

router = APIRouter()


# =============================================================================
# Generation Endpoints
# =============================================================================

@router.post("/", response_model=GenerateJobResponse)
async def create_generation(
    request: GenerateRequestAPI,
    background_tasks: BackgroundTasks,
):
    """
    Start an asset generation job.
    
    Generation runs in background - poll /status/{job_id} for result.
    """
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "queued",
        "request": request.model_dump(),
        "asset_id": None,
        "result": None,
        "error": None,
    }
    
    background_tasks.add_task(run_generation, job_id, request)
    
    print(f"📥 [*] Queued generation job {job_id}: {request.prompt[:50]}...")
    
    return GenerateJobResponse(job_id=job_id, status="queued")


@router.get("/status/{job_id}", response_model=GenerateStatusResponse)
async def get_generation_status(job_id: str):
    """Get the status of a generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
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
        for job_id, job in jobs.items()
    }


@router.get("/debug/{job_id}")
async def debug_job(job_id: str):
    """Debug endpoint to see full job state."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id].copy()
    job.pop("_task", None)
    return job


@router.post("/estimate", response_model=EstimateResponse)
async def estimate_generation_cost(request: EstimateRequest):
    """
    Estimate cost and time for asset generation.
    
    Pricing logic based on category complexity and style count.
    """
    base_costs = {
        "prop": 0.02,
        "weapon": 0.04,
        "vehicle": 0.06,
        "character": 0.08,
        "environment": 0.10,
    }
    
    base_cost = base_costs.get(request.category.lower(), 0.02)
    style_cost = len(request.styles) * 0.01
    prompt_cost = 0.01 if request.prompt_length > 50 else 0.0
    total_cost = base_cost + style_cost + prompt_cost
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
    materials: list[str] = []
    
    keyword_map: dict[tuple[str, ...], list[str]] = {
        ("metal", "steel", "iron", "gun", "weapon"): ["steel", "aluminum"],
        ("wood", "wooden", "stock", "handle"): ["oak", "walnut"],
        ("rust", "worn", "old", "weathered"): ["rusted steel"],
        ("plastic", "polymer", "synthetic"): ["ABS plastic"],
        ("stone", "concrete", "rock"): ["concrete"],
        ("glass", "transparent", "clear"): ["glass"],
        ("fabric", "cloth", "textile"): ["cotton"],
        ("leather", "hide"): ["leather"],
    }
    
    for keywords, mats in keyword_map.items():
        if any(word in prompt_lower for word in keywords):
            materials.extend(mats)
    
    # Deduplicate preserving order
    seen: set[str] = set()
    unique = [m for m in materials if not (m in seen or seen.add(m))]  # type: ignore[func-returns-value]
    
    return MaterialSuggestion(materials=unique)


# =============================================================================
# Stage Preview Endpoints
# =============================================================================

@router.get("/preview/{preview_id}")
async def get_stage_preview(preview_id: str):
    """
    Serve intermediate stage preview binary (.gve_bin format).
    
    Preview IDs are in format: {job_id}_stage_{stage_name}
    Previews are cached for 5 minutes after generation.
    """
    binary_data = stage_previews.get(preview_id)
    
    if not binary_data:
        raise HTTPException(
            status_code=404,
            detail=f"Preview not found or expired: {preview_id}"
        )
    
    return Response(
        content=binary_data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"inline; filename={preview_id}.gve_bin",
            "Cache-Control": "no-cache",
        }
    )


@router.get("/preview/{preview_id}/info")
async def get_stage_preview_info(preview_id: str):
    """Get metadata about a stage preview without downloading the binary."""
    binary_data = stage_previews.get(preview_id)
    
    if not binary_data:
        raise HTTPException(
            status_code=404,
            detail=f"Preview not found or expired: {preview_id}"
        )
    
    parts = preview_id.rsplit("_stage_", 1)
    job_id = parts[0] if len(parts) > 0 else None
    stage = parts[1] if len(parts) > 1 else None
    
    return {
        "preview_id": preview_id,
        "job_id": job_id,
        "stage": stage,
        "size_bytes": len(binary_data),
    }


# =============================================================================
# Concept Image Endpoints (Two-Phase Workflow)
# =============================================================================

@router.post("/concept", response_model=ConceptJobResponse)
async def create_concept(
    request: ConceptRequestAPI,
    background_tasks: BackgroundTasks,
):
    """
    Start a concept image generation job.
    
    This is Phase 1 of the two-phase generation workflow.
    After concept is ready, user can approve or regenerate.
    """
    job_id = f"concept-{str(uuid.uuid4())[:8]}"
    concept_jobs[job_id] = {
        "status": "queued",
        "request": request.model_dump(),
        "prompt": request.prompt,
        "concept_image": None,
        "prompt_used": None,
        "error": None,
    }
    
    background_tasks.add_task(run_concept_generation, job_id, request)
    
    print(f"🎨 [*] Queued concept job {job_id}: {request.prompt[:50]}...")
    
    return ConceptJobResponse(job_id=job_id, status="queued")


@router.get("/concept/{job_id}", response_model=ConceptStatusResponse)
async def get_concept_status(job_id: str):
    """
    Get the status of a concept generation job.
    
    When status is "ready", concept_image contains the base64 image data.
    """
    if job_id not in concept_jobs:
        raise HTTPException(status_code=404, detail="Concept job not found")
    
    job = concept_jobs[job_id]
    return ConceptStatusResponse(
        job_id=job_id,
        status=job["status"],
        concept_image=job.get("concept_image"),
        prompt=job.get("prompt"),
        prompt_used=job.get("prompt_used"),
        error=job.get("error"),
    )


@router.post("/concept/{job_id}/approve", response_model=ConceptApproveResponse)
async def approve_concept(
    job_id: str,
    request: ConceptApproveRequest,
    background_tasks: BackgroundTasks,
):
    """
    Approve a concept image and proceed to 3D generation.
    
    This is Phase 2 of the two-phase workflow.
    The concept image becomes the visual reference for all 3D generation stages.
    """
    if job_id not in concept_jobs:
        raise HTTPException(status_code=404, detail="Concept job not found")
    
    concept_job = concept_jobs[job_id]
    
    if concept_job["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Concept not ready for approval (status: {concept_job['status']})"
        )
    
    if not concept_job.get("concept_image"):
        raise HTTPException(status_code=400, detail="No concept image available")
    
    concept_job["status"] = "approved"
    
    gen_job_id = str(uuid.uuid4())[:8]
    original_request = concept_job["request"]
    
    gen_request = GenerateRequestAPI(
        prompt=original_request["prompt"],
        category=original_request.get("category"),
        style_reference=original_request.get("style"),
        track_override=request.track_override,
    )
    
    jobs[gen_job_id] = {
        "status": "queued",
        "request": gen_request.model_dump(),
        "concept_job_id": job_id,
        "asset_id": None,
        "result": None,
        "error": None,
    }
    
    background_tasks.add_task(
        run_generation_with_concept,
        gen_job_id,
        gen_request,
        concept_job["concept_image"],
    )
    
    print(f"✅ [*] Concept {job_id} approved, starting 3D generation {gen_job_id}")
    
    return ConceptApproveResponse(
        concept_job_id=job_id,
        generation_job_id=gen_job_id,
        status="approved",
    )


@router.post("/concept/{job_id}/regenerate", response_model=ConceptJobResponse)
async def regenerate_concept(
    job_id: str,
    request: ConceptRegenerateRequest,
    background_tasks: BackgroundTasks,
):
    """
    Regenerate concept image with user feedback.
    
    Can optionally use the previous image as reference for iterative refinement.
    """
    if job_id not in concept_jobs:
        raise HTTPException(status_code=404, detail="Concept job not found")
    
    concept_job = concept_jobs[job_id]
    
    if concept_job["status"] not in ("ready", "failed"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot regenerate (status: {concept_job['status']})"
        )
    
    new_job_id = f"concept-{str(uuid.uuid4())[:8]}"
    original_request = concept_job["request"]
    
    concept_jobs[new_job_id] = {
        "status": "queued",
        "request": original_request,
        "prompt": original_request["prompt"],
        "concept_image": None,
        "prompt_used": None,
        "error": None,
        "feedback": request.feedback,
        "previous_job_id": job_id,
    }
    
    async def _run_regeneration() -> None:
        import traceback
        from ..ai_pipeline.concept_artist import regenerate_concept_with_feedback
        
        print(f"🎨 [*] Regenerating concept {new_job_id} with feedback...")
        concept_jobs[new_job_id]["status"] = "generating"
        
        try:
            previous_image = None
            if request.use_previous_as_reference:
                previous_image = concept_job.get("concept_image")
            
            result = await regenerate_concept_with_feedback(
                original_prompt=original_request["prompt"],
                feedback=request.feedback,
                previous_image_base64=previous_image,
                style=original_request.get("style"),
                aspect_ratio=original_request.get("aspect_ratio", "1:1"),
            )
            
            concept_jobs[new_job_id]["status"] = "ready"
            concept_jobs[new_job_id]["concept_image"] = result.image_base64
            concept_jobs[new_job_id]["prompt_used"] = result.prompt_used
            
            print(f"✅ [+] Regenerated concept {new_job_id} ready")
            
        except Exception as e:
            concept_jobs[new_job_id]["status"] = "failed"
            concept_jobs[new_job_id]["error"] = str(e)
            print(f"❌ [-] Regeneration {new_job_id} failed: {e}")
    
    background_tasks.add_task(_run_regeneration)
    
    print(f"🔄 [*] Queued regeneration {new_job_id} with feedback: {request.feedback[:50]}...")
    
    return ConceptJobResponse(job_id=new_job_id, status="queued")


@router.delete("/concept/{job_id}")
async def cancel_concept(job_id: str):
    """
    Cancel/reject a concept job.
    
    Removes the job from tracking. User can start fresh with new prompt.
    """
    if job_id not in concept_jobs:
        raise HTTPException(status_code=404, detail="Concept job not found")
    
    concept_jobs[job_id]["status"] = "cancelled"
    
    print(f"🗑️ [*] Concept job {job_id} cancelled")
    
    return {"status": "cancelled", "job_id": job_id}


@router.get("/concept/{job_id}/styles")
async def get_available_styles():
    """Get available style presets for concept generation."""
    from ..ai_pipeline.concept_artist import get_available_styles
    
    return get_available_styles()
