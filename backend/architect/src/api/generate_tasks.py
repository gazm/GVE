# backend/architect/src/api/generate_tasks.py
"""
Background task runners for AI generation jobs.

These are async functions executed by FastAPI BackgroundTasks.
Each updates its job dict in the shared TTLCache stores.
Separated from generate.py route handlers for file-size discipline.
"""

from __future__ import annotations

from typing import Any

from cachetools import TTLCache

from .generate_schemas import ConceptRequestAPI, GenerateRequestAPI


# TTL-evicted job tracking — keeps at most 256 entries for 1 hour each.
# Prevents unbounded memory growth from accumulated jobs.
jobs: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=256, ttl=3600)

# Concept jobs — 256 entries, 2 hour TTL (users may take time to approve)
concept_jobs: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=256, ttl=7200)

# Stage preview binaries — 64 entries, 5 min TTL (auto-expires, no manual cleanup needed)
stage_previews: TTLCache[str, bytes] = TTLCache(maxsize=64, ttl=300)


async def run_concept_generation(job_id: str, request: ConceptRequestAPI) -> None:
    """Background task to generate concept image."""
    import traceback
    from ..ai_pipeline.concept_artist import generate_concept_image
    
    print(f"🎨 [*] Starting concept generation job {job_id}...")
    concept_jobs[job_id]["status"] = "generating"
    
    try:
        result = await generate_concept_image(
            prompt=request.prompt,
            style=request.style,
            category=request.category,
            aspect_ratio=request.aspect_ratio,
        )
        
        concept_jobs[job_id]["status"] = "ready"
        concept_jobs[job_id]["concept_image"] = result.image_base64
        concept_jobs[job_id]["prompt_used"] = result.prompt_used
        
        print(f"✅ [+] Concept job {job_id} ready for review")
        
    except Exception as e:
        concept_jobs[job_id]["status"] = "failed"
        concept_jobs[job_id]["error"] = str(e)
        error_trace = traceback.format_exc()
        print(f"❌ [-] Concept job {job_id} failed: {e}")
        print(f"[-] Traceback:\n{error_trace}")


async def run_generation_with_concept(
    job_id: str,
    request: GenerateRequestAPI,
    concept_image_base64: str,
) -> None:
    """
    Background task to run AI generation with concept image reference.
    
    Sends stage previews via WebSocket after each pipeline stage completes.
    After successful generation, indexes the concept image in RAG
    for the learning loop (future similar prompts can find this concept).
    """
    import traceback
    
    from ..ai_pipeline import generate_asset_with_concept, GenerateRequest, index_concept_image
    from ..ai_pipeline.orchestrator import GenerationTrack
    from .websocket import broadcast_event
    
    print(f"🚀 [*] Starting generation job {job_id} with concept reference...")
    jobs[job_id]["status"] = "running"
    jobs[job_id]["current_stage"] = None
    
    # Stage preview callback - stores preview and broadcasts WebSocket event
    async def on_stage_complete(stage: str, preview_binary: bytes) -> None:
        """Callback for pipeline stage completion - broadcasts preview to UI."""
        preview_id = f"{job_id}_stage_{stage}"
        
        stage_previews[preview_id] = preview_binary
        jobs[job_id]["current_stage"] = stage
        
        print(f"📺 [*] Stage {stage} preview ready ({len(preview_binary)} bytes)")
        
        try:
            await broadcast_event("generate:stage_complete", {
                "job_id": job_id,
                "stage": stage,
                "preview_url": f"/api/generate/preview/{preview_id}",
                "preview_bytes": len(preview_binary),
            })
        except Exception as ws_error:
            print(f"⚠️ WebSocket broadcast failed (non-fatal): {ws_error}")
    
    try:
        track = None
        if request.track_override:
            track = GenerationTrack(request.track_override)
        
        internal_request = GenerateRequest(
            prompt=request.prompt,
            category=request.category,
            style_reference=request.style_reference,
            track_override=track,
        )
        
        print(f"🔧 [*] Job {job_id}: Calling generate_asset_with_concept with stage previews...")
        
        result = await generate_asset_with_concept(
            internal_request,
            concept_image_base64=concept_image_base64,
            on_stage_complete=on_stage_complete,
        )
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["asset_id"] = result.asset_id
        jobs[job_id]["result"] = {
            "asset_id": result.asset_id,
            "confidence": result.confidence,
            "generation_time_sec": result.generation_time_sec,
            "track_used": result.track_used.value,
        }
        
        print(f"✅ [+] Generation job {job_id} completed: {result.asset_id}")
        
        try:
            await broadcast_event("generate:complete", {
                "job_id": job_id,
                "asset_id": result.asset_id,
                "result": jobs[job_id]["result"],
            })
        except Exception as ws_error:
            print(f"⚠️ WebSocket broadcast failed (non-fatal): {ws_error}")
        
        # Index concept image in RAG for learning loop
        try:
            await index_concept_image(
                asset_id=result.asset_id,
                prompt=request.prompt,
                concept_image_base64=concept_image_base64,
                dna=result.dna,
            )
            print(f"📚 [+] Concept indexed for RAG learning loop")
        except Exception as rag_error:
            print(f"⚠️ RAG indexing failed (non-fatal): {rag_error}")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        error_trace = traceback.format_exc()
        print(f"❌ [-] Generation job {job_id} failed: {e}")
        print(f"[-] Traceback:\n{error_trace}")
        
        try:
            await broadcast_event("generate:failed", {
                "job_id": job_id,
                "error": str(e),
            })
        except Exception:
            pass


async def run_generation(job_id: str, request: GenerateRequestAPI) -> None:
    """
    Background task to run AI generation (no concept reference).
    
    Updates job status in jobs dict.
    """
    import traceback
    
    from ..ai_pipeline import generate_asset, GenerateRequest
    from ..ai_pipeline.orchestrator import GenerationTrack
    
    print(f"🚀 [*] Starting generation job {job_id}...")
    jobs[job_id]["status"] = "running"
    
    try:
        track = None
        if request.track_override:
            track = GenerationTrack(request.track_override)
        
        internal_request = GenerateRequest(
            prompt=request.prompt,
            category=request.category,
            style_reference=request.style_reference,
            track_override=track,
        )
        
        print(f"🔧 [*] Job {job_id}: Calling generate_asset...")
        result = await generate_asset(internal_request)
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["asset_id"] = result.asset_id
        jobs[job_id]["result"] = {
            "asset_id": result.asset_id,
            "confidence": result.confidence,
            "generation_time_sec": result.generation_time_sec,
            "track_used": result.track_used.value,
        }
        
        print(f"✅ [+] Generation job {job_id} completed: {result.asset_id}")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        error_trace = traceback.format_exc()
        print(f"❌ [-] Generation job {job_id} failed: {e}")
        print(f"[-] Traceback:\n{error_trace}")
