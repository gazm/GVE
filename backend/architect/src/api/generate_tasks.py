# backend/architect/src/api/generate_tasks.py
"""
Background task runners for AI generation jobs.

These are async functions executed by FastAPI BackgroundTasks.
Each updates its job dict in the shared TTLCache stores.
Separated from generate.py route handlers for file-size discipline.
"""

from __future__ import annotations

import asyncio
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


def submit_stage_review(job_id: str, action: str) -> bool:
    """
    Resolve the A1 review future so the pipeline continues or aborts.
    Returns True if the future was resolved, False if job not awaiting review.
    """
    job = jobs.get(job_id)
    if not job:
        return False
    fut = job.get("stage_review_future")
    if not fut or fut.done():
        return False
    should_continue = action.lower() == "continue"
    fut.set_result(should_continue)
    job["stage_review_future"] = None
    return True


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
    jobs[job_id]["stage_review_future"] = None
    
    # Stage preview callback - stores preview, broadcasts, and for A1 waits for user review
    async def on_stage_complete(stage: str, preview_binary: bytes) -> bool:
        """Callback for pipeline stage completion. Returns True to continue, False to abort."""
        preview_id = f"{job_id}_stage_{stage}"
        
        stage_previews[preview_id] = preview_binary
        jobs[job_id]["current_stage"] = stage
        
        print(f"📺 [*] Stage {stage} preview ready ({len(preview_binary)} bytes)")
        
        try:
            payload = {
                "job_id": job_id,
                "stage": stage,
                "preview_url": f"/api/generate/preview/{preview_id}",
                "preview_bytes": len(preview_binary),
            }
            if stage == "A1":
                payload["awaiting_review"] = True
                jobs[job_id]["stage_review_future"] = asyncio.get_running_loop().create_future()
            await broadcast_event("generate:stage_complete", payload)
        except Exception as ws_error:
            print(f"⚠️ WebSocket broadcast failed (non-fatal): {ws_error}")
        
        if stage == "A1" and jobs[job_id].get("stage_review_future"):
            try:
                should_continue = await asyncio.wait_for(
                    jobs[job_id]["stage_review_future"],
                    timeout=300.0,
                )
                jobs[job_id]["stage_review_future"] = None
                return should_continue
            except asyncio.TimeoutError:
                print(f"⚠️ A1 review timed out (5 min) - continuing pipeline")
                return True
        return True
    
    try:
        track = None
        if request.track_override:
            track = GenerationTrack(request.track_override)
        
        internal_request = GenerateRequest(
            prompt=request.prompt,
            category=request.category,
            style_reference=request.style_reference,
            track_override=track,
            ai_provider=request.ai_provider,
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
        from ..ai_pipeline.track_matter_pipeline import GenerationRejectedByUser
        if isinstance(e, GenerationRejectedByUser):
            jobs[job_id]["status"] = "rejected"
            jobs[job_id]["error"] = "User rejected Blacksmith (A1) output"
            print(f"🚫 [-] Generation job {job_id} rejected by user")
            try:
                await broadcast_event("generate:rejected", {
                    "job_id": job_id,
                    "error": "User rejected Blacksmith output",
                })
            except Exception:
                pass
        else:
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
            ai_provider=request.ai_provider,
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
