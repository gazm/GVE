# backend/architect/src/ai_pipeline/orchestrator.py
"""
AI Pipeline Orchestrator - Multi-track generation flow coordination.

Entry point for asset generation. Routes requests to appropriate
track (Matter, Landscape, Audio) and manages generation state.

Supports stage preview callbacks for real-time viewport updates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable

from bson import ObjectId
from pydantic import BaseModel

from .agents import AgentContext
from .rag import inject_rag_context

# Type for stage completion callback: (stage_name, preview_binary) -> None
StageCompleteCallback = Callable[[str, bytes], Awaitable[None]]


class GenerationTrack(str, Enum):
    """Available generation tracks."""
    
    MATTER = "matter"       # Track A: 3D objects
    LANDSCAPE = "landscape" # Track B: Terrain
    AUDIO = "audio"         # Track C: Sound
    COMPOSITE = "composite" # Track D: Multi-track


class GenerateRequest(BaseModel):
    """Request to generate an asset from prompt."""
    
    prompt: str
    category: str | None = None  # Explicit category (e.g., "Weapon", "Vehicle")
    image_path: Path | None = None
    style_reference: str | None = None
    constraints: dict[str, Any] | None = None
    track_override: GenerationTrack | None = None

# ... (inside generate_asset)




class GenerateResult(BaseModel):
    """Result of asset generation."""
    
    asset_id: str  # ObjectId as string
    dna: dict[str, Any]
    confidence: float
    generation_time_sec: float
    track_used: GenerationTrack
    
    model_config = {"arbitrary_types_allowed": True}


@dataclass
class GenerationState:
    """
    Mutable state passed through generation pipeline.
    
    Accumulates outputs from each stage for downstream agents.
    """
    
    user_prompt: str
    selected_track: GenerationTrack
    rag_context: dict[str, Any] = field(default_factory=dict)
    stage_outputs: dict[str, Any] = field(default_factory=dict)
    validation_history: list[dict] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    style_token: str | None = None
    concept_image_base64: str | None = None  # Concept image for vision-guided generation
    
    def to_agent_context(self) -> AgentContext:
        """Convert state to AgentContext for agent execution."""
        return AgentContext(
            user_prompt=self.user_prompt,
            rag_context=self.rag_context,
            previous_outputs=self.stage_outputs,
            style_token=self.style_token,
        )


def classify_track(prompt: str) -> GenerationTrack:
    """
    Classify user prompt into generation track.
    
    Uses keyword matching for MVP. Can be upgraded to 
    fine-tuned classifier for >98% accuracy.
    """
    prompt_lower = prompt.lower()
    
    # Audio indicators
    audio_keywords = ["sound", "audio", "music", "sfx", "noise", "tone", "beep", "laser"]
    if any(kw in prompt_lower for kw in audio_keywords):
        return GenerationTrack.AUDIO
    
    # Landscape indicators
    landscape_keywords = [
        "terrain", "landscape", "island", "mountain", "valley", 
        "hill", "beach", "desert", "forest", "biome", "map"
    ]
    if any(kw in prompt_lower for kw in landscape_keywords):
        return GenerationTrack.LANDSCAPE
    
    # Default to Matter (3D objects)
    return GenerationTrack.MATTER


async def generate_asset(request: GenerateRequest) -> GenerateResult:
    """
    Main entry point for asset generation (without concept image).
    
    Routes to appropriate track and orchestrates multi-stage pipeline.
    Returns DNA JSON - does NOT compile (per module anti-patterns).
    
    For concept-guided generation, use generate_asset_with_concept() instead.
    """
    return await _generate_asset_internal(request, concept_image_base64=None)


async def generate_asset_with_concept(
    request: GenerateRequest,
    concept_image_base64: str,
    on_stage_complete: StageCompleteCallback | None = None,
) -> GenerateResult:
    """
    Generate asset with concept image as visual reference.
    
    This is the preferred entry point when using the two-phase workflow.
    The concept image guides all generation stages for better quality.
    
    Args:
        request: Generation request with prompt and options
        concept_image_base64: Base64 encoded concept image
        on_stage_complete: Optional callback for stage preview updates.
            Called after each stage with (stage_name, preview_binary_bytes).
    """
    return await _generate_asset_internal(
        request,
        concept_image_base64=concept_image_base64,
        on_stage_complete=on_stage_complete,
    )


async def _generate_asset_internal(
    request: GenerateRequest,
    concept_image_base64: str | None = None,
    on_stage_complete: StageCompleteCallback | None = None,
) -> GenerateResult:
    """
    Internal asset generation implementation.
    
    Routes to appropriate track and orchestrates multi-stage pipeline.
    Returns DNA JSON - does NOT compile (per module anti-patterns).
    
    Args:
        request: Generation request
        concept_image_base64: Optional concept image for vision guidance
        on_stage_complete: Optional callback for stage preview updates
    """
    start_time = time.time()
    
    # 1. Route to track
    # Auto-expand short prompts to avoid ambiguity (e.g. "m4a4" -> "m4a4, 3D game asset")
    expanded_prompt = _expand_short_prompt(request.prompt)
    if expanded_prompt != request.prompt:
        print(f"✨ Expanded prompt: '{request.prompt}' -> '{expanded_prompt}'")
        # Update request prompt for downstream usage
        request.prompt = expanded_prompt

    track = request.track_override or classify_track(request.prompt)
    print(f"🎯 Routing to track: {track.value}")
    if concept_image_base64:
        print(f"📷 Using concept image as visual reference")
    if on_stage_complete:
        print(f"📺 Stage preview callback enabled")
    
    # 2. Initialize state
    state = GenerationState(
        user_prompt=request.prompt,
        selected_track=track,
        style_token=request.style_reference,
        concept_image_base64=concept_image_base64,
    )
    
    # 3. Inject RAG context
    state.rag_context = await inject_rag_context(track, request.prompt)
    
    # 4. Execute track-specific pipeline
    if track == GenerationTrack.MATTER:
        from .track_matter import execute_matter_pipeline
        dna = await execute_matter_pipeline(state, on_stage_complete=on_stage_complete)
    elif track == GenerationTrack.LANDSCAPE:
        # TODO: Implement landscape pipeline
        raise NotImplementedError("Landscape track not yet implemented")
    elif track == GenerationTrack.AUDIO:
        # TODO: Implement audio pipeline
        raise NotImplementedError("Audio track not yet implemented")
    else:
        raise ValueError(f"Unknown track: {track}")
    
    # 5. Validate DNA structure before saving
    is_valid, error_msg = validate_dna_structure(dna)
    if not is_valid:
        raise ValueError(f"Invalid DNA structure: {error_msg}")
    
    # 6. Save via librarian (lazy import to avoid circular)
    from ..librarian import save_asset_doc
    
    # Asset doc must match AssetMetadata schema
    asset_doc = {
        "name": _generate_asset_name(request.prompt),
        "category": _infer_category(track),  # AssetCategory enum value
        "tags": _extract_tags(request.prompt),
        "settings": {
            "lod_count": 3,
            "resolution": 128,
        },
        "dna": dna,
        "is_draft": True,  # Mark as draft until explicitly saved
    }
    
    # Include concept image if available
    if concept_image_base64:
        asset_doc["concept_image"] = concept_image_base64
        asset_doc["concept_prompt"] = request.prompt
    
    asset_id = await save_asset_doc(asset_doc)
    
    # 7. Return result
    elapsed = time.time() - start_time
    
    return GenerateResult(
        asset_id=str(asset_id),
        dna=dna,
        confidence=0.85,  # TODO: Actual confidence from validation
        generation_time_sec=elapsed,
        track_used=track,
    )


def _generate_asset_name(prompt: str) -> str:
    """Generate a snake_case asset name from prompt."""
    # Take first few significant words
    words = prompt.lower().split()[:4]
    # Filter common words
    stopwords = {"a", "an", "the", "make", "create", "generate", "build"}
    words = [w for w in words if w not in stopwords]
    return "_".join(words) or "generated_asset"


def _infer_category(track: GenerationTrack) -> str:
    """Infer asset category from track. Returns AssetCategory enum value."""
    return {
        GenerationTrack.MATTER: "Prop",
        GenerationTrack.LANDSCAPE: "Environment",
        GenerationTrack.AUDIO: "Prop",  # No Audio category, fallback to Prop
        GenerationTrack.COMPOSITE: "Prop",
    }.get(track, "Prop")


def _extract_tags(prompt: str) -> list[str]:
    """Extract potential tags from prompt."""
    # Simple word extraction for MVP
    words = prompt.lower().split()
    # Filter short words and common terms
    tags = [w for w in words if len(w) > 3 and w.isalpha()]
    return tags[:10]  # Limit to 10 tags


def _expand_short_prompt(prompt: str) -> str:
    """
    Expand very short prompts to ensure they are interpreted as 3D assets.
    
    Example: "m4a4" -> "m4a4, 3D prop asset"
    """
    # Split by spaces to count words
    words = prompt.strip().split()
    
    # If 3 words or less, and doesn't already contain "asset", "prop", "3d"
    if len(words) <= 3:
        prompt_lower = prompt.lower()
        keywords = {"asset", "prop", "object", "item", "3d", "structure", "weapon", "tool"}
        if not any(k in prompt_lower for k in keywords):
            return f"{prompt}, 3D prop asset"
            
    return prompt


def validate_dna_structure(dna: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate DNA structure before saving.
    
    Returns (is_valid, error_message).
    If invalid, error_message contains specific details about what's wrong.
    """
    # Valid primitive shapes (includes fractals and revolution)
    VALID_SHAPES = {
        "box", "sphere", "cylinder", "capsule", "torus", "cone", "plane",
        "wedge", "revolution", "mandelbulb", "menger", "julia",
    }
    
    # Check root_node exists
    if "root_node" not in dna:
        return False, "DNA missing 'root_node' field"
    
    root_node = dna["root_node"]
    if not isinstance(root_node, dict):
        return False, f"'root_node' must be an object, got {type(root_node).__name__}"
    
    # Check root_node type
    if root_node.get("type") != "operation":
        return False, f"root_node.type must be 'operation', got '{root_node.get('type')}'"
    
    if root_node.get("op") != "union":
        return False, f"root_node.op must be 'union', got '{root_node.get('op')}'"
    
    # Check children exists and is array
    if "children" not in root_node:
        return False, "root_node missing 'children' field"
    
    children = root_node["children"]
    if not isinstance(children, list):
        return False, f"'children' must be an array, got {type(children).__name__}"
    
    if len(children) == 0:
        return False, "'children' array is empty (must have at least one child)"
    
    # Validate each child
    for idx, child in enumerate(children):
        # Check child is an object, not a string
        if not isinstance(child, dict):
            return False, (
                f"children[{idx}] must be an object with id, type, shape, params, "
                f"but got {type(child).__name__}: {repr(child)[:50]}"
            )
        
        # Check required fields
        if "id" not in child:
            return False, f"children[{idx}] missing required field 'id'"
        
        if "type" not in child:
            return False, f"children[{idx}] missing required field 'type'"
        
        if child.get("type") == "primitive":
            if "shape" not in child:
                return False, f"children[{idx}] (primitive) missing required field 'shape'"
            
            shape = child.get("shape")
            if shape not in VALID_SHAPES:
                return False, (
                    f"children[{idx}] has invalid shape '{shape}'. "
                    f"Valid shapes: {', '.join(sorted(VALID_SHAPES))}"
                )
            
            if "params" not in child:
                return False, f"children[{idx}] (primitive) missing required field 'params'"
            
            if not isinstance(child.get("params"), dict):
                return False, f"children[{idx}].params must be an object, got {type(child.get('params')).__name__}"
    
    return True, ""
