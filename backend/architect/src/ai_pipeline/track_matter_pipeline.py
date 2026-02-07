# backend/architect/src/ai_pipeline/track_matter_pipeline.py
"""
Track A: Matter Pipeline - Pipeline Execution

Orchestrates the three-stage Matter generation pipeline:
- A1: Blacksmith (Form & Massing) - Union operations only
- A2: Machinist (Function & Negative Space) - Subtract operations
- A3: Artist (Surface & Materials) - Material assignments

When a concept image is available, it guides all stages for better quality.
Supports on_stage_complete callback for intermediate viewport previews.
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any, Callable, Awaitable

from .orchestrator import GenerationState
from .track_matter_agents import BlacksmithAgent, MachinistAgent, ArtistAgent
from .track_matter_schemas import (
    BlacksmithOutput,
    MachinistOutput,
    ArtistOutput,
    SDFRootNode,
    SDFNode,
)

# Type for stage completion callback: (stage_name, preview_binary) -> None
StageCompleteCallback = Callable[[str, bytes], Awaitable[None]]


async def execute_matter_pipeline(
    state: GenerationState,
    on_stage_complete: StageCompleteCallback | None = None,
) -> dict[str, Any]:
    """
    Execute the full Matter pipeline (Track A).
    
    Returns combined DNA JSON ready for compiler.
    
    Pipeline Stages:
    - A1: Blacksmith - Base geometry (Union operations)
    - A2: Machinist - Functional details (Subtract operations)  
    - A3: Artist - Materials and surface appearance
    
    Args:
        state: Generation state with prompt, RAG context, and concept image
        on_stage_complete: Optional callback for stage previews. Called after
            each stage with (stage_name, preview_binary_bytes).
    
    If concept_image_base64 is in state, uses vision-guided generation.
    """
    has_concept = state.concept_image_base64 is not None
    
    if has_concept:
        print("🔨 Starting Matter Pipeline (Full 3-Stage with Concept Reference)...")
    else:
        print("🔨 Starting Matter Pipeline (Full 3-Stage)...")
    
    if on_stage_complete:
        print("  📺 Stage previews enabled")
    
    # Prepare concept image path if available
    concept_image_path: Path | None = None
    if has_concept:
        concept_image_path = _save_concept_to_temp(state.concept_image_base64)
        print(f"  📷 Concept image saved for reference")
    
    try:
        # =================================================================
        # Stage A1: Blacksmith (Form & Massing)
        # =================================================================
        print("  [A1] Initializing Blacksmith agent...")
        blacksmith = BlacksmithAgent()
        ctx = state.to_agent_context()
        
        if has_concept and concept_image_path:
            print("  [A1] Calling Blacksmith.generate_with_image()...")
            a1_output = await blacksmith.generate_with_image(ctx, concept_image_path)
        else:
            print("  [A1] Calling Blacksmith.generate()...")
            a1_output = await blacksmith.generate(ctx)
        
        print("  [A1] Blacksmith returned, processing output...")
        state.stage_outputs["a1"] = a1_output.model_dump()
        node_count = _count_nodes(a1_output.sdf_tree)
        print(f"  └─ Blacksmith: Generated {node_count} nodes")
        
        # DEBUG: Dump A1 output to check for "slats" hallucination
        try:
            debug_path = Path("C:/Users/Admin/.gemini/antigravity/brain/841a62e8-04ec-4fa6-a929-17ec9329c080/debug_a1_output.json")
            with open(debug_path, "w") as f:
                f.write(a1_output.model_dump_json(indent=2))
            print(f"  🐛 [DEBUG] Saved Blacksmith output to {debug_path}")
        except Exception as e:
            print(f"  ⚠️ Failed to save debug JSON: {e}")
        
        # Stage A1 preview callback
        if on_stage_complete:
            a1_dna = _build_intermediate_dna(a1_output, None, None)
            preview_bytes = await _draft_compile_for_preview(a1_dna, "A1")
            if preview_bytes:
                await on_stage_complete("A1", preview_bytes)
        
        # =================================================================
        # Stage A2: Machinist (Function & Negative Space)
        # =================================================================
        print("  [A2] Initializing Machinist agent...")
        machinist = MachinistAgent()
        ctx = state.to_agent_context()  # Updated with A1 output
        
        if has_concept and concept_image_path:
            print("  [A2] Calling Machinist.generate_with_image()...")
            a2_output = await machinist.generate_with_image(ctx, concept_image_path)
        else:
            print("  [A2] Calling Machinist.generate()...")
            a2_output = await machinist.generate(ctx)
        
        print("  [A2] Machinist returned, processing output...")
        state.stage_outputs["a2"] = a2_output.model_dump()
        patch_count = _count_patches(a2_output)
        print(f"  └─ Machinist: Generated {patch_count} patches")
        
        # Stage A2 preview callback
        if on_stage_complete:
            a2_dna = _build_intermediate_dna(a1_output, a2_output, None)
            preview_bytes = await _draft_compile_for_preview(a2_dna, "A2")
            if preview_bytes:
                await on_stage_complete("A2", preview_bytes)
        
        # =================================================================
        # Stage A3: Artist (Surface & Materials)
        # =================================================================
        print("  [A3] Initializing Artist agent...")
        artist = ArtistAgent()
        ctx = state.to_agent_context()  # Updated with A1 + A2 outputs
        
        if has_concept and concept_image_path:
            print("  [A3] Calling Artist.generate_with_image()...")
            a3_output = await artist.generate_with_image(ctx, concept_image_path)
        else:
            print("  [A3] Calling Artist.generate()...")
            a3_output = await artist.generate(ctx)
        
        print("  [A3] Artist returned, processing output...")
        state.stage_outputs["a3"] = a3_output.model_dump()
        material_count = _count_materials(a3_output)
        print(f"  └─ Artist: Assigned {material_count} materials")
        
        # =================================================================
        # Merge all outputs into final DNA
        # =================================================================
        print("  [Merge] Combining all stage outputs...")
        dna = _merge_pipeline_outputs(a1_output, a2_output, a3_output)
        
        print("✅ Matter Pipeline complete (Full 3-Stage)")
        return dna
        
    finally:
        # Cleanup temp file
        if concept_image_path and concept_image_path.exists():
            try:
                concept_image_path.unlink()
            except Exception:
                pass


async def _draft_compile_for_preview(dna: dict[str, Any], stage: str) -> bytes | None:
    """
    Draft compile intermediate DNA for stage preview.
    
    Returns binary bytes or None if compile fails.
    """
    try:
        from ..compiler.pipeline import draft_compile_dna
        
        result = await draft_compile_dna(dna, job_id=f"stage_{stage}", resolution=64)
        if result.success and result.binary_data:
            return result.binary_data
        else:
            print(f"  ⚠️ Stage {stage} preview compile failed: {result.error}")
            return None
    except Exception as e:
        print(f"  ⚠️ Stage {stage} preview compile error: {e}")
        return None


def _build_intermediate_dna(
    a1: BlacksmithOutput,
    a2: MachinistOutput | None,
    a3: ArtistOutput | None,
) -> dict[str, Any]:
    """
    Build intermediate DNA from available stage outputs.
    
    Used for stage previews before all stages are complete.
    """
    import json
    
    root_node = a1.sdf_tree.model_dump()
    
    dna = {
        "root_node": root_node,
        "metadata": a1.metadata if isinstance(a1.metadata, dict) else {},
    }
    
    # Debug: log the DNA structure
    children_count = len(root_node.get("children") or [])
    print(f"  [intermediate_dna] 🔍 Building DNA: root_node.op={root_node.get('op')}, children={children_count}", flush=True)
    if children_count > 0:
        first_child = root_node["children"][0]
        print(f"  [intermediate_dna] First child: type={first_child.get('type')}, shape={first_child.get('shape')}", flush=True)
    
    # Add A2 patches if available
    if a2 and a2.delta_patch:
        patches = a2.delta_patch.add_operations
        if patches:
            dna["machining_patches"] = [p.model_dump() for p in patches]
    
    # Add A3 materials if available
    if a3 and a3.material_config and isinstance(a3.material_config, dict):
        dna["materials"] = {
            node_id: mat.model_dump() if hasattr(mat, "model_dump") else mat
            for node_id, mat in a3.material_config.items()
        }
    
    return dna


def _save_concept_to_temp(concept_image_base64: str) -> Path:
    """Save concept image to temp file for vision agents."""
    image_bytes = base64.b64decode(concept_image_base64)
    
    # Create temp file with .png extension
    fd, path = tempfile.mkstemp(suffix=".png")
    with open(fd, "wb") as f:
        f.write(image_bytes)
    
    return Path(path)


def _count_nodes(sdf_tree: SDFRootNode) -> int:
    """Count nodes in SDF tree."""
    count = 0
    
    def traverse(node: SDFNode) -> None:
        nonlocal count
        count += 1
        if node.children:
            for child in node.children:
                traverse(child)
    
    # sdf_tree is SDFRootNode, count it and all children
    count += 1  # Count root node
    for child in sdf_tree.children:
        traverse(child)
    
    return count


def _count_patches(a2_output: MachinistOutput) -> int:
    """Count machining patches from Machinist output."""
    if not a2_output.delta_patch:
        return 0
    
    # delta_patch is now a MachinistDeltaPatchList with .add_operations attribute
    return len(a2_output.delta_patch.add_operations)


def _count_materials(a3_output: ArtistOutput) -> int:
    """Count material assignments from Artist output."""
    if not a3_output.material_config:
        return 0
    
    if isinstance(a3_output.material_config, dict):
        return len(a3_output.material_config)
    
    return 0


def _merge_pipeline_outputs(
    a1: BlacksmithOutput,
    a2: MachinistOutput,
    a3: ArtistOutput,
) -> dict[str, Any]:
    """
    Merge outputs from all three stages into final DNA.
    
    - A1 provides base SDF tree
    - A2 provides delta patches (subtract operations)
    - A3 provides material assignments
    """
    # Start with A1 SDF tree - convert SDFRootNode to dict
    root_node = a1.sdf_tree.model_dump()
    
    dna = {
        "root_node": root_node,
        "metadata": a1.metadata if isinstance(a1.metadata, dict) else {},
    }
    
    # Apply A2 delta patches - delta_patch is now MachinistDeltaPatchList
    patches = a2.delta_patch.add_operations if a2.delta_patch else []
    
    if patches:
        # For now, add patches as additional operations
        # Full implementation would inject into the tree at target nodes
        dna["machining_patches"] = [p.model_dump() for p in patches]
    
    # Apply A3 materials - safely handle dict structure
    if isinstance(a3.material_config, dict):
        dna["materials"] = {
            node_id: mat.model_dump() if hasattr(mat, "model_dump") else mat
            for node_id, mat in a3.material_config.items()
        }
    else:
        dna["materials"] = {}
    
    return dna
