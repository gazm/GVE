# backend/architect/src/ai_pipeline/track_matter_pipeline.py
"""
Track A: Matter Pipeline - Pipeline Execution

Orchestrates the three-stage Matter generation pipeline.
"""

from __future__ import annotations

from typing import Any

from .orchestrator import GenerationState
from .track_matter_agents import BlacksmithAgent, MachinistAgent, ArtistAgent
from .track_matter_schemas import (
    BlacksmithOutput,
    MachinistOutput,
    ArtistOutput,
    SDFRootNode,
    SDFNode,
)


async def execute_matter_pipeline(state: GenerationState) -> dict[str, Any]:
    """
    Execute the full Matter pipeline (Track A).
    
    Returns combined DNA JSON ready for compiler.
    
    NOTE: Currently only Blacksmith (A1) is active for basic shape testing.
    """
    print("🔨 Starting Matter Pipeline (Blacksmith only)...")
    
    # Stage A1: Blacksmith
    print("  [A1] Initializing Blacksmith agent...")
    blacksmith = BlacksmithAgent()
    ctx = state.to_agent_context()
    print("  [A1] Calling Blacksmith.generate()...")
    a1_output = await blacksmith.generate(ctx)
    print("  [A1] Blacksmith returned, processing output...")
    state.stage_outputs["a1"] = a1_output.model_dump()
    node_count = _count_nodes(a1_output.sdf_tree)
    print(f"  └─ Blacksmith: Generated {node_count} nodes")
    
    # Stages A2 and A3: BYPASSED for now - focus on basic shapes
    print("  [A2/A3] Machinist and Artist steps bypassed (focusing on basic shapes)")
    
    # Return just the Blacksmith output as DNA
    # sdf_tree is now SDFRootNode, convert to dict and wrap in root_node
    sdf_tree_dict = a1_output.sdf_tree.model_dump()
    
    dna = {
        "root_node": sdf_tree_dict,
        "metadata": a1_output.metadata if isinstance(a1_output.metadata, dict) else {},
    }
    
    print("✅ Matter Pipeline complete (Blacksmith only)")
    return dna


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
    
    # Apply A2 delta patches - safely handle dict structure
    patches = []
    if isinstance(a2.delta_patch, dict):
        add_ops = a2.delta_patch.get("add_operations", [])
        if isinstance(add_ops, list):
            patches = add_ops
    
    if patches:
        # For now, add patches as additional operations
        # Full implementation would inject into the tree at target nodes
        dna["machining_patches"] = [
            p.model_dump() if hasattr(p, "model_dump") else p 
            for p in patches
        ]
    
    # Apply A3 materials - safely handle dict structure
    if isinstance(a3.material_config, dict):
        dna["materials"] = {
            node_id: mat.model_dump() if hasattr(mat, "model_dump") else mat
            for node_id, mat in a3.material_config.items()
        }
    else:
        dna["materials"] = {}
    
    return dna
