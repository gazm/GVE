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

import asyncio
import base64
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Awaitable

from .agents import AgentContext, clean_json_schema_for_gemini, _get_client
from .orchestrator import GenerationState
from .track_matter_agents import BlacksmithAgent, MachinistAgent, ArtistAgent
from .track_matter_schemas import (
    BlacksmithOutput,
    MachinistOutput,
    MachinistDeltaPatchList,
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
        
        # DEBUG: A1-only dump (no Machinist patches or Artist materials; those are merged later)
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
        # Stage A2: Machinist (Function & Negative Space) - per part
        # =================================================================
        print("  [A2] Initializing Machinist agent...")
        machinist = MachinistAgent()
        parts = [c for c in a1_output.sdf_tree.children if getattr(c, "id", None)]
        
        if not parts:
            # Fallback: one global Machinist call (no top-level parts with id)
            print("  [A2] No parts with id found, using single asset-wide Machinist call...")
            ctx = state.to_agent_context()
            if has_concept and concept_image_path:
                a2_output = await machinist.generate_with_image(ctx, concept_image_path)
            else:
                a2_output = await machinist.generate(ctx)
            state.stage_outputs["a2"] = a2_output.model_dump()
        else:
            # Try Batch API first (parallel, ~50% cost); fall back to sequential on failure/timeout
            aggregated = await _run_machinist_batch(
                machinist, parts, state, concept_image_path if has_concept else None
            )
            if aggregated is None:
                print("  [A2] Running Machinist sequentially (fallback)...")
                aggregated = []
                for i, part in enumerate(parts):
                    part_id = getattr(part, "id", "")
                    print(f"  [A2] Machinist for part {i + 1}/{len(parts)}: {part_id}...")
                    part_ctx = AgentContext(
                        user_prompt=state.user_prompt,
                        rag_context=state.rag_context,
                        previous_outputs={
                            "a1": _strip_dna_for_context(state.stage_outputs["a1"]), # Full A1 can be large, use lite version
                            "a1_part": {"part_id": part_id, "part": part.model_dump()},
                        },
                        style_token=state.style_token,
                    )
                    if has_concept and concept_image_path:
                        a2_part_output = await machinist.generate_with_image(part_ctx, concept_image_path)
                    else:
                        a2_part_output = await machinist.generate(part_ctx)
                    ops = a2_part_output.delta_patch.add_operations if a2_part_output.delta_patch else []
                    aggregated.extend(ops)
            a2_output = MachinistOutput(delta_patch=MachinistDeltaPatchList(add_operations=aggregated))
            state.stage_outputs["a2"] = a2_output.model_dump()
        
        print("  [A2] Machinist returned, processing output...")
        patch_count = _count_patches(a2_output)
        print(f"  └─ Machinist: Generated {patch_count} patches")
        
        # Stage A2 preview callback
        if on_stage_complete:
            a2_dna = _build_intermediate_dna(a1_output, a2_output, None)
            preview_bytes = await _draft_compile_for_preview(a2_dna, "A2")
            if preview_bytes:
                await on_stage_complete("A2", preview_bytes)
        
        # =================================================================
        # Stage A2.5: Material Prep (Auto-Tune)
        # =================================================================
        print("  [A2.5] 🔧 Building material hints (auto-tune)...")
        material_prep = _build_material_prep(a1_output)
        state.stage_outputs["material_prep"] = material_prep
        print(f"  └─ Material Prep: Seeded {len(material_prep)} node hints")
        
        # =================================================================
        # Stage A3: Artist (Surface & Materials)
        # =================================================================
        print("  [A3] Initializing Artist agent...")
        artist = ArtistAgent()
        # Artist only needs IDs and types for material assignment. Strip params/transforms.
        artist_prev_outputs = {
            "a1": _strip_dna_for_context(state.stage_outputs["a1"], strip_params=True),
            "a2": state.stage_outputs.get("a2", {}),
        }
        ctx = AgentContext(
            user_prompt=state.user_prompt,
            rag_context=state.rag_context,
            previous_outputs=artist_prev_outputs,
            style_token=state.style_token,
        )
        
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
        
        # Merge material hints with Artist output (Artist wins)
        merged_materials = _merge_material_configs(material_prep, a3_output.material_config)
        if merged_materials:
            a3_output = a3_output.model_copy(update={"material_config": merged_materials})
        
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


    return dna


def _build_intermediate_dna(
    a1: BlacksmithOutput,
    a2: MachinistOutput | None,
    a3: ArtistOutput | None,
) -> dict[str, Any]:
    """
    Build intermediate DNA from available stage outputs.
    Used for stage previews before all stages are complete.
    """
    root_node = a1.sdf_tree.model_dump()
    
    dna = {
        "root_node": root_node,
        "metadata": a1.metadata if isinstance(a1.metadata, dict) else {},
    }
    
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


def _strip_dna_for_context(dna: dict[str, Any], strip_params: bool = False) -> dict[str, Any]:
    """
    Strip unneeded fields from DNA to reduce context size.
    
    If strip_params is True, removes 'params' and 'transform' from nodes.
    Used for Artist stage which only needs IDs and types for material assignment.
    """
    import copy
    dna_lite = copy.deepcopy(dna)
    
    def strip_recursive(node: dict):
        if strip_params:
            if "params" in node:
                node.pop("params")
            if "transform" in node:
                node.pop("transform")
        
        if "children" in node and isinstance(node["children"], list):
            for child in node["children"]:
                if isinstance(child, dict):
                    strip_recursive(child)
                    
    if "root_node" in dna_lite:
        strip_recursive(dna_lite["root_node"])
        
    return dna_lite


def _save_concept_to_temp(concept_image_base64: str) -> Path:
    """Save concept image to temp file for vision agents."""
    image_bytes = base64.b64decode(concept_image_base64)
    
    # Create temp file with .png extension
    fd, path = tempfile.mkstemp(suffix=".png")
    with open(fd, "wb") as f:
        f.write(image_bytes)
    
    return Path(path)


# Batch API: max wait then fall back to sequential (Batch target turnaround is up to 24h)
_MACHINIST_BATCH_POLL_INTERVAL_SEC = 15
_MACHINIST_BATCH_MAX_WAIT_SEC = 1800  # 30 minutes total
_MACHINIST_BATCH_PENDING_MAX_SEC = 300  # 5 minutes in PENDING → fall back to sequential


async def _run_machinist_batch(
    machinist: MachinistAgent,
    parts: list[SDFNode],
    state: GenerationState,
    concept_image_path: Path | None,
) -> list[Any] | None:
    """
    Run Machinist for all parts via Gemini Batch API (parallel, ~50% cost).
    Concept image is uploaded once via Files API and referenced by file_uri in each request.
    Returns aggregated add_operations or None on failure/timeout (caller should fall back to sequential).
    """
    schema = machinist.get_output_schema()
    json_schema = clean_json_schema_for_gemini(schema.model_json_schema())
    client = _get_client()

    # Upload concept image once and reference by key; on timeout/network failure fall back to inline base64
    concept_file_uri: str | None = None
    concept_image_b64: str | None = None
    if concept_image_path and concept_image_path.exists():
        try:
            upload_config: Any = {"mime_type": "image/png", "display_name": "machinist-concept"}
            try:
                from google.genai import types as genai_types
                upload_config = genai_types.UploadFileConfig(
                    mime_type="image/png",
                    display_name="machinist-concept",
                )
            except (ImportError, AttributeError):
                pass
            uploaded = await asyncio.to_thread(
                client.files.upload,
                file=str(concept_image_path),
                config=upload_config,
            )
            concept_file_uri = getattr(uploaded, "uri", None)
            if concept_file_uri and concept_file_uri.startswith("files/"):
                # Prefix with base URL for Batch API compatibility
                concept_file_uri = f"https://generativelanguage.googleapis.com/v1beta/{concept_file_uri}"
                
            if concept_file_uri:
                print(f"  [A2] 📤 Concept image uploaded: {concept_file_uri}")
        except Exception as e:
            print(f"  [A2] ⚠️ Concept image upload failed ({e}), inlining image in batch requests")
            try:
                concept_image_b64 = base64.b64encode(concept_image_path.read_bytes()).decode("ascii")
            except Exception:
                pass

    inline_requests: list[dict[str, Any]] = []
    for part in parts:
        part_id = getattr(part, "id", "")
        part_ctx = AgentContext(
            user_prompt=state.user_prompt,
            rag_context=state.rag_context,
            previous_outputs={
                "a1": state.stage_outputs["a1"],
                "a1_part": {"part_id": part_id, "part": part.model_dump()},
            },
            style_token=state.style_token,
        )
        system_instruction, user_prompt = machinist.build_prompt(part_ctx)
        parts_list: list[dict[str, Any]] = [{"text": user_prompt}]
        if concept_file_uri:
            parts_list.append({
                "file_data": {"file_uri": concept_file_uri, "mime_type": "image/png"},
            })
        elif concept_image_b64:
            parts_list.append({
                "inline_data": {"mime_type": "image/png", "data": concept_image_b64},
            })
        req: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts_list}],
            "config": {
                "system_instruction": {"parts": [{"text": system_instruction}]},
                "temperature": machinist.temperature,
                "response_mime_type": "application/json",
                "response_schema": json_schema,
            },
        }
        inline_requests.append(req)

    try:
        job = client.batches.create(
            model=machinist.model_name,
            src=inline_requests,
            config={"display_name": "machinist-parts"},
        )
        job_name = job.name
        print(f"  [A2] 📦 Batch job created: {job_name} (polling every {_MACHINIST_BATCH_POLL_INTERVAL_SEC}s, max {_MACHINIST_BATCH_MAX_WAIT_SEC}s)...")
        deadline = time.monotonic() + _MACHINIST_BATCH_MAX_WAIT_SEC
        poll_count = 0
        start = time.monotonic()
        pending_since: float | None = None
        while time.monotonic() < deadline:
            poll_count += 1
            batch_job = await asyncio.to_thread(client.batches.get, name=job_name)
            state_name = getattr(batch_job.state, "name", None) or str(getattr(batch_job, "state", ""))
            if state_name == "JOB_STATE_PENDING":
                if pending_since is None:
                    pending_since = time.monotonic()
                elif time.monotonic() - pending_since >= _MACHINIST_BATCH_PENDING_MAX_SEC:
                    print(f"  [A2] ⚠️ Batch still PENDING after 5 minutes, falling back to sequential")
                    return None
            else:
                pending_since = None
            # Progress every 2 polls (~30s) so the user sees activity
            if poll_count % 2 == 0:
                elapsed = int(time.monotonic() - start)
                print(f"  [A2] ⏳ Batch job still running ({state_name}, {elapsed}s elapsed)...")
            if state_name == "JOB_STATE_SUCCEEDED":
                aggregated: list[Any] = []
                total_input_tokens = 0
                dest = getattr(batch_job, "dest", None)
                inlined = getattr(dest, "inlined_responses", None) if dest else None
                if not inlined:
                    print("  [A2] ⚠️ Batch succeeded but no inlined_responses")
                    return None
                for i, inline_response in enumerate(inlined):
                    resp = getattr(inline_response, "response", None)
                    err = getattr(inline_response, "error", None)
                    if err:
                        print(f"  [A2] ⚠️ Batch response {i + 1} error: {err}")
                        continue
                    if not resp:
                        continue
                    
                    usage = getattr(resp, "usage_metadata", None)
                    if usage and hasattr(usage, "prompt_token_count"):
                        total_input_tokens += usage.prompt_token_count

                    text = getattr(resp, "text", None)
                    if not text:
                        continue
                    try:
                        data = json.loads(text)
                        out = MachinistOutput.model_validate(data)
                        ops = out.delta_patch.add_operations if out.delta_patch else []
                        aggregated.extend(ops)
                    except (json.JSONDecodeError, Exception) as e:
                        print(f"  [A2] ⚠️ Batch response {i + 1} parse error: {e}")
                
                print(f"  [A2] 📥 Total input tokens (batch): {total_input_tokens}")
                print(f"  [A2] ✅ Batch completed: {len(aggregated)} operations from {len(inlined)} parts")
                return aggregated
            if state_name in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
                err = getattr(batch_job, "error", None)
                print(f"  [A2] ⚠️ Batch job {state_name}: {err}")
                return None
            await asyncio.sleep(_MACHINIST_BATCH_POLL_INTERVAL_SEC)
        print("  [A2] ⚠️ Batch job timed out, falling back to sequential")
        return None
    except Exception as e:
        print(f"  [A2] ⚠️ Batch create failed: {e}")
        return None
    finally:
        if concept_file_uri:
            try:
                await asyncio.to_thread(client.files.delete, name=concept_file_uri)
            except Exception:
                pass


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


def _build_material_prep(a1_output: BlacksmithOutput) -> dict[str, dict[str, Any]]:
    """Create deterministic material hints from node IDs."""
    hints: dict[str, dict[str, Any]] = {}
    
    wood_keys = ("wood", "grip", "handle", "stock")
    rubber_keys = ("rubber", "seal", "gasket")
    glass_keys = ("glass", "lens", "optic")
    
    metal_texture = {
        "type": "perlin",
        "scale": 14.0,
        "intensity": 0.08,
        "color_variation": 0.04,
        "roughness_variation": 0.15,
        "metallic_variation": 0.04,
    }
    wood_texture = {
        "type": "wood_grain",
        "scale": 10.0,
        "intensity": 0.5,
        "color_variation": 0.3,
        "roughness_variation": 0.3,
    }
    metal_weather = {
        "edge_wear": 0.15,
        "cavity_grime": 0.15,
        "rust_amount": 0.0,
    }
    
    def classify(node_id: str) -> dict[str, Any]:
        lower = node_id.lower()
        if any(k in lower for k in wood_keys):
            return {
                "material_id": "WOOD_OAK",
                "procedural_texture": wood_texture,
            }
        if any(k in lower for k in rubber_keys):
            return {
                "material_id": "RUBBER_STANDARD",
            }
        if any(k in lower for k in glass_keys):
            return {
                "material_id": "GLASS_CLEAR",
            }
        return {
            "material_id": "METAL_STEEL",
            "procedural_texture": metal_texture,
            "texture_modifiers": metal_weather,
        }
    
    def traverse(node: SDFNode) -> None:
        node_id = getattr(node, "id", None)
        node_type = getattr(node, "type", None)
        if node_id and node_type == "primitive":
            hints[node_id] = classify(node_id)
        if node.children:
            for child in node.children:
                traverse(child)
    
    for child in a1_output.sdf_tree.children:
        traverse(child)
    
    return hints


def _merge_material_configs(
    defaults: dict[str, dict[str, Any]],
    artist_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Merge material hints with Artist output. Artist values win."""
    merged: dict[str, dict[str, Any]] = {}
    
    def to_dict(value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        if isinstance(value, dict):
            return value
        return {}
    
    all_keys = set(defaults.keys()) | set(artist_config.keys())
    for node_id in all_keys:
        artist_val = to_dict(artist_config.get(node_id))
        default_val = defaults.get(node_id, {})
        if not artist_val:
            merged[node_id] = default_val
            continue
        merged_val = dict(artist_val)
        for key, value in default_val.items():
            if key not in merged_val or merged_val.get(key) in (None, {}, []):
                merged_val[key] = value
        merged[node_id] = merged_val
    
    return merged


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
