"""
Asset Compilation Pipeline.

Orchestrates the conversion of DNA -> GVE Binary using:
1. CompilerContext: Shared state (SDF, Grid, Mesh).
2. Bakers: Pluggable logic for generating specific binary slots.
"""

from dataclasses import dataclass
from typing import Optional, Any, List, Dict
from pathlib import Path
from bson import ObjectId
import time
import asyncio
import traceback

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from .queue import CompilePriority
from .bakeries.base import CompilerContext, CompilerOptions
from .bakeries import VolumeBaker, MeshBaker, SplatBaker, TriplanarBaker
from .binary_writer import write_gve_bin, write_gve_bin_bytes
from .sdf_serializer import serialize_sdf_graph


@dataclass
class CompileRequest:
    asset_id: ObjectId
    priority: CompilePriority = CompilePriority.NORMAL
    force_recompile: bool = False
    options: Optional[dict] = None  # Transient overrides

@dataclass
class CompileResult:
    success: bool
    binary_path: Optional[Path]
    compile_time_sec: float
    error: Optional[str] = None


@dataclass
class DraftCompileResult:
    """Result of draft compilation (in-memory, no splats)."""
    success: bool
    binary_data: Optional[bytes]
    compile_time_sec: float
    error: Optional[str] = None


def _cleanup_cuda(ctx: Optional[CompilerContext] = None):
    """Aggressive GPU cleanup."""
    try:
        if ctx and ctx.sdf_graph and hasattr(ctx.sdf_graph, "to"):
             # Move to CPU to free GPU memory immediately even if context persists slightly longer
            ctx.sdf_graph = ctx.sdf_graph.to("cpu")
    except Exception:
        pass
    
    try:
        import gc
        gc.collect()
    except Exception:
        pass
        
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


async def draft_compile_dna(
    dna: dict[str, Any],
    job_id: str,
    resolution: int = 64,
) -> DraftCompileResult:
    """
    Fast preview compile - skips splat training, returns binary bytes directly.
    Used for intermediate stage previews during AI generation pipeline.
    """
    start = time.time()
    print(f"  [draft-compile] 🏃 Starting fast preview for {job_id}", flush=True)
    
    ctx = None
    try:
        if resolution <= 0:
            resolution = 64
        
        # Ensure torch is loaded
        from src.torch_preloader import preloader
        if not preloader.ensure_loaded():
            return DraftCompileResult(False, None, 0.0, "Torch unavailable")
            
        # Initialize Context
        # Draft mode forces low resolution and disables heavy features
        settings = {
            "resolution": resolution,
            "texture_mode": "procedural_triplanar", # Unused for draft but required by model
            "enable_splats": False,
        }
        ctx = CompilerContext(dna, settings)
        
        # 1. Build SDF & Bake Volume
        print(f"  [draft-compile] 1. Baking SDF (res={resolution})...", flush=True)
        # VolumeBaker handles baking the grid
        volume_baker = VolumeBaker()
        volume_result = await volume_baker.bake(ctx)
        
        # 2. Serialize SDF Bytecode (fast)
        if ctx.sdf_graph is None:
            await ctx.ensure_sdf_graph()
        sdf_bytecode = serialize_sdf_graph(ctx.sdf_graph)
        
        # 3. Generate Shell (simplified for preview)
        print(f"  [draft-compile] 2. Generating shell...", flush=True)
        # We manually call mesh generation with lower tri count for draft
        # MeshBaker uses default 5000, we want 2000. 
        # For now, we reuse the shared logic in context but maybe context needs 'target_tris' option?
        # Or we just accept 5000. It's fast enough.
        mesh_baker = MeshBaker()
        mesh_result = await mesh_baker.bake(ctx)
        
        # 4. Write to bytes
        print(f"  [draft-compile] 3. Building binary...", flush=True)
        binary_data = write_gve_bin_bytes(
            volume_data=volume_result.get("volume_data"),
            shell_data=mesh_result.get("shell_mesh"),
            splat_data=None, # No splats in draft
            sdf_bytecode=sdf_bytecode,
            triplanar_data=None # No textures in draft
        )
        
        elapsed = time.time() - start
        print(f"  [draft-compile] ✅ Draft done in {elapsed:.2f}s ({len(binary_data)} bytes)", flush=True)
        
        return DraftCompileResult(True, binary_data, elapsed)

    except Exception as e:
        print(f"  [draft-compile] ❌ ERROR: {e}", flush=True)
        traceback.print_exc()
        return DraftCompileResult(False, None, time.time() - start, str(e))
    finally:
        _cleanup_cuda(ctx)


async def compile_asset(request: CompileRequest) -> CompileResult:
    """
    Execute the full compilation pipeline using CompilerContext and Bakers.
    """
    start = time.time()
    print(f"  [compile] Starting pipeline for {request.asset_id}", flush=True)
    
    ctx = None
    try:
        # 1. Load Asset
        from ..librarian import load_asset_doc, resolve_cache_path as resolve_path
        doc = await load_asset_doc(str(request.asset_id))
        
        if not doc:
            return CompileResult(False, None, 0.0, "Asset not found")
        if "dna" not in doc:
            return CompileResult(False, None, 0.0, "Asset has no DNA field")

        # 2. Setup Context
        settings = doc.get("settings", {}).copy()
        if request.options:
            print(f"  [compile] ⚙️ Applying transient options: {request.options}", flush=True)
            settings.update(request.options)
            
        # Ensure defaults map to something valid if missing
        if "texture_mode" not in settings:
             # Map legacy 'splat_mode' or default
             settings["texture_mode"] = settings.get("splat_mode", "procedural_triplanar")
        
        # Inject concept image if present (for swatch/colorizing)
        dna = doc["dna"]

        # Initialize Context (validates settings)
        ctx = CompilerContext(dna, settings)
        
        # Ensure torch loaded
        from src.torch_preloader import preloader
        if not preloader.ensure_loaded():
            return CompileResult(False, None, 0.0, "Torch unavailable")

        # 3. Pipeline Execution
        output_path = resolve_path(doc)
        
        # Prepare Bakers
        bakers = [
            VolumeBaker(),
            MeshBaker(),
        ]
        
        # Add Texture Baker based on mode
        # The Options model ensures texture_mode is valid
        if ctx.options.texture_mode == "procedural_triplanar":
            bakers.append(TriplanarBaker())
        elif ctx.options.enable_splats:
            bakers.append(SplatBaker())

        # Execute Bakers
        artifacts: Dict[str, bytes] = {}
        
        # Always bake SDF bytecode first
        await ctx.ensure_sdf_graph()
        artifacts["sdf_bytecode"] = serialize_sdf_graph(ctx.sdf_graph)
        
        for baker in bakers:
            print(f"  [compile] 🔨 Running {baker.name()} baker...", flush=True)
            result = await baker.bake(ctx)
            artifacts.update(result)

        # 4. Write Binary
        print(f"  [compile] 💾 Writing binary to {output_path}...", flush=True)
        await asyncio.to_thread(output_path.parent.mkdir, parents=True, exist_ok=True)
        
        await asyncio.to_thread(
            write_gve_bin,
            output_path,
            volume_data=artifacts.get("volume_data"),
            shell_data=artifacts.get("shell_mesh"),
            splat_data=artifacts.get("splat_data"),
            sdf_bytecode=artifacts.get("sdf_bytecode"),
            triplanar_data=artifacts.get("triplanar_data"),
        )
        
        elapsed = time.time() - start
        print(f"  [compile] ✅ Done! Compiled in {elapsed:.2f}s", flush=True)
        
        return CompileResult(True, output_path, elapsed)

    except Exception as e:
        print(f"  [compile] ❌ ERROR: {e}", flush=True)
        traceback.print_exc()
        return CompileResult(False, None, time.time() - start, str(e))
    finally:
        _cleanup_cuda(ctx)
