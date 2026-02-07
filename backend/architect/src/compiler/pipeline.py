from dataclasses import dataclass
from typing import Optional, Any
from pathlib import Path
from bson import ObjectId
import time
import asyncio

from .queue import CompilePriority

@dataclass
class CompileRequest:
    asset_id: ObjectId
    priority: CompilePriority = CompilePriority.NORMAL
    force_recompile: bool = False

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


def _prepare_vdb_bounds(raw_bounds, target_resolution: int = 128):
    """
    Prepare bounds for VDB baking with padding and geometry-relative voxel size.
    
    Args:
        raw_bounds: Tuple of (min_xyz, max_xyz) lists
        target_resolution: Target voxels on longest axis
    
    Returns:
        (padded_bounds, voxel_size)
    """
    import numpy as np
    
    b_min = np.array(raw_bounds[0], dtype=np.float32)
    b_max = np.array(raw_bounds[1], dtype=np.float32)
    extent = b_max - b_min
    
    # Geometry-relative voxel size: longest axis / target_resolution
    longest_axis = float(extent.max())
    voxel_size = longest_axis / target_resolution
    
    # Add padding: 10% of extent or 3 voxels, whichever is larger
    padding = np.maximum(extent * 0.1, voxel_size * 3)
    
    b_min_padded = b_min - padding
    b_max_padded = b_max + padding
    
    return (b_min_padded.tolist(), b_max_padded.tolist()), float(voxel_size)


async def draft_compile_dna(
    dna: dict[str, Any],
    job_id: str,
    resolution: int = 64,
) -> DraftCompileResult:
    """
    Fast preview compile - skips splat training, returns binary bytes directly.
    
    Used for intermediate stage previews during AI generation pipeline.
    Does NOT write to disk or interact with database.
    
    Args:
        dna: DNA dictionary with root_node structure
        job_id: Job identifier for logging
        resolution: Voxel resolution (lower = faster, default 64 for previews)
    
    Returns:
        DraftCompileResult with binary_data bytes or error
    """
    start = time.time()
    print(f"  [draft-compile] 🏃 Starting fast preview for {job_id}", flush=True)
    
    try:
        if resolution <= 0:
            resolution = 64
        
        # Ensure torch is loaded
        from src.torch_preloader import preloader
        if not preloader.ensure_loaded():
            return DraftCompileResult(False, None, 0.0, "Torch unavailable")
        
        # 1. Build SDF graph from DNA
        print(f"  [draft-compile] 1. Building SDF graph...", flush=True)
        from .math_jit import build_sdf_graph
        sdf_graph = build_sdf_graph(dna)
        
        # 2. Bake SDF to dense grid + VDB (for mesh extraction)
        from .vdb_converter import bake_sdf, dense_grid_to_bytes
        
        # Get bounds from SDF graph and calculate geometry-relative voxel size
        raw_bounds = sdf_graph.bounds if sdf_graph.bounds else ([-1, -1, -1], [1, 1, 1])
        bounds, voxel_size = _prepare_vdb_bounds(raw_bounds, target_resolution=resolution)
        print(f"  [draft-compile] 2. Baking SDF (voxel_size={voxel_size:.4f})...", flush=True)
        print(f"  [draft-compile] Using bounds: {bounds}", flush=True)
        
        bake_result = await asyncio.to_thread(
            bake_sdf,
            sdf_graph,
            bounds_min=tuple(bounds[0]),
            bounds_max=tuple(bounds[1]),
            voxel_size=voxel_size
        )
        volume_data = dense_grid_to_bytes(bake_result)
        print(f"  [draft-compile] Dense grid: {len(volume_data)} bytes", flush=True)
        
        # 3. Generate shell mesh from VDB - blocking CPU operation
        print(f"  [draft-compile] 3. Generating shell...", flush=True)
        from .mesh_repair import repair_and_decimate
        
        shell_data = await asyncio.to_thread(
            repair_and_decimate,
            bake_result.vdb_volume,
            target_tris=2000,  # Lower tri count for preview
            voxel_size=voxel_size,
            bounds_min=tuple(bounds[0]),
        )
        print(f"  [draft-compile] Shell: {len(shell_data)} bytes", flush=True)
        
        # Debug: check shell_data header
        if shell_data and len(shell_data) >= 4:
            import struct
            shell_vertex_count = struct.unpack("<I", shell_data[0:4])[0]
            print(f"  [draft-compile] Shell vertex count from data: {shell_vertex_count}", flush=True)
        
        # 4. Skip splat training (draft mode)
        print(f"  [draft-compile] 4. Skipping splats (draft mode)", flush=True)
        
        # 5. Write to bytes (no disk)
        print(f"  [draft-compile] 5. Building binary...", flush=True)
        from .binary_writer import write_gve_bin_bytes
        
        binary_data = write_gve_bin_bytes(
            volume_data=volume_data,
            shell_data=shell_data,
            splat_data=None,
        )
        
        elapsed = time.time() - start
        print(f"  [draft-compile] ✅ Draft done in {elapsed:.2f}s ({len(binary_data)} bytes)", flush=True)
        
        return DraftCompileResult(
            success=True,
            binary_data=binary_data,
            compile_time_sec=elapsed,
        )
        
    except Exception as e:
        import traceback
        print(f"  [draft-compile] ❌ ERROR: {e}", flush=True)
        traceback.print_exc()
        return DraftCompileResult(
            success=False,
            binary_data=None,
            compile_time_sec=time.time() - start,
            error=str(e),
        )


async def compile_asset(request: CompileRequest) -> CompileResult:
    """
    Execute the full compilation pipeline.
    1. Load asset (raw document with DNA)
    2. Build SDF graph
    3. Bake volume
    4. Gen shell
    5. Write binary
    """
    start = time.time()
    print(f"  [compile] Starting pipeline for {request.asset_id}", flush=True)
    
    try:
        # Use librarian public API for all database access
        from ..librarian import load_asset_doc, resolve_cache_path

        # 1. Load raw asset document (includes DNA field)
        print(f"  [compile] 1. Loading asset from DB...", flush=True)
        doc = await load_asset_doc(str(request.asset_id))
        
        if not doc:
            print(f"  [compile] ❌ Asset not found!", flush=True)
            return CompileResult(False, None, 0.0, "Asset not found")
        
        if "dna" not in doc:
            print(f"  [compile] ❌ Asset has no DNA!", flush=True)
            return CompileResult(False, None, 0.0, "Asset has no DNA field")
        
        print(f"  [compile] ✅ Asset loaded: {doc.get('name')}", flush=True)

        # Get resolution from settings
        settings = doc.get("settings", {})
        resolution = settings.get("resolution", 128)
        if resolution <= 0: resolution = 128

        # Ensure torch is loaded before importing torch-dependent modules
        from src.torch_preloader import preloader
        if not preloader.ensure_loaded():
            return CompileResult(False, None, 0.0, "Torch unavailable for compilation")

        # 2. Build SDF graph from DNA
        print(f"  [compile] 2. Building SDF graph...", flush=True)
        from .math_jit import build_sdf_graph
        dna = doc["dna"]
        sdf_graph = build_sdf_graph(dna)
        print(f"  [compile] SDF graph built", flush=True)

        # 3. Bake SDF to dense grid + VDB
        from .vdb_converter import bake_sdf, dense_grid_to_bytes
        
        # Get bounds from SDF graph and calculate geometry-relative voxel size
        raw_bounds = sdf_graph.bounds if sdf_graph.bounds else ([-1, -1, -1], [1, 1, 1])
        bounds, voxel_size = _prepare_vdb_bounds(raw_bounds, target_resolution=resolution)
        print(f"  [compile] 3. Baking SDF (voxel_size={voxel_size:.4f})...", flush=True)
        print(f"  [compile] Using bounds: {bounds}", flush=True)
        
        # Bake to dense grid + VDB (blocking CPU operation - run in thread pool)
        bake_result = await asyncio.to_thread(
            bake_sdf, 
            sdf_graph, 
            bounds_min=tuple(bounds[0]),
            bounds_max=tuple(bounds[1]),
            voxel_size=voxel_size
        )
        print(f"  [compile] ✅ SDF baked: {bake_result.dims}", flush=True)
        
        # Serialize dense grid for GPU raymarching
        volume_data = dense_grid_to_bytes(bake_result)
        print(f"  [compile] Dense grid serialized: {len(volume_data)} bytes", flush=True)

        # 4. Generate shell from VDB (MeshLib)
        print(f"  [compile] 4. Generating shell from VDB...", flush=True)
        from .mesh_repair import repair_and_decimate
        
        # Generate, repair, and decimate mesh (blocking CPU operation - run in thread pool)
        shell_data = await asyncio.to_thread(
            repair_and_decimate,
            bake_result.vdb_volume,
            target_tris=5000,  # Increased from 1000 for higher quality
            voxel_size=voxel_size,
            bounds_min=tuple(bounds[0]),
        )
        print(f"  [compile] Shell generated: {len(shell_data)} bytes", flush=True)

        # 5. Train splats (Gaussian Splatting)
        # Check if splat training is enabled (can be disabled for draft mode)
        enable_splats = settings.get("enable_splats", True)
        splat_data = None
        
        if enable_splats:
            print(f"  [compile] 5. Training splats...", flush=True)
            from .splat_trainer import compile_splats
            
            # Get bounds from SDF graph or use default
            bounds = sdf_graph.bounds if sdf_graph.bounds else ([-1, -1, -1], [1, 1, 1])
            
            # Splat training parameters from settings
            splat_count = settings.get("splat_count", 1000)
            splat_iterations = settings.get("splat_iterations", 100)
            
            # Run splat training (always exports Oklab — shader handles conversion)
            splat_data = await asyncio.to_thread(
                compile_splats,
                sdf_graph,
                bounds,
                target_count=splat_count,
                iterations=splat_iterations,
            )
            print(f"  [compile] ✅ Splats trained: {len(splat_data)} bytes", flush=True)
        else:
            print(f"  [compile] 5. Splat training skipped (draft mode)", flush=True)

        # 6. Write binary
        print(f"  [compile] 6. Writing binary...", flush=True)
        from .binary_writer import write_gve_bin
        
        # Use resolve_cache_path from librarian to get output path
        output_path = resolve_cache_path(doc)
        
        print(f"  [compile] Output path: {output_path}", flush=True)
        
        # Ensure dir exists (non-blocking)
        await asyncio.to_thread(output_path.parent.mkdir, parents=True, exist_ok=True)
        
        # Write binary file (non-blocking)
        await asyncio.to_thread(
            write_gve_bin, 
            output_path, 
            volume_data=volume_data, 
            shell_data=shell_data,
            splat_data=splat_data,
        )
        
        elapsed = time.time() - start
        print(f"  [compile] Done! Compiled in {elapsed:.2f}s", flush=True)
        
        return CompileResult(
            success=True,
            binary_path=output_path,
            compile_time_sec=elapsed
        )
        
    except Exception as e:
        import traceback
        print(f"  [compile] ERROR: {e}", flush=True)
        traceback.print_exc()
        return CompileResult(
            success=False,
            binary_path=None,
            compile_time_sec=time.time() - start,
            error=str(e)
        )
