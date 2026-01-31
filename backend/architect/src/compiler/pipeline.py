from dataclasses import dataclass
from typing import Optional
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

        # Get resolution from settings (interpreted as voxel_size calculation base or direct size)
        settings = doc.get("settings", {})
        # For VDB, we might prefer 'voxel_size' directly, but map resolution 128 -> ~0.04m for 4m box
        resolution = settings.get("resolution", 128)
        if resolution <= 0: resolution = 128
        voxel_size = 4.0 / resolution # Heuristic: 4m scene / 128 = 3cm

        # 2. Build SDF graph from DNA
        print(f"  [compile] 2. Building SDF graph...", flush=True)
        from .math_jit import build_sdf_graph
        dna = doc["dna"]
        sdf_graph = build_sdf_graph(dna)
        print(f"  [compile] SDF graph built", flush=True)

        # 3. Bake volume (OpenVDB)
        print(f"  [compile] 3. Baking VDB volume (voxel_size={voxel_size:.3f})...", flush=True)
        from .vdb_converter import bake_sdf_to_vdb, vdb_to_bytes
        
        # Bake to VdbVolume (blocking CPU operation - run in thread pool)
        vdb_grid = await asyncio.to_thread(bake_sdf_to_vdb, sdf_graph, voxel_size=voxel_size)
        print(f"  [compile] ✅ VDB volume baked", flush=True)
        
        # Serialize (blocking I/O - run in thread pool)
        volume_data = await asyncio.to_thread(vdb_to_bytes, vdb_grid)
        print(f"  [compile] Volume serialized: {len(volume_data)} bytes", flush=True)

        # 4. Generate shell (MeshLib)
        print(f"  [compile] 4. Generating shell from VDB...", flush=True)
        from .mesh_repair import repair_and_decimate
        
        # Generate, repair, and decimate mesh (blocking CPU operation - run in thread pool)
        shell_data = await asyncio.to_thread(
            repair_and_decimate,
            vdb_grid,
            target_tris=1000,
            voxel_size=voxel_size,
        )
        print(f"  [compile] Shell generated: {len(shell_data)} bytes", flush=True)

        # 5. Write binary
        print(f"  [compile] 5. Writing binary...", flush=True)
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
            shell_data=shell_data
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
