"""
Compiler Context and Baker Interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from ..queue import CompilePriority

# =============================================================================
# Configuration Models
# =============================================================================

class CompilerOptions(BaseModel):
    """Configuration for the compilation process."""
    resolution: int = Field(128, ge=16, le=512, description="Voxel resolution for SDF baking")
    min_voxels_shortest_axis: int = Field(
        20, ge=4, le=128,
        description="Minimum voxels on thinnest axis. Prevents flat rendering of thin objects (e.g. pistols at real-world meter scale).",
    )
    texture_mode: str = Field("procedural_triplanar", pattern="^(dense|swatch|procedural_triplanar)$")
    
    # Triplanar settings
    triplanar_resolution: int = Field(512, ge=64, le=4096)
    triplanar_bake_mode: str = Field("gaussian", pattern="^(gaussian|point)$")
    triplanar_sample_count: int = Field(40000, ge=1000)
    use_cuda_triplanar: bool = True
    
    # Splat settings
    enable_splats: bool = True
    splat_count: Optional[int] = None
    splat_iterations: int = 1000
    target_rate: float = 0.17
    splat_batch_size: Optional[int] = None
    splat_overlap_interval: int = 50
    splat_accum_steps: int = 1
    splat_overlap_batch: bool = True
    splat_overlap_batch_size: Optional[int] = None
    
    
    # Swatch settings
    swatches_per_node: int = 1
    swatch_scale_factor: float = 0.4

    class Config:
        extra = "ignore"  # Allow unknown fields from legacy settings


# =============================================================================
# Compiler Context
# =============================================================================

class CompilerContext:
    """
    The shared state for a single compilation job.
    
    Holds the 'Ground Truth' geometry (SDF Graph) and lazily computed
    heavy intermediates (VDB Volume, Dense Grid, Mesh).
    """

    def __init__(self, dna: Dict[str, Any], settings: Union[Dict, CompilerOptions]):
        self.dna = dna
        if isinstance(settings, dict):
            self.options = CompilerOptions(**settings)
        else:
            self.options = settings

        # --- Geometry Ground Truth (Lazy) ---
        self.sdf_graph: Optional[nn.Module] = None
        
        # --- Heavy Intermediates (Lazy) ---
        self.dense_grid: Optional[Any] = None  # numpy array (H, W, D)
        self.vdb_volume: Optional[Any] = None  # mrmesh.VdbVolume
        self.shell_mesh: Optional[bytes] = None  # GVE Binary Shell Format bytes
        
        # --- Metrics ---
        # (min_xyz, max_xyz) tuples 
        self.bounds: Optional[Tuple[List[float], List[float]]] = None
        self.voxel_size: float = 0.0

    async def ensure_sdf_graph(self):
        """Build the SDF graph from DNA if not already built."""
        if self.sdf_graph is not None:
            return

        print(f"    [Context] Building SDF graph...", flush=True)
        # Avoid circular imports by importing inside method
        from ..math_jit import build_sdf_graph
        
        # We run this on the main thread as it constructs PyTorch graph (fast enough)
        self.sdf_graph = build_sdf_graph(self.dna)
        
        # Extract bounds from graph
        if hasattr(self.sdf_graph, "bounds") and self.sdf_graph.bounds:
            self.bounds = self.sdf_graph.bounds
        else:
            self.bounds = ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
            
        # Calculate voxel size based on resolution
        self._calculate_voxel_size()

    # WebGPU 3D texture hard limit per axis
    MAX_VOXELS_PER_AXIS: int = 256

    def _calculate_voxel_size(self):
        """Calculate voxel size and padding based on bounds and target resolution.

        Applies adaptive resolution: if the shortest axis would have fewer than
        ``min_voxels_shortest_axis`` voxels, the voxel size is reduced so thin
        objects (e.g. a pistol at real-world meter scale) retain visual detail.

        Final voxel size is clamped so no axis exceeds ``MAX_VOXELS_PER_AXIS``
        (WebGPU 3D texture limit = 256 per axis).
        """
        import numpy as np
        
        b_min = np.array(self.bounds[0], dtype=np.float32)
        b_max = np.array(self.bounds[1], dtype=np.float32)
        extent = b_max - b_min
        
        longest_axis = float(extent.max())
        shortest_axis = float(extent.min())
        target_res = self.options.resolution
        self.voxel_size = longest_axis / target_res
        
        # Adaptive: ensure minimum voxels on the thinnest axis
        min_voxels = self.options.min_voxels_shortest_axis
        if shortest_axis > 1e-6 and min_voxels > 0:
            detail_voxel_size = shortest_axis / min_voxels
            if detail_voxel_size < self.voxel_size:
                print(
                    f"    [Context] Adaptive voxel: shortest axis {shortest_axis:.4f}m "
                    f"needs {min_voxels} voxels -> voxel_size {detail_voxel_size:.5f}m",
                    flush=True,
                )
                self.voxel_size = detail_voxel_size

        # Clamp: no axis may exceed MAX_VOXELS_PER_AXIS (WebGPU 3D texture limit)
        max_limit = self.MAX_VOXELS_PER_AXIS
        min_voxel_for_limit = longest_axis / max_limit
        if self.voxel_size < min_voxel_for_limit:
            print(
                f"    [Context] Clamped voxel_size {self.voxel_size:.5f}m -> "
                f"{min_voxel_for_limit:.5f}m (longest axis {longest_axis:.3f}m "
                f"capped to {max_limit} voxels)",
                flush=True,
            )
            self.voxel_size = min_voxel_for_limit

        # Log final resolution estimate
        est_res = [int(e / self.voxel_size) for e in extent.tolist()]
        print(
            f"    [Context] Final voxel_size={self.voxel_size:.5f}m, "
            f"est grid ~{est_res[0]}x{est_res[1]}x{est_res[2]}",
            flush=True,
        )

        # Add padding: 10% or 3 voxels
        padding = np.maximum(extent * 0.1, self.voxel_size * 3)
        b_min -= padding
        b_max += padding
        
        self.bounds = (b_min.tolist(), b_max.tolist())

    async def ensure_dense_grid_and_vdb(self):
        """
        Bake the SDF graph into a Dense Grid and NanoVDB volume.
        Populates self.dense_grid and self.vdb_volume.
        """
        if self.dense_grid is not None and self.vdb_volume is not None:
            return

        await self.ensure_sdf_graph()
        
        import asyncio
        from .volume.generator import bake_sdf
        
        print(f"    [Context] Baking SDF to VDB/Grid (res={self.options.resolution})...", flush=True)
        
        # Blocking CPU/GPU operation -> Run in thread
        bake_result = await asyncio.to_thread(
            bake_sdf,
            self.sdf_graph,
            bounds_min=tuple(self.bounds[0]),
            bounds_max=tuple(self.bounds[1]),
            voxel_size=self.voxel_size
        )
        
        self.dense_grid = bake_result.dense_grid
        self.vdb_volume = bake_result.vdb_volume

    async def ensure_shell_mesh(self):
        """
        Generate the shell mesh from the VDB volume.
        Populates self.shell_mesh (bytes).
        """
        if self.shell_mesh is not None:
            return

        await self.ensure_dense_grid_and_vdb()
        
        import asyncio
        from .mesh.generator import repair_and_decimate
        
        print(f"    [Context] Generating shell mesh...", flush=True)
        
        # Blocking CPU operation -> Run in thread
        self.shell_mesh = await asyncio.to_thread(
            repair_and_decimate,
            self.vdb_volume,
            target_tris=5000,
            voxel_size=self.voxel_size,
            bounds_min=tuple(self.bounds[0])
        )


# =============================================================================
# Baker Interface
# =============================================================================

class Baker(ABC):
    """
    A Baker takes a populated CompilerContext and produces specific binary artifacts.
    """
    
    @abstractmethod
    def name(self) -> str:
        """Unique name of this baker (e.g. 'splat', 'mesh')."""
        pass

    @abstractmethod
    async def bake(self, ctx: 'CompilerContext') -> Dict[str, bytes]:
        """
        Produce artifacts.
        Returns:
            Dict[str, bytes]: A mapping of binary slot names to byte content.
            Example: {"splat_data": b'...', "triplanar_data": b'...'}
        """
        pass




