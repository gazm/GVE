"""
Compiler Context and Baker Interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from .queue import CompilePriority

# =============================================================================
# Configuration Models
# =============================================================================

class CompilerOptions(BaseModel):
    """Configuration for the compilation process."""
    resolution: int = Field(128, ge=16, le=512, description="Voxel resolution for SDF baking")
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
    
    # Concept/Swatch settings
    use_concept_texture: bool = True
    concept_texture_blend: float = 0.7
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
        from .math_jit import build_sdf_graph
        
        # We run this on the main thread as it constructs PyTorch graph (fast enough)
        self.sdf_graph = build_sdf_graph(self.dna)
        
        # Extract bounds from graph
        if hasattr(self.sdf_graph, "bounds") and self.sdf_graph.bounds:
            self.bounds = self.sdf_graph.bounds
        else:
            self.bounds = ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
            
        # Calculate voxel size based on resolution
        self._calculate_voxel_size()

    def _calculate_voxel_size(self):
        """Calculate voxel size and padding based on bounds and target resolution."""
        import numpy as np
        
        b_min = np.array(self.bounds[0], dtype=np.float32)
        b_max = np.array(self.bounds[1], dtype=np.float32)
        extent = b_max - b_min
        
        longest_axis = float(extent.max())
        target_res = self.options.resolution
        self.voxel_size = longest_axis / target_res
        
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
        from .vdb_converter import bake_sdf
        
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
        from .mesh_repair import repair_and_decimate
        
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
    async def bake(self, ctx: CompilerContext) -> Dict[str, bytes]:
        """
        Produce artifacts.
        Returns:
            Dict[str, bytes]: A mapping of binary slot names to byte content.
            Example: {"splat_data": b'...', "triplanar_data": b'...'}
        """
        pass
