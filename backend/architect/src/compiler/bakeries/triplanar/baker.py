from typing import Dict, Any
import asyncio
import torch
from ..base import Baker
# Avoid circular types
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..base import CompilerContext

class TriplanarBaker(Baker):
    """
    Bakes triplanar textures by raymarching the SDF/Grid.
    """
    def name(self) -> str:
        return "triplanar" # Formerly iso_surface

    async def bake(self, ctx: 'CompilerContext') -> Dict[str, bytes]:
        await ctx.ensure_dense_grid_and_vdb()
        await ctx.ensure_sdf_graph()

        from .generator import bake_procedural_triplanar
        # We need to import BakeResult from volume logic to type hint or reconstruct it?
        # Actually in triplanar_baker.py it expects an object with .dims, .dense_grid etc.
        # The VolumeBaker logic constructs BakeResult.
        # But here we assume ctx has the data. 
        # We can loosely type it or import BakeResult from ...bakeries.volume.generator
        from ..volume.generator import BakeResult

        # Reconstruct BakeResult (TODO: Refactor generator to take ctx directly)
        bake_result = BakeResult(
            vdb_volume=ctx.vdb_volume,
            dense_grid=ctx.dense_grid,
            dims=(ctx.dense_grid.shape[0], ctx.dense_grid.shape[1], ctx.dense_grid.shape[2]),
            bounds_min=tuple(ctx.bounds[0]),
            bounds_max=tuple(ctx.bounds[1]),
            voxel_size=ctx.voxel_size
        )

        # Determine device
        use_cuda = ctx.options.use_cuda_triplanar and torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        
        triplanar_data = await asyncio.to_thread(
            bake_procedural_triplanar,
            sdf_graph=ctx.sdf_graph,
            bounds=ctx.bounds,
            resolution=ctx.options.triplanar_resolution,
            target_sample_count=ctx.options.triplanar_sample_count,
            device=device,
            bake_result=bake_result,
            dna=ctx.dna,
            mode=ctx.options.triplanar_bake_mode
        )
        
        return {
            "triplanar_data": triplanar_data
        }
