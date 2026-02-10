from typing import Dict, Any
from ..base import Baker
# Avoid circular type hint imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..base import CompilerContext

class VolumeBaker(Baker):
    """
    Bakes the NanoVDB grid and dense grid data.
    """
    def name(self) -> str:
        return "volume"

    async def bake(self, ctx: 'CompilerContext') -> Dict[str, bytes]:
        # Ensure data exists in context
        await ctx.ensure_dense_grid_and_vdb()
        
        # Import moved logic
        from .generator import dense_grid_to_bytes, BakeResult
        
        # Reconstruct a BakeResult 
        bake_result = BakeResult(
            vdb_volume=ctx.vdb_volume,
            dense_grid=ctx.dense_grid,
            dims=(ctx.dense_grid.shape[0], ctx.dense_grid.shape[1], ctx.dense_grid.shape[2]),
            bounds_min=tuple(ctx.bounds[0]),
            bounds_max=tuple(ctx.bounds[1]),
            voxel_size=ctx.voxel_size
        )
        
        volume_data = dense_grid_to_bytes(bake_result)
        
        return {
            "volume_data": volume_data
        }
