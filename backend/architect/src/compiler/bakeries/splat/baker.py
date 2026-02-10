
import asyncio
from typing import Dict, Any, Optional

from ..base import CompilerContext, Baker
from .trainer import compile_splats, compile_splats_swatch_mode

class SplatBaker(Baker):
    """
    Bakes Gaussian Splats (Point Cloud).
    """
    def name(self) -> str:
        return "splat"
        
    async def bake(self, ctx: CompilerContext) -> Dict[str, bytes]:
        # Helper to get safe float from list or scalar
        def _get_extent(bounds):
            b_min = bounds[0]
            b_max = bounds[1]
            return [b_max[i] - b_min[i] for i in range(3)]

        # --- Auto-calculate splat count if missing ---
        splat_count = ctx.options.splat_count
        if splat_count is None:
            extent = _get_extent(ctx.bounds)
            # Surface area approx: 2 * (xy + xz + yz)
            surface_area = 2.0 * (
                extent[0] * extent[1] +
                extent[0] * extent[2] +
                extent[1] * extent[2]
            )
            # Density per m^2
            density_per_m2 = 150000.0
            splat_count = int(surface_area * density_per_m2)
            splat_count = max(10000, min(80000, splat_count))
            print(f"    [SplatBaker] 🧮 Auto splat_count={splat_count} (area={surface_area:.4f} m^2)", flush=True)

        # --- Check for Swatch Mode ---
        if ctx.options.texture_mode == "swatch":
             print(f"    [SplatBaker] Swatch mode active...", flush=True)
             splat_data = await asyncio.to_thread(
                compile_splats_swatch_mode,
                sdf_fn=ctx.sdf_graph,
                bounds=ctx.bounds,
                dna=ctx.dna,
                swatches_per_node=ctx.options.swatches_per_node,
                swatch_scale_factor=ctx.options.swatch_scale_factor
             )
             return {"splat_data": splat_data}

        splat_data = await asyncio.to_thread(
            compile_splats,
            sdf_fn=ctx.sdf_graph,
            bounds=ctx.bounds,
            target_count=splat_count,
            iterations=ctx.options.splat_iterations,
            target_loss=ctx.options.target_rate,
            batch_size=ctx.options.splat_batch_size,
            overlap_interval=ctx.options.splat_overlap_interval,
            accum_steps=ctx.options.splat_accum_steps,
            overlap_batch=ctx.options.splat_overlap_batch,
            overlap_batch_size=ctx.options.splat_overlap_batch_size
        )
        
        return {
            "splat_data": splat_data
        }
