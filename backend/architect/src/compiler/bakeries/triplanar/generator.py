"""
Triplanar Baker - Surface sample SDF + attributes, bake to triplanar.

When texture_mode is "procedural_triplanar", the compiler uses this path instead
of Gaussian splat training: sample the SDF surface, get material+procedural
attributes at each point via query_attributes, then bake into triplanar textures.
No optimization step; uses existing procedural libraries only.

Reference: docs/workflows/compiler-pipeline.md (texture mode: procedural_triplanar)
"""

from __future__ import annotations

import copy
import numpy as np
import torch
from typing import Tuple, List, Optional, Any

# Adjusted imports for new location src/compiler/bakeries/triplanar/
from ...splat_rasterizer import (
    TriplanarTextures,
    bake_triplanar_textures_oklab,
    bake_triplanar_from_voxel_oklab,
    pack_triplanar_textures,
    SplatBakeMode,
)

# Chunk size for batch query_attributes on surface voxels
_VOXEL_ATTRS_CHUNK = 50_000

# Default attrs when no material: Orange Oklab [0.75, 0.06, 0.12, 0.0, 0.5]
_DEFAULT_ATTRS = np.array([0.75, 0.06, 0.12, 0.0, 0.3], dtype=np.float32)


def _bake_triplanar_from_voxels(
    bake_result: Any,
    sdf_graph: Any,
    resolution: int,
    surface_tolerance: float,
    dna: Optional[dict] = None,
    device: str = "cpu",
    mode: str = "gaussian",
) -> bytes:
    """
    Color surface voxels via query_attributes and bake into triplanar.
    Uses the existing dense SDF grid; no surface sampling or splat training.
    """
    dense_grid = bake_result.dense_grid  # (nx, ny, nz) float32
    nx, ny, nz = bake_result.dims
    b_min = np.array(bake_result.bounds_min, dtype=np.float32)
    b_max = np.array(bake_result.bounds_max, dtype=np.float32)
    extent = b_max - b_min

    # Surface voxels: |SDF| < tolerance
    mask = np.abs(dense_grid) < surface_tolerance
    if not np.any(mask):
        # Fallback: use a slightly larger tolerance
        surface_tolerance = min(surface_tolerance * 2.0, np.abs(dense_grid).max() * 0.1)
        mask = np.abs(dense_grid) < surface_tolerance
    ii, jj, kk = np.where(mask)
    n_surface = len(ii)
    print(f"      [voxel-triplanar] {n_surface} surface voxels (|SDF| < {surface_tolerance:.6f})", flush=True)

    # World positions: grid index -> world (same as vdb_converter linspace)
    def idx_to_world(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
        x = b_min[0] + (i.astype(np.float32) / max(nx - 1, 1)) * extent[0]
        y = b_min[1] + (j.astype(np.float32) / max(ny - 1, 1)) * extent[1]
        z = b_min[2] + (k.astype(np.float32) / max(nz - 1, 1)) * extent[2]
        return np.stack([x, y, z], axis=1)

    positions_np = idx_to_world(ii, jj, kk)  # (N, 3)

    attrs_fn = sdf_graph if hasattr(sdf_graph, "query_attributes") else None
    if attrs_fn is None:
        # Default Orange Oklab [0.75, 0.06, 0.12, 0.0, 0.5]
        attrs_np = np.tile(_DEFAULT_ATTRS, (n_surface, 1))
    else:
        # Batch query_attributes (CPU or CUDA). On CUDA, TextureModifierNode no-ops so edge_wear/cavity_grime/rust are skipped.
        use_cuda = device == "cuda" and torch.cuda.is_available()
        if use_cuda and hasattr(sdf_graph, "to"):
            attrs_fn = sdf_graph.to("cuda")
        else:
            attrs_fn = sdf_graph
        attrs_list = []
        for start in range(0, n_surface, _VOXEL_ATTRS_CHUNK):
            end = min(start + _VOXEL_ATTRS_CHUNK, n_surface)
            pts = torch.from_numpy(positions_np[start:end]).float()
            if use_cuda:
                pts = pts.cuda()
            with torch.no_grad():
                attrs = attrs_fn.query_attributes(pts)  # (chunk, 5)
            attrs_list.append(attrs.cpu().numpy())
        attrs_np = np.concatenate(attrs_list, axis=0)
        # Diagnostic: are we getting varied attributes per voxel?
        if n_surface >= 2:
            first_row = attrs_np[0]
            same = np.allclose(attrs_np, first_row, atol=1e-5)
            if same:
                print(f"      [voxel-triplanar] ⚠️ query_attributes constant (all voxels same L,a,b)", flush=True)
            else:
                uniq = np.unique(attrs_np[:, :3].round(decimals=4), axis=0)
                print(f"      [voxel-triplanar] query_attributes vary: ~{len(uniq)} distinct (L,a,b)", flush=True)

    # attrs_np: (N, 5) [L, a, b, metallic, roughness] in Oklab. Bake blends in Oklab, outputs sRGB.
    use_cuda = device == "cuda" and torch.cuda.is_available()
    print(
        f"      [voxel-triplanar] Baking {n_surface} voxel Oklab {{'on CUDA' if use_cuda else ''}} to 3×{resolution}×{resolution}...",
        flush=True,
    )
    bake_mode = SplatBakeMode(mode) if mode in ("gaussian", "point") else SplatBakeMode.GAUSSIAN
    textures = bake_triplanar_from_voxel_oklab(
        positions_np, attrs_np, b_min, b_max, resolution=resolution, device=device, mode=bake_mode
    )
    return pack_triplanar_textures(textures)


def bake_procedural_triplanar(
    sdf_graph: Any,
    bounds: Tuple[List[float], List[float]],
    resolution: int = 512,
    target_sample_count: int = 40000,
    device: str = "cpu",
    bake_result: Optional[Any] = None,
    dna: Optional[dict] = None,
    mode: str = "gaussian",
) -> bytes:
    """
    Bake procedural colors into triplanar bytes.

    When bake_result is provided (dense SDF grid from pipeline), colors surface
    voxels via query_attributes and bakes to triplanar (fast, no surface sampling).
    Otherwise falls back to batched surface sampling + Gaussian splat bake.

    Args:
        sdf_graph: SDF module with query_attributes(x) -> (N, 5).
        bounds: (bounds_min, bounds_max) world-space AABB.
        resolution: Triplanar texture resolution (square).
        target_sample_count: Used only when bake_result is None (surface-sample path).
        device: Used only when bake_result is None.
        bake_result: Optional BakeResult from vdb_converter.bake_sdf; when set, use voxel path.
        dna: Optional DNA for material blending.
        mode: Splat bake mode ("gaussian" or "point").

    Returns:
        Packed triplanar binary (TRI1) for .gve_bin embedding.
    """
    if bake_result is not None:
        surface_tolerance = max(bake_result.voxel_size * 0.5, 1e-6)
        return _bake_triplanar_from_voxels(
            bake_result, sdf_graph, resolution, surface_tolerance, dna=dna, device=device, mode=mode
        )

    # Adjusted import for splat trainer
    from ..splat.trainer import initialize_splats_batched

    sdf_fn = sdf_graph
    attrs_fn = sdf_graph if hasattr(sdf_graph, "query_attributes") else None
    if device == "cuda" and hasattr(sdf_graph, "to"):
        # Keep original on CPU for attribute queries; use a GPU copy for batched SDF eval.
        # (Moving the same module to CPU after .to(cuda) would leave sdf_fn on CPU and break device consistency.)
        sdf_fn = copy.deepcopy(sdf_graph).to(device)
        attrs_fn = sdf_graph  # unchanged on CPU; initialize_splats_batched will pass positions.cpu() for attrs

    positions, attrs, avg_spacing = initialize_splats_batched(
        sdf_fn,
        bounds,
        target_count=target_sample_count,
        min_radius=0.02,
        device=device,
        attrs_fn=attrs_fn,
    )

    # attrs: (N, 5) float [L, a, b, metallic, roughness]. Bake in Oklab, output sRGB.
    attrs_np = attrs.detach().cpu().numpy().astype(np.float32)
    pos_np = positions.detach().cpu().numpy().astype(np.float32)
    scale_val = max(avg_spacing * 0.5, 1e-4)
    scales = np.full((len(pos_np), 3), scale_val, dtype=np.float32)
    bmin = np.array(bounds[0], dtype=np.float32)
    bmax = np.array(bounds[1], dtype=np.float32)
    
    bake_mode = SplatBakeMode(mode) if mode in ("gaussian", "point") else SplatBakeMode.GAUSSIAN

    textures = bake_triplanar_textures_oklab(
        pos_np, attrs_np, scales, bmin, bmax, resolution=resolution, mode=bake_mode
    )
    return pack_triplanar_textures(textures)
