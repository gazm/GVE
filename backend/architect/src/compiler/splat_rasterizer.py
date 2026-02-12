"""
Triplanar Texture Baker - Bakes splat colors into 3 orthogonal textures.

This module implements compile-time baking of Gaussian splat colors into
triplanar textures (XY, XZ, YZ planes) for world-space SDF texturing.

The baked textures replace screen-space splat sampling in `SdfTextured` mode,
providing view-independent coloring with O(1) texture lookups at runtime.

Color pipeline: blend in Oklab (perceptually uniform), output sRGB to binary.
Reference: docs/workflows/compiler-pipeline.md Phase 4
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class SplatBakeMode(Enum):
    """Accumulation strategy for splat baking."""
    GAUSSIAN = "gaussian"  # Weighted average (soft/smooth)
    POINT = "point"        # Max-weight winner-takes-all (faceted/pixel)


@dataclass
class TriplanarTextures:
    """Container for baked triplanar textures."""
    xy: np.ndarray   # (H, W, 4) uint8 RGBA - view from +Z
    xz: np.ndarray   # (H, W, 4) uint8 RGBA - view from +Y
    yz: np.ndarray   # (H, W, 4) uint8 RGBA - view from +X
    bounds_min: np.ndarray  # (3,) float32
    bounds_max: np.ndarray  # (3,) float32
    resolution: int         # Texture resolution (assumed square)





def bake_triplanar_textures_oklab(
    positions: np.ndarray,      # (N, 3) float32
    attrs_oklab: np.ndarray,    # (N, 5) float [L, a, b, metallic, roughness]
    scales: np.ndarray,         # (N, 3) float32
    bounds_min: np.ndarray,     # (3,) float32
    bounds_max: np.ndarray,     # (3,) float32
    resolution: int = 512,
    mode: SplatBakeMode = SplatBakeMode.GAUSSIAN,
) -> TriplanarTextures:
    """
    Bake Oklab attributes into 3 orthogonal textures via Gaussian splatting.
    Blends in Oklab, outputs sRGB uint8. Alpha = (roughness_4bit << 4) | metallic_4bit.
    """
    n_splats = len(positions)
    print(f"      [triplanar] Baking {n_splats} Oklab splats ({mode.value}) to 3×{resolution}×{resolution}...", flush=True)
    bounds_extent = bounds_max - bounds_min
    # Bounds expansion removed to match volume bounds (v2.3)
    # bounds_center = (bounds_min + bounds_max) * 0.5
    # bounds_min = bounds_center - bounds_extent * 0.55
    # bounds_max = bounds_center + bounds_extent * 0.55
    bounds_extent = bounds_max - bounds_min

    normalized = (positions - bounds_min) / (bounds_extent + 1e-8)
    normalized_scales = bounds_extent / (bounds_extent + 1e-8)  # Unused for point mode but kept for compat

    xy_accum = np.zeros((resolution, resolution, 5), dtype=np.float32)
    xz_accum = np.zeros((resolution, resolution, 5), dtype=np.float32)
    yz_accum = np.zeros((resolution, resolution, 5), dtype=np.float32)
    xy_weight = np.zeros((resolution, resolution), dtype=np.float32)
    xz_weight = np.zeros((resolution, resolution), dtype=np.float32)
    yz_weight = np.zeros((resolution, resolution), dtype=np.float32)

    log_interval = max(1, n_splats // 20)
    
    # Point Mode: scales are ignored (except for radius calc, use fixed small splat size maybe?)
    # Or reuse scales for influence radius?
    normalized_scales = scales / (bounds_extent + 1e-8)

    for i in range(n_splats):

        if log_interval and (i + 1) % log_interval == 0:
            print(f"      [triplanar] Baking Oklab splat {i + 1}/{n_splats}...", flush=True)
        px, py, pz = normalized[i]
        sx, sy, sz = normalized_scales[i]
        L = np.clip(attrs_oklab[i, 0], 0.0, 1.0)
        a = np.clip(attrs_oklab[i, 1], -0.4, 0.4)
        b = np.clip(attrs_oklab[i, 2], -0.4, 0.4)
        roughness = np.clip(attrs_oklab[i, 4], 0.0, 1.0)
        metallic = np.clip(attrs_oklab[i, 3], 0.0, 1.0)

        if mode == SplatBakeMode.POINT:
            _splat_point_2d_oklab(xy_accum, xy_weight, px, py, sx, sy, L, a, b, roughness, metallic, resolution)
            _splat_point_2d_oklab(xz_accum, xz_weight, px, pz, sx, sz, L, a, b, roughness, metallic, resolution)
            _splat_point_2d_oklab(yz_accum, yz_weight, py, pz, sy, sz, L, a, b, roughness, metallic, resolution)
        else:
            _splat_gaussian_2d_oklab(xy_accum, xy_weight, px, py, sx, sy, L, a, b, roughness, metallic, resolution)
            _splat_gaussian_2d_oklab(xz_accum, xz_weight, px, pz, sx, sz, L, a, b, roughness, metallic, resolution)
            _splat_gaussian_2d_oklab(yz_accum, yz_weight, py, pz, sy, sz, L, a, b, roughness, metallic, resolution)

    xy = _finalize_texture_oklab(xy_accum, xy_weight)
    xz = _finalize_texture_oklab(xz_accum, xz_weight)
    yz = _finalize_texture_oklab(yz_accum, yz_weight)
    
    # Fill logic (same for both modes, though point mode should have full coverage ideally)
    _fill_empty_triplanar_texels(xy, xy_weight > 0.001, use_oklab_fill=True)
    _fill_empty_triplanar_texels(xz, xz_weight > 0.001, use_oklab_fill=True)
    _fill_empty_triplanar_texels(yz, yz_weight > 0.001, use_oklab_fill=True)

    return TriplanarTextures(
        xy=xy, xz=xz, yz=yz,
        bounds_min=bounds_min.astype(np.float32),
        bounds_max=bounds_max.astype(np.float32),
        resolution=resolution,
    )


def _splat_gaussian_2d_oklab(
    accum: np.ndarray,
    weight: np.ndarray,
    cx: float, cy: float,
    sx: float, sy: float,
    L: float, a: float, b: float, roughness: float, metallic: float,
    resolution: int,
) -> None:
    """Splat Oklab (L,a,b), roughness, metallic using Gaussian-weighted accumulation."""
    px = cx * resolution
    py = cy * resolution
    rx = max(sx * resolution * 3, 1.5)
    ry = max(sy * resolution * 3, 1.5)
    x0 = max(0, int(px - rx))
    x1 = min(resolution, int(px + rx) + 1)
    y0 = max(0, int(py - ry))
    y1 = min(resolution, int(py + ry) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    for yi in range(y0, y1):
        for xi in range(x0, x1):
            dx = (xi + 0.5 - px) / max(rx, 0.5)
            dy = (yi + 0.5 - py) / max(ry, 0.5)
            d2 = dx * dx + dy * dy
            if d2 > 1.0:
                continue
            w = np.exp(-d2 * _SPLAT_GAUSSIAN_EXP)
            accum[yi, xi, 0] += L * w
            accum[yi, xi, 1] += a * w
            accum[yi, xi, 2] += b * w
            accum[yi, xi, 3] += roughness * w
            accum[yi, xi, 4] += metallic * w
            weight[yi, xi] += w


def _splat_point_2d_oklab(
    accum: np.ndarray,
    weight: np.ndarray,
    cx: float, cy: float,
    sx: float, sy: float,
    L: float, a: float, b: float, roughness: float, metallic: float,
    resolution: int,
) -> None:
    """Splat Oklab (L,a,b) using Max-Weight (Voronoi) accumulation."""
    px = cx * resolution
    py = cy * resolution
    # Use fixed small radius for point mode to ensure sharp edges if density is high,
    # or use scale-based radius if we want "large crystals".
    # For "Voronoi" look, we usually want the splat to cover its cell.
    # So we use the same radius logic as Gaussian to cover holes.
    rx = max(sx * resolution * 3, 1.5)
    ry = max(sy * resolution * 3, 1.5)
    
    x0 = max(0, int(px - rx))
    x1 = min(resolution, int(px + rx) + 1)
    y0 = max(0, int(py - ry))
    y1 = min(resolution, int(py + ry) + 1)
    
    if x0 >= x1 or y0 >= y1:
        return

    for yi in range(y0, y1):
        for xi in range(x0, x1):
            dx = (xi + 0.5 - px) / max(rx, 0.5)
            dy = (yi + 0.5 - py) / max(ry, 0.5)
            d2 = dx * dx + dy * dy
            if d2 > 1.0:
                continue
            
            # Weight falls off with distance (creating Voronoi cells where perpendicular bisector usually separates)
            # We use same Gaussian weight curve for consistency
            w = np.exp(-d2 * _SPLAT_GAUSSIAN_EXP)
            
            # Check against existing max weight at this pixel
            if w > weight[yi, xi]:
                weight[yi, xi] = w
                # We store pre-multiplied color because _finalize divides by weight
                # If we store pure L, then finalize(L/w) will explode.
                # So we store L*w, and finalize retrieves (L*w)/w = L. Correct.
                accum[yi, xi, 0] = L * w
                accum[yi, xi, 1] = a * w
                accum[yi, xi, 2] = b * w
                accum[yi, xi, 3] = roughness * w
                accum[yi, xi, 4] = metallic * w





def _finalize_texture(
    accum: np.ndarray, weight: np.ndarray, *, alpha_is_roughness: bool = False
) -> np.ndarray:
    """Normalize accumulated colors and convert to uint8 RGBA. Uses rounding for unbiased quantization."""
    result = np.zeros_like(accum, dtype=np.uint8)
    mask = weight > 0.001

    for c in range(3):
        val = accum[:, :, c][mask] / weight[mask]
        result[:, :, c][mask] = np.round(np.clip(val, 0, 255)).astype(np.uint8)

    if alpha_is_roughness:
        # Legacy path (unused now)
        val = (accum[:, :, 3][mask] / weight[mask]) * 255
        result[:, :, 3][mask] = np.round(np.clip(val, 0, 255)).astype(np.uint8)
    else:
        val = weight[mask] * 255
        result[:, :, 3][mask] = np.round(np.clip(val, 0, 255)).astype(np.uint8)

    return result


def _fill_empty_triplanar_texels(
    tex: np.ndarray, written_mask: np.ndarray, *, use_oklab_fill: bool = False
) -> None:
    """Fill unwritten pixels with average of written pixels so no sample is pure black."""
    if np.all(written_mask):
        return
    empty = ~written_mask
    if np.any(written_mask):
        if use_oklab_fill:
            from .oklab import srgb_to_oklab, oklab_to_srgb
            written_rgb = tex[:, :, :3][written_mask].astype(np.float32) / 255.0
            oklab = srgb_to_oklab(torch.from_numpy(written_rgb).float())
            mean_oklab = oklab.mean(dim=0)
            mean_srgb = oklab_to_srgb(mean_oklab.unsqueeze(0))[0].numpy()
            mean_rgb = np.round(np.clip(mean_srgb * 255, 0, 255)).astype(np.uint8)
        else:
            mean_rgb = np.round(np.array([
                tex[:, :, 0][written_mask].mean(),
                tex[:, :, 1][written_mask].mean(),
                tex[:, :, 2][written_mask].mean(),
            ], dtype=np.float32)).clip(0, 255).astype(np.uint8)
    else:
        mean_rgb = np.array([128, 128, 128], dtype=np.uint8)
    for c in range(3):
        tex[:, :, c][empty] = mean_rgb[c]
    tex[:, :, 3][empty] = 255


# Kernel radius for voxel splat (1 = 3x3) so coverage is higher and fill doesn't dominate
_VOXEL_SPLAT_RADIUS = 1
# Shared Gaussian splat params (numpy and CUDA paths must match).
# Larger radius = more overlap on flat axis-aligned surfaces (e.g. top slide), fewer visible bands.
_SPLAT_GAUSSIAN_EXP = 8
_SPLAT_RADIUS_TEXELS = 12
# Bias baked roughness down so wide splats don't wash out shine (weighted avg pulls in cavity/edge roughness).
_ROUGHNESS_SHINE_BIAS = 1.25  # rough_final = rough_avg ** this; >1 => shinier


def _splat_at_oklab(
    accum: np.ndarray,
    weight: np.ndarray,
    cx: int,
    cy: int,
    L: float,
    a: float,
    b: float,
    roughness: float,
    w: float,
    res: int,
) -> None:
    """Splat Oklab (L,a,b) and roughness with blend weight w at (cx, cy) and neighbors."""
    Lw, aw, bw = L * w, a * w, b * w
    rough_w = roughness * w
    metal_w = metallic * w
    for dy in range(-_VOXEL_SPLAT_RADIUS, _VOXEL_SPLAT_RADIUS + 1):
        for dx in range(-_VOXEL_SPLAT_RADIUS, _VOXEL_SPLAT_RADIUS + 1):
            yi = cy + dy
            xi = cx + dx
            if 0 <= yi < res and 0 <= xi < res:
                accum[yi, xi, 0] += Lw
                accum[yi, xi, 1] += aw
                accum[yi, xi, 2] += bw
                accum[yi, xi, 3] += rough_w
                accum[yi, xi, 4] += metal_w
                weight[yi, xi] += w


def _finalize_texture_oklab(accum: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Blend Oklab accum, convert to sRGB, output uint8 RGBA. Alpha = (roughness << 4) | metallic."""
    from .oklab import oklab_to_srgb

    result = np.zeros((*accum.shape[:2], 4), dtype=np.uint8)
    mask = weight > 0.001

    L = accum[:, :, 0][mask] / weight[mask]
    a = accum[:, :, 1][mask] / weight[mask]
    b = accum[:, :, 2][mask] / weight[mask]
    rough = accum[:, :, 3][mask] / weight[mask]
    rough = np.power(np.clip(rough, 0.0, 1.0), _ROUGHNESS_SHINE_BIAS)
    metal = accum[:, :, 4][mask] / weight[mask]
    metal = np.clip(metal, 0.0, 1.0)
    
    # 4-bit packing: 0..15
    rough_4 = np.round(rough * 15).astype(np.uint8)
    metal_4 = np.round(metal * 15).astype(np.uint8)
    packed_alpha = (rough_4 << 4) | metal_4

    oklab = torch.stack([
        torch.from_numpy(L.flatten()).float().clamp(0, 1),
        torch.from_numpy(a.flatten()).float().clamp(-0.4, 0.4),
        torch.from_numpy(b.flatten()).float().clamp(-0.4, 0.4),
    ], dim=1)
    srgb = oklab_to_srgb(oklab).numpy()
    result[:, :, 0][mask] = np.round(np.clip(srgb[:, 0] * 255, 0, 255)).astype(np.uint8)
    result[:, :, 1][mask] = np.round(np.clip(srgb[:, 1] * 255, 0, 255)).astype(np.uint8)
    result[:, :, 2][mask] = np.round(np.clip(srgb[:, 2] * 255, 0, 255)).astype(np.uint8)
    result[:, :, 3][mask] = packed_alpha

    return result


def _bake_one_plane_cuda(
    cx: torch.Tensor,
    cy: torch.Tensor,
    attrs: torch.Tensor,
    res: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bake one triplanar plane (e.g. XY) on CUDA with batched Gaussian splat. Returns (accum, weight) as (res*res, 5) and (res*res). Uses _SPLAT_RADIUS_TEXELS and _SPLAT_GAUSSIAN_EXP."""
    # (cx, cy) in [0,1]; convert to texel space
    px = cx * res
    py = cy * res
    rx = max(float(_SPLAT_RADIUS_TEXELS), 1.5)
    # Offsets -radius .. +radius
    off = torch.arange(-_SPLAT_RADIUS_TEXELS, _SPLAT_RADIUS_TEXELS + 1, device=device, dtype=torch.float32)
    # 2D offset grid (169 for radius 6)
    dy, dx = torch.meshgrid(off, off, indexing="ij")
    dx = dx.flatten()  # (169,)
    dy = dy.flatten()
    # For each voxel: (N, 169) texel indices
    xi = (px.unsqueeze(1) + dx.unsqueeze(0)).round().long().clamp(0, res - 1)
    yi = (py.unsqueeze(1) + dy.unsqueeze(0)).round().long().clamp(0, res - 1)
    # Gaussian weight: same formula as _splat_gaussian_2d_oklab (see _SPLAT_GAUSSIAN_EXP)
    px_ = px.unsqueeze(1)
    py_ = py.unsqueeze(1)
    d2 = ((xi.float() + 0.5 - px_) / rx).pow(2) + ((yi.float() + 0.5 - py_) / rx).pow(2)
    w = torch.exp(-d2 * _SPLAT_GAUSSIAN_EXP)
    w = torch.where(d2 <= 1.0, w, torch.zeros_like(w))
    index_flat = yi * res + xi  # (N, 169)
    # value (N, 169, 4): L*w, a*w, b*w, rough*w
    L = attrs[:, 0].clamp(0.0, 1.0).unsqueeze(1)
    a = attrs[:, 1].clamp(-0.4, 0.4).unsqueeze(1)
    b = attrs[:, 2].clamp(-0.4, 0.4).unsqueeze(1)
    rough = attrs[:, 4].clamp(0.0, 1.0).unsqueeze(1)
    metal = attrs[:, 3].clamp(0.0, 1.0).unsqueeze(1)
    value = torch.stack([L * w, a * w, b * w, rough * w, metal * w], dim=2)  # (N, 169, 5)
    # Flatten
    index_flat = index_flat.reshape(-1)
    value = value.reshape(-1, 5)
    # Scatter add
    accum = torch.zeros(res * res, 5, device=device, dtype=torch.float32)
    weight_accum = torch.zeros(res * res, device=device, dtype=torch.float32)
    accum.scatter_add_(0, index_flat.unsqueeze(1).expand(-1, 5), value)
    weight_accum.scatter_add_(0, index_flat, w.reshape(-1))
    return accum, weight_accum


def _bake_one_plane_cuda_point(
    cx: torch.Tensor,
    cy: torch.Tensor,
    attrs: torch.Tensor,
    res: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bake one triplanar plane (e.g. XY) on CUDA using Point/Voronoi logic.
    Instead of summing weights, we take the color contributing the max weight per pixel.
    """
    # 1. Generate point cloud of all potential texel writes
    px = cx * res
    py = cy * res
    rx = max(float(_SPLAT_RADIUS_TEXELS), 1.5)
    
    # Offsets (-R..R)
    off = torch.arange(-_SPLAT_RADIUS_TEXELS, _SPLAT_RADIUS_TEXELS + 1, device=device, dtype=torch.float32)
    dy, dx = torch.meshgrid(off, off, indexing="ij")
    dx = dx.flatten()
    dy = dy.flatten()
    
    # Broadcast to (N, 169)
    # xi: (N, 169)
    xi = (px.unsqueeze(1) + dx.unsqueeze(0)).round().long()
    yi = (py.unsqueeze(1) + dy.unsqueeze(0)).round().long()
    
    # Filter out-of-bounds
    mask = (xi >= 0) & (xi < res) & (yi >= 0) & (yi < res)
    
    # Calculate weights (N, 169)
    px_ = px.unsqueeze(1)
    py_ = py.unsqueeze(1)
    d2 = ((xi.float() + 0.5 - px_) / rx).pow(2) + ((yi.float() + 0.5 - py_) / rx).pow(2)
    w = torch.exp(-d2 * _SPLAT_GAUSSIAN_EXP)
    mask = mask & (d2 <= 1.0)
    
    # Flatten valid entries
    valid_indices = torch.where(mask)  # tuple of (n_idx, k_idx)
    n_idx = valid_indices[0]
    k_idx = valid_indices[1]
    
    flat_xi = xi[mask] # (M,)
    flat_yi = yi[mask] # (M,)
    flat_w = w[mask]   # (M,)
    
    # Pixel index flat (0..res*res)
    pixel_idx = flat_yi * res + flat_xi
    
    # 2. Sort by (pixel_idx, weight) -> robust way to find max weight per pixel
    # We want to keep the entry with largest 'flat_w' for each 'pixel_idx'.
    # Lexical sort: sort by pixel_idx first, then by weight.
    # Since we can't easily do lexsort in one go with raw tensors, we can pack them or sort twice.
    # Stable sort by weight, then sort by pixel_idx.
    
    # Sort by weight ASC
    sort_w_idx = torch.argsort(flat_w)
    pixel_idx_sorted = pixel_idx[sort_w_idx]
    
    # Sort by pixel_idx ASC (stable=True keeps weight order relative)
    sort_p_idx = torch.argsort(pixel_idx_sorted, stable=True)
    
    # Final combined permutation
    perm = sort_w_idx[sort_p_idx]
    
    pixel_idx_sorted = pixel_idx[perm]
    # flat_w_sorted = flat_w[perm]
    
    # 3. Unique consecutive on pixel_idx
    # The LAST entry for each unique pixel_idx is the one with the highest weight
    # because of the previous stable sort by weight.
    unique_pixels, inverse_indices, counts = torch.unique_consecutive(pixel_idx_sorted, return_inverse=True, return_counts=True)
    
    # Indices of the winners in the 'perm' array
    # cumulative sum of counts gives the end boundaries. separate - 1 gives usage indices?
    # No, unique_consecutive returns the first occurrence if we don't be careful.
    # Actually, `unique_consecutive` collapses. We want the indices of the *last* element of each run.
    run_ends = torch.cumsum(counts, dim=0) - 1
    winner_indices_in_perm = run_ends
    
    winner_perm_indices = perm[winner_indices_in_perm]
    
    # 4. Gather data for winners
    winner_n_idx = n_idx[winner_perm_indices]
    winner_w = flat_w[winner_perm_indices]
    winner_pixel_idx = flat_yi[winner_perm_indices] * res + flat_xi[winner_perm_indices]
    
    # Get Attributes only for winners
    win_L = attrs[winner_n_idx, 0]
    win_a = attrs[winner_n_idx, 1]
    win_b = attrs[winner_n_idx, 2]
    win_rough = attrs[winner_n_idx, 4]
    win_metal = attrs[winner_n_idx, 3]
    
    # 5. Scatter to output
    # accum stores L*w, etc.
    accum = torch.zeros(res * res, 5, device=device, dtype=torch.float32)
    weight_accum = torch.zeros(res * res, device=device, dtype=torch.float32)
    
    # We use simple indexing instead of scatter_add because indices are unique now
    accum[winner_pixel_idx, 0] = win_L * winner_w
    accum[winner_pixel_idx, 1] = win_a * winner_w
    accum[winner_pixel_idx, 2] = win_b * winner_w
    accum[winner_pixel_idx, 3] = win_rough * winner_w
    accum[winner_pixel_idx, 4] = win_metal * winner_w
    weight_accum[winner_pixel_idx] = winner_w
    
    return accum, weight_accum


def bake_triplanar_from_voxel_oklab_cuda(
    positions: torch.Tensor,
    attrs_oklab: torch.Tensor,
    bounds_min: Union[np.ndarray, torch.Tensor],
    bounds_max: Union[np.ndarray, torch.Tensor],
    resolution: int,
    device: Union[str, torch.device],
    mode: SplatBakeMode = SplatBakeMode.GAUSSIAN,
) -> TriplanarTextures:
    """
    Bake voxel Oklab into triplanar on CUDA. positions (N,3), attrs_oklab (N,5) on device.
    Reuses _finalize_texture_oklab for Oklab->sRGB so logic is not duplicated.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    res = resolution
    if isinstance(bounds_min, np.ndarray):
        bounds_min = torch.from_numpy(bounds_min.astype(np.float32)).to(dev)
    if isinstance(bounds_max, np.ndarray):
        bounds_max = torch.from_numpy(bounds_max.astype(np.float32)).to(dev)
    extent = (bounds_max - bounds_min).clamp(min=1e-8)
    normalized = (positions - bounds_min) / extent
    normalized = normalized.clamp(0.0, 1.0)

    # XY / XZ / YZ: same algorithm as numpy path, batched on GPU
    fn = _bake_one_plane_cuda_point if mode == SplatBakeMode.POINT else _bake_one_plane_cuda
    
    axy, wxy = fn(normalized[:, 0], normalized[:, 1], attrs_oklab, res, dev)
    axz, wxz = fn(normalized[:, 0], normalized[:, 2], attrs_oklab, res, dev)
    ayz, wyz = fn(normalized[:, 1], normalized[:, 2], attrs_oklab, res, dev)

    # Reuse existing finalize + fill (no duplicated Oklab->sRGB logic)
    xy_accum_np = axy.cpu().numpy().reshape(res, res, 5)
    xy_weight_np = wxy.cpu().numpy().reshape(res, res)
    xz_accum_np = axz.cpu().numpy().reshape(res, res, 5)
    xz_weight_np = wxz.cpu().numpy().reshape(res, res)
    yz_accum_np = ayz.cpu().numpy().reshape(res, res, 5)
    yz_weight_np = wyz.cpu().numpy().reshape(res, res)

    xy = _finalize_texture_oklab(xy_accum_np, xy_weight_np)
    xz = _finalize_texture_oklab(xz_accum_np, xz_weight_np)
    yz = _finalize_texture_oklab(yz_accum_np, yz_weight_np)
    _fill_empty_triplanar_texels(xy, xy_weight_np > 0.001, use_oklab_fill=True)
    _fill_empty_triplanar_texels(xz, xz_weight_np > 0.001, use_oklab_fill=True)
    _fill_empty_triplanar_texels(yz, yz_weight_np > 0.001, use_oklab_fill=True)

    bmin_np = bounds_min.cpu().numpy() if bounds_min.is_cuda else np.array(bounds_min)
    bmax_np = bounds_max.cpu().numpy() if bounds_max.is_cuda else np.array(bounds_max)
    return TriplanarTextures(
        xy=xy, xz=xz, yz=yz,
        bounds_min=bmin_np.astype(np.float32),
        bounds_max=bmax_np.astype(np.float32),
        resolution=resolution,
    )


def _splat_at(
    accum: np.ndarray,
    weight: np.ndarray,
    cx: int,
    cy: int,
    r: float,
    g: float,
    b: float,
    roughness: float,
    metallic: float,
    w: float,
    res: int,
) -> None:
    """Splat color (r,g,b) and roughness with blend weight w at (cx, cy) and neighbors."""
    rw, gw, bw = r * w, g * w, b * w
    rough_w = roughness * w
    metal_w = metallic * w
    for dy in range(-_VOXEL_SPLAT_RADIUS, _VOXEL_SPLAT_RADIUS + 1):
        for dx in range(-_VOXEL_SPLAT_RADIUS, _VOXEL_SPLAT_RADIUS + 1):
            yi = cy + dy
            xi = cx + dx
            if 0 <= yi < res and 0 <= xi < res:
                accum[yi, xi, 0] += rw
                accum[yi, xi, 1] += gw
                accum[yi, xi, 2] += bw
                accum[yi, xi, 3] += rough_w
                accum[yi, xi, 4] += metal_w
                weight[yi, xi] += w


def bake_triplanar_from_voxel_oklab(
    positions: np.ndarray,      # (N, 3) float32 voxel center positions
    attrs_oklab: np.ndarray,    # (N, 5) float [L, a, b, metallic, roughness]
    bounds_min: np.ndarray,     # (3,) float32
    bounds_max: np.ndarray,     # (3,) float32
    resolution: int = 512,
    device: str = "cpu",
    mode: SplatBakeMode = SplatBakeMode.GAUSSIAN,
) -> TriplanarTextures:
    """
    Bake voxel Oklab attributes into 3 orthogonal textures. Blends in Oklab,
    outputs sRGB uint8. Alpha channel stores (roughness << 4) | metallic.
    When device=="cuda" and torch.cuda.is_available(), runs accumulation on GPU (same algorithm).
    """
    use_cuda = device == "cuda" and torch.cuda.is_available()
    if use_cuda:
        pos_t = torch.from_numpy(positions.astype(np.float32)).cuda()
        attrs_t = torch.from_numpy(attrs_oklab.astype(np.float32)).cuda()
        return bake_triplanar_from_voxel_oklab_cuda(
            pos_t, attrs_t, bounds_min, bounds_max, resolution=resolution, device="cuda", mode=mode
        )
    bounds_extent = bounds_max - bounds_min
    bounds_extent = np.maximum(bounds_extent, 1e-8)
    normalized = (positions - bounds_min) / bounds_extent

    xy_accum = np.zeros((resolution, resolution, 5), dtype=np.float32)
    xz_accum = np.zeros((resolution, resolution, 5), dtype=np.float32)
    yz_accum = np.zeros((resolution, resolution, 5), dtype=np.float32)
    xy_weight = np.zeros((resolution, resolution), dtype=np.float32)
    xz_weight = np.zeros((resolution, resolution), dtype=np.float32)
    yz_weight = np.zeros((resolution, resolution), dtype=np.float32)

    n = len(positions)
    res = resolution
    # Gaussian splat scale so each voxel spreads over ~6 texels; reduces grid pattern from voxel quantization
    splat_scale = 4.0 / res
    log_interval = max(1, n // 20)
    for i in range(n):
        if log_interval and (i + 1) % log_interval == 0:
            print(f"      [triplanar] Voxel Oklab {i + 1}/{n}...", flush=True)
        px, py, pz = np.clip(normalized[i], 0.0, 1.0)
        L = np.clip(attrs_oklab[i, 0], 0.0, 1.0)
        a = np.clip(attrs_oklab[i, 1], -0.4, 0.4)
        b = np.clip(attrs_oklab[i, 2], -0.4, 0.4)
        roughness = np.clip(attrs_oklab[i, 4], 0.0, 1.0)
        if mode == SplatBakeMode.POINT:
            _splat_point_2d_oklab(xy_accum, xy_weight, px, py, splat_scale, splat_scale, L, a, b, roughness, metallic, resolution)
            _splat_point_2d_oklab(xz_accum, xz_weight, px, pz, splat_scale, splat_scale, L, a, b, roughness, metallic, resolution)
            _splat_point_2d_oklab(yz_accum, yz_weight, py, pz, splat_scale, splat_scale, L, a, b, roughness, metallic, resolution)
        else:
            _splat_gaussian_2d_oklab(xy_accum, xy_weight, px, py, splat_scale, splat_scale, L, a, b, roughness, metallic, resolution)
            _splat_gaussian_2d_oklab(xz_accum, xz_weight, px, pz, splat_scale, splat_scale, L, a, b, roughness, metallic, resolution)
            _splat_gaussian_2d_oklab(yz_accum, yz_weight, py, pz, splat_scale, splat_scale, L, a, b, roughness, metallic, resolution)

    xy = _finalize_texture_oklab(xy_accum, xy_weight)
    xz = _finalize_texture_oklab(xz_accum, xz_weight)
    yz = _finalize_texture_oklab(yz_accum, yz_weight)
    _fill_empty_triplanar_texels(xy, xy_weight > 0.001, use_oklab_fill=True)
    _fill_empty_triplanar_texels(xz, xz_weight > 0.001, use_oklab_fill=True)
    _fill_empty_triplanar_texels(yz, yz_weight > 0.001, use_oklab_fill=True)
    return TriplanarTextures(
        xy=xy, xz=xz, yz=yz,
        bounds_min=bounds_min.astype(np.float32),
        bounds_max=bounds_max.astype(np.float32),
        resolution=resolution,
    )





def pack_triplanar_textures(textures: TriplanarTextures) -> bytes:
    """
    Pack triplanar textures into binary format for .gve_bin.
    
    Format:
        Header (28 bytes):
            magic: [u8; 4]     "TRI1"
            resolution: u32
            bounds_min: [f32; 3]
            bounds_max: [f32; 3]
        Data:
            xy_data: [u8; resolution * resolution * 4]
            xz_data: [u8; resolution * resolution * 4]
            yz_data: [u8; resolution * resolution * 4]
    
    Returns:
        Binary data ready for embedding in .gve_bin
    """
    import struct
    
    header = struct.pack(
        "<4sI3f3f",
        b"TRI1",
        textures.resolution,
        *textures.bounds_min.tolist(),
        *textures.bounds_max.tolist(),
    )
    
    # Pack textures as raw RGBA bytes
    return (
        header +
        textures.xy.tobytes() +
        textures.xz.tobytes() +
        textures.yz.tobytes()
    )


def bake_from_splat_data(
    positions: np.ndarray,
    scales: np.ndarray,
    colors: np.ndarray,
    bounds_min: Optional[np.ndarray] = None,
    bounds_max: Optional[np.ndarray] = None,
    resolution: int = 512,
) -> bytes:
    """
    High-level API: Bake splat data into packed triplanar texture bytes.
    
    Args:
        positions: (N, 3) float32 splat centers
        scales: (N, 3) float32 ellipsoid radii
        colors: (N, 4) uint8 RGBA colors
        bounds_min: Optional scene bounds (auto-computed if None)
        bounds_max: Optional scene bounds (auto-computed if None)
        resolution: Texture resolution (default 512)
    
    Returns:
        Packed binary data for .gve_bin embedding
    """
    # Auto-compute bounds if not provided
    if bounds_min is None:
        bounds_min = positions.min(axis=0) - scales.max(axis=0)
    if bounds_max is None:
        bounds_max = positions.max(axis=0) + scales.max(axis=0)
    
    textures = bake_triplanar_textures(
        positions, colors, scales,
        bounds_min, bounds_max,
        resolution
    )
    
    return pack_triplanar_textures(textures)
