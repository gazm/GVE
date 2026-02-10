"""
Splat Trainer - Gaussian Splatting Generation from SDF.

This module implements Stage 3 of the GVE Compiler Pipeline:
1. Curvature-weighted surface initialization (batched GPU or Poisson CPU).
2. Adam optimisation with eikonal-weighted surface/normal loss and
   voxel-hashed overlap penalty.
3. Post-training Gaussian coverage analysis.

Several improvements are inspired by SplatSDF (arXiv:2411.15468):
- Curvature-guided adaptive densification  (Eq. 9)
- Voxel-hashed spatial overlap detection   (Appendix B)
- Eikonal gradient confidence weighting     (Eq. 8)
- Gaussian-weighted coverage analysis       (Eq. 3)

GPU Acceleration: Uses torch_preloader to auto-detect CUDA and runs all
tensor operations on GPU when available for 10-50x speedup.

Reference: docs/workflows/compiler-pipeline.md §3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import math
import copy
from typing import List, Tuple, Optional, Callable, Union
from dataclasses import dataclass

# Adjusted imports for src/compiler/bakeries/splat/trainer.py
from ....torch_preloader import preloader
from ...math_jit_builder import collect_node_bounds


@dataclass
class SplatData:
    """Container for trained splat data (always Oklab-encoded)."""
    positions: np.ndarray    # (N, 3) float32
    scales: np.ndarray       # (N, 3) float32
    rotations: np.ndarray    # (N, 4) float32 (quaternions)
    colors: np.ndarray       # (N, 4) uint8 — Oklab8+A: [L, a, b, alpha]
    metallic: np.ndarray     # (N,) uint8 (0-255 -> 0.0-1.0)
    roughness: np.ndarray    # (N,) uint8 (0-255 -> 0.0-1.0)
    flags: np.ndarray        # (N,) uint8 (always 0x01 = Oklab)


# ============================================================================
# Surface Projection
# ============================================================================

def _compute_surface_tolerance(
    bounds: Tuple[List[float], List[float]],
    min_radius: Optional[float] = None,
    avg_spacing: Optional[float] = None,
) -> float:
    """Adaptive surface tolerance in SDF units.

    Uses spacing/radius when available, otherwise falls back to bounds size.
    """
    min_xyz, max_xyz = bounds
    extent = [max_xyz[i] - min_xyz[i] for i in range(3)]
    diag = math.sqrt(sum(e * e for e in extent))
    fallback = diag * 0.002
    if avg_spacing is not None and avg_spacing > 0:
        tol = avg_spacing * 0.5
    elif min_radius is not None and min_radius > 0:
        tol = min_radius * 0.5
    else:
        tol = fallback
    tol = max(1e-5, min(tol, fallback * 2.0))
    return float(tol)


def project_to_surface(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    points: torch.Tensor,
    max_steps: int = 10,
    tolerance: float = 1e-4,
) -> torch.Tensor:
    """
    Project points onto the SDF zero-isosurface using gradient descent.
    
    Args:
        sdf_fn: SDF evaluation function (N, 3) -> (N,)
        points: Initial points (N, 3)
        max_steps: Maximum Newton iterations
        tolerance: Convergence threshold
        
    Returns:
        Projected points (N, 3)
    """
    p = points.clone().detach().requires_grad_(True)
    
    for _ in range(max_steps):
        # Compute SDF values
        d = sdf_fn(p)
        
        # Compute gradients (surface normals)
        grads = torch.autograd.grad(
            d.sum(), p, create_graph=False, retain_graph=False
        )[0]
        
        # Normalize gradients
        grad_norm = torch.norm(grads, dim=1, keepdim=True).clamp(min=1e-8)
        normals = grads / grad_norm
        
        # Newton step: move along normal by -d
        step = d.unsqueeze(1) * normals
        p = p.detach() - step
        p.requires_grad_(True)
        
        # Check convergence
        if torch.abs(d).max() < tolerance:
            break
    
    return p.detach()


# ============================================================================
# SDF Curvature & Spatial Overlap (SplatSDF-inspired)
# Reference: arXiv:2411.15468 — Eq. 9 (curvature), Appendix B (sparse KNN)
# ============================================================================

@torch.no_grad()
def compute_curvature(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    positions: torch.Tensor,
    epsilon: float = 0.001,
) -> torch.Tensor:
    """Estimate mean curvature |nabla^2 f(x)| via finite differences.

    High curvature => complex geometry needing denser splat coverage.
    Uses 6 off-axis SDF evaluations (Laplacian approximation).

    Args:
        sdf_fn: SDF evaluation function ``(N, 3) -> (N,)``.
        positions: Surface points ``(N, 3)``.
        epsilon: Finite difference step size.

    Returns:
        Absolute curvature ``(N,)`` -- higher = sharper features.
    """
    f_center = sdf_fn(positions)
    laplacian = torch.zeros(len(positions), device=positions.device)
    for axis in range(3):
        offset = torch.zeros(1, 3, device=positions.device)
        offset[0, axis] = epsilon
        f_plus = sdf_fn(positions + offset)
        f_minus = sdf_fn(positions - offset)
        laplacian += (f_plus + f_minus - 2.0 * f_center) / (epsilon ** 2)
    return torch.abs(laplacian)


def _voxel_overlap_loss(
    positions: torch.Tensor,
    cell_size: float,
    k_check: int = 8,
) -> torch.Tensor:
    """Voxel-hashed overlap penalty using sort-based spatial grouping.

    Sorts positions by spatial hash so same-cell splats become adjacent,
    then checks K nearest sorted neighbors for proximity violations.
    Fully GPU-vectorized, ``O(N*K)`` instead of ``O(N^2)``.

    Args:
        positions: Splat positions ``(N, 3)``, requires_grad for backprop.
        cell_size: Overlap threshold -- penalises pairs closer than this.
        k_check: Number of sorted neighbors to check per splat.

    Returns:
        Scalar overlap penalty (differentiable w.r.t. positions).
    """
    n = len(positions)
    if n < 2:
        return torch.tensor(0.0, device=positions.device)

    # Spatial hash: floor to cell, pack via large primes
    cells = torch.floor(positions.detach() / cell_size).int()
    cell_hash = (
        cells[:, 0].long() * 73856093
        + cells[:, 1].long() * 19349663
        + cells[:, 2].long() * 83492791
    )

    # Sort by hash -> same-cell splats become neighbours in array
    _, sort_idx = torch.sort(cell_hash)
    sorted_pos = positions[sort_idx]  # gradient flows through fancy indexing

    # Compare each splat with K nearest in sorted order
    k = min(k_check, n - 1)
    penalties: list[torch.Tensor] = []
    for offset in range(1, k + 1):
        shifted = torch.roll(sorted_pos, -offset, dims=0)
        dists = torch.norm(sorted_pos - shifted, dim=1)
        # Soft penalty: relu(1 - dist/cell_size) -- 0 far, 1 coincident
        penalties.append(F.relu(1.0 - dists / cell_size))

    return torch.stack(penalties).mean()


def _auto_batch_size(n: int, device: str) -> int:
    """Heuristic batch size for splat training."""
    if device != "cuda":
        return min(n, 8192)
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return min(n, 2048)
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < 6.0:
        return min(n, 1024)
    if free_gb < 10.0:
        return min(n, 2048)
    return min(n, 4096)


# ============================================================================
# Fast Batched Initialization (GPU-optimized)
# ============================================================================

def initialize_splats_batched(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    bounds: Tuple[List[float], List[float]],
    target_count: int = 10000,
    min_radius: float = 0.02,
    device: str = "cpu",
    attrs_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Fast GPU-batched splat initialization using grid sampling + farthest point.
    Returns: (positions, attributes, avg_spacing)
    """
    """
    Fast GPU-batched splat initialization using grid sampling + farthest point.
    
    Strategy:
    1. Generate dense random points in volume (batched)
    2. Project ALL to surface in single batch (GPU parallel)
    3. Filter to near-surface points
    4. Use farthest-point sampling for even spacing
    
    This is 10-50x faster than sequential Poisson on GPU.
    """
    min_xyz = torch.tensor(bounds[0], dtype=torch.float32, device=device)
    max_xyz = torch.tensor(bounds[1], dtype=torch.float32, device=device)
    extent = max_xyz - min_xyz
    surface_tolerance = _compute_surface_tolerance(bounds, min_radius=min_radius)
    print(f"      [batch] 🧭 Surface tol: {surface_tolerance:.6f}", flush=True)
    
    # Generate 5x oversampled random points
    oversample = max(5, int(50000 / target_count))  # More oversampling for small counts
    n_candidates = target_count * oversample
    
    print(f"      [batch] Generating {n_candidates} candidate points...", flush=True)
    
    # Random points in bounding box
    random_points = torch.rand(n_candidates, 3, device=device)
    random_points = min_xyz + random_points * extent
    
    # Project ALL to surface in one batch
    print(f"      [batch] Projecting {n_candidates} points to surface (batched)...", flush=True)
    surface_points = project_to_surface(sdf_fn, random_points, max_steps=10)
    
    # Filter: keep only points that actually reached the surface
    with torch.no_grad():
        distances = sdf_fn(surface_points)
    
    surface_mask = torch.abs(distances) < surface_tolerance
    surface_points = surface_points[surface_mask]
    print(f"      [batch] {surface_points.shape[0]} points on surface", flush=True)
    
    if len(surface_points) < target_count:
        print(f"      [batch] Warning: only {len(surface_points)} surface points, using all", flush=True)
        positions = surface_points
    else:
        # Compute curvature for density-adaptive sampling
        print(f"      [batch] Computing surface curvature for adaptive sampling...", flush=True)
        curvature = compute_curvature(sdf_fn, surface_points)
        high_curv_pct = (curvature > curvature.median()).float().mean().item()
        print(f"      [batch] Curvature range: [{curvature.min():.2f}, {curvature.max():.2f}], "
              f"{high_curv_pct:.0%} above median", flush=True)

        # Curvature-weighted farthest point sampling
        print(f"      [batch] Curvature-weighted FPS: {target_count} from {len(surface_points)}...", flush=True)
        positions = farthest_point_sample_weighted(surface_points, curvature, target_count)
    
    # Query material attributes (color[3] + metallic + roughness = 5 channels)
    attrs_source = attrs_fn if attrs_fn is not None else sdf_fn
    if hasattr(attrs_source, "query_attributes"):
        if device == "cuda" and attrs_fn is not None:
            attrs = attrs_source.query_attributes(positions.cpu()).to(device)
        else:
            attrs = attrs_source.query_attributes(positions)
    else:
        default = torch.tensor([0.627, 0.0, 0.0, 0.0, 0.5], dtype=torch.float32, device=device)
        attrs = default.unsqueeze(0).expand(len(positions), 5)
    
    # Compute Average Spacing (KNN-1)
    # Use a subset for speed if n is large
    print(f"      [batch] Computing average spacing for initial scale...", flush=True)
    with torch.no_grad():
        # Sample random subset of 1000 points to estimate density
        n_est = min(len(positions), 1000)
        subset = positions[:n_est]
        # Dist matrix (n_est, n_est)
        dists = torch.cdist(subset, subset)
        # Set diagonal to inf
        dists.fill_diagonal_(float('inf'))
        # Nearest neighbor dist
        nn_dists = dists.min(dim=1).values
        avg_spacing = nn_dists.mean().item()
        print(f"      [batch] Avg spacing: {avg_spacing:.5f}", flush=True)

    print(f"    [splat_trainer] Batch init: {len(positions)} splats on {device} (spacing={avg_spacing:.4f})", flush=True)
    return positions, attrs, avg_spacing


def farthest_point_sample(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Farthest point sampling - select well-spaced subset of points.
    
    O(n * n_samples) but fully vectorized on GPU.
    """
    device = points.device
    n_points = len(points)
    
    if n_points <= n_samples:
        return points
    
    # Start with random point
    selected_indices = torch.zeros(n_samples, dtype=torch.long, device=device)
    selected_indices[0] = torch.randint(n_points, (1,), device=device)
    
    # Track min distance to any selected point
    min_distances = torch.full((n_points,), float('inf'), device=device)
    
    for i in range(1, n_samples):
        # Update distances with last selected point
        last_selected = points[selected_indices[i-1]]
        distances = torch.norm(points - last_selected, dim=1)
        min_distances = torch.minimum(min_distances, distances)
        
        # Select farthest point
        selected_indices[i] = torch.argmax(min_distances)
        
        # Progress for large counts
        if i % 500 == 0:
            print(f"      [fps] {i}/{n_samples} points sampled...", flush=True)
    
    return points[selected_indices]


def farthest_point_sample_weighted(
    points: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """Curvature-weighted farthest point sampling.

    Selects ``argmax(min_distance * weight)`` instead of ``argmax(min_distance)``,
    biasing toward high-curvature regions that need denser coverage.

    Inspired by SplatSDF's observation that complex geometry (holes, thin
    features) needs more representation density (arXiv:2411.15468 Sec. 5.4).

    Args:
        points: Candidate positions ``(M, 3)``.
        weights: Per-point importance weights ``(M,)``, e.g. curvature.
        n_samples: Number of points to select.

    Returns:
        Selected points ``(n_samples, 3)``.
    """
    device = points.device
    n_points = len(points)

    if n_points <= n_samples:
        return points

    # Normalise weights to [1, 3] range so flat regions still get coverage
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min > 1e-8:
        w_norm = (weights - w_min) / (w_max - w_min)
        boost = 1.0 + 2.0 * w_norm  # high curvature gets 3x priority
    else:
        boost = torch.ones(n_points, device=device)

    selected_indices = torch.zeros(n_samples, dtype=torch.long, device=device)
    selected_indices[0] = torch.argmax(weights)  # seed with sharpest feature
    min_distances = torch.full((n_points,), float('inf'), device=device)

    for i in range(1, n_samples):
        last_selected = points[selected_indices[i - 1]]
        distances = torch.norm(points - last_selected, dim=1)
        min_distances = torch.minimum(min_distances, distances)

        # Weighted criterion: prefer far AND high-curvature points
        selected_indices[i] = torch.argmax(min_distances * boost)

        if i % 500 == 0:
            print(f"      [fps-w] {i}/{n_samples} points sampled...", flush=True)

    return points[selected_indices]


# ============================================================================
# Legacy Poisson Disk Sampling (slower, kept for reference)
# ============================================================================

def initialize_splats_poisson(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    bounds: Tuple[List[float], List[float]],
    target_count: int = 10000,
    min_radius: float = 0.02,
    k_candidates: int = 30,
    device: str = "cpu",
    attrs_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Generate initial splat positions using Poisson disk sampling on SDF surface.
    Returns: (positions, attributes, avg_spacing)
    
    NOTE: This is slower than initialize_splats_batched on GPU due to sequential
    point-by-point processing. Kept for compatibility and CPU fallback.
    """
    min_xyz = torch.tensor(bounds[0], dtype=torch.float32, device=device)
    max_xyz = torch.tensor(bounds[1], dtype=torch.float32, device=device)
    extent = max_xyz - min_xyz
    
    # Grid for spatial lookup (kept on CPU for dict operations)
    cell_size = min_radius / np.sqrt(3)
    grid_dims = (extent / cell_size).ceil().int().cpu().tolist()
    grid = {}  # (i, j, k) -> point index
    
    points: List[torch.Tensor] = []
    active: List[int] = []
    
    # helper functions - use CPU for grid lookups
    def get_cell(p: torch.Tensor) -> Tuple[int, int, int]:
        cell = ((p.cpu() - min_xyz.cpu()) / cell_size).int()
        return tuple(cell.tolist())
    
    def is_valid(p: torch.Tensor) -> bool:
        cell = get_cell(p)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                for dk in range(-2, 3):
                    neighbor = (cell[0] + di, cell[1] + dj, cell[2] + dk)
                    if neighbor in grid:
                        idx = grid[neighbor]
                        if torch.norm(p - points[idx]) < min_radius:
                            return False
        return True
    
    # Find initial point
    print(f"      [poisson] Finding initial surface point on {device}...", flush=True)
    initial_p = (min_xyz + max_xyz) / 2
    initial_p = initial_p.unsqueeze(0)
    initial_p = project_to_surface(sdf_fn, initial_p)[0]
    print(f"      [poisson] Initial point found, starting sampling loop...", flush=True)
    
    points.append(initial_p)
    active.append(0)
    grid[get_cell(initial_p)] = 0
    
    # Sampling loop with progress logging
    last_log = 0
    while active and len(points) < target_count:
        # Log progress every 500 points
        if len(points) - last_log >= 500:
            print(f"      [poisson] Sampled {len(points)}/{target_count} points...", flush=True)
            last_log = len(points)
        active_idx = np.random.randint(len(active))
        base_idx = active[active_idx]
        base_point = points[base_idx]
        
        found = False
        for _ in range(k_candidates):
            # Generate random direction on device
            direction = torch.randn(3, device=device)
            direction = F.normalize(direction, dim=0)
            distance = min_radius * (1 + np.random.random())
            candidate = base_point + direction * distance
            
            if (candidate < min_xyz).any() or (candidate > max_xyz).any():
                continue
            
            candidate = candidate.unsqueeze(0)
            candidate = project_to_surface(sdf_fn, candidate)[0]
            
            if is_valid(candidate):
                idx = len(points)
                points.append(candidate)
                active.append(idx)
                grid[get_cell(candidate)] = idx
                found = True
                break
        
        if not found:
            active.pop(active_idx)
    
    positions = torch.stack(points)
    
    # Query material attributes (color[3] + metallic + roughness = 5 channels)
    attrs_source = attrs_fn if attrs_fn is not None else sdf_fn
    if hasattr(attrs_source, "query_attributes"):
        if device == "cuda" and attrs_fn is not None:
            attrs = attrs_source.query_attributes(positions.cpu()).to(device)
        else:
            attrs = attrs_source.query_attributes(positions)
    else:
        default = torch.tensor([0.627, 0.0, 0.0, 0.0, 0.5], dtype=torch.float32, device=device)
        attrs = default.unsqueeze(0).expand(len(positions), 5)
        
    print(f"    [splat_trainer] Poisson sampling: {len(points)} splats on {device}", flush=True)
    # Estimate spacing from cell_size (approximate)
    avg_spacing = min_radius 
    return positions, attrs, avg_spacing



# ============================================================================
# Splat Optimizer
# ============================================================================

class SplatOptimizer:
    """
    Optimizes Gaussian splat parameters against an SDF.
    
    GPU Acceleration: All learnable parameters and computations run on the
    specified device (CUDA or CPU) for optimal performance.
    """
    
    def __init__(
        self,
        sdf_fn: Callable[[torch.Tensor], torch.Tensor],
        initial_positions: torch.Tensor,
        initial_attrs: torch.Tensor,
        initial_scale: float = 0.02,
        avg_spacing: float = 0.0,
        device: str = "cpu",
    ):
        self.sdf_fn = sdf_fn
        self.device = device
        n = len(initial_positions)

        # Voxel-hashed overlap cell size (2x avg spacing for neighbour detection)
        self.overlap_cell_size = avg_spacing * 2.0 if avg_spacing > 0 else initial_scale * 4.0
        
        # Learnable parameters - ensure all on correct device
        self.positions = initial_positions.clone().to(device).requires_grad_(True)
        
        # Scales: Derived from local density (isotropic) represents sphere "radius"
        # 3-sigma rule: Gaussian radius r covers 99%, so sigma = r/3 ?
        # Standard 3DGS init: log(scale) optimized.
        # Here we use fixed scales based on density.
        # If avg_spacing is provided, use it.
        base_scale = initial_scale
        if avg_spacing > 0:
            # Overlap factor: 1.0 means radius = spacing (touching). 
            # We want slight overlap => 1.5x spacing ?
            # Wait, scale in 3DGS is typically Standard Deviation.
            # Visually, size ~= 3 * scale.
            # So if we want size = spacing, scale = spacing / 3.
            # Let's try scale = spacing * 0.35 (~1.05 sigma coverage = spacing)
            base_scale = avg_spacing * 0.35
            print(f"    [SplatOptimizer] Using density-based scale: {base_scale:.5f} (from spacing {avg_spacing:.5f})", flush=True)

        self.scales = torch.full(
            (n, 3), base_scale, dtype=torch.float32, device=device
        )
        # self.scales.requires_grad_(False) # Explicitly not optimizable
        
        # Identity quaternion (w, x, y, z) = (1, 0, 0, 0)
        self.rotations = torch.zeros(n, 4, dtype=torch.float32, device=device)
        self.rotations[:, 0] = 1.0
        self.rotations = self.rotations.requires_grad_(True)
        
        # Unpack 5-channel attributes: color[3] + metallic + roughness
        if initial_attrs.shape[0] != n:
            initial_attrs = initial_attrs[0:1].expand(n, 5)
        # Ensure 5 channels (backward compat with old 3-channel callers)
        if initial_attrs.shape[1] < 5:
            pad = torch.zeros(n, 5 - initial_attrs.shape[1], device=device)
            initial_attrs = torch.cat([initial_attrs, pad], dim=1)
        
        attrs = initial_attrs.clone().to(device)
        self.colors_oklab = attrs[:, :3].contiguous().requires_grad_(True)
        self.metallic = attrs[:, 3].contiguous().requires_grad_(True)
        self.roughness = attrs[:, 4].contiguous().requires_grad_(True)
        
        # Opacities - Start opaque (logit 4.0 ~= 0.98) to avoid "ghostly" look
        self.opacities_logit = torch.full((n,), 4.0, dtype=torch.float32, device=device).requires_grad_(True)
        
        # Optimizer - SCALES REMOVED
        self.optimizer = torch.optim.Adam([
            {'params': self.positions, 'lr': 0.005},
            # {'params': self.scales, 'lr': 0.002}, # Removed: no gradient signal matches density
            {'params': self.rotations, 'lr': 0.005},
            {'params': self.colors_oklab, 'lr': 0.01},
            {'params': self.metallic, 'lr': 0.005},
            {'params': self.roughness, 'lr': 0.005},
            {'params': self.opacities_logit, 'lr': 0.05},
        ])
    
    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacities_logit)
    
    def compute_sdf_gradients(
        self, positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute SDF normals and gradient magnitudes.

        Enables Eikonal validation: ``||grad f||`` should be ~1.0 for
        well-formed SDFs. Deviations flag CSG seams or modifier distortions.

        Returns:
            normals: ``(N, 3)`` unit surface normals.
            grad_magnitudes: ``(N,)`` ``||grad f||`` values (1.0 = valid SDF).
        """
        p = positions.detach().requires_grad_(True)
        d = self.sdf_fn(p)
        grads = torch.autograd.grad(d.sum(), p, create_graph=False)[0]
        grad_mag = torch.norm(grads, dim=1)
        normals = grads / grad_mag.unsqueeze(1).clamp(min=1e-8)
        return normals, grad_mag
    
    def quaternion_to_normal(self, q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        zx = 2 * (x * z + w * y)
        zy = 2 * (y * z - w * x)
        zz = 1 - 2 * (x * x + y * y)
        return F.normalize(torch.stack([zx, zy, zz], dim=1), dim=1)
    
    def compute_loss_for(
        self,
        indices: Optional[torch.Tensor] = None,
        include_overlap_full: bool = True,
        overlap_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined training loss with SplatSDF-inspired improvements.

        Loss terms:
            surface:  ``(sdf_value)^2`` weighted by eikonal confidence.
            normal:   ``1 - cos(sdf_normal, splat_normal)`` weighted by eikonal confidence.
            eikonal:  ``(||grad f|| - 1)^2`` -- diagnoses degenerate SDF regions.
            overlap:  Voxel-hashed proximity penalty (SplatSDF Appendix B).
            color:    Oklab bounds regularisation.
        """
        if indices is None:
            positions = self.positions
            rotations = self.rotations
            colors_oklab = self.colors_oklab
            metallic = self.metallic
            roughness = self.roughness
            opacities = self.opacities
        else:
            positions = self.positions[indices]
            rotations = self.rotations[indices]
            colors_oklab = self.colors_oklab[indices]
            metallic = self.metallic[indices]
            roughness = self.roughness[indices]
            opacities = self.opacities[indices]

        # SDF values for surface alignment (Forward pass 1)
        # We need retain_graph=True because we'll backward through this graph for surface loss
        sdf_values = self.sdf_fn(positions)

        # Compute gradients (normals) from the SAME graph
        # create_graph=False: We don't need second derivatives for optimization
        # retain_graph=True: We still need the graph for loss.backward() later
        grads = torch.autograd.grad(
            sdf_values.sum(), 
            positions, 
            create_graph=False, 
            retain_graph=True
        )[0]
        
        grad_mag = torch.norm(grads, dim=1)
        sdf_normals = grads / grad_mag.unsqueeze(1).clamp(min=1e-8)

        # Eikonal confidence: down-weight loss at degenerate SDF regions
        # where ||grad f|| != 1 (CSG seams, modifier distortions)
        # Detach to avoid trying to optimize positions to fix eikonal issues
        eikonal_deviation = (grad_mag.detach() - 1.0) ** 2
        eikonal_confidence = 1.0 / (1.0 + 10.0 * eikonal_deviation)  # (N,)

        # Surface loss (weighted by eikonal confidence)
        loss_surface = (eikonal_confidence * sdf_values ** 2).mean()

        # Normal loss (weighted by eikonal confidence)
        # Rotates splats to align with surface normals
        splat_normals = self.quaternion_to_normal(rotations)
        cosine = (sdf_normals.detach() * splat_normals).sum(dim=1)
        loss_normal = (eikonal_confidence * (1.0 - cosine)).mean()

        # Eikonal diagnostic (||grad f|| - 1)^2 -- diagnostic only
        loss_eikonal = eikonal_deviation.mean()

        # Voxel-hashed overlap penalty
        if include_overlap_full:
            loss_overlap = _voxel_overlap_loss(self.positions, self.overlap_cell_size)
        elif overlap_positions is not None:
            loss_overlap = _voxel_overlap_loss(overlap_positions, self.overlap_cell_size)
        else:
            loss_overlap = torch.tensor(0.0, device=self.device)

        # Color regularisation
        L = colors_oklab[:, 0]
        ab = colors_oklab[:, 1:]
        loss_color = (
            F.relu(L - 1.0).mean() + F.relu(-L).mean()
            + F.relu(torch.abs(ab) - 0.4).mean()
        )

        # Opacity regularization: encourage solid surface (1.0)
        loss_opacity = (1.0 - opacities).mean()

        total = (
            10.0 * loss_surface
            + 1.0 * loss_normal
            + 0.1 * loss_overlap
            + 0.01 * loss_color
            + 0.05 * loss_opacity
            # eikonal is diagnostic only (SDF is frozen, not trained)
        )

        return total, {
            'surface': loss_surface.item(),
            'normal': loss_normal.item(),
            'eikonal': loss_eikonal.item(),
            'overlap': loss_overlap.item(),
            'color': loss_color.item(),
            'opacity': loss_opacity.item(),
        }
    
    def train(
        self,
        iterations: int = 300,
        target_loss: float = 0.0,
        log_interval: int = 50,
        batch_size: Optional[int] = None,
        overlap_interval: int = 50,
        accum_steps: int = 1,
        overlap_batch: bool = True,
        overlap_batch_size: Optional[int] = None,
    ) -> List[float]:
        """Run optimization loop on the configured device.
        
        Args:
            iterations: Max iterations.
            target_loss: Stop if total loss drops below this value. 0.0 to disable.
        """
        """Run optimization loop on the configured device."""
        loss_history = []
        n = int(self.positions.shape[0])
        if batch_size is None or batch_size <= 0:
            batch_size = _auto_batch_size(n, self.device)
        batch_size = max(1, min(int(batch_size), n))
        overlap_interval = max(1, int(overlap_interval))
        use_batches = batch_size < n
        accum_steps = max(1, int(accum_steps))
        overlap_batch = bool(overlap_batch)
        if overlap_batch_size is None or overlap_batch_size <= 0:
            overlap_batch_size = batch_size
        overlap_batch_size = max(1, min(int(overlap_batch_size), n))
        if use_batches:
            print(
                f"    [splat_trainer] 🧩 Mini-batch training enabled: batch={batch_size}, overlap_every={overlap_interval}",
                flush=True,
            )
            if accum_steps > 1:
                print(
                    f"    [splat_trainer] 🔁 Gradient accumulation: steps={accum_steps}",
                    flush=True,
                )
        else:
            print("    [splat_trainer] 🧠 Full-batch training enabled", flush=True)

        for i in range(iterations):
            self.optimizer.zero_grad()
            total_loss = 0.0
            comp_accum = {
                "surface": 0.0,
                "normal": 0.0,
                "eikonal": 0.0,
                "overlap": 0.0,
                "color": 0.0,
                "opacity": 0.0,
            }
            for step in range(accum_steps):
                if use_batches:
                    indices = torch.randint(0, n, (batch_size,), device=self.device)
                else:
                    indices = None

                include_overlap_full = (i % overlap_interval == 0) and step == 0
                overlap_positions = None
                if use_batches and overlap_batch and not include_overlap_full:
                    overlap_indices = torch.randint(0, n, (overlap_batch_size,), device=self.device)
                    overlap_positions = self.positions[overlap_indices]

                loss, components = self.compute_loss_for(
                    indices=indices,
                    include_overlap_full=include_overlap_full,
                    overlap_positions=overlap_positions,
                )
                (loss / accum_steps).backward()
                total_loss += loss.item()
                for key in comp_accum:
                    comp_accum[key] += components[key]

            self.optimizer.step()
            
            with torch.no_grad():
                self.rotations.data = F.normalize(self.rotations.data, dim=1)
            
            loss_history.append(total_loss / accum_steps)
            if i % log_interval == 0:
                mem_info = ""
                if self.device == "cuda":
                    # Report current and peak memory usage in MB
                    cur_mem = torch.cuda.memory_allocated() / 1024 / 1024
                    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                    mem_info = f" mem={cur_mem:.0f}/{peak_mem:.0f}MB"
                
                avg_components = {k: v / accum_steps for k, v in comp_accum.items()}
                print(
                    f"    [splat_trainer] Iter {i}: loss={total_loss / accum_steps:.6f}"
                    f" srf={avg_components['surface']:.4f} nrm={avg_components['normal']:.4f}"
                    f" eik={avg_components['eikonal']:.3f} ovl={avg_components['overlap']:.3f}"
                    f"{mem_info} ({self.device})", flush=True,
                )
            
            # Check target rate/loss
            if target_loss > 0.0 and (total_loss / accum_steps) < target_loss:
                print(
                    f"    [splat_trainer] 🎯 Reached target loss {target_loss} "
                    f"(val={total_loss / accum_steps:.6f}) at iter {i}",
                    flush=True,
                )
                break
            
            # Clean up GPU memory every 100 iterations to prevent accumulation
            if self.device == "cuda" and i % 100 == 0 and i > 0:
                torch.cuda.empty_cache()
                
        return loss_history
    
    def export(self) -> SplatData:
        """Export splats as Oklab u8. No color-space conversion needed.

        Quantisation:
            L  in [0, 1]       -> [0, 255]
            a  in [-0.4, 0.4]  -> [0, 255]  (offset + scale)
            b  in [-0.4, 0.4]  -> [0, 255]
        """
        oklab = self.colors_oklab.detach().cpu().numpy()  # (N, 3)
        alpha = (self.opacities.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        L_u8 = (oklab[:, 0] * 255).clip(0, 255).astype(np.uint8)
        a_u8 = ((oklab[:, 1] + 0.4) / 0.8 * 255).clip(0, 255).astype(np.uint8)
        b_u8 = ((oklab[:, 2] + 0.4) / 0.8 * 255).clip(0, 255).astype(np.uint8)
        colors = np.column_stack([L_u8, a_u8, b_u8, alpha])

        metallic_u8 = (self.metallic.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        roughness_u8 = (self.roughness.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        flags = np.full(len(oklab), 0x01, dtype=np.uint8)

        return SplatData(
            positions=self.positions.detach().cpu().numpy(),
            scales=self.scales.detach().cpu().numpy(),
            rotations=self.rotations.detach().cpu().numpy(),
            colors=colors,
            metallic=metallic_u8,
            roughness=roughness_u8,
            flags=flags,
        )




# ============================================================================
# Post-Training Coverage Analysis (SplatSDF-inspired)
# Reference: arXiv:2411.15468 — Eq. 3 (weighted blending / coverage)
# ============================================================================

@torch.no_grad()
def compute_coverage(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    positions: torch.Tensor,
    scales: torch.Tensor,
    bounds: Tuple[List[float], List[float]],
    n_probes: int = 5000,
    coverage_threshold: float = 0.1,
) -> dict:
    """Analyze Gaussian-weighted coverage of the SDF surface.

    Probes random surface points and measures cumulative Gaussian weight
    from nearby splats.  Under-covered regions indicate gaps that could
    benefit from adaptive densification or LOD budget reallocation.

    Args:
        sdf_fn: SDF evaluation function ``(N, 3) -> (N,)``.
        positions: Trained splat positions ``(N, 3)``.
        scales: Splat scales ``(N, 3)``.
        bounds: ``(min_xyz, max_xyz)`` bounding box.
        n_probes: Number of surface points to probe.
        coverage_threshold: Minimum acceptable cumulative weight.

    Returns:
        Dictionary with coverage statistics:
            mean_coverage: Average cumulative Gaussian weight.
            min_coverage: Worst-case coverage.
            pct_under_covered: Fraction of probes below threshold.
            under_covered_positions: ``(M, 3)`` ndarray of gap locations.
    """
    device = positions.device
    min_xyz = torch.tensor(bounds[0], dtype=torch.float32, device=device)
    max_xyz = torch.tensor(bounds[1], dtype=torch.float32, device=device)
    extent = max_xyz - min_xyz

    # Generate random probe points on the surface via iterative batching to avoid OOM
    # Generate in small chunks until we have enough valid surface points
    valid_probes_list = []
    total_found = 0
    batch_size = 2000
    max_attempts = max(5, (n_probes * 5) // batch_size)  # Limit attempts
    
    for _ in range(max_attempts):
        if total_found >= n_probes:
            break
            
        # Generate varied random points
        chunk_pts = min_xyz + torch.rand(batch_size, 3, device=device) * extent
        
        # Project to surface (enable grad for normals)
        with torch.enable_grad():
            chunk_proj = project_to_surface(sdf_fn, chunk_pts, max_steps=8)
            
        # Check distance to valid surface (no grad needed for distances)
        chunk_dists = torch.abs(sdf_fn(chunk_proj))
        mask = chunk_dists < 0.01
        
        valid_batch = chunk_proj[mask]
        if len(valid_batch) > 0:
            valid_probes_list.append(valid_batch)
            total_found += len(valid_batch)
            
        # Cleanup
        del chunk_pts, chunk_proj, chunk_dists, mask
        
    if not valid_probes_list:
        probes = torch.empty((0, 3), device=device)
    else:
        probes = torch.cat(valid_probes_list)[:n_probes]
            
    n_actual = len(probes)

    if n_actual == 0:
        return {
            'mean_coverage': 0.0,
            'min_coverage': 0.0,
            'pct_under_covered': 1.0,
            'under_covered_positions': np.empty((0, 3), dtype=np.float32),
        }

    # Average scale per splat for isotropic Gaussian approximation
    avg_scale = scales.mean(dim=1).clamp(min=1e-6)  # (N,)

    # Cumulative Gaussian weight at each probe (batched for memory)
    batch_size = 2000
    all_coverage: list[torch.Tensor] = []

    for start in range(0, n_actual, batch_size):
        end = min(start + batch_size, n_actual)
        probe_batch = probes[start:end]  # (B, 3)
        # Pairwise distances (B, N)
        dists = torch.cdist(probe_batch, positions)
        # Gaussian weight: exp(-0.5 * (dist / scale)^2)
        weights = torch.exp(-0.5 * (dists / avg_scale.unsqueeze(0)) ** 2)
        all_coverage.append(weights.sum(dim=1))  # (B,)

    coverage = torch.cat(all_coverage)
    under_mask = coverage < coverage_threshold

    return {
        'mean_coverage': coverage.mean().item(),
        'min_coverage': coverage.min().item(),
        'pct_under_covered': under_mask.float().mean().item(),
        'under_covered_positions': probes[under_mask].cpu().numpy(),
    }


# ============================================================================
# Swatch mode: one or few large splats per node (reuses dense surface logic)
# ============================================================================

def _swatch_surface_sample(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    bmin_t: torch.Tensor,
    bmax_t: torch.Tensor,
    surface_tolerance: float,
    n_sample: int,
    device: str,
) -> Tuple[torch.Tensor, float]:
    """
    Dense-style sampling within node AABB: random points → project to surface
    → filter by tolerance. Returns (valid_surface_points, avg_spacing).
    If fewer than 2 valid points, returns (empty tensor, 0.0).
    """
    extent = bmax_t - bmin_t
    random_points = torch.rand(n_sample, 3, device=device, dtype=torch.float32) * extent + bmin_t
    surface_points = project_to_surface(sdf_fn, random_points, max_steps=10)
    with torch.no_grad():
        distances = sdf_fn(surface_points)
    mask = torch.abs(distances) < surface_tolerance
    valid = surface_points[mask]
    if len(valid) < 2:
        return valid, 0.0
    # Avg nearest-neighbor spacing (same as dense init)
    dists = torch.cdist(valid, valid)
    dists.fill_diagonal_(float("inf"))
    nn_dists = dists.min(dim=1).values
    avg_spacing = nn_dists.mean().item()
    return valid, avg_spacing


def initialize_swatches_per_node(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    node_bounds_list: List[Tuple[str, List[float], List[float]]],
    swatches_per_node: int = 1,
    swatch_scale_factor: float = 0.4,
    device: str = "cpu",
    attrs_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    max_swatch_scale: Optional[float] = None,
    bounds: Optional[Tuple[List[float], List[float]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Place one or several large swatch splats per node. When bounds is provided,
    reuses dense-mode logic: sample points in node AABB, project to surface,
    filter by surface tolerance, then use centroid of valid surface points and
    spacing-derived scale (0.35 * avg_spacing) so splats stay attached to SDF.
    Otherwise falls back to centroid projection + extent-based scale.
    """
    positions_list: List[torch.Tensor] = []
    attrs_list: List[torch.Tensor] = []
    scales_list: List[torch.Tensor] = []
    query_fn = attrs_fn if attrs_fn is not None and hasattr(attrs_fn, "query_attributes") else sdf_fn
    if not hasattr(query_fn, "query_attributes"):
        default_attrs = torch.tensor([0.25, 0.0, 0.0, 0.0, 0.5], dtype=torch.float32, device=device)
    use_dense_style = bounds is not None
    surface_tolerance = _compute_surface_tolerance(bounds, min_radius=0.02) if use_dense_style else 1e-4
    n_sample = 80

    for node_id, bmin, bmax in node_bounds_list:
        bmin_t = torch.tensor(bmin, dtype=torch.float32, device=device)
        bmax_t = torch.tensor(bmax, dtype=torch.float32, device=device)
        extent = (bmax_t - bmin_t).tolist()
        max_extent = max(extent) if extent else 0.05
        scale_val = max(max_extent * swatch_scale_factor, 0.005)
        if max_swatch_scale is not None:
            scale_val = min(scale_val, max_swatch_scale)

        if use_dense_style:
            valid, avg_spacing = _swatch_surface_sample(
                sdf_fn, bmin_t, bmax_t, surface_tolerance, n_sample, device
            )
            if len(valid) >= 2 and avg_spacing > 0:
                # Larger than dense (0.35) so swatches are visible patches, not tiny dots
                scale_val = 0.9 * avg_spacing
                if max_swatch_scale is not None:
                    scale_val = min(scale_val, max_swatch_scale)
                scale_val = max(scale_val, 0.02)
                first_centroid = valid.mean(dim=0)
            else:
                first_centroid = (bmin_t + bmax_t) * 0.5

        for sw in range(swatches_per_node):
            if use_dense_style:
                if sw == 0:
                    centroid = first_centroid.unsqueeze(0)
                else:
                    jitter = (torch.rand(3, device=device) - 0.5) * torch.tensor(extent, device=device) * 0.3
                    centroid = (first_centroid + jitter).unsqueeze(0)
                proj = project_to_surface(sdf_fn, centroid, max_steps=10)
            else:
                centroid = (bmin_t + bmax_t) * 0.5
                if swatches_per_node > 1:
                    jitter = (torch.rand(3, device=device) - 0.5) * torch.tensor(extent, device=device) * 0.3
                    centroid = centroid + jitter
                proj = project_to_surface(sdf_fn, centroid.unsqueeze(0), max_steps=10)
            if hasattr(query_fn, "query_attributes"):
                attrs = query_fn.query_attributes(proj)
            else:
                attrs = default_attrs.unsqueeze(0).expand(1, 5)
            positions_list.append(proj)
            attrs_list.append(attrs)
            scales_list.append(torch.full((1, 3), scale_val, dtype=torch.float32, device=device))

    positions = torch.cat(positions_list, dim=0)
    attrs = torch.cat(attrs_list, dim=0)
    scales = torch.cat(scales_list, dim=0)
    # Surface normals at each position so splats can be oriented on the SDF (not billboarded)
    with torch.enable_grad():
        p = positions.detach().requires_grad_(True)
        d = sdf_fn(p)
        grads = torch.autograd.grad(d.sum(), p, create_graph=False)[0]
    grad_mag = torch.norm(grads, dim=1).unsqueeze(1).clamp(min=1e-8)
    normals = (grads / grad_mag).detach()
    return positions, attrs, scales, normals


def _normal_to_quat_z(normals: torch.Tensor) -> torch.Tensor:
    """Unit normals (N,3) -> quaternions (N,4) w,x,y,z that rotate +Z to normal."""
    # q = (1 + Nz, -Ny, Nx, 0); normalize
    n = normals
    qw = 1.0 + n[:, 2]
    qx = -n[:, 1]
    qy = n[:, 0]
    qz = torch.zeros_like(qw)
    q = torch.stack([qw, qx, qy, qz], dim=1)
    q = F.normalize(q, dim=1)
    return q


def build_splat_data_from_swatches(
    positions: torch.Tensor,
    attrs: torch.Tensor,
    scales: torch.Tensor,
    normals: Optional[torch.Tensor] = None,
) -> SplatData:
    """Build SplatData from swatch (positions, attrs, scales). If normals given, rotations orient splats on surface."""
    positions_np = positions.detach().cpu().numpy().astype(np.float32)
    scales_np = scales.detach().cpu().numpy().astype(np.float32)
    n = len(positions_np)
    oklab = attrs[:, :3].detach().cpu().numpy()
    L_u8 = (oklab[:, 0] * 255).clip(0, 255).astype(np.uint8)
    a_u8 = ((oklab[:, 1] + 0.4) / 0.8 * 255).clip(0, 255).astype(np.uint8)
    b_u8 = ((oklab[:, 2] + 0.4) / 0.8 * 255).clip(0, 255).astype(np.uint8)
    alpha = np.full(n, 255, dtype=np.uint8)
    colors = np.column_stack([L_u8, a_u8, b_u8, alpha])
    metallic_u8 = (attrs[:, 3].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    roughness_u8 = (attrs[:, 4].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    flags = np.full(n, 0x01, dtype=np.uint8)
    if normals is not None:
        quats = _normal_to_quat_z(normals).detach().cpu().numpy().astype(np.float32)
        rotations = quats  # (N,4) w,x,y,z
    else:
        rotations = np.zeros((n, 4), dtype=np.float32)
        rotations[:, 0] = 1.0
    return SplatData(
        positions=positions_np,
        scales=scales_np,
        rotations=rotations,
        colors=colors,
        metallic=metallic_u8,
        roughness=roughness_u8,
        flags=flags,
    )


def pack_splat_data(splat_data: SplatData) -> bytes:
    """Pack splat data to binary matching ``engine/shared/src/binary_format.rs`` Splat (48 bytes each).

    Layout per splat::

        position:  12 bytes (f32x3)
        scale:     12 bytes (f32x3)
        rotation:  16 bytes (f32x4)
        color:      4 bytes (Oklab8+A: L, a, b, alpha)
        metallic:   1 byte  (u8, 0-255 -> 0.0-1.0)
        roughness:  1 byte  (u8, 0-255 -> 0.0-1.0)
        flags:      1 byte  (u8, 0x01 = Oklab)
        _pad:       1 byte
        Total: 48 bytes
    """
    n = len(splat_data.positions)
    buf = bytearray()
    buf.extend(struct.pack('<I', n))
    for i in range(n):
        buf.extend(struct.pack('<fff', *splat_data.positions[i]))       # 12 bytes
        buf.extend(struct.pack('<fff', *splat_data.scales[i]))          # 12 bytes
        buf.extend(struct.pack('<ffff', *splat_data.rotations[i]))      # 16 bytes
        buf.extend(splat_data.colors[i].tobytes())                      # 4 bytes
        buf.extend(struct.pack('<BBBx',
            splat_data.metallic[i],
            splat_data.roughness[i],
            splat_data.flags[i],
        ))                                                              # 4 bytes
    return bytes(buf)


def _knn_mean_distance_chunked(
    positions: torch.Tensor,
    k: int = 3,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Compute mean k-NN distance without full NxN cdist allocation."""
    n = int(positions.shape[0])
    if n < 2:
        return torch.zeros(n, device=positions.device, dtype=positions.dtype)
    k = max(1, min(k, n - 1))
    batch_size = max(1, int(batch_size))

    out = torch.empty(n, device=positions.device, dtype=positions.dtype)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = positions[start:end]
        dists = torch.cdist(chunk, positions)

        # Mask self-distances for rows that correspond to the same indices
        row_idx = torch.arange(end - start, device=positions.device)
        col_idx = torch.arange(start, end, device=positions.device)
        dists[row_idx, col_idx] = float("inf")

        knn = dists.topk(k, dim=1, largest=False).values
        out[start:end] = knn.mean(dim=1)
        del dists

    return out


def compile_splats_swatch_mode(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    bounds: Tuple[List[float], List[float]],
    dna: dict,
    swatches_per_node: int = 1,
    swatch_scale_factor: float = 0.4,
    device: Optional[str] = None,
) -> bytes:
    """
    Compile splats as one or few large swatches per material node (no dense training).
    Uses collect_node_bounds(dna), places swatches at node centroids (projected to surface),
    then builds SplatData and optionally applies concept-image recolor.
    """
    if device is None:
        device = "cpu"
    print(f"    [splat_trainer] 🧩 Swatch mode: collecting node bounds...", flush=True)
    node_bounds_list = collect_node_bounds(dna)
    if not node_bounds_list:
        print(f"    [splat_trainer] ⚠️ No nodes with id in DNA; returning empty splat payload", flush=True)
        return pack_splat_data(SplatData(
            positions=np.empty((0, 3), dtype=np.float32),
            scales=np.empty((0, 3), dtype=np.float32),
            rotations=np.empty((0, 4), dtype=np.float32),
            colors=np.empty((0, 4), dtype=np.uint8),
            metallic=np.empty(0, dtype=np.uint8),
            roughness=np.empty(0, dtype=np.uint8),
            flags=np.empty(0, dtype=np.uint8),
        ))
    # Cap swatch scale so one node doesn't become a giant blob (wrap object instead)
    bmin_b, bmax_b = bounds[0], bounds[1]
    global_max_extent = max((bmax_b[i] - bmin_b[i]) for i in range(3)) if bounds else 2.0
    max_swatch_scale = global_max_extent * 0.15
    sdf_cpu = sdf_fn.to("cpu") if hasattr(sdf_fn, "to") else sdf_fn
    attrs_fn = sdf_cpu if hasattr(sdf_cpu, "query_attributes") else None
    positions, attrs, scales, normals = initialize_swatches_per_node(
        sdf_cpu,
        node_bounds_list,
        swatches_per_node=swatches_per_node,
        swatch_scale_factor=swatch_scale_factor,
        device=device,
        attrs_fn=attrs_fn,
        max_swatch_scale=max_swatch_scale,
        bounds=bounds,
    )
    print(f"    [splat_trainer] 🧩 Swatch mode: {len(positions)} swatches from {len(node_bounds_list)} nodes", flush=True)
    splat_data = build_splat_data_from_swatches(positions, attrs, scales, normals=normals)
    return pack_splat_data(splat_data)


def compile_splats(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    bounds: Tuple[List[float], List[float]],
    target_count: int = 10000,
    iterations: int = 300,
    target_loss: float = 0.0,
    batch_size: Optional[int] = None,
    overlap_interval: int = 50,
    accum_steps: int = 1,
    overlap_batch: bool = True,
    overlap_batch_size: Optional[int] = None,
    device: Optional[str] = None,
) -> bytes:
    """Compile splats from an SDF function.

    Colors are always exported as Oklab u8 -- the shader handles the
    single Oklab -> linear RGB conversion for PBR lighting.

    Args:
        sdf_fn: SDF evaluation function ``(N, 3) -> (N,)``, typically an ``SdfGraph``.
        bounds: ``(min_xyz, max_xyz)`` bounding box.
        target_count: Target number of splats to generate.
        iterations: Number of optimisation iterations.
        target_loss: Stop training if loss drops below this.
        device: ``"cuda"``, ``"cpu"``, or ``None`` for auto-detect.

    Returns:
        Packed binary splat data.
    """
    if device is None:
        device = preloader.get_device()
    
    print(f"    [splat_trainer] 🚀 Starting splat compilation on {device.upper()}...", flush=True)
    print(f"    [splat_trainer] 1. Initializing {target_count} splats via batched sampling...", flush=True)
    
    attrs_fn = None
    optimizer = None
    positions = None
    attrs = None
    try:
        # Move SDF graph to GPU if it's an nn.Module
        if hasattr(sdf_fn, 'to'):
            if device == "cuda":
                try:
                    attrs_fn = copy.deepcopy(sdf_fn).to("cpu")
                    print("    [splat_trainer] 🧊 Using CPU SDF for texture modifiers", flush=True)
                except Exception as e:
                    attrs_fn = None
                    print(f"    [splat_trainer] ⚠️ CPU attrs fallback unavailable: {e}", flush=True)
            sdf_fn = sdf_fn.to(device)
            print(f"    [splat_trainer] 📦 Moved SDF graph to {device}", flush=True)
        
        # Use fast batched init on GPU, sequential Poisson on CPU
        if device == "cuda":
            positions, attrs, avg_spacing = initialize_splats_batched(
                sdf_fn, bounds, target_count=target_count, device=device, attrs_fn=attrs_fn
            )
        else:
            positions, attrs, avg_spacing = initialize_splats_poisson(
                sdf_fn, bounds, target_count=target_count, device=device, attrs_fn=attrs_fn
            )
        print(f"    [splat_trainer] 2. Initialized {len(positions)} splats, starting optimization...", flush=True)
        
        # Use computed average spacing (smaller to reduce over-blur)
        initial_scale = float(avg_spacing * 0.5)
            
        print(f"    [splat_trainer] 📏 Calculated dynamic scale: {initial_scale:.5f} (avg spacing: {avg_spacing:.5f})", flush=True)
        
        optimizer = SplatOptimizer(
            sdf_fn, positions, attrs,
            initial_scale=initial_scale, avg_spacing=avg_spacing, device=device,
        )
        optimizer.train(
            iterations=iterations,
            target_loss=target_loss,
            batch_size=batch_size,
            overlap_interval=overlap_interval,
            accum_steps=accum_steps,
            overlap_batch=overlap_batch,
            overlap_batch_size=overlap_batch_size,
        )

        # Final projection pass to keep splats on the surface
        post_tol = _compute_surface_tolerance(bounds, avg_spacing=avg_spacing)
        print(f"    [splat_trainer] 🧭 Final surface projection (tol={post_tol:.6f})", flush=True)
        projected = project_to_surface(
            sdf_fn, optimizer.positions.detach(), max_steps=8, tolerance=post_tol
        )
        with torch.no_grad():
            distances = sdf_fn(projected)
            drift_mask = torch.abs(distances) > post_tol
        if drift_mask.any():
            drift_count = int(drift_mask.sum().item())
            print(f"    [splat_trainer] ⚠️ Re-projecting {drift_count} off-surface splats", flush=True)
            reprojected = project_to_surface(
                sdf_fn, projected[drift_mask], max_steps=12, tolerance=post_tol * 0.5
            )
            projected = projected.clone()
            projected[drift_mask] = reprojected
        optimizer.positions = projected.detach()
        
        # Recompute per-splat adaptive scale from trained positions
        print(f"    [splat_trainer] 📏 Computing adaptive scales (K=3 NN)...", flush=True)
        with torch.no_grad():
            trained_pos = optimizer.positions.detach()
            # Chunked k-NN to avoid O(N^2) memory spikes on GPU
            knn_batch = 2048 if device == "cuda" else 4096
            print(f"    [splat_trainer] 🧩 KNN spacing (chunked, batch={knn_batch})", flush=True)
            local_spacing = _knn_mean_distance_chunked(
                trained_pos, k=3, batch_size=knn_batch
            )
            
            # Debug: Log actual spacing stats
            print(f"    [splat_trainer] 📏 Spacing stats: min={local_spacing.min():.5f}, mean={local_spacing.mean():.5f}, max={local_spacing.max():.5f}", flush=True)

            # Set scale to 0.35 * local_spacing (tighter splats for sharper detail)
            adaptive_scale = (local_spacing * 0.35).clamp(
                min=avg_spacing * 0.2,
                max=avg_spacing * 0.6
            ).unsqueeze(1).expand(-1, 3)
            optimizer.scales = adaptive_scale.clone()
            
        print(f"    [splat_trainer] 3. Optimization complete, running coverage analysis...", flush=True)

        # Post-training coverage analysis
        coverage = compute_coverage(
            sdf_fn, optimizer.positions.detach(), optimizer.scales,
            bounds, n_probes=min(5000, target_count),
        )
        print(
            f"    [splat_trainer] 📊 Coverage: mean={coverage['mean_coverage']:.2f}"
            f" min={coverage['min_coverage']:.2f}"
            f" under-covered={coverage['pct_under_covered']:.1%}"
            f" ({len(coverage['under_covered_positions'])} gap points)",
            flush=True,
        )

        print(f"    [splat_trainer] 4. Exporting Oklab...", flush=True)
        splat_data = optimizer.export()
        binary_data = pack_splat_data(splat_data)

        print(f"    [splat_trainer] ✅ Done: {len(splat_data.positions)} splats on {device.upper()}", flush=True)
        return binary_data
    finally:
        try:
            if hasattr(sdf_fn, "to"):
                sdf_fn = sdf_fn.to("cpu")
        except Exception:
            pass
        optimizer = None
        positions = None
        attrs = None
        if device == "cuda":
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print(f"    [splat_trainer] 🧹 Cleared CUDA cache", flush=True)
            except Exception as e:
                print(f"    [splat_trainer] ⚠️ CUDA cache cleanup failed: {e}", flush=True)


