"""
Splat Trainer - Gaussian Splatting Generation from SDF.

This module implements Stage 3 of the GVE Compiler Pipeline:
1. Poisson Disk Sampling on the SDF surface for initialization.
2. Adam optimization to align splats to the surface with Oklab colors.
3. Adaptive densification and pruning.

GPU Acceleration: Uses torch_preloader to auto-detect CUDA and runs all
tensor operations on GPU when available for 10-50x speedup.

Reference: docs/workflows/compiler-pipeline.md §3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
from typing import List, Tuple, Optional, Callable, Union
from dataclasses import dataclass

from ..torch_preloader import preloader


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
# Fast Batched Initialization (GPU-optimized)
# ============================================================================

def initialize_splats_batched(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    bounds: Tuple[List[float], List[float]],
    target_count: int = 10000,
    min_radius: float = 0.02,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    
    surface_mask = torch.abs(distances) < 0.01  # Within 1cm of surface
    surface_points = surface_points[surface_mask]
    print(f"      [batch] {surface_points.shape[0]} points on surface", flush=True)
    
    if len(surface_points) < target_count:
        print(f"      [batch] Warning: only {len(surface_points)} surface points, using all", flush=True)
        positions = surface_points
    else:
        # Farthest point sampling for even distribution
        print(f"      [batch] Farthest-point sampling {target_count} from {len(surface_points)}...", flush=True)
        positions = farthest_point_sample(surface_points, target_count)
    
    # Query material attributes (color[3] + metallic + roughness = 5 channels)
    if hasattr(sdf_fn, "query_attributes"):
        attrs = sdf_fn.query_attributes(positions)
    else:
        default = torch.tensor([0.627, 0.0, 0.0, 0.0, 0.5], dtype=torch.float32, device=device)
        attrs = default.unsqueeze(0).expand(len(positions), 5)
    
    print(f"    [splat_trainer] Batch init: {len(positions)} splats on {device}", flush=True)
    return positions, attrs


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate initial splat positions using Poisson disk sampling on SDF surface.
    
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
    if hasattr(sdf_fn, "query_attributes"):
        attrs = sdf_fn.query_attributes(positions)
    else:
        default = torch.tensor([0.627, 0.0, 0.0, 0.0, 0.5], dtype=torch.float32, device=device)
        attrs = default.unsqueeze(0).expand(len(positions), 5)
        
    print(f"    [splat_trainer] Poisson sampling: {len(points)} splats on {device}", flush=True)
    return positions, attrs


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
        device: str = "cpu",
    ):
        self.sdf_fn = sdf_fn
        self.device = device
        n = len(initial_positions)
        
        # Learnable parameters - ensure all on correct device
        self.positions = initial_positions.clone().to(device).requires_grad_(True)
        
        # Scales are derived from local density and NOT optimized (no loss term for them)
        # Using a fixed scale based on average spacing prevents "bloated" splats
        self.scales = torch.full(
            (n, 3), initial_scale, dtype=torch.float32, device=device
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
        
        # Opacities
        self.opacities_logit = torch.zeros(n, dtype=torch.float32, device=device).requires_grad_(True)
        
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
    
    def compute_sdf_normals(self, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.detach().requires_grad_(True)
        d = self.sdf_fn(positions)
        grads = torch.autograd.grad(d.sum(), positions, create_graph=True)[0]
        return F.normalize(grads, dim=1)
    
    def quaternion_to_normal(self, q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        zx = 2 * (x * z + w * y)
        zy = 2 * (y * z - w * x)
        zz = 1 - 2 * (x * x + y * y)
        return F.normalize(torch.stack([zx, zy, zz], dim=1), dim=1)
    
    def compute_loss(self) -> Tuple[torch.Tensor, dict]:
        # Surface alignment
        sdf_values = self.sdf_fn(self.positions)
        loss_surface = (sdf_values ** 2).mean()
        
        # Normal alignment
        sdf_normals = self.compute_sdf_normals(self.positions)
        splat_normals = self.quaternion_to_normal(self.rotations)
        loss_normal = (1 - (sdf_normals * splat_normals).sum(dim=1)).mean()
        
        # Color regularization
        L = self.colors_oklab[:, 0]
        ab = self.colors_oklab[:, 1:]
        loss_color = (
            F.relu(L - 1.0).mean() + F.relu(-L).mean() +
            F.relu(torch.abs(ab) - 0.4).mean()
        )
        
        # Simplified overlap (too expensive for strict check every step)
        loss_overlap = torch.tensor(0.0, device=self.positions.device)
        
        total = (
            10.0 * loss_surface +
            1.0 * loss_normal +
            0.01 * loss_color
        )
        
        return total, {
            'surface': loss_surface.item(),
            'normal': loss_normal.item(),
            'overlap': loss_overlap.item(),
            'color': loss_color.item(),
        }
    
    def train(self, iterations: int = 300, log_interval: int = 50) -> List[float]:
        """Run optimization loop on the configured device."""
        loss_history = []
        for i in range(iterations):
            self.optimizer.zero_grad()
            loss, components = self.compute_loss()
            loss.backward()
            self.optimizer.step()
            
            with torch.no_grad():
                self.rotations.data = F.normalize(self.rotations.data, dim=1)
            
            loss_history.append(loss.item())
            if i % log_interval == 0:
                print(f"    [splat_trainer] Iter {i}: loss={loss.item():.6f} ({self.device})", flush=True)
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


def compile_splats(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    bounds: Tuple[List[float], List[float]],
    target_count: int = 10000,
    iterations: int = 300,
    device: Optional[str] = None,
) -> bytes:
    """Compile splats from an SDF function.

    Colors are always exported as Oklab u8 — the shader handles the
    single Oklab -> linear RGB conversion for PBR lighting.

    Args:
        sdf_fn: SDF evaluation function ``(N, 3) -> (N,)``, typically an ``SdfGraph``.
        bounds: ``(min_xyz, max_xyz)`` bounding box.
        target_count: Target number of splats to generate.
        iterations: Number of optimisation iterations.
        device: ``"cuda"``, ``"cpu"``, or ``None`` for auto-detect.

    Returns:
        Packed binary splat data.
    """
    if device is None:
        device = preloader.get_device()
    
    print(f"    [splat_trainer] 🚀 Starting splat compilation on {device.upper()}...", flush=True)
    print(f"    [splat_trainer] 1. Initializing {target_count} splats via batched sampling...", flush=True)
    
    # Move SDF graph to GPU if it's an nn.Module
    if hasattr(sdf_fn, 'to'):
        sdf_fn = sdf_fn.to(device)
        print(f"    [splat_trainer] 📦 Moved SDF graph to {device}", flush=True)
    
    # Use fast batched init on GPU, sequential Poisson on CPU
    if device == "cuda":
        positions, attrs = initialize_splats_batched(
            sdf_fn, bounds, target_count=target_count, device=device
        )
    else:
        positions, attrs = initialize_splats_poisson(
            sdf_fn, bounds, target_count=target_count, device=device
        )
    print(f"    [splat_trainer] 2. Initialized {len(positions)} splats, starting optimization...", flush=True)
    
    # Calculate sparse density to determine optimal scale
    with torch.no_grad():
        n_density_samples = min(2000, len(positions))
        subset = positions[:n_density_samples]
        dists = torch.cdist(subset, subset)
        dists.fill_diagonal_(float('inf'))
        min_dists = dists.min(dim=1).values
        avg_spacing = min_dists.mean().item()
        initial_scale = float(avg_spacing * 0.6)
        
    print(f"    [splat_trainer] 📏 Calculated dynamic scale: {initial_scale:.5f} (avg spacing: {avg_spacing:.5f})", flush=True)
    
    optimizer = SplatOptimizer(sdf_fn, positions, attrs, initial_scale=initial_scale, device=device)
    optimizer.train(iterations=iterations)
    print(f"    [splat_trainer] 3. Optimization complete, exporting Oklab...", flush=True)
    
    splat_data = optimizer.export()
    binary_data = pack_splat_data(splat_data)
    
    print(f"    [splat_trainer] ✅ Done: {len(splat_data.positions)} splats on {device.upper()}", flush=True)
    return binary_data
