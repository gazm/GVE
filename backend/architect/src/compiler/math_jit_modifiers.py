"""SDF Domain Modifiers - Space warping transformations."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional
from .math_jit_nodes import GeometryNode, ModifierNode, MaterialNode, _DEFAULT_OKLAB

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# =============================================================================
# Base Classes for Modifiers
# =============================================================================

class DomainWarpNode(ModifierNode):
    """Base for modifiers that just warp space before calling child."""
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        warped_x = self.warp(x)
        return self.child(warped_x)

    def warp(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# =============================================================================
# Transforms & Warps
# =============================================================================

class TransformNode(DomainWarpNode):
    """Wrapper that applies translation and rotation to a child SDF."""
    def __init__(self, child: GeometryNode, pos: List[float], rot: List[float]):
        super().__init__(child)
        self.register_buffer('translation', torch.tensor(pos, dtype=torch.float32))
        q = torch.tensor(rot, dtype=torch.float32)
        self.register_buffer('quat', q / (torch.norm(q) + 1e-8))
        
    def warp(self, x: torch.Tensor) -> torch.Tensor:
        local_p = x - self.translation
        return self._rotate_by_inverse_quat(local_p)
    
    def _rotate_by_inverse_quat(self, p: torch.Tensor) -> torch.Tensor:
        q = self.quat
        qx, qy, qz, qw = -q[0], -q[1], -q[2], q[3]
        q_xyz = torch.stack([qx, qy, qz])
        t = 2.0 * torch.cross(q_xyz.unsqueeze(0).expand(p.shape[0], -1), p, dim=1)
        return p + qw * t + torch.cross(q_xyz.unsqueeze(0).expand(p.shape[0], -1), t, dim=1)

    def _rotate_by_quat(self, p: torch.Tensor) -> torch.Tensor:
        q = self.quat
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        q_xyz = torch.stack([qx, qy, qz])
        t = 2.0 * torch.cross(q_xyz.unsqueeze(0).expand(p.shape[0], -1), p, dim=1)
        return p + qw * t + torch.cross(q_xyz.unsqueeze(0).expand(p.shape[0], -1), t, dim=1)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        b_min, b_max = self.child.compute_bounds()
        
        corners = []
        for x in [b_min[0], b_max[0]]:
            for y in [b_min[1], b_max[1]]:
                for z in [b_min[2], b_max[2]]:
                    corners.append(torch.stack([x, y, z]))
        corners = torch.stack(corners) # [8, 3]
        
        rotated = self._rotate_by_quat(corners)
        transformed = rotated + self.translation
        
        return (
            torch.min(transformed, dim=0)[0],
            torch.max(transformed, dim=0)[0]
        )


class TwistModifier(DomainWarpNode):
    """Twist space around an axis."""
    def __init__(self, child: GeometryNode, axis: str = "y", rate: float = 1.0):
        super().__init__(child)
        self.axis_idx = AXIS_INDEX.get(axis.lower(), 1)
        self.rate = rate
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Calculate perpendicular distance to axis for Lipschitz correction
        if self.axis_idx == 0: perp = x[:, [1, 2]]
        elif self.axis_idx == 1: perp = x[:, [0, 2]]
        else: perp = x[:, [0, 1]]
        
        r = torch.norm(perp, dim=1)
        # Correction: twisting expands space by sqrt(1 + (r*rate)^2)
        # We must scale distance by the reciprocal to maintain Lipschitz <= 1.0
        correction = 1.0 / torch.sqrt(1.0 + (r * self.rate)**2)
        
        warped_x = self.warp(x)
        res = self.child(warped_x)
        
        if isinstance(res, tuple):
            dist, attrs = res
            return dist * correction, attrs
        return res * correction

    def warp(self, x: torch.Tensor) -> torch.Tensor:
        axis_val = x[:, self.axis_idx]
        angle = axis_val * self.rate
        c = torch.cos(angle)
        s = torch.sin(angle)
        
        if self.axis_idx == 0:  # X axis
            return torch.stack([x[:, 0], c * x[:, 1] - s * x[:, 2], s * x[:, 1] + c * x[:, 2]], dim=1)
        elif self.axis_idx == 1:  # Y axis
            return torch.stack([c * x[:, 0] - s * x[:, 2], x[:, 1], s * x[:, 0] + c * x[:, 2]], dim=1)
        else:  # Z axis
            return torch.stack([c * x[:, 0] - s * x[:, 1], s * x[:, 0] + c * x[:, 1], x[:, 2]], dim=1)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        b_min, b_max = self.child.compute_bounds()
        pad = 0.5
        return (b_min - pad, b_max + pad)


class BendModifier(DomainWarpNode):
    """Bend space around an axis."""
    def __init__(self, child: GeometryNode, axis: str = "x", angle: float = 0.5):
        super().__init__(child)
        self.axis_idx = AXIS_INDEX.get(axis.lower(), 0)
        self.k = angle
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Bend as implemented is a vertical-dependent tilt/roll.
        # It expands space significantly. We use a conservative correction.
        # Max stretching is approx 1 + |k * r|
        if self.axis_idx == 0: r = torch.norm(x[:, [1, 2]], dim=1)
        elif self.axis_idx == 1: r = torch.norm(x[:, [0, 2]], dim=1)
        else: r = torch.norm(x[:, [0, 1]], dim=1)
        
        correction = 1.0 / (1.0 + torch.abs(r * self.k))
        
        warped_x = self.warp(x)
        res = self.child(warped_x)
        
        if isinstance(res, tuple):
            dist, attrs = res
            return dist * correction, attrs
        return res * correction

    def warp(self, x: torch.Tensor) -> torch.Tensor:
        if self.axis_idx == 0:
            c = torch.cos(self.k * x[:, 1])
            s = torch.sin(self.k * x[:, 1])
            return torch.stack([x[:, 0], c * x[:, 1] - s * x[:, 2], s * x[:, 1] + c * x[:, 2]], dim=1)
        elif self.axis_idx == 1:
            c = torch.cos(self.k * x[:, 0])
            s = torch.sin(self.k * x[:, 0])
            return torch.stack([c * x[:, 0] - s * x[:, 2], x[:, 1], s * x[:, 0] + c * x[:, 2]], dim=1)
        else:
            c = torch.cos(self.k * x[:, 0])
            s = torch.sin(self.k * x[:, 0])
            return torch.stack([c * x[:, 0] - s * x[:, 1], s * x[:, 0] + c * x[:, 1], x[:, 2]], dim=1)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        b_min, b_max = self.child.compute_bounds()
        pad = 1.0 
        return (b_min - pad, b_max + pad)


def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    t = torch.clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class TaperModifier(DomainWarpNode):
    """Taper (scale cross-section) along an axis."""
    def __init__(self, child: GeometryNode, axis: str = "y", scale_min: float = 0.5, scale_max: float = 1.0):
        super().__init__(child)
        self.axis_idx = AXIS_INDEX.get(axis.lower(), 1)
        self.scale_min = scale_min
        self.scale_max = scale_max
        
        # Cache child axis extent for stabilization
        if hasattr(child, "compute_bounds"):
            b_min, b_max = child.compute_bounds()
            axis_min, axis_max = b_min[self.axis_idx].item(), b_max[self.axis_idx].item()
        else:
            axis_min, axis_max = -1.0, 1.0
            
        self.register_buffer("_axis_min", torch.tensor(axis_min, dtype=torch.float32))
        self.register_buffer("_axis_max", torch.tensor(axis_max, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Calculate local scale factor
        axis_val = x[:, self.axis_idx]
        axis_min = self._axis_min.to(x.device)
        axis_max = self._axis_max.to(x.device)
        extent = axis_max - axis_min + 1e-8
        t = (axis_val - axis_min) / extent
        t = _smoothstep(t)

        scale = self.scale_min + t * (self.scale_max - self.scale_min)
        scale = torch.clamp(scale, min=1e-4).unsqueeze(1)
        
        # Warp domain
        if self.axis_idx == 0:
            warped_x = torch.stack([x[:, 0], x[:, 1] / scale.squeeze(), x[:, 2] / scale.squeeze()], dim=1)
        elif self.axis_idx == 1:
            warped_x = torch.stack([x[:, 0] / scale.squeeze(), x[:, 1], x[:, 2] / scale.squeeze()], dim=1)
        else:
            warped_x = torch.stack([x[:, 0] / scale.squeeze(), x[:, 1] / scale.squeeze(), x[:, 2]], dim=1)

        # Evaluate child
        res = self.child(warped_x)
        if isinstance(res, tuple):
            dist, attrs = res
        else:
            dist, attrs = res, None
            
        # Lipschitz correction: multiply distance by min(1.0, scale)
        # When expanding space (scale < 1.0), we must reduce distance to avoid overstepping.
        # When compressing space (scale > 1.0), distance is conservative (understepping) so we clamp to 1.0.
        correction = torch.clamp(scale, max=1.0).squeeze(1)
        dist = dist * correction
        
        if attrs is not None:
            return dist, attrs
        return dist

    def warp(self, x: torch.Tensor) -> torch.Tensor:
        # Kept for compatibility / bounds computation helper
        axis_val = x[:, self.axis_idx]
        axis_min = self._axis_min.to(x.device)
        axis_max = self._axis_max.to(x.device)
        extent = axis_max - axis_min + 1e-8
        t = (axis_val - axis_min) / extent
        t = _smoothstep(t)

        scale = self.scale_min + t * (self.scale_max - self.scale_min)
        scale = torch.clamp(scale, min=1e-4).unsqueeze(1)
        
        if self.axis_idx == 0:
            return torch.stack([x[:, 0], x[:, 1] / scale.squeeze(), x[:, 2] / scale.squeeze()], dim=1)
        elif self.axis_idx == 1:
            return torch.stack([x[:, 0] / scale.squeeze(), x[:, 1], x[:, 2] / scale.squeeze()], dim=1)
        else:
            return torch.stack([x[:, 0] / scale.squeeze(), x[:, 1] / scale.squeeze(), x[:, 2]], dim=1)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        b_min, b_max = self.child.compute_bounds()
        scale_limit = max(self.scale_max, self.scale_min)
        extent = torch.max(torch.abs(b_min), torch.abs(b_max))
        new_extent = extent.clone()
        for i in range(3):
            if i != self.axis_idx:
                new_extent[i] *= scale_limit
        return (-new_extent, new_extent)


def _smooth_abs(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    ax = torch.abs(x)
    blend = torch.where(ax < eps, 0.5 * x * x / (eps + 1e-8), ax - 0.5 * eps)
    return torch.sign(x + 1e-8) * blend


class MirrorModifier(DomainWarpNode):
    """Mirror space across an axis plane."""
    def __init__(self, child: GeometryNode, axis: str = "x"):
        super().__init__(child)
        self.axis_idx = AXIS_INDEX.get(axis.lower(), 0)
        
    def warp(self, x: torch.Tensor) -> torch.Tensor:
        mirrored = x.clone()
        mirrored[:, self.axis_idx] = _smooth_abs(mirrored[:, self.axis_idx])
        return mirrored

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        b_min, b_max = self.child.compute_bounds()
        extent_val = max(abs(b_min[self.axis_idx]), abs(b_max[self.axis_idx]))
        new_min = b_min.clone()
        new_max = b_max.clone()
        new_min[self.axis_idx] = -extent_val
        new_max[self.axis_idx] = extent_val
        return (new_min, new_max)


# =============================================================================
# Value Modifiers (Affect Distance)
# =============================================================================

class RoundModifier(ModifierNode):
    """Round edges by subtracting from the SDF distance."""
    def __init__(self, child: GeometryNode, radius: float = 0.02):
        super().__init__(child)
        self.radius = radius
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        res = self.child(x)
        if isinstance(res, tuple):
            dist, attrs = res
            return dist - self.radius, attrs
        return res - self.radius

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        b_min, b_max = self.child.compute_bounds()
        r = self.radius
        return (b_min - r, b_max + r)


class ChamferModifier(ModifierNode):
    """Chamfer edges with a flat 45-degree bevel (linear offset method).

    Unlike round (constant SDF offset producing curved edges), chamfer applies
    a distance-dependent offset that linearly tapers from full at the surface
    to zero at ``width``, creating a flat bevel — same as a 45-degree chamfer
    cut on a milling machine.
    """

    def __init__(self, child: GeometryNode, width: float = 0.02):
        super().__init__(child)
        self.width = width

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        res = self.child(x)
        if isinstance(res, tuple):
            dist, attrs = res
        else:
            dist, attrs = res, None

        # Linear offset: full at surface (d=0), zero at d=width
        t = torch.clamp(dist / self.width, 0.0, 1.0)
        dist = dist - self.width * (1.0 - t)

        if attrs is not None:
            return dist, attrs
        return dist

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        b_min, b_max = self.child.compute_bounds()
        w = self.width
        return (b_min - w, b_max + w)


# =============================================================================
# Complex/Procedural Modifiers (Voronoi, etc)
# =============================================================================

# Pre-computed 27-neighbor offsets
_NEIGHBOR_OFFSETS = torch.tensor(
    [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
    dtype=torch.float32,
)

def _voronoi_hash(p: torch.Tensor) -> torch.Tensor:
    p_int = (p * 127.1 + 311.7).sin() * 43758.5453
    return p_int - torch.floor(p_int)

class VoronoiModifier(ModifierNode):
    """Voronoi cellular pattern modifier."""
    def __init__(self, child: GeometryNode, cell_size: float = 0.2,
                 wall_thickness: float = 0.02, mode: str = "subtract"):
        super().__init__(child)
        self.cell_size = cell_size
        self.wall_thickness = wall_thickness
        self.mode = mode
        self.register_buffer('offsets', _NEIGHBOR_OFFSETS.clone())

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # TODO: This logic assumes child only returns dist for now unless we unpack?
        res = self.child(x)
        if isinstance(res, tuple):
            child_dist, attrs = res
        else:
            child_dist, attrs = res, None
            
        scaled = x / self.cell_size
        cell = torch.floor(scaled)
        neighbors = cell.unsqueeze(1) + self.offsets.unsqueeze(0)
        flat_neighbors = neighbors.reshape(-1, 3)
        jitter = _voronoi_hash(flat_neighbors)
        centers = (flat_neighbors + jitter) * self.cell_size
        centers = centers.reshape(x.shape[0], 27, 3)
        diffs = x.unsqueeze(1) - centers
        dists_to_centers = torch.norm(diffs, dim=2)
        voronoi_dist = torch.min(dists_to_centers, dim=1)[0] - self.wall_thickness
        
        if self.mode == "intersect":
            result_dist = torch.max(child_dist, voronoi_dist)
        else:
            result_dist = torch.max(child_dist, -voronoi_dist)
            
        if attrs is not None:
            return result_dist, attrs
        return result_dist


def _get_child_min_half_extent(child: GeometryNode) -> float:
    """Get the smallest half-extent of a child node for proportional clamping."""
    if hasattr(child, "compute_bounds"):
        try:
            b_min, b_max = child.compute_bounds()
            extents = (b_max - b_min) / 2.0
            positive = extents[extents > 1e-6]
            if len(positive) > 0:
                return positive.min().item()
        except Exception:
            pass
    return 0.05  # Conservative fallback


def build_modifier(child: GeometryNode, modifier_data: Dict) -> GeometryNode:
    """Build a modifier node from JSON data with defensive parameter clamping."""
    mod_type = modifier_data.get("type", "").lower()
    
    if mod_type == "twist":
        rate = float(modifier_data.get("rate", 1.0))
        rate = max(-10.0, min(10.0, rate))
        return TwistModifier(
            child,
            axis=modifier_data.get("axis", "y"),
            rate=rate,
        )
    elif mod_type == "bend":
        angle = float(modifier_data.get("angle", 0.5))
        angle = max(-3.0, min(3.0, angle))
        return BendModifier(
            child,
            axis=modifier_data.get("axis", "x"),
            angle=angle,
        )
    elif mod_type == "taper":
        scale_min = float(modifier_data.get("scale_min", 0.5))
        scale_max = float(modifier_data.get("scale_max", 1.0))
        scale_min_clamped = max(0.3, min(2.0, scale_min))
        scale_max_clamped = max(0.3, min(2.0, scale_max))
        if scale_min_clamped != scale_min or scale_max_clamped != scale_max:
            print(f"      [modifier] Clamped taper scale: [{scale_min},{scale_max}] -> [{scale_min_clamped},{scale_max_clamped}]", flush=True)
        # Skip taper if child bounds are degenerate
        if hasattr(child, "compute_bounds"):
            try:
                b_min, b_max = child.compute_bounds()
                axis_idx = AXIS_INDEX.get(str(modifier_data.get("axis", "y")).lower(), 1)
                extent = (b_max[axis_idx] - b_min[axis_idx]).item()
                if extent < 0.001:
                    print(f"      [modifier] Skipping taper: degenerate axis extent ({extent:.4f})", flush=True)
                    return child
            except Exception:
                pass
        return TaperModifier(
            child,
            axis=modifier_data.get("axis", "y"),
            scale_min=scale_min_clamped,
            scale_max=scale_max_clamped,
        )
    elif mod_type == "mirror":
        return MirrorModifier(
            child,
            axis=modifier_data.get("axis", "x"),
        )
    elif mod_type == "round":
        radius = float(modifier_data.get("radius", 0.02))
        max_radius = _get_child_min_half_extent(child) * 0.4
        if radius > max_radius and max_radius > 0.001:
            print(f"      [modifier] Clamped round radius: {radius:.4f} -> {max_radius:.4f} (40% of min half-extent)", flush=True)
            radius = max_radius
        radius = max(0.0005, radius)  # Floor to avoid invisible rounding
        return RoundModifier(child, radius=radius)
    elif mod_type == "chamfer":
        width = float(modifier_data.get("width", 0.02))
        max_width = _get_child_min_half_extent(child) * 0.4
        if width > max_width and max_width > 0.001:
            print(f"      [modifier] Clamped chamfer width: {width:.4f} -> {max_width:.4f} (40% of min half-extent)", flush=True)
            width = max_width
        width = max(0.0005, width)  # Floor to avoid invisible chamfer
        return ChamferModifier(child, width=width)
    elif mod_type == "voronoi":
        return VoronoiModifier(
            child,
            cell_size=float(modifier_data.get("cell_size", 0.2)),
            wall_thickness=float(modifier_data.get("wall_thickness", 0.02)),
            mode=modifier_data.get("mode", "subtract"),
        )
    else:
        return child
