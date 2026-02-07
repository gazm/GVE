"""SDF Domain Modifiers - Space warping transformations."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class TransformNode(nn.Module):
    """Wrapper that applies translation and rotation to a child SDF."""
    def __init__(self, child: nn.Module, pos: List[float], rot: List[float]):
        super().__init__()
        self.child = child
        # Store translation
        self.register_buffer('translation', torch.tensor(pos, dtype=torch.float32))
        # Store rotation as quaternion [x, y, z, w]
        q = torch.tensor(rot, dtype=torch.float32)
        self.register_buffer('quat', q / (torch.norm(q) + 1e-8))  # Normalize
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Translate points to local space (subtract position)
        local_p = x - self.translation
        # Rotate points by inverse quaternion
        local_p = self._rotate_by_inverse_quat(local_p)
        return self.child(local_p)
    
    def _rotate_by_inverse_quat(self, p: torch.Tensor) -> torch.Tensor:
        """Rotate points by inverse of stored quaternion."""
        # Inverse of unit quaternion is conjugate: [-x, -y, -z, w]
        q = self.quat
        qx, qy, qz, qw = -q[0], -q[1], -q[2], q[3]
        
        # Rodrigues' rotation formula via quaternion
        # p' = p + 2*qw*(q_xyz x p) + 2*(q_xyz x (q_xyz x p))
        q_xyz = torch.stack([qx, qy, qz])
        
        # Cross product: q_xyz x p (for each point)
        # p is [N, 3]
        t = 2.0 * torch.cross(q_xyz.unsqueeze(0).expand(p.shape[0], -1), p, dim=1)
        return p + qw * t + torch.cross(q_xyz.unsqueeze(0).expand(p.shape[0], -1), t, dim=1)


class TwistModifier(nn.Module):
    """Twist space around an axis. Rotation angle is proportional to position along axis."""
    def __init__(self, child: nn.Module, axis: str = "y", rate: float = 1.0):
        super().__init__()
        self.child = child
        self.axis_idx = AXIS_INDEX.get(axis.lower(), 1)
        self.rate = rate
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the component along the twist axis
        axis_val = x[:, self.axis_idx]
        angle = axis_val * self.rate
        
        c = torch.cos(angle)
        s = torch.sin(angle)
        
        # Twist rotates the other two axes
        if self.axis_idx == 0:  # X axis - twist YZ
            twisted = torch.stack([x[:, 0], c * x[:, 1] - s * x[:, 2], s * x[:, 1] + c * x[:, 2]], dim=1)
        elif self.axis_idx == 1:  # Y axis - twist XZ
            twisted = torch.stack([c * x[:, 0] - s * x[:, 2], x[:, 1], s * x[:, 0] + c * x[:, 2]], dim=1)
        else:  # Z axis - twist XY
            twisted = torch.stack([c * x[:, 0] - s * x[:, 1], s * x[:, 0] + c * x[:, 1], x[:, 2]], dim=1)
        
        return self.child(twisted)


class BendModifier(nn.Module):
    """Bend space around an axis."""
    def __init__(self, child: nn.Module, axis: str = "x", angle: float = 0.5):
        super().__init__()
        self.child = child
        self.axis_idx = AXIS_INDEX.get(axis.lower(), 0)
        self.k = angle  # Bend rate
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Bending is more complex - simplified version
        if self.axis_idx == 0:  # Bend around X - affects YZ based on Y
            c = torch.cos(self.k * x[:, 1])
            s = torch.sin(self.k * x[:, 1])
            bent = torch.stack([x[:, 0], c * x[:, 1] - s * x[:, 2], s * x[:, 1] + c * x[:, 2]], dim=1)
        elif self.axis_idx == 1:  # Bend around Y - affects XZ based on X
            c = torch.cos(self.k * x[:, 0])
            s = torch.sin(self.k * x[:, 0])
            bent = torch.stack([c * x[:, 0] - s * x[:, 2], x[:, 1], s * x[:, 0] + c * x[:, 2]], dim=1)
        else:  # Bend around Z - affects XY based on X
            c = torch.cos(self.k * x[:, 0])
            s = torch.sin(self.k * x[:, 0])
            bent = torch.stack([c * x[:, 0] - s * x[:, 1], s * x[:, 0] + c * x[:, 1], x[:, 2]], dim=1)
        
        return self.child(bent)


class TaperModifier(nn.Module):
    """Taper (scale cross-section) along an axis."""
    def __init__(self, child: nn.Module, axis: str = "y", scale_min: float = 0.5, scale_max: float = 1.0):
        super().__init__()
        self.child = child
        self.axis_idx = AXIS_INDEX.get(axis.lower(), 1)
        self.scale_min = scale_min
        self.scale_max = scale_max
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Normalize axis position to 0-1 range (assuming object spans -1 to 1)
        axis_val = x[:, self.axis_idx]
        t = (axis_val + 1.0) / 2.0
        t = torch.clamp(t, 0.0, 1.0)
        
        # Interpolate scale (clamped to avoid division-by-zero for aggressive tapers)
        scale = self.scale_min + t * (self.scale_max - self.scale_min)
        scale = torch.clamp(scale, min=1e-4)
        scale = scale.unsqueeze(1)  # [N, 1] for broadcasting
        
        # Scale the perpendicular axes
        if self.axis_idx == 0:  # Taper along X - scale YZ
            tapered = torch.stack([x[:, 0], x[:, 1] / scale.squeeze(), x[:, 2] / scale.squeeze()], dim=1)
        elif self.axis_idx == 1:  # Taper along Y - scale XZ
            tapered = torch.stack([x[:, 0] / scale.squeeze(), x[:, 1], x[:, 2] / scale.squeeze()], dim=1)
        else:  # Taper along Z - scale XY
            tapered = torch.stack([x[:, 0] / scale.squeeze(), x[:, 1] / scale.squeeze(), x[:, 2]], dim=1)
        
        return self.child(tapered)


class MirrorModifier(nn.Module):
    """Mirror space across an axis plane."""
    def __init__(self, child: nn.Module, axis: str = "x"):
        super().__init__()
        self.child = child
        self.axis_idx = AXIS_INDEX.get(axis.lower(), 0)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Take absolute value of the mirrored axis
        mirrored = x.clone()
        mirrored[:, self.axis_idx] = torch.abs(mirrored[:, self.axis_idx])
        return self.child(mirrored)


class RoundModifier(nn.Module):
    """Round edges by subtracting from the SDF distance."""
    def __init__(self, child: nn.Module, radius: float = 0.02):
        super().__init__()
        self.child = child
        self.radius = radius
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Rounding just subtracts from the distance
        dist, mat = self.child(x)
        return dist - self.radius, mat


# Pre-computed 27-neighbor offsets for vectorized Voronoi evaluation
_NEIGHBOR_OFFSETS = torch.tensor(
    [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
    dtype=torch.float32,
)  # [27, 3]


def _voronoi_hash(p: torch.Tensor) -> torch.Tensor:
    """Deterministic 3D hash: [N, 3] -> [N, 3] in [0, 1).
    
    Uses integer-arithmetic-based hash for GPU-reproducible results.
    """
    # Modular hashing via large primes (matches WGSL implementation)
    p_int = (p * 127.1 + 311.7).sin() * 43758.5453
    return p_int - torch.floor(p_int)


class VoronoiModifier(nn.Module):
    """Voronoi cellular pattern modifier. Vectorized 27-neighbor evaluation.
    
    Applies a 3D Voronoi pattern to the child SDF, creating cell structures.
    
    Args:
        child: Child SDF node to modify.
        cell_size: Size of each Voronoi cell in world units.
        wall_thickness: Thickness of cell walls.
        mode: 'subtract' carves cells, 'intersect' keeps only walls.
    """
    def __init__(self, child: nn.Module, cell_size: float = 0.2,
                 wall_thickness: float = 0.02, mode: str = "subtract"):
        super().__init__()
        self.child = child
        self.cell_size = cell_size
        self.wall_thickness = wall_thickness
        self.mode = mode
        # Register as buffer so it moves to correct device with model
        self.register_buffer('offsets', _NEIGHBOR_OFFSETS.clone())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        child_dist, child_color = self.child(x)
        
        # Compute Voronoi cell distance - fully vectorized
        scaled = x / self.cell_size
        cell = torch.floor(scaled)  # [N, 3]
        
        # All 27 neighbor centers: [N, 27, 3]
        neighbors = cell.unsqueeze(1) + self.offsets.unsqueeze(0)  # broadcast [N,1,3] + [1,27,3]
        
        # Hash neighbor cells to get jittered centers
        flat_neighbors = neighbors.reshape(-1, 3)
        jitter = _voronoi_hash(flat_neighbors)
        centers = (flat_neighbors + jitter) * self.cell_size  # world-space centers
        centers = centers.reshape(x.shape[0], 27, 3)  # [N, 27, 3]
        
        # Distance from each point to each center: [N, 27]
        diffs = x.unsqueeze(1) - centers  # [N, 27, 3]
        dists_to_centers = torch.norm(diffs, dim=2)  # [N, 27]
        
        # Minimum distance to nearest cell center
        voronoi_dist = torch.min(dists_to_centers, dim=1)[0] - self.wall_thickness
        
        # Combine with child SDF
        if self.mode == "intersect":
            result_dist = torch.max(child_dist, voronoi_dist)
        else:  # subtract (default)
            result_dist = torch.max(child_dist, -voronoi_dist)
        
        return result_dist, child_color


def build_modifier(child: nn.Module, modifier_data: Dict) -> nn.Module:
    """Build a modifier node from JSON data."""
    mod_type = modifier_data.get("type", "").lower()
    
    if mod_type == "twist":
        return TwistModifier(
            child,
            axis=modifier_data.get("axis", "y"),
            rate=float(modifier_data.get("rate", 1.0))
        )
    elif mod_type == "bend":
        return BendModifier(
            child,
            axis=modifier_data.get("axis", "x"),
            angle=float(modifier_data.get("angle", 0.5))
        )
    elif mod_type == "taper":
        return TaperModifier(
            child,
            axis=modifier_data.get("axis", "y"),
            scale_min=float(modifier_data.get("scale_min", 0.5)),
            scale_max=float(modifier_data.get("scale_max", 1.0))
        )
    elif mod_type == "mirror":
        return MirrorModifier(
            child,
            axis=modifier_data.get("axis", "x")
        )
    elif mod_type == "round":
        return RoundModifier(
            child,
            radius=float(modifier_data.get("radius", 0.02))
        )
    elif mod_type == "voronoi":
        return VoronoiModifier(
            child,
            cell_size=float(modifier_data.get("cell_size", 0.2)),
            wall_thickness=float(modifier_data.get("wall_thickness", 0.02)),
            mode=modifier_data.get("mode", "subtract"),
        )
    else:
        # Unknown modifier, return child unchanged
        return child
