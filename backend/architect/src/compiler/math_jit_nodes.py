"""SDF Primitive Nodes - Basic geometry shapes."""

import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional

print("🟢 Loading math_jit_nodes.py...", flush=True)

# Oklab mid-gray (matches math_jit_builder._resolve_material default). Do not use sRGB [0.5,0.5,0.5].
_DEFAULT_OKLAB = [0.627, 0.0, 0.0]


class PrimitiveNode(nn.Module):
    """Base class for primitive nodes handling material attributes.

    Attributes tensor layout: [oklab_L, oklab_a, oklab_b, metallic, roughness]
    """
    def __init__(
        self,
        color: List[float] = None,
        metallic: float = 0.0,
        roughness: float = 0.5,
    ):
        super().__init__()
        c = color if color is not None else _DEFAULT_OKLAB
        attrs = c + [metallic, roughness]
        self.register_buffer('attrs', torch.tensor(attrs, dtype=torch.float32))

    def _return_with_attributes(self, dist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attach [N, 5] attribute tensor (color + metallic + roughness) to distances."""
        return dist, self.attrs.unsqueeze(0).expand(dist.shape[0], 5)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate local AABB (min_xyz, max_xyz).
        
        Returns:
            (min, max) tuple of [3] tensors.
        """
        # Default unit box if not implemented
        return (
            torch.tensor([-1.0, -1.0, -1.0], device=self.attrs.device),
            torch.tensor([1.0, 1.0, 1.0], device=self.attrs.device)
        )


class SphereNode(PrimitiveNode):
    def __init__(self, radius: float, color: List[float] = None, metallic: float = 0.0, roughness: float = 0.5):
        super().__init__(color, metallic, roughness)
        self.radius = radius

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ||x|| - r
        dist = torch.norm(x, dim=1) - self.radius
        return self._return_with_attributes(dist)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.radius
        return (
            torch.tensor([-r, -r, -r], device=self.attrs.device),
            torch.tensor([r, r, r], device=self.attrs.device)
        )


class BoxNode(PrimitiveNode):
    def __init__(self, size: List[float], color: List[float] = None, metallic: float = 0.0, roughness: float = 0.5):
        super().__init__(color, metallic, roughness)
        # Size IS already half-extents from AI output (box spans from -size to +size)
        self.register_buffer('b', torch.tensor(size, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = torch.abs(x) - self.b
        dist = torch.norm(torch.clamp(q, min=0.0), dim=1) + \
               torch.clamp(torch.max(q, dim=1)[0], max=0.0)
        return self._return_with_attributes(dist)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (-self.b, self.b)


class CylinderNode(PrimitiveNode):
    def __init__(self, radius: float, height: float, color: List[float] = None, metallic: float = 0.0, roughness: float = 0.5):
        super().__init__(color, metallic, roughness)
        self.radius = radius
        self.height = height

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cylinder aligned with Z axis (forward direction for weapons/barrels)
        d_xy = torch.norm(x[:, [0, 1]], dim=1) - self.radius
        d_z = torch.abs(x[:, 2]) - self.height / 2.0
        
        d_vec = torch.stack([d_xy, d_z], dim=1)
        
        inside = torch.min(torch.max(d_vec, dim=1)[0], torch.tensor(0.0, device=x.device))
        outside = torch.norm(torch.clamp(d_vec, min=0.0), dim=1)
        
        dist = inside + outside
        return self._return_with_attributes(dist)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.radius
        h = self.height / 2.0
        return (
            torch.tensor([-r, -r, -h], device=self.attrs.device),
            torch.tensor([r, r, h], device=self.attrs.device)
        )


class TorusNode(PrimitiveNode):
    """Torus (donut) SDF - ring in XZ plane, Y is up."""
    def __init__(self, major_r: float, minor_r: float, color: List[float] = None, metallic: float = 0.0, roughness: float = 0.5):
        super().__init__(color, metallic, roughness)
        self.major_r = major_r
        self.minor_r = minor_r

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Distance to ring center in XZ plane
        q_xz = torch.norm(x[:, [0, 2]], dim=1) - self.major_r
        # Distance to tube surface
        q = torch.stack([q_xz, x[:, 1]], dim=1)
        dist = torch.norm(q, dim=1) - self.minor_r
        return self._return_with_attributes(dist)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # XZ plane ring
        R = self.major_r + self.minor_r
        r = self.minor_r
        return (
            torch.tensor([-R, -r, -R], device=self.attrs.device),
            torch.tensor([R, r, R], device=self.attrs.device)
        )


class ConeNode(PrimitiveNode):
    """Cone SDF - base at -Z, tip at +Z, opens along Z axis."""
    def __init__(self, radius: float, height: float, color: List[float] = None, metallic: float = 0.0, roughness: float = 0.5):
        super().__init__(color, metallic, roughness)
        self.radius = radius
        self.height = height
        # Compute cone angle
        self.sin_cos = torch.tensor([radius, height], dtype=torch.float32)
        self.sin_cos = self.sin_cos / torch.norm(self.sin_cos)  # [sin, cos]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shift so base is at z=0, tip at z=height (Z-axis aligned)
        p_z = x[:, 2] - self.height / 2.0
        
        # 2D distance in XY plane
        q = torch.stack([torch.norm(x[:, [0, 1]], dim=1), p_z], dim=1)
        
        # Simple bounded cone SDF
        d_side = q[:, 0] * self.sin_cos[1] - (self.height - q[:, 1]) * self.sin_cos[0]
        d_base = -q[:, 1]
        d_top = q[:, 1] - self.height
        
        # Inside: max of all constraints (negative when inside)
        inside = torch.max(torch.stack([d_side, d_base], dim=1), dim=1)[0]
        
        # Clamp to reasonable bounds
        dist = torch.clamp(inside, min=-self.height, max=self.height)
        return self._return_with_attributes(dist)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Bounds based on previous analysis: z in [h/2, 3h/2]
        # Base (widest) at z=h/2 with radius r.
        r = self.radius
        z_min = self.height / 2.0
        z_max = self.height * 1.5
        return (
            torch.tensor([-r, -r, z_min], device=self.attrs.device),
            torch.tensor([r, r, z_max], device=self.attrs.device)
        )


class CapsuleNode(PrimitiveNode):
    """Capsule SDF - cylinder with hemispherical caps, aligned with Z axis."""
    def __init__(self, radius: float, height: float, color: List[float] = None, metallic: float = 0.0, roughness: float = 0.5):
        super().__init__(color, metallic, roughness)
        self.radius = radius
        self.half_height = height / 2.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Clamp Z to line segment (Z-axis aligned)
        clamped_z = torch.clamp(x[:, 2], -self.half_height, self.half_height)
        # Build clamped point tensor
        p = torch.stack([x[:, 0], x[:, 1], clamped_z], dim=1)
        # Distance from clamped point
        dist = torch.norm(x - p, dim=1) - self.radius
        return self._return_with_attributes(dist)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.radius
        h = self.half_height
        # Capsule extends from -h to h in Z, plus radius caps
        return (
            torch.tensor([-r, -r, -h - r], device=self.attrs.device),
            torch.tensor([r, r, h + r], device=self.attrs.device)
        )


class PlaneNode(PrimitiveNode):
    def __init__(self, normal: List[float], distance: float, color: List[float] = None, metallic: float = 0.0, roughness: float = 0.5):
        super().__init__(color, metallic, roughness)
        n = torch.tensor(normal, dtype=torch.float32)
        self.register_buffer('n', torch.nn.functional.normalize(n, dim=0))
        self.distance = distance

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = torch.matmul(x, self.n) + self.distance
        return self._return_with_attributes(dist)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Infinite. Return large box.
        inf = 1000.0
        return (
            torch.tensor([-inf, -inf, -inf], device=self.attrs.device),
            torch.tensor([inf, inf, inf], device=self.attrs.device)
        )


class WedgeNode(PrimitiveNode):
    """Wedge/ramp SDF — box intersected with a diagonal cutting plane.
    
    Creates a triangular prism (wedge) by slicing a box along a diagonal.
    ``taper_axis`` shrinks from full-width to zero across ``taper_dir``.

    Args:
        size: Half-extents ``[x, y, z]``.
        taper_axis: Axis that tapers to zero (``"x"``, ``"y"``, or ``"z"``).
        taper_dir: Axis along which the taper progresses (``"x"``, ``"y"``, or ``"z"``).
    """

    _AXIS = {"x": 0, "y": 1, "z": 2}

    def __init__(
        self,
        size: List[float],
        taper_axis: str = "y",
        taper_dir: str = "z",
        color: List[float] = None,
        metallic: float = 0.0,
        roughness: float = 0.5,
    ):
        super().__init__(color, metallic, roughness)
        self.register_buffer("b", torch.tensor(size, dtype=torch.float32))
        self.taper_idx = self._AXIS.get(taper_axis.lower(), 1)
        self.dir_idx = self._AXIS.get(taper_dir.lower(), 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Standard box SDF
        q = torch.abs(x) - self.b
        box_dist = (
            torch.norm(torch.clamp(q, min=0.0), dim=1)
            + torch.clamp(torch.max(q, dim=1)[0], max=0.0)
        )

        # 2. Diagonal cutting plane: taper_axis <= size * (1 - t)
        #    where t = (dir_pos + size_dir) / (2 * size_dir) ∈ [0, 1]
        size_dir = self.b[self.dir_idx]
        size_tap = self.b[self.taper_idx]
        t = (x[:, self.dir_idx] + size_dir) / (2.0 * size_dir + 1e-8)
        t = torch.clamp(t, 0.0, 1.0)
        # At t=0 full width allowed; at t=1 width is 0
        allowed = size_tap * (1.0 - t)
        plane_dist = torch.abs(x[:, self.taper_idx]) - allowed

        # 3. Smooth intersection of box and half-space (avoids gradient discontinuity at diagonal)
        #    Sharp max(box, plane) causes jagged artifacts; smooth_intersect gives C1 continuity.
        k = 0.02
        h = torch.clamp(0.5 - 0.5 * (plane_dist - box_dist) / (k + 1e-8), 0.0, 1.0)
        dist = torch.lerp(plane_dist, box_dist, h) + k * h * (1.0 - h)
        return self._return_with_attributes(dist)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Bounded by the box
        return (-self.b, self.b)


class RevolutionNode(nn.Module):
    """Revolution SDF - spins a child 2D profile around an axis."""
    def __init__(self, child: nn.Module, axis: str = "y", offset: float = 0.0):
        super().__init__()
        self.child = child
        self.axis = axis.lower()
        self.offset = offset

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project 3D point to 2D (radial_distance - offset, height)
        if self.axis == "y":
            radial = torch.norm(x[:, [0, 2]], dim=1) - self.offset
            p2d = torch.stack([radial, x[:, 1]], dim=1)
        elif self.axis == "x":
            radial = torch.norm(x[:, [1, 2]], dim=1) - self.offset
            p2d = torch.stack([radial, x[:, 0]], dim=1)
        else:  # z
            radial = torch.norm(x[:, [0, 1]], dim=1) - self.offset
            p2d = torch.stack([radial, x[:, 2]], dim=1)
        # Pad to 3D with zero (child expects [N, 3])
        p3d = torch.zeros_like(x)
        p3d[:, 0] = p2d[:, 0]
        p3d[:, 1] = p2d[:, 1]
        return self.child(p3d)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Assume child is 2D shape defined in XY (or first 2 axes of 3D)
        if hasattr(self.child, "compute_bounds"):
            c_min, c_max = self.child.compute_bounds()
        else:
            c_min = torch.tensor([-1.0, -1.0, -1.0])
            c_max = torch.tensor([1.0, 1.0, 1.0])
        
        # Child 2D bounds:
        # x (radial) -> expand by offset
        # y (axial) -> keep
        
        # Taking "radial" bounds from child's X
        r_min, r_max = c_min[0], c_max[0]
        # Correct radial bounds including negative side of revolution
        # If child extends from r_min to r_max, and we offset by O:
        # The swept profile goes from (r_min+O) to (r_max+O).
        # Since it revolves, the max radius is max(abs(r_min+O), abs(r_max+O))
        max_rad = max(abs(r_min + self.offset), abs(r_max + self.offset))
        
        # Axial bounds from child's Y
        h_min, h_max = c_min[1], c_max[1]
        
        device = c_min.device
        if self.axis == "y":
            # x, z are radial. y is axial.
            return (
                torch.tensor([-max_rad, h_min, -max_rad], device=device),
                torch.tensor([max_rad, h_max, max_rad], device=device)
            )
        elif self.axis == "x":
            # y, z are radial. x is axial.
            # Using child coords: x->radial, y->axial
            # So child Y maps to world X.
            return (
                torch.tensor([h_min, -max_rad, -max_rad], device=device),
                torch.tensor([h_max, max_rad, max_rad], device=device)
            )
        else: # z
            # x, y are radial. z is axial.
            return (
                torch.tensor([-max_rad, -max_rad, h_min], device=device),
                torch.tensor([max_rad, max_rad, h_max], device=device)
            )


# =============================================================================
# CSG Operations
# =============================================================================

def _union_bounds(nodes):
    if not nodes:
        return (torch.tensor([-1.]*3), torch.tensor([1.]*3))
    
    mins, maxs = [], []
    for n in nodes:
        if hasattr(n, "compute_bounds"):
            b_min, b_max = n.compute_bounds()
            mins.append(b_min)
            maxs.append(b_max)
    
    if not mins:
        return (torch.tensor([-1.]*3), torch.tensor([1.]*3))
        
    # Stack and find extents
    all_mins = torch.stack(mins)
    all_maxs = torch.stack(maxs)
    
    return (
        torch.min(all_mins, dim=0)[0],
        torch.max(all_maxs, dim=0)[0]
    )


class UnionNode(nn.Module):
    """Boolean union: min(d1, d2). Attributes from closest child."""
    def __init__(self, children: List[nn.Module]):
        super().__init__()
        self.child_nodes = nn.ModuleList(children)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        results = [child(x) for child in self.child_nodes]
        dists = torch.stack([r[0] for r in results], dim=1)
        attrs = torch.stack([r[1] for r in results], dim=1)  # [N, C, 5]
        min_vals, min_indices = torch.min(dists, dim=1)
        n_attrs = attrs.shape[2]
        s_indices = min_indices.view(-1, 1, 1).expand(-1, 1, n_attrs)
        selected = torch.gather(attrs, 1, s_indices).squeeze(1)
        return min_vals, selected

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return _union_bounds(self.child_nodes)


class SubtractNode(nn.Module):
    """Boolean subtract: max(d1, -d2). Color from the first child."""
    def __init__(self, children: List[nn.Module]):
        super().__init__()
        self.child_nodes = nn.ModuleList(children)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.child_nodes) < 2:
            return self.child_nodes[0](x)
        d1, c1 = self.child_nodes[0](x)
        d2, _c2 = self.child_nodes[1](x)
        return torch.max(d1, -d2), c1

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.child_nodes:
            return (torch.tensor([-1.]*3), torch.tensor([1.]*3))
        # Bounded by Positive shape (first child)
        if hasattr(self.child_nodes[0], "compute_bounds"):
            return self.child_nodes[0].compute_bounds()
        return (torch.tensor([-1.]*3), torch.tensor([1.]*3))


class IntersectNode(nn.Module):
    """Boolean intersect: max(d1, d2). Attributes from furthest child."""
    def __init__(self, children: List[nn.Module]):
        super().__init__()
        self.child_nodes = nn.ModuleList(children)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.child_nodes) < 2:
            return self.child_nodes[0](x)
        d1, a1 = self.child_nodes[0](x)
        d2, a2 = self.child_nodes[1](x)
        dist = torch.max(d1, d2)
        mask = (d1 > d2).unsqueeze(1).expand_as(a1)
        attrs = torch.where(mask, a1, a2)
        return dist, attrs

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.child_nodes:
            return (torch.tensor([-1.]*3), torch.tensor([1.]*3))
        
        # Intersection of bounds
        mins, maxs = [], []
        for n in self.child_nodes:
            if hasattr(n, "compute_bounds"):
                b_min, b_max = n.compute_bounds()
                mins.append(b_min)
                maxs.append(b_max)
        
        if not mins:
             return (torch.tensor([-1.]*3), torch.tensor([1.]*3))

        # Max of mins, Min of maxs
        return (
            torch.max(torch.stack(mins), dim=0)[0],
            torch.min(torch.stack(maxs), dim=0)[0]
        )


class SmoothUnionNode(nn.Module):
    """Smooth union: IQ polynomial smooth min. Attributes from closest child (per-node materials)."""
    def __init__(self, children: List[nn.Module], k: float = 0.5):
        super().__init__()
        self.child_nodes = nn.ModuleList(children)
        self.k = k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.child_nodes) < 2:
            return self.child_nodes[0](x)
        d1, a1 = self.child_nodes[0](x)
        d2, a2 = self.child_nodes[1](x)
        k = self.k
        h = torch.clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
        mix_dist = torch.lerp(d2, d1, h) - k * h * (1.0 - h)
        closer_first = (d1 <= d2).unsqueeze(1).expand_as(a1)
        mix_attrs = torch.where(closer_first, a1, a2)
        return mix_dist, mix_attrs

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Union + padding k
        min_b, max_b = _union_bounds(self.child_nodes)
        pad = self.k
        return (min_b - pad, max_b + pad)


class SmoothSubtractNode(nn.Module):
    """Smooth subtract: IQ filleted concave edges. Color from first child."""
    def __init__(self, children: List[nn.Module], k: float = 0.5):
        super().__init__()
        self.child_nodes = nn.ModuleList(children)
        self.k = k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.child_nodes) < 2:
            return self.child_nodes[0](x)
        d1, c1 = self.child_nodes[0](x)
        d2, _c2 = self.child_nodes[1](x)
        k = self.k
        h = torch.clamp(0.5 - 0.5 * (d1 + d2) / k, 0.0, 1.0)
        mix_dist = torch.lerp(d1, -d2, h) + k * h * (1.0 - h)
        return mix_dist, c1

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Positive shape + padding k
        if not self.child_nodes:
            return (torch.tensor([-1.]*3), torch.tensor([1.]*3))
        if hasattr(self.child_nodes[0], "compute_bounds"):
            b_min, b_max = self.child_nodes[0].compute_bounds()
            return (b_min - self.k, b_max + self.k)
        return (torch.tensor([-1.]*3), torch.tensor([1.]*3))


class SmoothIntersectNode(nn.Module):
    """Smooth intersect: IQ filleted convex edges. Attributes from dominating child (per-node materials)."""
    def __init__(self, children: List[nn.Module], k: float = 0.5):
        super().__init__()
        self.child_nodes = nn.ModuleList(children)
        self.k = k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.child_nodes) < 2:
            return self.child_nodes[0](x)
        d1, a1 = self.child_nodes[0](x)
        d2, a2 = self.child_nodes[1](x)
        k = self.k
        h = torch.clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
        mix_dist = torch.lerp(d2, d1, h) + k * h * (1.0 - h)
        # At intersection surface, take attrs from child with larger d (dominant interior)
        dominant_first = (d1 >= d2).unsqueeze(1).expand_as(a1)
        mix_attrs = torch.where(dominant_first, a1, a2)
        return mix_dist, mix_attrs

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Intersection - but smoothing might expand?
        # Safe to just use intersection of padded?
        # Actually min(d1,d2) - k ... 
        # Smooth intersection is approx max(d1, d2) + ...
        # If we take intersection of bounds, we are safe.
        # Maybe pad by k just in case.
        
        mins, maxs = [], []
        for n in self.child_nodes:
            if hasattr(n, "compute_bounds"):
                b_min, b_max = n.compute_bounds()
                mins.append(b_min)
                maxs.append(b_max)
        
        if not mins:
             return (torch.tensor([-1.]*3), torch.tensor([1.]*3))

        b_min = torch.max(torch.stack(mins), dim=0)[0]
        b_max = torch.min(torch.stack(maxs), dim=0)[0]
        
        pad = self.k
        return (b_min - pad, b_max + pad)


# =============================================================================
# Fractal Primitives
# =============================================================================

class MandelbulbNode(PrimitiveNode):
    """Mandelbulb fractal SDF - 3D power-N Mandelbrot. Vectorized iteration."""
    def __init__(self, power: float = 8.0, iterations: int = 8,
                 scale: float = 1.0, color: List[float] = None,
                 metallic: float = 0.0, roughness: float = 0.5):
        super().__init__(color, metallic, roughness)
        self.power = power
        self.iterations = min(int(iterations), 12)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Scale input to unit space
        p = x / self.scale
        z = p.clone()
        dr = torch.ones(x.shape[0], device=x.device)
        r = torch.zeros(x.shape[0], device=x.device)

        for _ in range(self.iterations):
            r = torch.norm(z, dim=1)
            # Mask: skip escaped points (but keep tensor shape for vectorization)
            escaped = r > 2.0

            # Spherical coordinates
            theta = torch.acos(torch.clamp(z[:, 2] / (r + 1e-8), -1.0, 1.0))
            phi = torch.atan2(z[:, 1], z[:, 0])

            # Power transform
            r_pow = torch.pow(r, self.power)
            dr = torch.where(escaped, dr, r_pow * self.power * dr + 1.0)

            theta_n = theta * self.power
            phi_n = phi * self.power

            # Back to cartesian
            sin_theta = torch.sin(theta_n)
            new_z = torch.stack([
                sin_theta * torch.cos(phi_n),
                sin_theta * torch.sin(phi_n),
                torch.cos(theta_n),
            ], dim=1) * r_pow.unsqueeze(1) + p

            z = torch.where(escaped.unsqueeze(1).expand_as(z), z, new_z)

        r = torch.norm(z, dim=1)
        dist = 0.5 * torch.log(r + 1e-8) * r / (dr + 1e-8) * self.scale
        return self._return_with_attributes(dist)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Mandelbulb is roughly radius 1.2 * scale
        r = 1.5 * self.scale
        return (
            torch.tensor([-r, -r, -r], device=self.attrs.device),
            torch.tensor([r, r, r], device=self.attrs.device)
        )


class MengerSpongeNode(PrimitiveNode):
    """Menger Sponge fractal SDF - recursive cross subtraction. Vectorized."""
    def __init__(self, iterations: int = 3, scale: float = 1.0,
                 color: List[float] = None, metallic: float = 0.0, roughness: float = 0.5):
        super().__init__(color, metallic, roughness)
        self.iterations = min(int(iterations), 5)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = x / self.scale
        # Start with unit box
        q = torch.abs(p)
        dist = torch.max(torch.max(q[:, 0], q[:, 1]), q[:, 2]) - 1.0

        s = 1.0
        for _ in range(self.iterations):
            # Fold into positive octant and center on repeated cell
            a = torch.remainder(q * s, 2.0) - 1.0
            s *= 3.0
            r = torch.abs(1.0 - 3.0 * torch.abs(a))

            # Cross distances (infinite cross along each axis)
            da = torch.max(r[:, 0], r[:, 1])
            db = torch.max(r[:, 1], r[:, 2])
            dc = torch.max(r[:, 0], r[:, 2])
            c = (torch.min(torch.min(da, db), dc) - 1.0) / s

            dist = torch.max(dist, c)

        return self._return_with_attributes(dist * self.scale)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Unit box [-1, 1] scaled
        s = self.scale
        return (
            torch.tensor([-s, -s, -s], device=self.attrs.device),
            torch.tensor([s, s, s], device=self.attrs.device)
        )


class JuliaSetNode(PrimitiveNode):
    """Quaternion Julia set fractal SDF. Vectorized iteration."""
    def __init__(self, c: List[float] = None, iterations: int = 8,
                 scale: float = 1.0, color: List[float] = None,
                 metallic: float = 0.0, roughness: float = 0.5):
        super().__init__(color, metallic, roughness)
        c_val = c if c is not None else [0.3, 0.5, 0.2, 0.1]
        self.register_buffer('c', torch.tensor(c_val[:4], dtype=torch.float32))
        self.iterations = min(int(iterations), 12)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = x / self.scale
        # Quaternion z = (p.x, p.y, p.z, 0)
        z = torch.zeros(x.shape[0], 4, device=x.device)
        z[:, :3] = p
        dz = torch.ones(x.shape[0], device=x.device)

        for _ in range(self.iterations):
            r = torch.norm(z, dim=1)
            escaped = r > 4.0

            # Quaternion square: z*z
            # (a + bi + cj + dk)^2 = a^2 - b^2 - c^2 - d^2 + 2a(bi + cj + dk)
            a, b, c_q, d = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
            new_z = torch.stack([
                a*a - b*b - c_q*c_q - d*d + self.c[0],
                2.0*a*b + self.c[1],
                2.0*a*c_q + self.c[2],
                2.0*a*d + self.c[3],
            ], dim=1)

            dz = torch.where(escaped, dz, 2.0 * r * dz)
            z = torch.where(escaped.unsqueeze(1).expand_as(z), z, new_z)

        r = torch.norm(z, dim=1)
        dist = 0.5 * r * torch.log(r + 1e-8) / (dz + 1e-8) * self.scale
        return self._return_with_attributes(dist)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Approximate bounds for Julia set (escape radius 2.0 approx)
        r = 2.0 * self.scale
        return (
            torch.tensor([-r, -r, -r], device=self.attrs.device),
            torch.tensor([r, r, r], device=self.attrs.device)
        )
