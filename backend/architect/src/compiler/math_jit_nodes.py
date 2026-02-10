"""SDF Primitive Nodes - Basic geometry shapes and CSG operations."""

import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional



# Safety Orange for missing materials
_DEFAULT_FALLBACK_COLOR = [1.0, 0.4, 0.0] 
# Oklab mid-gray default for initialized MaterialNodes
_DEFAULT_OKLAB = [0.627, 0.0, 0.0]


# =============================================================================
# Base Classes
# =============================================================================

class GeometryNode(nn.Module):
    """Base class for pure geometry (distance only)."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            dist: [N] tensor of signed distances.
            (optional) attrs: [N, 5] tensor if this node has embedded material (e.g. MaterialNode).
        """
        raise NotImplementedError

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate local AABB (min_xyz, max_xyz)."""
        # Default unit box safety fallback
        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        return (
            torch.tensor([-1.0, -1.0, -1.0], device=device),
            torch.tensor([1.0, 1.0, 1.0], device=device)
        )


class MaterialNode(GeometryNode):
    """Wraps a GeometryNode and assigns it a uniform material."""
    def __init__(
        self,
        child: GeometryNode,
        color: List[float] = None,
        metallic: float = 0.0,
        roughness: float = 0.5,
    ):
        super().__init__()
        self.child = child
        c = color if color is not None else _DEFAULT_OKLAB
        attrs = c + [metallic, roughness]
        self.register_buffer('attrs', torch.tensor(attrs, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.child(x)
        # If child returns tuple, it already has material - we override it? 
        # Or we act as a "paint" operation. For now, we override.
        if isinstance(result, tuple):
            dist, _ = result
        else:
            dist = result
            
        # Expand attributes to [N, 5]
        return dist, self.attrs.unsqueeze(0).expand(dist.shape[0], 5)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.child.compute_bounds()


class ModifierNode(GeometryNode):
    """Base for nodes that modify a single child."""
    def __init__(self, child: GeometryNode):
        super().__init__()
        self.child = child

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.child.compute_bounds()


class CSGNode(GeometryNode):
    """Base for boolean operations (Union, Subtract, Intersect)."""
    def __init__(self, children: List[GeometryNode]):
        super().__init__()
        self.child_nodes = nn.ModuleList(children)

    def _eval_children(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate all children. 
        Returns (stacked_dists, stacked_attrs). 
        stacked_attrs is None if NO children have attributes.
        If some have attributes and others don't, we inject default orange for the naked ones.
        """
        results = [child(x) for child in self.child_nodes]
        dists = []
        attrs_list = []
        has_any_attrs = False

        for res in results:
            if isinstance(res, tuple):
                dists.append(res[0])
                attrs_list.append(res[1])
                has_any_attrs = True
            else:
                dists.append(res)
                attrs_list.append(None)

        stacked_dists = torch.stack(dists, dim=1)  # [N, num_children]
        
        if not has_any_attrs:
            return stacked_dists, None

        # Homogenize attributes
        final_attrs = []
        device = x.device
        
        # Attribute breakdown: [L, a, b, metallic, roughness]
        # Safety Orange Oklab approx: 0.7, 0.15, 0.15 (Rough approx, verify later)
        # Actually let's use the constant defined at top.
        orange = torch.tensor(_DEFAULT_FALLBACK_COLOR + [0.0, 0.5], device=device)
        
        for i, a in enumerate(attrs_list):
            if a is None:
                # Expand orange to [N, 5]
                final_attrs.append(orange.unsqueeze(0).expand(dists[i].shape[0], 5))
            else:
                final_attrs.append(a)
                
        stacked_attrs = torch.stack(final_attrs, dim=1) # [N, num_children, 5]
        return stacked_dists, stacked_attrs

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Simple union bounds for all CSG operations as safe default
        if not self.child_nodes:
             return (torch.tensor([-1.]*3), torch.tensor([1.]*3))
        
        mins, maxs = [], []
        for n in self.child_nodes:
            b_min, b_max = n.compute_bounds()
            mins.append(b_min)
            maxs.append(b_max)
            
        return (
            torch.min(torch.stack(mins), dim=0)[0],
            torch.max(torch.stack(maxs), dim=0)[0]
        )


# =============================================================================
# Primitives (Angle-free, pure distances)
# =============================================================================

class SphereNode(GeometryNode):
    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.norm(x, dim=1) - self.radius

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.radius
        device = next(self.parameters(), torch.tensor(0)).device
        return (
            torch.tensor([-r, -r, -r], device=device),
            torch.tensor([r, r, r], device=device)
        )


class BoxNode(GeometryNode):
    def __init__(self, size: List[float]):
        super().__init__()
        self.register_buffer('b', torch.tensor(size, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.abs(x) - self.b
        return torch.norm(torch.clamp(q, min=0.0), dim=1) + \
               torch.clamp(torch.max(q, dim=1)[0], max=0.0)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (-self.b, self.b)


class CylinderNode(GeometryNode):
    def __init__(self, radius: float, height: float):
        super().__init__()
        self.radius = radius
        self.height = height

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_xy = torch.norm(x[:, [0, 1]], dim=1) - self.radius
        d_z = torch.abs(x[:, 2]) - self.height / 2.0
        d_vec = torch.stack([d_xy, d_z], dim=1)
        return torch.min(torch.max(d_vec, dim=1)[0], torch.tensor(0.0, device=x.device)) + \
               torch.norm(torch.clamp(d_vec, min=0.0), dim=1)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.radius
        h = self.height / 2.0
        device = next(self.parameters(), torch.tensor(0)).device
        return (
            torch.tensor([-r, -r, -h], device=device),
            torch.tensor([r, r, h], device=device)
        )


class TorusNode(GeometryNode):
    def __init__(self, major_r: float, minor_r: float):
        super().__init__()
        self.major_r = major_r
        self.minor_r = minor_r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_xz = torch.norm(x[:, [0, 2]], dim=1) - self.major_r
        q = torch.stack([q_xz, x[:, 1]], dim=1)
        return torch.norm(q, dim=1) - self.minor_r

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        R = self.major_r + self.minor_r
        r = self.minor_r
        device = next(self.parameters(), torch.tensor(0)).device
        return (
            torch.tensor([-R, -r, -R], device=device),
            torch.tensor([R, r, R], device=device)
        )


class ConeNode(GeometryNode):
    def __init__(self, radius: float, height: float):
        super().__init__()
        self.radius = radius
        self.height = height
        self.sin_cos = torch.tensor([radius, height], dtype=torch.float32)
        self.sin_cos = self.sin_cos / torch.norm(self.sin_cos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_z = x[:, 2] - self.height / 2.0
        q = torch.stack([torch.norm(x[:, [0, 1]], dim=1), p_z], dim=1)
        d_side = q[:, 0] * self.sin_cos[1] - (self.height - q[:, 1]) * self.sin_cos[0]
        d_base = -q[:, 1]
        d_top = q[:, 1] - self.height
        inside = torch.max(torch.stack([d_side, d_base], dim=1), dim=1)[0]
        return torch.clamp(inside, min=-self.height, max=self.height)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.radius
        z_min = self.height / 2.0
        z_max = self.height * 1.5
        device = next(self.parameters(), torch.tensor(0)).device
        return (
            torch.tensor([-r, -r, z_min], device=device),
            torch.tensor([r, r, z_max], device=device)
        )


class CapsuleNode(GeometryNode):
    def __init__(self, radius: float, height: float):
        super().__init__()
        self.radius = radius
        self.half_height = height / 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clamped_z = torch.clamp(x[:, 2], -self.half_height, self.half_height)
        p = torch.stack([x[:, 0], x[:, 1], clamped_z], dim=1)
        return torch.norm(x - p, dim=1) - self.radius

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.radius
        h = self.half_height
        device = next(self.parameters(), torch.tensor(0)).device
        return (
            torch.tensor([-r, -r, -h - r], device=device),
            torch.tensor([r, r, h + r], device=device)
        )


class PlaneNode(GeometryNode):
    def __init__(self, normal: List[float], distance: float):
        super().__init__()
        n = torch.tensor(normal, dtype=torch.float32)
        self.register_buffer('n', torch.nn.functional.normalize(n, dim=0))
        self.distance = distance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.n) + self.distance

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        inf = 1000.0
        device = self.n.device
        return (
            torch.tensor([-inf, -inf, -inf], device=device),
            torch.tensor([inf, inf, inf], device=device)
        )


class WedgeNode(GeometryNode):
    _AXIS = {"x": 0, "y": 1, "z": 2}
    def __init__(self, size: List[float], taper_axis: str = "y", taper_dir: str = "z"):
        super().__init__()
        self.register_buffer("b", torch.tensor(size, dtype=torch.float32))
        self.taper_idx = self._AXIS.get(taper_axis.lower(), 1)
        self.dir_idx = self._AXIS.get(taper_dir.lower(), 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.abs(x) - self.b
        box_dist = torch.norm(torch.clamp(q, min=0.0), dim=1) + torch.clamp(torch.max(q, dim=1)[0], max=0.0)
        
        size_dir = self.b[self.dir_idx]
        size_tap = self.b[self.taper_idx]
        t = (x[:, self.dir_idx] + size_dir) / (2.0 * size_dir + 1e-8)
        t = torch.clamp(t, 0.0, 1.0)
        allowed = size_tap * (1.0 - t)
        plane_dist = torch.abs(x[:, self.taper_idx]) - allowed
        
        k = 0.02
        h = torch.clamp(0.5 - 0.5 * (plane_dist - box_dist) / (k + 1e-8), 0.0, 1.0)
        return torch.lerp(plane_dist, box_dist, h) + k * h * (1.0 - h)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (-self.b, self.b)


class RevolutionNode(ModifierNode):
    def __init__(self, child: GeometryNode, axis: str = "y", offset: float = 0.0):
        super().__init__(child)
        self.axis = axis.lower()
        self.offset = offset

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.axis == "y":
            radial = torch.norm(x[:, [0, 2]], dim=1) - self.offset
            p2d = torch.stack([radial, x[:, 1]], dim=1)
        elif self.axis == "x":
            radial = torch.norm(x[:, [1, 2]], dim=1) - self.offset
            p2d = torch.stack([radial, x[:, 0]], dim=1)
        else: # z
            radial = torch.norm(x[:, [0, 1]], dim=1) - self.offset
            p2d = torch.stack([radial, x[:, 2]], dim=1)
            
        p3d = torch.zeros_like(x)
        p3d[:, 0] = p2d[:, 0]
        p3d[:, 1] = p2d[:, 1]
        
        return self.child(p3d)

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        c_min, c_max = self.child.compute_bounds()
        r_min, r_max = c_min[0], c_max[0]
        max_rad = max(abs(r_min + self.offset), abs(r_max + self.offset))
        h_min, h_max = c_min[1], c_max[1]
        device = c_min.device

        if self.axis == "y":
             return (torch.tensor([-max_rad, h_min, -max_rad], device=device),
                     torch.tensor([max_rad, h_max, max_rad], device=device))
        elif self.axis == "x":
             return (torch.tensor([h_min, -max_rad, -max_rad], device=device),
                     torch.tensor([h_max, max_rad, max_rad], device=device))
        else:
             return (torch.tensor([-max_rad, -max_rad, h_min], device=device),
                     torch.tensor([max_rad, max_rad, h_max], device=device))


# =============================================================================
# Operations
# =============================================================================

class UnionNode(CSGNode):
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        stacked_dists, stacked_attrs = self._eval_children(x)
        
        # Min dist
        min_vals, min_indices = torch.min(stacked_dists, dim=1)
        
        if stacked_attrs is None:
            return min_vals
            
        # Select attrs corresponding to min dist
        # stacked_attrs: [N, C, 5]
        s_indices = min_indices.view(-1, 1, 1).expand(-1, 1, 5)
        selected = torch.gather(stacked_attrs, 1, s_indices).squeeze(1)
        return min_vals, selected


class SubtractNode(CSGNode):
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if len(self.child_nodes) < 2:
            return self.child_nodes[0](x)
            
        stacked_dists, stacked_attrs = self._eval_children(x)
        
        d1 = stacked_dists[:, 0]
        d2 = stacked_dists[:, 1] # Only support binary or strictly iterative? Assuming binary for now per old code
        
        # max(d1, -d2)
        dist = torch.max(d1, -d2)
        
        if stacked_attrs is None:
            return dist
            
        # Color from positive shape (0)
        return dist, stacked_attrs[:, 0, :]


class IntersectNode(CSGNode):
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if len(self.child_nodes) < 2:
            return self.child_nodes[0](x)
            
        stacked_dists, stacked_attrs = self._eval_children(x)
        
        # max(d1, d2)
        max_vals, max_indices = torch.max(stacked_dists, dim=1)
        
        if stacked_attrs is None:
            return max_vals
            
        # Select attrs from max dist (interior dominance)
        s_indices = max_indices.view(-1, 1, 1).expand(-1, 1, 5)
        selected = torch.gather(stacked_attrs, 1, s_indices).squeeze(1)
        return max_vals, selected


class SmoothUnionNode(CSGNode):
    def __init__(self, children: List[GeometryNode], k: float = 0.5):
        super().__init__(children)
        self.k = k

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if len(self.child_nodes) < 2:
            return self.child_nodes[0](x)

        stacked_dists, stacked_attrs = self._eval_children(x)
        d1, d2 = stacked_dists[:, 0], stacked_dists[:, 1]
        
        k = self.k
        h = torch.clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
        mix_dist = torch.lerp(d2, d1, h) - k * h * (1.0 - h)
        
        if stacked_attrs is None:
            return mix_dist
            
        # Attributes from closest
        closer_first = (d1 <= d2).unsqueeze(1).expand(-1, 5)
        a1, a2 = stacked_attrs[:, 0], stacked_attrs[:, 1]
        mix_attrs = torch.where(closer_first, a1, a2)
        return mix_dist, mix_attrs

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        b_min, b_max = super().compute_bounds()
        return (b_min - self.k, b_max + self.k)


class SmoothSubtractNode(CSGNode):
    def __init__(self, children: List[GeometryNode], k: float = 0.5):
        super().__init__(children)
        self.k = k

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if len(self.child_nodes) < 2:
            return self.child_nodes[0](x)

        stacked_dists, stacked_attrs = self._eval_children(x)
        d1, d2 = stacked_dists[:, 0], stacked_dists[:, 1]
        
        k = self.k
        h = torch.clamp(0.5 - 0.5 * (d1 + d2) / k, 0.0, 1.0)
        mix_dist = torch.lerp(d1, -d2, h) + k * h * (1.0 - h)
        
        if stacked_attrs is None:
            return mix_dist
            
        # Attributes from positive shape
        return mix_dist, stacked_attrs[:, 0]

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.child_nodes: return super().compute_bounds()
        b_min, b_max = self.child_nodes[0].compute_bounds()
        return (b_min - self.k, b_max + self.k)


class SmoothIntersectNode(CSGNode):
    def __init__(self, children: List[GeometryNode], k: float = 0.5):
        super().__init__(children)
        self.k = k

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if len(self.child_nodes) < 2:
            return self.child_nodes[0](x)

        stacked_dists, stacked_attrs = self._eval_children(x)
        d1, d2 = stacked_dists[:, 0], stacked_dists[:, 1]
        
        k = self.k
        h = torch.clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
        mix_dist = torch.lerp(d2, d1, h) + k * h * (1.0 - h)
        
        if stacked_attrs is None:
            return mix_dist
            
        # Attributes from dominant interior (max d)
        dominates = (d1 >= d2).unsqueeze(1).expand(-1, 5)
        a1, a2 = stacked_attrs[:, 0], stacked_attrs[:, 1]
        mix_attrs = torch.where(dominates, a1, a2)
        return mix_dist, mix_attrs
        
    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        b_min, b_max = super().compute_bounds()
        return (b_min - self.k, b_max + self.k)


# =============================================================================
# Fractals (Pure Geometry for now, coloring handled by Material wrapper or procedural noise later)
# =============================================================================

class MandelbulbNode(GeometryNode):
    def __init__(self, power: float = 8.0, iterations: int = 8, scale: float = 1.0):
        super().__init__()
        self.power = power
        self.iterations = min(int(iterations), 12)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = x / self.scale
        z = p.clone()
        dr = torch.ones(x.shape[0], device=x.device)
        r = torch.zeros(x.shape[0], device=x.device)

        for _ in range(self.iterations):
            r = torch.norm(z, dim=1)
            escaped = r > 2.0
            
            theta = torch.acos(torch.clamp(z[:, 2] / (r + 1e-8), -1.0, 1.0))
            phi = torch.atan2(z[:, 1], z[:, 0])

            r_pow = torch.pow(r, self.power)
            dr = torch.where(escaped, dr, r_pow * self.power * dr + 1.0)
            
            theta_n = theta * self.power
            phi_n = phi * self.power
            
            sin_theta = torch.sin(theta_n)
            new_z = torch.stack([
                sin_theta * torch.cos(phi_n),
                sin_theta * torch.sin(phi_n),
                torch.cos(theta_n),
            ], dim=1) * r_pow.unsqueeze(1) + p
            
            z = torch.where(escaped.unsqueeze(1).expand_as(z), z, new_z)

        r = torch.norm(z, dim=1)
        return 0.5 * torch.log(r + 1e-8) * r / (dr + 1e-8) * self.scale

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r = 1.5 * self.scale
        device = next(self.parameters(), torch.tensor(0)).device
        return (
            torch.tensor([-r, -r, -r], device=device),
            torch.tensor([r, r, r], device=device)
        )


class MengerSpongeNode(GeometryNode):
    def __init__(self, iterations: int = 3, scale: float = 1.0):
        super().__init__()
        self.iterations = min(int(iterations), 5)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = x / self.scale
        q = torch.abs(p)
        dist = torch.max(torch.max(q[:, 0], q[:, 1]), q[:, 2]) - 1.0

        s = 1.0
        for _ in range(self.iterations):
            a = torch.remainder(q * s, 2.0) - 1.0
            s *= 3.0
            r = torch.abs(1.0 - 3.0 * torch.abs(a))
            da = torch.max(r[:, 0], r[:, 1])
            db = torch.max(r[:, 1], r[:, 2])
            dc = torch.max(r[:, 0], r[:, 2])
            c = (torch.min(torch.min(da, db), dc) - 1.0) / s
            dist = torch.max(dist, c)

        return dist * self.scale

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.scale
        device = next(self.parameters(), torch.tensor(0)).device
        return (
            torch.tensor([-s, -s, -s], device=device),
            torch.tensor([s, s, s], device=device)
        )


class JuliaSetNode(GeometryNode):
    def __init__(self, c: List[float] = None, iterations: int = 8, scale: float = 1.0):
        super().__init__()
        c_val = c if c is not None else [0.3, 0.5, 0.2, 0.1]
        self.register_buffer('c', torch.tensor(c_val[:4], dtype=torch.float32))
        self.iterations = min(int(iterations), 12)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = x / self.scale
        z = torch.zeros(x.shape[0], 4, device=x.device)
        z[:, :3] = p
        dz = torch.ones(x.shape[0], device=x.device)

        for _ in range(self.iterations):
            r = torch.norm(z, dim=1)
            escaped = r > 4.0
            
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
        return 0.5 * r * torch.log(r + 1e-8) / (dz + 1e-8) * self.scale

    def compute_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r = 2.0 * self.scale
        device = self.c.device
        return (
            torch.tensor([-r, -r, -r], device=device),
            torch.tensor([r, r, r], device=device)
        )

# Added to verify full loading


class PrimitiveNode(GeometryNode):
    # Dummy class for re-export if needed
    pass
