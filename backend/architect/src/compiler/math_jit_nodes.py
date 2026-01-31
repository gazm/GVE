"""SDF Primitive Nodes - Basic geometry shapes."""

import torch
import torch.nn as nn
from typing import List


class SphereNode(nn.Module):
    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius
        print(f"    🔵 SphereNode: radius={radius}", flush=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ||x|| - r
        return torch.norm(x, dim=1) - self.radius


class BoxNode(nn.Module):
    def __init__(self, size: List[float]):
        super().__init__()
        # Size IS already half-extents from AI output (box spans from -size to +size)
        # DO NOT divide by 2 - the AI prompt says "half-extents"
        self.register_buffer('b', torch.tensor(size, dtype=torch.float32))
        print(f"    📦 BoxNode: half-extents={size}", flush=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.abs(x) - self.b
        return torch.norm(torch.clamp(q, min=0.0), dim=1) + \
               torch.clamp(torch.max(q, dim=1)[0], max=0.0)


class CylinderNode(nn.Module):
    def __init__(self, radius: float, height: float):
        super().__init__()
        self.radius = radius
        self.height = height
        print(f"    🔷 CylinderNode: radius={radius}, height={height}", flush=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cylinder aligned with Y axis
        d_xz = torch.norm(x[:, [0, 2]], dim=1) - self.radius
        d_y = torch.abs(x[:, 1]) - self.height / 2.0
        
        d_vec = torch.stack([d_xz, d_y], dim=1)
        
        inside = torch.min(torch.max(d_vec, dim=1)[0], torch.tensor(0.0, device=x.device))
        outside = torch.norm(torch.clamp(d_vec, min=0.0), dim=1)
        
        return inside + outside


class TorusNode(nn.Module):
    """Torus (donut) SDF - ring in XZ plane, Y is up."""
    def __init__(self, major_r: float, minor_r: float):
        super().__init__()
        self.major_r = major_r  # Distance from center to tube center
        self.minor_r = minor_r  # Tube radius
        print(f"    🍩 TorusNode: major_r={major_r}, minor_r={minor_r}", flush=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Distance to ring center in XZ plane
        q_xz = torch.norm(x[:, [0, 2]], dim=1) - self.major_r
        # Distance to tube surface
        q = torch.stack([q_xz, x[:, 1]], dim=1)
        return torch.norm(q, dim=1) - self.minor_r


class ConeNode(nn.Module):
    """Cone SDF - tip at origin, opens along +Y axis."""
    def __init__(self, radius: float, height: float):
        super().__init__()
        self.radius = radius
        self.height = height
        # Compute cone angle
        self.sin_cos = torch.tensor([radius, height], dtype=torch.float32)
        self.sin_cos = self.sin_cos / torch.norm(self.sin_cos)  # [sin, cos]
        print(f"    🔺 ConeNode: radius={radius}, height={height}", flush=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shift so base is at y=0, tip at y=height
        p = x.clone()
        p[:, 1] = p[:, 1] - self.height / 2.0
        
        # 2D distance in XZ
        q = torch.stack([torch.norm(p[:, [0, 2]], dim=1), p[:, 1]], dim=1)
        
        # Simple bounded cone SDF
        d_side = q[:, 0] * self.sin_cos[1] - (self.height - q[:, 1]) * self.sin_cos[0]
        d_base = -q[:, 1]
        d_top = q[:, 1] - self.height
        
        # Inside: max of all constraints (negative when inside)
        inside = torch.max(torch.stack([d_side, d_base], dim=1), dim=1)[0]
        
        # Clamp to reasonable bounds
        return torch.clamp(inside, min=-self.height, max=self.height)


class CapsuleNode(nn.Module):
    """Capsule SDF - cylinder with hemispherical caps, aligned with Y axis."""
    def __init__(self, radius: float, height: float):
        super().__init__()
        self.radius = radius
        self.half_height = height / 2.0
        print(f"    💊 CapsuleNode: radius={radius}, height={height}", flush=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp Y to line segment
        p = x.clone()
        p[:, 1] = torch.clamp(p[:, 1], -self.half_height, self.half_height)
        # Distance from clamped point
        return torch.norm(x - p, dim=1) - self.radius


class PlaneNode(nn.Module):
    def __init__(self, normal: List[float], distance: float):
        super().__init__()
        n = torch.tensor(normal, dtype=torch.float32)
        self.register_buffer('n', torch.nn.functional.normalize(n, dim=0))
        self.distance = distance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.n) + self.distance


class UnionNode(nn.Module):
    def __init__(self, children: List[nn.Module]):
        super().__init__()
        self.child_nodes = nn.ModuleList(children)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dists = [child(x) for child in self.child_nodes]
        stack = torch.stack(dists, dim=1)
        return torch.min(stack, dim=1)[0]
