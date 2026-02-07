"""
Procedural Noise Functions - GPU-accelerated via PyTorch.

Provides Perlin, Simplex, and Voronoi noise for modulating material
attributes during SDF graph evaluation. Noise is sampled at world-space
positions so each splat gets unique surface detail (wood grain, marble
veins, rust patches) without UV coordinates.

Reference: docs/workflows/compiler-pipeline.md (Phase 4)
"""

import torch
import torch.nn as nn
from typing import List, Tuple

# =============================================================================
# Low-level noise primitives (vectorised for GPU)
# =============================================================================


def _hash_3d(p: torch.Tensor) -> torch.Tensor:
    """Integer-arithmetic style hash for 3D -> 3D. Input shape (..., 3)."""
    # Dot-product based hash (deterministic, no sin precision issues)
    x, y, z = p[..., 0:1], p[..., 1:2], p[..., 2:3]
    h = torch.cat([
        x * 127.1 + y * 311.7 + z * 74.7,
        x * 269.5 + y * 183.3 + z * 246.1,
        x * 113.5 + y * 271.9 + z * 124.6,
    ], dim=-1)
    return torch.frac(torch.sin(h) * 43758.5453)


def _fade(t: torch.Tensor) -> torch.Tensor:
    """Perlin smootherstep: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def perlin_noise_3d(p: torch.Tensor) -> torch.Tensor:
    """
    Classic Perlin noise in 3D. Fully vectorised on GPU.

    Args:
        p: [N, 3] world positions.

    Returns:
        [N] noise values in approximately [-1, 1].
    """
    pi = torch.floor(p)
    pf = p - pi

    # Smootherstep interpolation weights
    u = _fade(pf)

    # Hash 8 corners of the unit cube
    offsets = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=p.dtype, device=p.device)  # (8, 3)

    # Gradients at each corner: hash -> random unit vector
    grads = []
    for i in range(8):
        corner = pi + offsets[i]
        g = _hash_3d(corner) * 2.0 - 1.0  # [N, 3] in [-1, 1]
        diff = pf - offsets[i]              # [N, 3]
        grads.append((g * diff).sum(dim=-1))  # [N] dot product

    # Trilinear interpolation
    x0 = torch.lerp(grads[0], grads[1], u[:, 0])
    x1 = torch.lerp(grads[2], grads[3], u[:, 0])
    x2 = torch.lerp(grads[4], grads[5], u[:, 0])
    x3 = torch.lerp(grads[6], grads[7], u[:, 0])

    y0 = torch.lerp(x0, x1, u[:, 1])
    y1 = torch.lerp(x2, x3, u[:, 1])

    return torch.lerp(y0, y1, u[:, 2])


def voronoi_noise_3d(p: torch.Tensor) -> torch.Tensor:
    """
    Voronoi (Worley) noise in 3D -- returns distance to nearest cell centre.

    Args:
        p: [N, 3] world positions.

    Returns:
        [N] distances in [0, ~1.5].
    """
    cell = torch.floor(p)
    frac = p - cell

    min_dist = torch.full((p.shape[0],), 999.0, device=p.device, dtype=p.dtype)

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                offset = torch.tensor([dx, dy, dz], dtype=p.dtype, device=p.device)
                neighbor = cell + offset
                jitter = _hash_3d(neighbor)
                centre = offset + jitter - frac
                dist = torch.norm(centre, dim=-1)
                min_dist = torch.minimum(min_dist, dist)

    return min_dist


def fbm_perlin_3d(
    p: torch.Tensor,
    octaves: int = 4,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
) -> torch.Tensor:
    """
    Fractal Brownian Motion using Perlin noise.

    Args:
        p: [N, 3] world positions.
        octaves: number of noise layers.
        lacunarity: frequency multiplier per octave.
        persistence: amplitude multiplier per octave.

    Returns:
        [N] accumulated noise.
    """
    total = torch.zeros(p.shape[0], device=p.device, dtype=p.dtype)
    amplitude = 1.0
    frequency = 1.0
    for _ in range(octaves):
        total = total + perlin_noise_3d(p * frequency) * amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return total


# =============================================================================
# Preset pattern functions
# =============================================================================


def wood_grain(p: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    """Concentric ring pattern (wood cross-section) with noise perturbation.

    Returns [N] in [0, 1] -- 0 = early-wood (lighter), 1 = late-wood (darker).
    """
    noise = perlin_noise_3d(p * 2.0) * 0.3
    radial = torch.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) * scale + noise
    rings = torch.frac(radial)
    return rings * rings


def marble_veins(p: torch.Tensor, scale: float = 5.0) -> torch.Tensor:
    """Marble veining pattern (sine + turbulence).

    Returns [N] in [0, 1] -- 0 = base stone, 1 = vein.
    """
    turbulence = fbm_perlin_3d(p * 3.0, octaves=4)
    pattern = torch.sin(p[:, 1] * scale + turbulence * 5.0)
    return pattern * 0.5 + 0.5


def rust_patches(p: torch.Tensor, scale: float = 4.0) -> torch.Tensor:
    """Patchy weathering noise (voronoi edges + perlin blend).

    Returns [N] in [0, 1] -- 0 = clean surface, 1 = rusty.
    """
    vor = voronoi_noise_3d(p * scale)
    pnoise = perlin_noise_3d(p * scale * 2.0) * 0.5 + 0.5
    edge = 1.0 - torch.clamp(vor * 2.0, 0.0, 1.0)
    return torch.clamp(edge * pnoise, 0.0, 1.0)


# =============================================================================
# ProceduralTextureNode -- wraps a child SDF and modulates attributes
# =============================================================================

# All pattern functions share the same (p, scale) signature.
_PATTERN_FNS = {
    "perlin": lambda p, scale: perlin_noise_3d(p * scale) * 0.5 + 0.5,
    "wood_grain": wood_grain,
    "marble": marble_veins,
    "rust": rust_patches,
}

# Target Oklab chroma for each pattern (L, a, b).
# The noise intensity lerps the surface toward these values.
_PATTERN_COLORS = {
    "perlin": torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32),
    "rust": torch.tensor([0.6, 0.2, 0.15], dtype=torch.float32),
    "wood_grain": torch.tensor([0.65, 0.05, 0.08], dtype=torch.float32),
    "marble": torch.tensor([0.4, 0.0, 0.0], dtype=torch.float32),
}


class ProceduralTextureNode(nn.Module):
    """Wraps a child SDF node and modulates material attributes with noise.

    Supports patterns: ``perlin``, ``wood_grain``, ``marble``, ``rust``.

    All five attribute channels ``[L, a, b, metallic, roughness]`` can be
    perturbed.  The ``intensity`` parameter gates how strongly the noise
    affects each channel.
    """

    def __init__(
        self,
        child: nn.Module,
        pattern: str = "perlin",
        scale: float = 5.0,
        intensity: float = 0.3,
        color_variation: float = 0.2,
        roughness_variation: float = 0.1,
        metallic_variation: float = 0.0,
    ):
        super().__init__()
        self.child = child
        self.pattern = pattern
        self.scale = scale
        self.intensity = intensity
        self.color_var = color_variation
        self.roughness_var = roughness_variation
        self.metallic_var = metallic_variation

        target = _PATTERN_COLORS.get(pattern, torch.zeros(3))
        self.register_buffer('target_color', target)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist, attrs = self.child(x)

        pattern_fn = _PATTERN_FNS.get(self.pattern)
        if pattern_fn is None:
            return dist, attrs

        # Uniform call — every pattern fn takes (p, scale)
        noise = pattern_fn(x, self.scale)  # [N] in ~[0, 1]

        n = (noise * self.intensity).unsqueeze(1)  # [N, 1]

        # --- Modulate [L, a, b, metallic, roughness] -------------------------
        new_attrs = attrs.clone()

        # 1. Lightness
        new_attrs[:, 0] = attrs[:, 0] + n.squeeze(1) * self.color_var

        # 2. Chroma — lerp towards pattern target color
        mix = torch.clamp(n, 0.0, 1.0)
        target = self.target_color.unsqueeze(0).expand(attrs.shape[0], 3)
        new_attrs[:, :3] = torch.lerp(new_attrs[:, :3], target, mix)

        # 3. Metallic
        new_attrs[:, 3] = torch.clamp(
            attrs[:, 3] + n.squeeze(1) * self.metallic_var, 0.0, 1.0
        )

        # 4. Roughness
        new_attrs[:, 4] = torch.clamp(
            attrs[:, 4] + n.squeeze(1) * self.roughness_var, 0.0, 1.0
        )

        # Keep L in valid range
        new_attrs[:, 0] = torch.clamp(new_attrs[:, 0], 0.0, 1.0)

        return dist, new_attrs
