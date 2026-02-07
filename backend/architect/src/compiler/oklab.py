"""
Oklab Color Space Utilities - Differentiable PyTorch Implementation.

Oklab is a perceptually uniform color space designed by Björn Ottosson.
It ensures that interpolating between colors produces aesthetically pleasing results
(e.g., red -> green goes through yellow, not muddy brown).

References:
- https://bottosson.github.io/posts/oklab/
- GVE Pipeline Docs: docs/workflows/compiler-pipeline.md §3.2
"""

import torch
import numpy as np
from typing import Union


# ============================================================================
# sRGB <-> Linear RGB
# ============================================================================

def srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    """
    Convert sRGB (gamma-encoded) to linear RGB.
    
    Args:
        srgb: Tensor of shape (..., 3) with values in [0, 1].
        
    Returns:
        Linear RGB tensor of same shape.
    """
    # Standard sRGB transfer function
    threshold = 0.04045
    low = srgb / 12.92
    high = torch.pow((srgb + 0.055) / 1.055, 2.4)
    return torch.where(srgb <= threshold, low, high)


def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """
    Convert linear RGB to sRGB (gamma-encoded).
    
    Args:
        linear: Tensor of shape (..., 3) with values in [0, 1].
        
    Returns:
        sRGB tensor of same shape.
    """
    threshold = 0.0031308
    low = linear * 12.92
    high = 1.055 * torch.pow(linear.clamp(min=1e-8), 1.0 / 2.4) - 0.055
    return torch.where(linear <= threshold, low, high)


# ============================================================================
# Linear RGB <-> Oklab
# ============================================================================

# Matrix M1: Linear sRGB -> LMS (cone response)
_M1 = torch.tensor([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=torch.float32)

# Matrix M2: LMS' -> Oklab
_M2 = torch.tensor([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
], dtype=torch.float32)

# Inverse matrices
_M1_inv = torch.linalg.inv(_M1)
_M2_inv = torch.linalg.inv(_M2)


def linear_rgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert linear RGB to Oklab color space.
    
    Args:
        rgb: Tensor of shape (..., 3) with linear RGB values.
        
    Returns:
        Oklab tensor of shape (..., 3) where:
        - L: Lightness [0, 1]
        - a: Green-Red axis [-0.4, 0.4]
        - b: Blue-Yellow axis [-0.4, 0.4]
    """
    # Ensure matrices are on same device
    M1 = _M1.to(rgb.device)
    M2 = _M2.to(rgb.device)
    
    # RGB -> LMS
    lms = torch.matmul(rgb, M1.T)
    
    # Apply cube root (with safety for negative values)
    lms_cbrt = torch.sign(lms) * torch.pow(torch.abs(lms).clamp(min=1e-10), 1.0 / 3.0)
    
    # LMS' -> Oklab
    oklab = torch.matmul(lms_cbrt, M2.T)
    
    return oklab


def oklab_to_linear_rgb(oklab: torch.Tensor) -> torch.Tensor:
    """
    Convert Oklab to linear RGB color space.
    
    Args:
        oklab: Tensor of shape (..., 3) with Oklab values (L, a, b).
        
    Returns:
        Linear RGB tensor of shape (..., 3).
    """
    # Ensure matrices are on same device
    M1_inv_d = _M1_inv.to(oklab.device)
    M2_inv_d = _M2_inv.to(oklab.device)
    
    # Oklab -> LMS'
    lms_cbrt = torch.matmul(oklab, M2_inv_d.T)
    
    # Cube (inverse of cube root)
    lms = lms_cbrt ** 3
    
    # LMS -> RGB
    rgb = torch.matmul(lms, M1_inv_d.T)
    
    return rgb


# ============================================================================
# Convenience: sRGB <-> Oklab (combined)
# ============================================================================

def srgb_to_oklab(srgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB to Oklab."""
    linear = srgb_to_linear(srgb)
    return linear_rgb_to_oklab(linear)


def oklab_to_srgb(oklab: torch.Tensor) -> torch.Tensor:
    """Convert Oklab to sRGB."""
    linear = oklab_to_linear_rgb(oklab)
    return linear_to_srgb(linear.clamp(0, 1))


# ============================================================================
# NumPy convenience wrappers (for non-differentiable use)
# ============================================================================

def np_linear_rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """NumPy wrapper for linear_rgb_to_oklab."""
    t = torch.from_numpy(rgb.astype(np.float32))
    result = linear_rgb_to_oklab(t)
    return result.numpy()


def np_oklab_to_linear_rgb(oklab: np.ndarray) -> np.ndarray:
    """NumPy wrapper for oklab_to_linear_rgb."""
    t = torch.from_numpy(oklab.astype(np.float32))
    result = oklab_to_linear_rgb(t)
    return result.numpy()
