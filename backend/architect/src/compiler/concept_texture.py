"""
Concept-image texture – recolor splats from the concept image at compile time.

When an asset has a stored concept image, we project each splat's 3D position
onto the image (orthographic front view), sample color, convert to Oklab, and
blend with the existing procedural color so the result matches the reference.
"""

import base64
import io
import numpy as np
from typing import Tuple

from .oklab import srgb_to_oklab


def _decode_image_to_srgb(base64_str: str) -> np.ndarray:
    """Decode base64 image to sRGB float (H, W, 3) in [0, 1]."""
    raw = base64.b64decode(base64_str)
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
    except Exception:
        raise ValueError("Concept image decode failed (need PIL); install Pillow") from None
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def _positions_to_uv_front(
    positions: np.ndarray,
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D positions to 2D UV [0,1] using orthographic front view (-Z).
    u = (x - min_x) / ext_x, v = (y - min_y) / ext_y.
    """
    (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
    ext_x = max_x - min_x
    ext_y = max_y - min_y
    if ext_x <= 0:
        ext_x = 1.0
    if ext_y <= 0:
        ext_y = 1.0
    u = (positions[:, 0] - min_x) / ext_x
    v = (positions[:, 1] - min_y) / ext_y
    u = np.clip(u, 0.0, 1.0).astype(np.float32)
    v = np.clip(v, 0.0, 1.0).astype(np.float32)
    return u, v


def _sample_bilinear(rgb: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample RGB image at (u,v) in [0,1] with bilinear interpolation. Returns (N, 3)."""
    H, W = rgb.shape[0], rgb.shape[1]
    # Image row 0 = top; we use v=0 as bottom of bbox so row = (1-v)*(H-1)
    rows = (1.0 - v) * (H - 1)
    cols = u * (W - 1)
    r0 = np.floor(rows).astype(np.int32).clip(0, H - 2)
    r1 = r0 + 1
    c0 = np.floor(cols).astype(np.int32).clip(0, W - 2)
    c1 = c0 + 1
    wr = (rows - r0).astype(np.float32)
    wc = (cols - c0).astype(np.float32)
    c00 = rgb[r0, c0]
    c01 = rgb[r0, c1]
    c10 = rgb[r1, c0]
    c11 = rgb[r1, c1]
    out = (
        (1 - wr)[:, None] * (1 - wc)[:, None] * c00
        + (1 - wr)[:, None] * wc[:, None] * c01
        + wr[:, None] * (1 - wc)[:, None] * c10
        + wr[:, None] * wc[:, None] * c11
    )
    return out.astype(np.float32)


def _rgb_to_oklab_np(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (N, 3) float to Oklab (N, 3) float using existing oklab module."""
    import torch
    t = torch.from_numpy(rgb.astype(np.float32))
    oklab = srgb_to_oklab(t)
    return oklab.numpy()


def recolor_splats_from_concept_image(
    positions: np.ndarray,
    colors: np.ndarray,
    concept_image_base64: str,
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    blend: float = 0.7,
) -> np.ndarray:
    """
    Recolor splat colors by sampling the concept image at projected 2D UVs.

    positions: (N, 3) float32
    colors: (N, 4) uint8 Oklab8+A [L, a, b, alpha]
    concept_image_base64: base64-encoded RGB image
    bounds: (min_xyz, max_xyz) for projection
    blend: 0 = keep procedural only, 1 = full image; default 0.7

    Returns:
        New colors (N, 4) uint8 [L, a, b, alpha]; alpha and encoding unchanged.
    """
    rgb = _decode_image_to_srgb(concept_image_base64)
    u, v = _positions_to_uv_front(positions, bounds)
    rgb_sampled = _sample_bilinear(rgb, u, v)
    img_oklab = _rgb_to_oklab_np(rgb_sampled)

    # Decode current Oklab u8 -> float
    L = colors[:, 0].astype(np.float32) / 255.0
    a = (colors[:, 1].astype(np.float32) / 255.0) * 0.8 - 0.4
    b = (colors[:, 2].astype(np.float32) / 255.0) * 0.8 - 0.4
    alpha = colors[:, 3]

    # Blend
    new_L = np.clip(blend * img_oklab[:, 0] + (1.0 - blend) * L, 0.0, 1.0)
    new_a = np.clip(blend * img_oklab[:, 1] + (1.0 - blend) * a, -0.4, 0.4)
    new_b = np.clip(blend * img_oklab[:, 2] + (1.0 - blend) * b, -0.4, 0.4)

    # Encode back to u8
    L_u8 = (new_L * 255).clip(0, 255).astype(np.uint8)
    a_u8 = ((new_a + 0.4) / 0.8 * 255).clip(0, 255).astype(np.uint8)
    b_u8 = ((new_b + 0.4) / 0.8 * 255).clip(0, 255).astype(np.uint8)
    return np.column_stack([L_u8, a_u8, b_u8, alpha])
