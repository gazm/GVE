
import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend", "architect"))

from src.compiler.triplanar_baker import (
    bake_triplanar_textures_oklab,
    SplatBakeMode,
    TriplanarTextures
)

def test_bake_modes():
    print("🧪 Testing SplatBakeMode...")
    
    # 1. Setup random splat data
    N = 1000
    res = 128
    
    # Grid of points
    x = np.linspace(0.1, 0.9, int(np.sqrt(N)))
    y = np.linspace(0.1, 0.9, int(np.sqrt(N)))
    xx, yy = np.meshgrid(x, y)
    positions = np.stack([xx.flatten(), yy.flatten(), np.zeros_like(xx.flatten()) + 0.5], axis=1)
    # Jitter
    positions += np.random.uniform(-0.02, 0.02, positions.shape)
    
    # Colors: bright red and bright blue
    attrs = np.zeros((len(positions), 5), dtype=np.float32)
    # Half red, half blue
    mask = positions[:, 0] < 0.5
    attrs[mask, 0] = 0.6 # L
    attrs[mask, 1] = 0.2 # a (reddish)
    attrs[mask, 2] = 0.1 # b
    
    attrs[~mask, 0] = 0.6 # L
    attrs[~mask, 1] = -0.1 # a (greenish/bluish)
    attrs[~mask, 2] = -0.2 # b (bluish)
    
    attrs[:, 4] = 0.5 # roughness
    
    scales = np.full((len(positions), 3), 0.05, dtype=np.float32)
    
    bmin = np.array([0, 0, 0], dtype=np.float32)
    bmax = np.array([1, 1, 1], dtype=np.float32)
    
    # 2. Bake GAUSSIAN
    print("\n--- Gaussian Mode ---")
    tex_g = bake_triplanar_textures_oklab(
        positions, attrs, scales, bmin, bmax, resolution=res, mode=SplatBakeMode.GAUSSIAN
    )
    
    # 3. Bake POINT
    print("\n--- Point Mode ---")
    tex_p = bake_triplanar_textures_oklab(
        positions, attrs, scales, bmin, bmax, resolution=res, mode=SplatBakeMode.POINT
    )
    
    # 4. Compare
    # In point mode, we expect sharper transitions. 
    # Let's count unique colors in the middle row of XY plane
    row = res // 2
    
    uniq_g = len(np.unique(tex_g.xy[row], axis=0))
    uniq_p = len(np.unique(tex_p.xy[row], axis=0))
    
    print(f"\nUnique colors in middle row:")
    print(f"  Gaussian: {uniq_g}")
    print(f"  Point:    {uniq_p}")
    
    # Point mode should ideally have fewer unique colors if the splats form solid Voronoi cells, 
    # whereas Gaussian blends creating gradients.
    
    if uniq_p < uniq_g:
        print("✅ Point mode produced fewer unique colors (crisper) as expected.")
    else:
        print("⚠️ Point mode not significantly improved in crispness (check distribution).")

    # Check CUDA path if available
    if torch.cuda.is_available():
        print("\n--- CUDA Point Mode ---")
        tex_cuda = bake_triplanar_textures_oklab(
            positions, attrs, scales, bmin, bmax, resolution=res, device="cuda", mode=SplatBakeMode.POINT
        )
        diff = np.mean(np.abs(tex_p.xy.astype(float) - tex_cuda.xy.astype(float)))
        print(f"CPU vs CUDA Mean Diff: {diff:.4f}")
        if diff < 2.0:
            print("✅ CUDA matches CPU.")
        else:
            print("❌ CUDA mismatch!")

if __name__ == "__main__":
    test_bake_modes()
