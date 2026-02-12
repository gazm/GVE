
import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend", "architect"))

from src.compiler.splat_rasterizer import (
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
    
    # Half roughness=0.5, Metallic=0.0
    attrs[:, 4] = 0.5 
    attrs[:, 3] = 0.0

    # Test metallic on a subset
    metal_mask = positions[:, 1] > 0.5
    attrs[metal_mask, 3] = 1.0 # Metallic
    
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
    col = res // 2
    
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

    # 5. Check Packed Alpha
    # Rough=0.5 -> 4-bit 7 or 8. Metal=0.0 -> 0.
    # Alpha ~ (8<<4)|0 = 128.
    # Metal=1.0 -> 15. Rough=0.5 -> 8.
    # Alpha ~ (8<<4)|15 = 128+15 = 143.
    
    # Sample a metallic pixel (Y > 0.5 means row > res/2)
    meta_pixel_alpha = tex_p.xy[3*res//4, res//4, 3]
    print(f"Sampled Metallic Alpha: {meta_pixel_alpha}")
    
    rough_4 = (meta_pixel_alpha >> 4) & 0xF
    metal_4 = meta_pixel_alpha & 0xF
    print(f"  -> Rough4: {rough_4} (~8), Metal4: {metal_4} (15)")
    
    if metal_4 > 10 and 6 <= rough_4 <= 9:
        print("✅ Packed alpha looks correct.")
    else:
        print("❌ Packed alpha incorrect!")

    # Check CUDA path if available
    if torch.cuda.is_available():
        print("\n--- CUDA Point Mode ---")
        from src.compiler.splat_rasterizer import bake_triplanar_from_voxel_oklab
        
        # bake_triplanar_from_voxel_oklab expects attrs_oklab as (N, 5)
        # It doesn't use scales (uses fixed splat radius logic for voxels)
        # So results might differ slightly due to scale differences vs explicit baker
        tex_cuda = bake_triplanar_from_voxel_oklab(
            positions, attrs, bmin, bmax, resolution=res, device="cuda", mode=SplatBakeMode.POINT
        )
        # Convert to float for diff
        cpu_float = tex_p.xy.astype(float)
        cuda_float = tex_cuda.xy.astype(float)
        
        # Ignore empty pixels where CPU/CUDA might differ on boundary
        mask = (cpu_float[..., 3] > 0) & (cuda_float[..., 3] > 0)
        
        if np.any(mask):
            diff = np.mean(np.abs(cpu_float[mask] - cuda_float[mask]))
            print(f"CPU vs CUDA Mean Diff (masked): {diff:.4f}")
            if diff < 10.0: # Allow some diff due to scale/implementation variance
                print("✅ CUDA matches CPU reasonably well.")
            else:
                print("❌ CUDA mismatch!")
        else:
            print("⚠️ No overlap between CPU and CUDA outputs?")

if __name__ == "__main__":
    test_bake_modes()
