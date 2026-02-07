
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(r"c:\Users\Admin\Documents\OG\Projects\GVE\backend\architect")

from src.compiler.math_jit import build_sdf_graph
from src.compiler.oklab import oklab_to_linear_rgb

def verify_explicit_color():
    print("\n--- Verifying Explicit Color ---", flush=True)
    dna = {
        "root_node": {
            "type": "primitive",
            "shape": "sphere",
            "params": {"radius": 1.0},
            "color": [1.0, 0.0, 0.0]  # RED
        }
    }
    
    graph = build_sdf_graph(dna)
    points = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32) # Surface point
    
    attrs = graph.query_attributes(points)
    oklab = attrs[0, :3].detach().unsqueeze(0)
    linear = oklab_to_linear_rgb(oklab)[0]
    
    print(f"Input Color: [1.0, 0.0, 0.0]", flush=True)
    print(f"Output Linear: {linear.tolist()}", flush=True)
    
    # Check if red channel is dominant
    if linear[0] > 0.8 and linear[1] < 0.1 and linear[2] < 0.1:
        print("✅ Explicit color verified (Red)", flush=True)
    else:
        print("❌ Explicit color failed", flush=True)

def verify_rust_texture():
    print("\n--- Verifying Rust Texture (Chromatic) ---", flush=True)
    dna = {
        "root_node": {
            "type": "primitive",
            "shape": "sphere",
            "params": {"radius": 1.0},
            "color": [0.5, 0.5, 0.5], # Gray base
            "texture_pattern": {
                "type": "rust",
                "scale": 5.0,
                "intensity": 1.0 
            }
        }
    }
    
    graph = build_sdf_graph(dna)
    
    # Sample multiple points to catch rust patches
    points = torch.randn(10, 3)
    points = torch.nn.functional.normalize(points, dim=1) # project to unit sphere surface
    
    attrs = graph.query_attributes(points)
    oklab_colors = attrs[:, :3]
    
    print("Oklab Colors (Sample):", flush=True)
    print(oklab_colors.tolist(), flush=True)
    
    # Check scales
    # We need to access the optimizer or just check the export logic
    # Since we can't easily access the optimizer internals here without running it, 
    # we will rely on the fact that if this passes, we know existing logic works.
    # But for THIS specific test, we want to know the scales.
    # We can inspect the graph... wait, graph doesn't store scales.
    # We need to inspect `compile_splats` output but that returns bytes.
    # Let's blindly trust the analysis via code review first, 
    # or better, just run a quick `SplatOptimizer` test here.
    
    from src.compiler.splat_trainer import SplatOptimizer
    
    # Mock data for optimizer test
    print("\n--- Verifying Splat Optimizer Scales ---", flush=True)
    # create grid of points to force known spacing
    # grid with 0.1 spacing
    x = torch.linspace(0, 1, 10)
    y = torch.zeros(10)
    z = torch.zeros(10)
    init_pos = torch.stack([x, y, z], dim=1) # 10 points at 0.0, 0.111, ...
    init_attrs = torch.zeros(10, 5)
    
    # spacing is 1.0/9 = 0.111
    # scale should be ~ 0.111 * 0.6 = 0.066
    
    # We need to manually calculate what compile_splats WOULD do, because we can't easily run compile_splats 
    # without a real SDF graph that supports bounds.
    
    # But we can verify that SplatOptimizer respects the passed initial_scale
    target_scale = 0.005 # Force a tiny scale
    optimizer = SplatOptimizer(graph, init_pos, init_attrs, initial_scale=target_scale)
    
    scales = optimizer.scales.detach().numpy()
    print(f"Scales (Mean): {scales.mean()}", flush=True)
    
    if abs(scales.mean() - target_scale) < 1e-6:
        print(f"✅ Optimizer initialized with correct dynamic scale ({scales.mean()})", flush=True)
    else:
        print(f"❌ Optimizer ignored initial_scale (Got {scales.mean()}, Expected {target_scale})", flush=True)

    # Check variance in a/b channels (chroma)
    a_var = torch.var(oklab_colors[:, 1]).item()
    b_var = torch.var(oklab_colors[:, 2]).item()
    
    print(f"Chroma Variance (a): {a_var:.6f}", flush=True)
    print(f"Chroma Variance (b): {b_var:.6f}", flush=True)
    
    if a_var > 1e-4 or b_var > 1e-4:
        print("✅ Chromatic variance detected (Rust texture is colored)", flush=True)
    else:
        print("❌ No chromatic variance (Rust texture is monochrome)", flush=True)

if __name__ == "__main__":
    with open("verification_result.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        verify_explicit_color()
        verify_rust_texture()
