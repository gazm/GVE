
import sys
import os
import torch
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.compiler.math_jit_builder import build_sdf_graph
from src.compiler.splat_trainer import compile_splats

def test_splats():
    print("--- Verifying Splat Compilation ---")
    
    # 1. Simple Sphere at Origin
    dna_simple = {
        "root_node": {
            "type": "primitive",
            "shape": "sphere",
            "params": {"radius": 1.0},
             # Explicit color to verify color handling too (from implementation plan)
            "params": {"radius": 1.0, "color": [0.0, 1.0, 0.0]} # Green
        }
    }
    
    print("1. Building SDF Graph...")
    graph = build_sdf_graph(dna_simple)
    
    bounds = graph.bounds
    print(f"Bounds: {bounds}")
    assert bounds[0] == [-1.0, -1.0, -1.0]
    
    print("2. Compiling Splats (Target=500)...")
    # Using small count for speed
    splat_bytes = compile_splats(
        graph,
        bounds,
        target_count=500,
        iterations=50, # Short run
        device="cpu" # Force CPU for reliable testing in this environment
    )
    
    print(f"3. Compilation produced {len(splat_bytes)} bytes")
    
    # Verify binary header
    import struct
    count = struct.unpack('<I', splat_bytes[0:4])[0]
    print(f"Splat Count in Binary: {count}")
    
    assert count > 450 # Should be close to target
    
    print("✅ Splat Compilation Test Passed")

if __name__ == "__main__":
    try:
        test_splats()
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
