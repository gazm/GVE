
import sys
import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

# Add parent directory to path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.compiler.math_jit_builder import build_sdf_graph
from src.compiler.math_jit_nodes import SphereNode

def test_bounds():
    print("--- Verifying SDF Bounds Calculation ---")
    
    import src.compiler.math_jit_nodes as nodes_module
    print(f"SphereNode module: {nodes_module.__file__}")
    
    # Debug: Check if SphereNode has compute_bounds
    attributes = dir(SphereNode)
    print(f"Attributes start with c: {[a for a in attributes if a.startswith('c')]}")
    print(f"Has compute_bounds? {'compute_bounds' in attributes}")
    print(f"Is SphereNode.compute_bounds callable? {callable(getattr(SphereNode, 'compute_bounds', None))}")
    
    # 1. Simple Sphere at Origin
    dna_simple = {
        "root_node": {
            "type": "primitive",
            "shape": "sphere",
            "params": {"radius": 1.0}
        }
    }
    graph = build_sdf_graph(dna_simple)
    b_min, b_max = graph.bounds
    print(f"Sphere(r=1) Bounds: {b_min} to {b_max}")
    
    assert b_min == [-1.0, -1.0, -1.0]
    assert b_max == [1.0, 1.0, 1.0]
    print("✅ Simple Sphere Passed")
    
    # 2. Transformed Sphere (Translate x=5)
    dna_trans = {
        "root_node": {
            "type": "primitive",
            "shape": "sphere",
            "params": {"radius": 1.0},
            "transform": {"pos": [5.0, 0.0, 0.0]}
        }
    }
    graph = build_sdf_graph(dna_trans)
    b_min, b_max = graph.bounds
    print(f"Sphere(r=1, x=5) Bounds: {b_min} to {b_max}")
    
    # Check with tolerance for float errors
    assert abs(b_min[0] - 4.0) < 1e-5
    assert abs(b_max[0] - 6.0) < 1e-5
    print("✅ Transformed Sphere Passed")
    
    # 3. Union of two spheres
    dna_union = {
        "root_node": {
            "type": "operation",
            "op": "union",
            "children": [
                {
                    "type": "primitive", "shape": "sphere", "params": {"radius": 1.0},
                    "transform": {"pos": [-2.0, 0.0, 0.0]}
                },
                {
                    "type": "primitive", "shape": "sphere", "params": {"radius": 1.0},
                    "transform": {"pos": [2.0, 0.0, 0.0]}
                }
            ]
        }
    }
    graph = build_sdf_graph(dna_union)
    b_min, b_max = graph.bounds
    print(f"Union Bounds: {b_min} to {b_max}")
    
    # Expect [-3, -1, -1] to [3, 1, 1]
    assert abs(b_min[0] - (-3.0)) < 1e-5
    assert abs(b_max[0] - 3.0) < 1e-5
    print("✅ Union Passed")

if __name__ == "__main__":
    try:
        test_bounds()
        print("\n🎉 All bounds tests passed!")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
