"""
Test GPU-accelerated splat training.

Run from backend/architect directory:
    python -m pytest tests/test_gpu_splat.py -v -s
    
Or directly:
    python tests/test_gpu_splat.py
"""

import sys
import os
import time

# Force UTF-8 for Windows console
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, ".")

import torch


def test_preloader_device():
    """Verify torch_preloader detects GPU correctly."""
    from src.torch_preloader import preloader
    
    preloader.ensure_loaded()
    status = preloader.get_status()
    device = preloader.get_device()
    
    print(f"\n[torch] Preloader Status:")
    print(f"   Status: {status['status']}")
    print(f"   Device: {device}")
    print(f"   Device Name: {status['device_name']}")
    
    assert status['status'] == 'ready', f"Torch not ready: {status}"
    assert device in ('cuda', 'cpu'), f"Unknown device: {device}"
    
    if device == 'cuda':
        print(f"   [OK] GPU acceleration available!")
    else:
        print(f"   [WARN] Running on CPU (no GPU detected)")


def test_sdf_graph_to_device():
    """Test that SdfGraph can be moved to GPU."""
    from src.compiler.math_jit_builder import SdfGraph, build_sdf_graph
    from src.torch_preloader import preloader
    
    device = preloader.get_device()
    
    # Build a simple SDF graph
    dna = {
        "root_node": {
            "type": "primitive",
            "shape": "sphere",
            "params": {"radius": 0.5}
        }
    }
    
    graph = build_sdf_graph(dna)
    print(f"\n[graph] SdfGraph created on CPU")
    
    # Move to device
    graph = graph.to(device)
    print(f"   Moved to {device}")
    
    # Test evaluation
    test_points = torch.randn(100, 3, device=device)
    with torch.no_grad():
        distances = graph(test_points)
    
    print(f"   Evaluated 100 points, result device: {distances.device}")
    assert str(distances.device).startswith(device), f"Result on wrong device: {distances.device}"
    print(f"   [OK] SdfGraph works on {device}")


def test_splat_training_gpu():
    """Test full splat training pipeline with GPU."""
    from src.compiler.math_jit_builder import build_sdf_graph
    from src.compiler.splat_trainer import compile_splats
    from src.torch_preloader import preloader
    
    device = preloader.get_device()
    
    # Build a simple SDF graph
    dna = {
        "root_node": {
            "type": "primitive",
            "shape": "sphere",
            "params": {"radius": 0.3}
        }
    }
    
    graph = build_sdf_graph(dna)
    bounds = ([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    
    print(f"\n[splat] Testing splat compilation on {device.upper()}...")
    print(f"   Target: 500 splats, 50 iterations")
    
    start = time.time()
    splat_data = compile_splats(
        graph,
        bounds,
        target_count=500,
        iterations=50,
        device=device,
    )
    elapsed = time.time() - start
    
    print(f"\n[results] Results:")
    print(f"   Compile time: {elapsed:.2f}s")
    print(f"   Output size: {len(splat_data)} bytes")
    print(f"   Device used: {device}")
    
    # Verify output is valid binary
    assert len(splat_data) > 0, "Empty output"
    
    # Decode splat count from header
    import struct
    splat_count = struct.unpack('<I', splat_data[:4])[0]
    print(f"   Splat count: {splat_count}")
    
    assert splat_count > 0, "No splats generated"
    print(f"   [OK] GPU splat training works!")


if __name__ == "__main__":
    print("=" * 60)
    print("GPU Splat Training Test")
    print("=" * 60)
    
    test_preloader_device()
    test_sdf_graph_to_device()
    test_splat_training_gpu()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
