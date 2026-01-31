import sys
import os
import torch
import numpy as np
import struct

# Add path
sys.path.append(r"c:\Users\Admin\Documents\OG\Projects\GVE\backend\architect\src")

from compiler.math_jit import build_sdf_graph
from compiler.shell_gen import generate_shell

def verify():
    # Test DNA
    dna = {
        "nodes": [
            {"type": "sphere", "radius": 0.5},
            {"type": "box", "size": [0.8, 0.8, 0.8]},
            # Add cylinder to test new primitive
            {"type": "cylinder", "radius": 0.3, "height": 1.0}
        ]
    }
    # Note: implicitly a union of these.
    
    print("Building SDF Graph...")
    graph = build_sdf_graph(dna)
    
    print("Evaluating Graph at Origin...")
    # Origin is inside all of them (Union -> min dist). 
    # Sphere: -0.5
    # Box: -0.4 (half size 0.4)
    # Cylinder: -0.3
    # Result should be roughly -0.5 (min)
    # ... (existing code for building graph and evaluating point) ...
    # This part is covered by existing checks in previous steps (if any)
    # Re-verify point just in case
    pts = torch.tensor([[0.0, 0.0, 0.0]])
    d = graph(pts)
    print(f"Dist at origin: {d.item()} (Expected ~ -0.5)")

    print("\n--- Testing Volume Bake ---")
    from compiler.volume_bake import bake_volume
    # Bake small volume
    res = 32
    bounds = (-1.0, 1.0)
    vol_bytes = bake_volume(graph, resolution=res, bounds=bounds)
    print(f"Volume Baked: {len(vol_bytes)} bytes")
    
    expected_size = res * res * res * 4 # float32
    if len(vol_bytes) == expected_size:
        print("SUCCESS: Volume size correct.")
    else:
        print(f"FAILURE: Volume size incorrect. Got {len(vol_bytes)}, expected {expected_size}")

    print("\n--- Testing Shell Gen (Optimized Path) ---")
    # Provide volume_data to generate_shell
    shell_bytes = generate_shell(graph, resolution=res, bounds=bounds, volume_data=vol_bytes)
    
    print(f"Shell Size: {len(shell_bytes)} bytes")
    
    # Parse shell to verify it's not empty
    if len(shell_bytes) > 8:
        num_verts = struct.unpack('I', shell_bytes[:4])[0]
        print(f"Num Verts: {num_verts}")
        if num_verts > 0:
            print("SUCCESS: Mesh generated from pre-baked volume.")
        else:
            print("FAILURE: Mesh empty.")
    else:
        print("FAILURE: Output too small.")

    print("\n--- Testing Binary Writer ---")
    from compiler.binary_writer import write_gve_bin
    import tempfile
    from pathlib import Path

    # Create temp file
    with tempfile.NamedTemporaryFile(suffix=".gve_bin", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        write_gve_bin(tmp_path, volume_data=vol_bytes, shell_data=shell_bytes)
        print(f"Binary written to: {tmp_path}")
        
        # Verify header
        with open(tmp_path, "rb") as f:
            header = f.read(24)
            # Magic(4s), Version(I), Flags(I), VolOff(I), ShellOff(I), SplatOff(I)
            magic, ver, flags, vol_off, shell_off, splat_off = struct.unpack("<4s5I", header)
            
            print(f"Magic: {magic}")
            print(f"Version: {ver}")
            print(f"Volume Offset: {vol_off}")
            print(f"Shell Offset: {shell_off}")
            
            if magic == b"GVE\0" and vol_off == 24: # Header is 24 bytes
                print("SUCCESS: Binary header verified.")
            else:
                print("FAILURE: Binary header invalid.")

            # Verify size
            file_size = os.path.getsize(tmp_path)
            expected_file_size = 24 + len(vol_bytes) + len(shell_bytes)
            if file_size == expected_file_size:
                 print("SUCCESS: File size matches content.")
            else:
                 print(f"FAILURE: File size mismatch. Got {file_size}, expected {expected_file_size}")

    finally:
        # Cleanup
        if tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except:
                pass

if __name__ == "__main__":
    verify()
