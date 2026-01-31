import sys
import os
import struct
import torch
from pathlib import Path

# Add path to src (one level up)
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

def verify():
    print("--- Verifying OpenVDB Integration (Phase 1) ---")

    # 1. Imports
    try:
        from compiler.math_jit import build_sdf_graph
        from compiler.vdb_converter import bake_sdf_to_vdb, vdb_to_bytes
        from compiler.mesh_repair import repair_and_decimate
        from compiler.binary_writer import write_gve_bin
        print("✅ Imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Please ensure meshlib is installed.")
        return

    # 2. Build SDF Graph
    print("\n--- Step 1: Building SDF Graph ---")
    dna = {
        "nodes": [
            {"type": "sphere", "radius": 0.5},
            {"type": "box", "size": [0.3, 1.2, 0.3]} # Cross shape
        ],
        "root": {"type": "union", "children": [0, 1]}
    }
    # Mocking DNA structure if different from actual parser expectation
    # existing verify_compiler.py used flattened list structure with implicit union?
    # Let's check verify_compiler.py again.
    # It used: "nodes": [{"type": "sphere"...}, ...] which implies implicit union in build_sdf_graph?
    # verify_compiler.py:23 "# Note: implicitly a union of these."
    
    dna_simple = {
        "nodes": [
            {"type": "sphere", "radius": 0.5},
            {"type": "box", "size": [0.4, 0.4, 0.4]}
        ]
    }
    
    try:
        graph = build_sdf_graph(dna_simple)
        print("✅ SDF Graph built")
    except Exception as e:
        print(f"❌ SDF Build failed: {e}")
        return

    # 3. Bake VDB
    print(f"\n--- Step 2: Baking SDF to VDB (MeshLib) ---")
    try:
        voxel_size = 0.05
        # bounds derived from graph, but let's be explicit
        grid = bake_sdf_to_vdb(
            lambda p: graph(p),
            bounds_min=(-1.0, -1.0, -1.0),
            bounds_max=(1.0, 1.0, 1.0),
            voxel_size=voxel_size
        )
        
        print(f"✅ VDB Grid created: {type(grid)}")
        # MeshLib VdbVolume doesn't expose activeVoxelCount directly in python?
        # We can check heapBytes() which returns size in bytes
        print(f"    📦 Grid Size: {grid.heapBytes()} bytes")
            
        # Serialize
        vdb_bytes = vdb_to_bytes(grid)
        print(f"✅ Serialized VDB size: {len(vdb_bytes)} bytes")
    except Exception as e:
        print(f"❌ VDB Baking failed: {e}")
        # Continue for other checks? No, crucial.
        return

    # 4. Mesh Repair
    print("\n--- Step 3: Mesh Repair (MeshLib) ---")
    try:
        shell_bytes = repair_and_decimate(grid, target_tris=500)
        print(f"✅ Shell Bytes: {len(shell_bytes)} bytes")
        if len(shell_bytes) > 8:
             num_verts = struct.unpack('<I', shell_bytes[:4])[0]
             print(f"   Vertex Count: {num_verts}")
    except Exception as e:
        print(f"❌ Mesh Repair failed: {e}")
        # Use dummy bytes to proceed to binary check
        shell_bytes = b"\x00"*12 

    # 5. Write Binary
    print("\n--- Step 4: Writing Binary (.gve_bin) ---")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".gve_bin", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        write_gve_bin(tmp_path, volume_data=vdb_bytes, shell_data=shell_bytes)
        
        file_size = os.path.getsize(tmp_path)
        print(f"✅ Binary written: {tmp_path} ({file_size} bytes)")
        
        # Verify Header
        with open(tmp_path, "rb") as f:
            header = f.read(84)
            # <4s I I Q Q Q Q Q Q I I I I 8x
            # Magic(4), Version(4), Flags(4), offsets(6*8), sizes(4*4)
            unpacked = struct.unpack("<4s I I Q Q Q Q Q Q I I I I", header[:76])
            
            magic = unpacked[0]
            version = unpacked[1]
            flags = unpacked[2]
            sdf_code_off = unpacked[3]
            vol_off = unpacked[4]
            shell_off = unpacked[6] # Index 6 corresponds to shell_mesh_offset in struct
            
            vol_size = unpacked[10] # sdf_bytecode_size (9), volume_size (10)
            
            print(f"   Magic: {magic}")
            print(f"   Version: 0x{version:08x} (Expected: 0x00021000)")
            print(f"   Volume Offset: {vol_off}")
            print(f"   Volume Size: {vol_size}")
            
            if magic == b"GVE1":
                print("✅ Magic Verified")
            else:
                print(f"❌ Magic Mismatch: {magic}")
                
            if version == 0x00021000:
                print("✅ Version Verified (2.1)")
            else:
                print(f"❌ Version Mismatch: {version:08x}")
                
            if vol_off > 0 and vol_size > 0:
                 print("✅ Volume chunk present")
            else:
                 print("❌ Volume chunk missing")

        # Cleanup
        os.unlink(tmp_path)
            
    except Exception as e:
        print(f"❌ Binary Write failed: {e}")

if __name__ == "__main__":
    verify()
