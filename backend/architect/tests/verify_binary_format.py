
import sys
import os
import struct
from pathlib import Path
import tempfile

# Bypass compiler package to avoid torch dependency
# Add backend/architect/src/compiler to sys.path
current_file = Path(__file__).resolve()
# backend/architect/tests/verify_binary_writer_direct.py
# ../../src/compiler -> backend/architect/src/compiler
compiler_path = current_file.parent.parent / "src" / "compiler"
sys.path.append(str(compiler_path))

try:
    import binary_writer
    print("✅ Successfully imported binary_writer directly")
except ImportError as e:
    print(f"❌ Failed to import binary_writer: {e}")
    sys.exit(1)

def verify_writer():
    print("--- Verifying GVE Binary Writer (v2.3) ---")
    
    tmp_path = None
    try:
        # Create temp file
        fd, tmp_path_str = tempfile.mkstemp(suffix=".gve_bin")
        os.close(fd)
        tmp_path = Path(tmp_path_str)
            
        print(f"  📝 Writing to temporary file: {tmp_path}")
        
        # Test 1: Write via GVEBinaryWriter class
        writer = binary_writer.GVEBinaryWriter(tmp_path)
        
        # Add dummy data
        dummy_shell_verts = b"v"*24  # 1 vertex
        dummy_shell_indices = b"i"*12 # 1 triangle (3 ints) -> 12 bytes? No indices are u32? Assuming 4 bytes per index?
        # binary_writer.py logic:
        # vertices = shell_data[4:vertices_end]
        # So I should populate it correctly if I use set_shell_mesh.
        # set_shell_mesh(self, vertices: bytes, indices: bytes, ...
        writer.set_shell_mesh(dummy_shell_verts, dummy_shell_indices, 1, 3)
        writer.set_volume_data(b"volume_data")
        writer.set_splat_data(b"splat_data", 5)
        # writer.set_triplanar_data(b"triplanar") 
        
        writer.write()
        print("✅ Writer.write() successful")
        
        # Verify Header
        with open(tmp_path, "rb") as f:
            header = f.read(72)
            
        verify_header(header, "GVEBinaryWriter.write()")

        # Test 2: Write via write_gve_bin_bytes (Bytes generation)
        print("  📝 Testing write_gve_bin_bytes...")
        bytes_out = binary_writer.write_gve_bin_bytes(
            volume_data=b"volume_data",
            shell_data=None, # simpler
            splat_data=b"\x05\x00\x00\x00splat_data", # 4 bytes count + data
            triplanar_data=None
        )
        
        verify_header(bytes_out[:72], "write_gve_bin_bytes()")
        
        # Verify sizes
        expected_size = 72 + len(b"volume_data") + len(b"splat_data")
        if len(bytes_out) != expected_size:
             print(f"❌ Size Mismatch: {len(bytes_out)} (Expected {expected_size})")
        else:
             print(f"✅ Size Verified: {len(bytes_out)} bytes")

    except Exception as e:
        print(f"❌ Verification failed with exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print("  🗑️ Cleaned up temporary file")

def verify_header(header, source):
    if len(header) != 72:
        print(f"❌ [{source}] Header size mismatch: {len(header)} bytes (Expected 72)")
        return
        
    # Unpack
    # <4s I I Q Q Q Q Q I I I 8x
    unpacked = struct.unpack(binary_writer.HEADER_FMT, header)
    
    magic = unpacked[0]
    version = unpacked[1]
    # flags = unpacked[2]
    # vol_off = unpacked[3]
    # splat_off = unpacked[4]
    
    if magic != b"GVE1":
            print(f"❌ [{source}] Invalid Magic: {magic}")
    else:
            print(f"✅ [{source}] Magic Verified: {magic}")
            
    if version == 0x00023000:
            print(f"✅ [{source}] Version Verified: 0x{version:08x} (v2.3)")
    else:
            print(f"❌ [{source}] Version Mismatch: 0x{version:08x} (Expected 0x00023000)")

if __name__ == "__main__":
    verify_writer()
