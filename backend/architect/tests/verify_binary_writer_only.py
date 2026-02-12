
import sys
import os
import struct
from pathlib import Path

# Add src to path so we can import compiler.binary_writer
# Expecting to be run from project root or checks/tests dir
# If run from project root: backend/architect/tests/verify_binary_writer_only.py
# File path: backend/architect/tests/verify_binary_writer_only.py
# Root is ../../..
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent # GVE root? No.
# backend/architect/src is where compiler is.
# backend/architect/tests/.. -> backend/architect
architect_root = current_file.parent.parent
src_path = architect_root / "src"
sys.path.append(str(src_path))

try:
    from compiler.binary_writer import GVEBinaryWriter
    print("✅ Successfully imported GVEBinaryWriter")
except ImportError as e:
    print(f"❌ Failed to import GVEBinaryWriter: {e}")
    sys.exit(1)

def verify_writer():
    import tempfile
    
    print("--- Verifying GVE Binary Writer (v2.3) ---")
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".gve_bin", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        print(f"  📝 Writing to temporary file: {tmp_path}")
        
        writer = GVEBinaryWriter(tmp_path)
        
        # Add dummy data
        writer.set_shell_mesh(b"vertices", b"indices", 10, 20)
        writer.set_volume_data(b"volume")
        writer.set_splat_data(b"splat", 5)
        
        writer.write()
        
        print("✅ Write successful")
        
        # Read back and verify header
        with open(tmp_path, "rb") as f:
            header = f.read(72)
            
        if len(header) != 72:
            print(f"❌ Header size mismatch: {len(header)} bytes (Expected 72)")
            return
            
        # Unpack
        # <4s I I Q Q Q Q Q I I I 8x
        unpacked = struct.unpack("<4sIIQQQQQIII8x", header)
        
        magic = unpacked[0]
        version = unpacked[1]
        
        if magic != b"GVE1":
             print(f"❌ Invalid Magic: {magic}")
        else:
             print(f"✅ Magic Verified: {magic}")
             
        if version == 0x00023000:
             print(f"✅ Version Verified: 0x{version:08x} (v2.3)")
        else:
             print(f"❌ Version Mismatch: 0x{version:08x} (Expected 0x00023000)")

    except Exception as e:
        print(f"❌ Verification failed with exception: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print("  🗑️ Cleaned up temporary file")

if __name__ == "__main__":
    verify_writer()
