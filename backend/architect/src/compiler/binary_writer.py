"""
Binary writer for .gve_bin files.
Must match engine/shared/src/binary_format.rs GVEBinaryHeader
"""
import struct
from pathlib import Path
from typing import Optional

# Header constants - must match Rust
GVE_MAGIC = b"GVE1"  # Changed from GVE\0 to match Rust
HEADER_SIZE = 84      # 84 bytes to match Rust struct (not 64!)

# Header format (84 bytes total):
# magic: [u8; 4]              - 4 bytes
# version: u32                - 4 bytes  
# flags: u32                  - 4 bytes
# sdf_bytecode_offset: u64    - 8 bytes
# sdf_texture_offset: u64     - 8 bytes
# splat_data_offset: u64      - 8 bytes
# shell_mesh_offset: u64      - 8 bytes
# audio_patch_offset: u64     - 8 bytes
# metadata_offset: u64        - 8 bytes
# sdf_bytecode_size: u32      - 4 bytes
# sdf_texture_size: u32       - 4 bytes
# splat_count: u32            - 4 bytes
# vertex_count: u32           - 4 bytes
# _padding: [u8; 8]           - 8 bytes
# Total: 4+4+4+(6*8)+(4*4)+8 = 84 bytes

HEADER_FMT = "<4s I I Q Q Q Q Q Q I I I I 8x"


class GVEBinaryWriter:
    """
    Writes .gve_bin files according to engine/shared/src/binary_format.rs
    """
    VERSION = 0x00021000  # v2.1 - OpenVDB support
    
    def __init__(self, path: Path):
        self.path = path
        self.shell_vertices = b""   # Raw vertex data
        self.shell_indices = b""    # Raw index data  
        self.sdf_bytecode = b""     # SDF instruction bytecode
        self.volume_data = b""      # VDB Grid data
        self.splat_data = b""       # Gaussian splat data
        self.vertex_count = 0
        self.index_count = 0
        self.splat_count = 0
        self.flags = 0
        
    def set_shell_mesh(self, vertices: bytes, indices: bytes, vertex_count: int, index_count: int):
        """Set shell mesh data (vertices + indices)."""
        self.shell_vertices = vertices
        self.shell_indices = indices
        self.vertex_count = vertex_count
        self.index_count = index_count
        
    def set_sdf_bytecode(self, bytecode: bytes):
        """Set SDF bytecode data."""
        self.sdf_bytecode = bytecode
        
    def set_volume_data(self, data: bytes):
        """Set Volume data (VDB/NanoVDB)."""
        self.volume_data = data
        
    def set_splat_data(self, data: bytes, count: int):
        """Set Gaussian splat data."""
        self.splat_data = data
        self.splat_count = count
        
    def write(self):
        """Write the binary file to disk."""
        # Calculate offsets (all relative to start of file)
        offset = HEADER_SIZE
        
        # SDF bytecode
        sdf_bytecode_offset = offset if self.sdf_bytecode else 0
        sdf_bytecode_size = len(self.sdf_bytecode)
        offset += sdf_bytecode_size
        
        # Volume Data (formerly SDF Texture)
        # In v2.1 this is VDB data
        volume_offset = offset if self.volume_data else 0
        volume_size = len(self.volume_data)
        offset += volume_size
        
        # Shell mesh
        shell_mesh_offset = offset if self.shell_vertices else 0
        shell_mesh_size = len(self.shell_vertices) + len(self.shell_indices)
        if self.shell_vertices:
            shell_mesh_size += 8  # Add header (vertex_count + index_count)
        offset += shell_mesh_size
        
        # Splat data
        splat_data_offset = offset if self.splat_data else 0
        offset += len(self.splat_data)
        
        # Pack header
        header = struct.pack(
            HEADER_FMT,
            GVE_MAGIC,
            self.VERSION,
            self.flags,
            sdf_bytecode_offset,    # sdf_bytecode_offset
            volume_offset,           # sdf_texture_offset (Used for VDB in V2.1)
            splat_data_offset,       # splat_data_offset
            shell_mesh_offset,       # shell_mesh_offset
            0,                       # audio_patch_offset
            0,                       # metadata_offset
            sdf_bytecode_size,       # sdf_bytecode_size
            volume_size,             # sdf_texture_size (VDB size)
            self.splat_count,        # splat_count
            self.vertex_count,       # vertex_count
        )
        
        # Write file
        with open(self.path, "wb") as f:
            f.write(header)
            
            # Write SDF bytecode
            if self.sdf_bytecode:
                f.write(self.sdf_bytecode)
                
            # Write Volume Data
            if self.volume_data:
                f.write(self.volume_data)
            
            # Write shell mesh (vertex_count + index_count header, then data)
            if self.shell_vertices:
                # Shell mesh section starts with counts
                f.write(struct.pack("<II", self.vertex_count, self.index_count))
                f.write(self.shell_vertices)
                if self.shell_indices:
                    f.write(self.shell_indices)
            
            # Write splat data
            if self.splat_data:
                f.write(self.splat_data)


def _prepare_writer(
    volume_data: Optional[bytes] = None,
    shell_data: Optional[bytes] = None,
    splat_data: Optional[bytes] = None
) -> GVEBinaryWriter:
    """
    Prepare a GVEBinaryWriter with the given data.
    Internal helper used by both write_gve_bin and write_gve_bin_bytes.
    """
    writer = GVEBinaryWriter(Path("/dev/null"))  # Path not used for bytes output
    
    if volume_data:
        writer.set_volume_data(volume_data)
    
    if shell_data and len(shell_data) > 8:
        # Parse shell_data from shell_gen format
        vertex_count = struct.unpack("<I", shell_data[0:4])[0]
        
        # Each vertex is 6 floats (24 bytes): pos(3) + normal(3)
        vertex_size = 24
        vertices_end = 4 + vertex_count * vertex_size
        vertices = shell_data[4:vertices_end]
        
        # Index count follows vertices
        index_count_offset = vertices_end
        
        print(f"  [binary_writer] 🔍 shell_data len={len(shell_data)}, vertex_count={vertex_count}, vertices_end={vertices_end}, index_count_offset={index_count_offset}", flush=True)
        
        if index_count_offset + 4 <= len(shell_data):
            index_count = struct.unpack("<I", shell_data[index_count_offset:index_count_offset+4])[0]
            indices = shell_data[index_count_offset+4:]
            print(f"  [binary_writer] ✅ Parsed: index_count={index_count}, indices_len={len(indices)}, vertices_len={len(vertices)}", flush=True)
            writer.set_shell_mesh(vertices, indices, vertex_count, index_count)
        else:
            # Fallback if shell data is malformed
            print(f"  [binary_writer] ❌ Shell data malformed: index_count_offset ({index_count_offset}) + 4 > len ({len(shell_data)})", flush=True)
            writer.set_shell_mesh(b"", b"", 0, 0)
    else:
        print(f"  [binary_writer] ⚠️ No shell_data or too short: shell_data={shell_data is not None}, len={len(shell_data) if shell_data else 0}", flush=True)
    
    # Parse and set splat data
    if splat_data and len(splat_data) >= 4:
        # Splat data starts with u32 count header
        splat_count = struct.unpack("<I", splat_data[0:4])[0]
        writer.set_splat_data(splat_data, splat_count)
    
    return writer


def write_gve_bin(
    path: Path, 
    volume_data: Optional[bytes] = None,
    shell_data: Optional[bytes] = None,
    splat_data: Optional[bytes] = None
) -> Path:
    """
    Write a .gve_bin file from compiled data.
    """
    writer = _prepare_writer(volume_data, shell_data, splat_data)
    writer.path = path
    writer.write()
    return path


def write_gve_bin_bytes(
    volume_data: Optional[bytes] = None,
    shell_data: Optional[bytes] = None,
    splat_data: Optional[bytes] = None
) -> bytes:
    """
    Build .gve_bin data and return as bytes (no disk write).
    
    Used for stage previews during AI generation pipeline.
    """
    writer = _prepare_writer(volume_data, shell_data, splat_data)
    
    # Calculate offsets (all relative to start of file)
    offset = HEADER_SIZE
    
    # SDF bytecode
    sdf_bytecode_offset = offset if writer.sdf_bytecode else 0
    sdf_bytecode_size = len(writer.sdf_bytecode)
    offset += sdf_bytecode_size
    
    # Volume Data (VDB)
    volume_offset = offset if writer.volume_data else 0
    volume_size = len(writer.volume_data)
    offset += volume_size
    
    # Shell mesh
    shell_mesh_offset = offset if writer.shell_vertices else 0
    shell_mesh_size = len(writer.shell_vertices) + len(writer.shell_indices)
    if writer.shell_vertices:
        shell_mesh_size += 8  # Add header (vertex_count + index_count)
    offset += shell_mesh_size
    
    # Splat data
    splat_data_offset = offset if writer.splat_data else 0
    offset += len(writer.splat_data)
    
    # Pack header
    header = struct.pack(
        HEADER_FMT,
        GVE_MAGIC,
        writer.VERSION,
        writer.flags,
        sdf_bytecode_offset,
        volume_offset,
        splat_data_offset,
        shell_mesh_offset,
        0,  # audio_patch_offset
        0,  # metadata_offset
        sdf_bytecode_size,
        volume_size,
        writer.splat_count,
        writer.vertex_count,
    )
    
    # Build bytes buffer
    parts = [header]
    
    if writer.sdf_bytecode:
        parts.append(writer.sdf_bytecode)
    
    if writer.volume_data:
        parts.append(writer.volume_data)
    
    if writer.shell_vertices:
        parts.append(struct.pack("<II", writer.vertex_count, writer.index_count))
        parts.append(writer.shell_vertices)
        if writer.shell_indices:
            parts.append(writer.shell_indices)
    
    if writer.splat_data:
        parts.append(writer.splat_data)
    
    return b"".join(parts)

