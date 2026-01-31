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
        self.vertex_count = 0
        self.index_count = 0
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
        offset += len(self.shell_vertices) + len(self.shell_indices)
        
        # Pack header
        header = struct.pack(
            HEADER_FMT,
            GVE_MAGIC,
            self.VERSION,
            self.flags,
            sdf_bytecode_offset,    # sdf_bytecode_offset
            volume_offset,           # sdf_texture_offset (Used for VDB in V2.1)
            0,                       # splat_data_offset
            shell_mesh_offset,       # shell_mesh_offset
            0,                       # audio_patch_offset
            0,                       # metadata_offset
            sdf_bytecode_size,       # sdf_bytecode_size
            volume_size,             # sdf_texture_size (VDB size)
            0,                       # splat_count
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


def write_gve_bin(
    path: Path, 
    volume_data: Optional[bytes] = None,
    shell_data: Optional[bytes] = None,
    splat_data: Optional[bytes] = None
) -> Path:
    """
    Write a .gve_bin file from compiled data.
    """
    writer = GVEBinaryWriter(path)
    
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
        if index_count_offset + 4 <= len(shell_data):
            index_count = struct.unpack("<I", shell_data[index_count_offset:index_count_offset+4])[0]
            indices = shell_data[index_count_offset+4:]
            writer.set_shell_mesh(vertices, indices, vertex_count, index_count)
        else:
             # Fallback if shell data is malformed
             writer.set_shell_mesh(b"", b"", 0, 0)
        
    writer.write()
    return path
