"""
Binary writer for .gve_bin files.
GVE 3.0 Chunk-Based Format — must match engine/shared/src/binary_format.rs

File layout:
  [GVE3Header]  16 bytes: magic("GVE3") + version + chunk_count + reserved
  [ChunkHeader] 16 bytes: fourcc + size(u64) + reserved
  [Chunk Data]  N bytes
  [Padding]     0-15 bytes (align to 16)
  ... repeat for each chunk ...
"""
import struct
from pathlib import Path
from typing import Optional, List, Tuple
import io

# GVE 3.0 constants
GVE3_MAGIC = b"GVE3"
GVE3_VERSION = 0x00030000  # v3.0

# Header: magic(4) + version(4) + chunk_count(4) + reserved(4) = 16
GVE3_HEADER_FMT = "<4sIII"
GVE3_HEADER_SIZE = 16

# Chunk header: fourcc(4) + size(8) + reserved(4) = 16
CHUNK_HEADER_FMT = "<4sQI"
CHUNK_HEADER_SIZE = 16

# Standard chunk FourCC IDs (must match binary_format.rs chunk_id)
CHUNK_VOLM = b"VOLM"  # Volume VDB data (LZ4 compressed)
CHUNK_MESH = b"MESH"  # Shell mesh (vertices + indices)
CHUNK_SPLT = b"SPLT"  # Gaussian splat data
CHUNK_TRIP = b"TRIP"  # Triplanar textures
CHUNK_ROPS = b"ROPS"  # Runtime operations (baked patches)
CHUNK_META = b"META"  # Metadata


def _pad_to_16(size: int) -> int:
    """Calculate padding bytes needed for 16-byte alignment."""
    return (16 - (size % 16)) % 16


def _write_chunk(f, fourcc: bytes, data: bytes):
    """Write a single chunk: header + data + padding."""
    chunk_header = struct.pack(CHUNK_HEADER_FMT, fourcc, len(data), 0)
    f.write(chunk_header)
    f.write(data)
    padding = _pad_to_16(len(data))
    if padding > 0:
        f.write(b"\x00" * padding)


class GVEBinaryWriter:
    """
    Writes .gve_bin files in GVE 3.0 chunk-based format.
    See engine/shared/src/binary_format.rs
    """
    
    def __init__(self, path: Path):
        self.path = path
        self.shell_vertices = b""
        self.shell_indices = b""
        self.volume_data = b""
        self.splat_data = b""
        self.triplanar_data = b""
        self.runtime_ops_data = b""
        self.vertex_count = 0
        self.index_count = 0
        self.splat_count = 0
        self.runtime_ops_count = 0
        self.flags = 0
        
    def set_shell_mesh(self, vertices: bytes, indices: bytes, vertex_count: int, index_count: int):
        """Set shell mesh data (vertices + indices)."""
        self.shell_vertices = vertices
        self.shell_indices = indices
        self.vertex_count = vertex_count
        self.index_count = index_count
        
    def set_volume_data(self, data: bytes):
        """Set Volume data (VDB/NanoVDB)."""
        self.volume_data = data
        
    def set_splat_data(self, data: bytes, count: int):
        """Set Gaussian splat data."""
        self.splat_data = data
        self.splat_count = count
        
    def set_triplanar_data(self, data: bytes):
        """Set triplanar texture data."""
        self.triplanar_data = data

    def _collect_chunks(self) -> List[Tuple[bytes, bytes]]:
        """Collect all non-empty data as (fourcc, data) pairs."""
        chunks = []
        
        if self.volume_data:
            chunks.append((CHUNK_VOLM, self.volume_data))
        
        if self.shell_vertices:
            # MESH chunk: vertex_count(u32) + index_count(u32) + vertices + indices
            mesh_data = struct.pack("<II", self.vertex_count, self.index_count)
            mesh_data += self.shell_vertices
            if self.shell_indices:
                mesh_data += self.shell_indices
            chunks.append((CHUNK_MESH, mesh_data))
        
        if self.splat_data:
            chunks.append((CHUNK_SPLT, self.splat_data))
        
        if self.triplanar_data:
            chunks.append((CHUNK_TRIP, self.triplanar_data))
        
        if self.runtime_ops_data:
            chunks.append((CHUNK_ROPS, self.runtime_ops_data))
        
        return chunks

    def write(self):
        """Write the binary file to disk."""
        chunks = self._collect_chunks()
        
        with open(self.path, "wb") as f:
            # Write GVE3 header
            header = struct.pack(GVE3_HEADER_FMT,
                GVE3_MAGIC,
                GVE3_VERSION,
                len(chunks),
                0,  # reserved
            )
            f.write(header)
            
            # Write chunks
            for fourcc, data in chunks:
                _write_chunk(f, fourcc, data)

    def to_bytes(self) -> bytes:
        """Build .gve_bin data and return as bytes (no disk write)."""
        chunks = self._collect_chunks()
        
        buf = io.BytesIO()
        
        # Write GVE3 header
        header = struct.pack(GVE3_HEADER_FMT,
            GVE3_MAGIC,
            GVE3_VERSION,
            len(chunks),
            0,  # reserved
        )
        buf.write(header)
        
        # Write chunks
        for fourcc, data in chunks:
            _write_chunk(buf, fourcc, data)
        
        return buf.getvalue()


def _prepare_writer(
    volume_data: Optional[bytes] = None,
    shell_data: Optional[bytes] = None,
    splat_data: Optional[bytes] = None,
    triplanar_data: Optional[bytes] = None,
    runtime_ops_data: Optional[bytes] = None,
) -> GVEBinaryWriter:
    """
    Prepare a GVEBinaryWriter with the given data.
    Internal helper used by both write_gve_bin and write_gve_bin_bytes.
    """
    writer = GVEBinaryWriter(Path("/dev/null"))  # Path not used for bytes output
    
    if volume_data:
        writer.set_volume_data(volume_data)
    
    if shell_data and len(shell_data) > 8:
        # Parse shell_data from binary shell format
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
    
    # Parse and set splat data (strip the 4-byte count header — count is
    # already stored separately)
    if splat_data and len(splat_data) >= 4:
        splat_count = struct.unpack("<I", splat_data[0:4])[0]
        writer.set_splat_data(splat_data[4:], splat_count)
    
    # Set triplanar texture data
    if triplanar_data:
        writer.set_triplanar_data(triplanar_data)

    if runtime_ops_data:
        # Direct set as raw bytes
        writer.runtime_ops_data = runtime_ops_data
    
    return writer


def write_gve_bin(
    path: Path, 
    volume_data: Optional[bytes] = None,
    shell_data: Optional[bytes] = None,
    splat_data: Optional[bytes] = None,
    triplanar_data: Optional[bytes] = None,
    runtime_ops_data: Optional[bytes] = None,
) -> Path:
    """
    Write a .gve_bin file from compiled data.
    """
    writer = _prepare_writer(volume_data, shell_data, splat_data, triplanar_data, runtime_ops_data)
    writer.path = path
    writer.write()
    return path


def write_gve_bin_bytes(
    volume_data: Optional[bytes] = None,
    shell_data: Optional[bytes] = None,
    splat_data: Optional[bytes] = None,
    triplanar_data: Optional[bytes] = None,
    runtime_ops_data: Optional[bytes] = None,
) -> bytes:
    """
    Build .gve_bin data and return as bytes (no disk write).
    
    Used for stage previews during AI generation pipeline.
    """
    writer = _prepare_writer(volume_data, shell_data, splat_data, triplanar_data, runtime_ops_data)
    return writer.to_bytes()
