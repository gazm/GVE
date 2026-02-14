#!/usr/bin/env python3
"""
GVE Binary Inspector - Analyze .gve_bin file structure and slot usage

Supports GVE1 (legacy) and GVE3 (chunk-based) formats.
Processes VOLM, MESH, SPLT, TRIP, ROPS, SKEL chunks.

Usage:
    python bin_inspector.py <file.gve_bin>
    python bin_inspector.py <directory>  # Scan all .gve_bin files
"""

import struct
import sys
import io
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass


# GVE1 (legacy) constants
GVE1_MAGIC = b"GVE1"
GVE1_HEADER_SIZE = 84
GVE1_HEADER_FMT = "<4s I I Q Q Q Q Q Q I I I I 8x"

# GVE3 (chunk-based) constants — matches binary_writer.py
GVE3_MAGIC = b"GVE3"
GVE3_HEADER_FMT = "<4sIII"
GVE3_HEADER_SIZE = 16
CHUNK_HEADER_FMT = "<4sQI"
CHUNK_HEADER_SIZE = 16
CHUNK_VOLM = b"VOLM"
CHUNK_MESH = b"MESH"
CHUNK_SPLT = b"SPLT"
CHUNK_TRIP = b"TRIP"
CHUNK_ROPS = b"ROPS"
CHUNK_SKEL = b"SKEL"
CHUNK_META = b"META"


def _pad_to_16(size: int) -> int:
    return (16 - (size % 16)) % 16


@dataclass
class BinarySlot:
    """Represents a data slot in the binary"""
    name: str
    offset: int
    size: int
    count: Optional[int] = None


@dataclass
class SkeletonInfo:
    """Parsed SKEL chunk details"""
    bone_count: int
    binding_count: int
    rigid_count: int
    skinned_count: int


@dataclass
class BinaryInfo:
    """Parsed binary file information"""
    path: Path
    file_size: int
    version: int
    flags: int
    slots: List[BinarySlot]
    format: str = "GVE1"
    skeleton: Optional[SkeletonInfo] = None

    @property
    def total_data_size(self) -> int:
        return sum(slot.size for slot in self.slots)


def parse_skel_chunk(data: bytes) -> Optional[SkeletonInfo]:
    """Parse SKEL chunk bytes; returns SkeletonInfo or None on error."""
    if len(data) < 2:
        return None
    buf = io.BytesIO(data)
    bone_count = struct.unpack("<H", buf.read(2))[0]
    const_bone_size = 2 + 12 + 16  # parent_idx + rest_pos + rest_rot
    if len(data) < 2 + bone_count * const_bone_size + 4:
        return None
    buf.seek(2 + bone_count * const_bone_size)
    mapping_count = struct.unpack("<I", buf.read(4))[0]
    rigid = 0
    skinned = 0
    cursor = buf.tell()
    for _ in range(mapping_count):
        if cursor + 8 > len(data):
            break
        kind = data[cursor]
        if kind == 0:
            rigid += 1
            cursor += 8
        else:
            n = data[cursor + 5]
            cursor += 8 + n * (2 + 4)
            skinned += 1
    return SkeletonInfo(bone_count, mapping_count, rigid, skinned)


def parse_gve3_bin(path: Path, f) -> Optional[BinaryInfo]:
    """Parse GVE3 chunk-based format."""
    header = f.read(GVE3_HEADER_SIZE)
    if len(header) < GVE3_HEADER_SIZE:
        return None
    magic, version, chunk_count, _ = struct.unpack(GVE3_HEADER_FMT, header)
    if magic != GVE3_MAGIC:
        return None
    f.seek(0, 2)
    file_size = f.tell()
    f.seek(GVE3_HEADER_SIZE)

    slots: List[BinarySlot] = []
    skeleton: Optional[SkeletonInfo] = None

    for _ in range(chunk_count):
        chdr = f.read(CHUNK_HEADER_SIZE)
        if len(chdr) < CHUNK_HEADER_SIZE:
            break
        fourcc, size, _ = struct.unpack(CHUNK_HEADER_FMT, chdr)
        data_start = f.tell()
        if data_start + size > file_size:
            break
        chunk_data = f.read(size) if size > 0 else b""
        padding = _pad_to_16(size)
        if padding > 0:
            f.read(padding)

        if fourcc == CHUNK_VOLM:
            slots.append(BinarySlot("Volume (VOLM)", data_start, size))
        elif fourcc == CHUNK_MESH:
            if len(chunk_data) >= 8:
                vc, ic = struct.unpack("<II", chunk_data[:8])
                mesh_size = 8 + vc * 24 + ic * 4
                slots.append(BinarySlot("Shell Mesh", data_start, min(mesh_size, size), vc))
            else:
                slots.append(BinarySlot("Shell Mesh", data_start, size))
        elif fourcc == CHUNK_SPLT:
            if len(chunk_data) >= 4:
                splat_count = struct.unpack("<I", chunk_data[:4])[0]
                splat_size = splat_count * 48
                slots.append(BinarySlot("Gaussian Splats (SPLT)", data_start, size, splat_count))
            else:
                slots.append(BinarySlot("Splat (SPLT)", data_start, size))
        elif fourcc == CHUNK_TRIP:
            slots.append(BinarySlot("Triplanar (TRIP)", data_start, size))
        elif fourcc == CHUNK_ROPS:
            op_size = 96
            count = size // op_size if op_size else 0
            slots.append(BinarySlot("Runtime Ops (ROPS)", data_start, size, count))
        elif fourcc == CHUNK_SKEL:
            skel_info = parse_skel_chunk(chunk_data)
            if skel_info:
                skeleton = skel_info
            bone_cnt = skeleton.bone_count if skeleton else 0
            slots.append(BinarySlot("Skeleton (SKEL)", data_start, size, bone_cnt))
        elif fourcc == CHUNK_META:
            slots.append(BinarySlot("Metadata (META)", data_start, size))

    return BinaryInfo(path, file_size, version, 0, slots, format="GVE3", skeleton=skeleton)


def parse_gve_bin(path: Path) -> Optional[BinaryInfo]:
    """Parse a .gve_bin file and extract metadata (GVE1 or GVE3 format)."""
    try:
        with open(path, "rb") as f:
            magic_bytes = f.read(4)
            if len(magic_bytes) < 4:
                return None
            f.seek(0)

            if magic_bytes == GVE3_MAGIC:
                return parse_gve3_bin(path, f)

            # GVE1 format
            header_data = f.read(GVE1_HEADER_SIZE)
            if len(header_data) < GVE1_HEADER_SIZE:
                return None
            (magic, version, flags,
             volume_offset, splat_offset, shell_offset,
             audio_offset, triplanar_offset,
             volume_size, splat_count, vertex_count) = struct.unpack(GVE1_HEADER_FMT, header_data)
            if magic != GVE1_MAGIC:
                return None
            
            # Get file size
            f.seek(0, 2)
            file_size = f.tell()
            
            # Build slot list
            slots = []
            
            # Volume Data
            if volume_offset > 0:
                slots.append(BinarySlot("Volume (NanoVDB)", volume_offset, volume_size))
            
            # Shell Mesh
            if shell_offset > 0:
                # Read mesh header to get actual size
                f.seek(shell_offset)
                mesh_vertex_count, mesh_index_count = struct.unpack("<II", f.read(8))
                mesh_size = 8 + (mesh_vertex_count * 24) + (mesh_index_count * 4)
                slots.append(BinarySlot("Shell Mesh", shell_offset, mesh_size, mesh_vertex_count))
            
            # Splat Data
            if splat_offset > 0:
                splat_size = splat_count * 48  # 48 bytes per splat
                slots.append(BinarySlot("Gaussian Splats", splat_offset, splat_size, splat_count))
            
            # Triplanar Textures
            if triplanar_offset > 0:
                # Calculate size as difference to next section or EOF
                next_offset = file_size
                triplanar_size = next_offset - triplanar_offset
                slots.append(BinarySlot("Triplanar Textures", triplanar_offset, triplanar_size))
            
            return BinaryInfo(path, file_size, version, flags, slots)
            
    except Exception as e:
        print(f"Error parsing {path}: {e}", file=sys.stderr)
        return None


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def print_bar(label: str, value: float, max_width: int = 50):
    """Print a horizontal bar chart"""
    bar_width = int(value * max_width)
    bar = "█" * bar_width + "░" * (max_width - bar_width)
    print(f"  {label:20s} {bar} {value:6.1%}")


def display_binary_info(info: BinaryInfo):
    """Display binary information in a nice terminal format"""
    print(f"\n{'='*70}")
    print(f"📦 {info.path.name}")
    print(f"{'='*70}")
    
    # File info
    header_size = GVE3_HEADER_SIZE if info.format == "GVE3" else GVE1_HEADER_SIZE
    print(f"\n📊 File Information:")
    print(f"  Total Size:    {format_size(info.file_size)}")
    print(f"  Format:        {info.format}")
    print(f"  Version:       0x{info.version:08X}")
    print(f"  Header:        {header_size} bytes")
    print(f"  Data Payload:  {format_size(info.total_data_size)}")

    if info.skeleton:
        print(f"\n🦴 Skeleton (SKEL):")
        print(f"  Bones:         {info.skeleton.bone_count}")
        print(f"  Bindings:      {info.skeleton.binding_count} (rigid: {info.skeleton.rigid_count}, skinned: {info.skeleton.skinned_count})")

    # Slot breakdown
    if info.slots:
        print(f"\n📂 Data Slots:")
        print(f"  {'Slot':<20s} {'Size':>12s} {'Count':>10s} {'%':>6s}")
        print(f"  {'-'*52}")
        
        for slot in sorted(info.slots, key=lambda s: s.size, reverse=True):
            size_str = format_size(slot.size)
            count_str = f"{slot.count:,}" if slot.count else "-"
            pct = (slot.size / info.file_size) * 100
            print(f"  {slot.name:<20s} {size_str:>12s} {count_str:>10s} {pct:5.1f}%")
        
        # Visual distribution
        print(f"\n📈 Space Distribution:")
        header_pct = header_size / info.file_size
        print_bar("Header", header_pct)
        
        for slot in sorted(info.slots, key=lambda s: s.size, reverse=True):
            pct = slot.size / info.file_size
            print_bar(slot.name, pct)
    else:
        print(f"\n⚠️  No data slots found (empty binary)")
    
    # GPU memory estimate
    print(f"\n💾 GPU Memory Estimate:")
    gpu_size = info.total_data_size  # Rough estimate (uncompressed)
    print(f"  Estimated VRAM: {format_size(gpu_size)}")
    
    print()


def scan_directory(directory: Path):
    """Scan directory for .gve_bin files and display summary"""
    bin_files = list(directory.rglob("*.gve_bin"))
    
    if not bin_files:
        print(f"No .gve_bin files found in {directory}")
        return
    
    print(f"\n🔍 Found {len(bin_files)} .gve_bin files in {directory}")
    
    total_size = 0
    total_splats = 0
    total_vertices = 0
    
    for bin_file in sorted(bin_files):
        info = parse_gve_bin(bin_file)
        if info:
            display_binary_info(info)
            total_size += info.file_size
            
            # Accumulate stats
            for slot in info.slots:
                if slot.name == "Gaussian Splats" and slot.count:
                    total_splats += slot.count
                elif slot.name == "Shell Mesh" and slot.count:
                    total_vertices += slot.count
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 Summary ({len(bin_files)} files)")
    print(f"{'='*70}")
    print(f"  Total Size:      {format_size(total_size)}")
    print(f"  Total Splats:    {total_splats:,}")
    print(f"  Total Vertices:  {total_vertices:,}")
    print(f"  Avg File Size:   {format_size(total_size // len(bin_files))}")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if not path.exists():
        print(f"Error: {path} does not exist")
        sys.exit(1)
    
    if path.is_file():
        info = parse_gve_bin(path)
        if info:
            display_binary_info(info)
        else:
            print(f"Error: Could not parse {path} as .gve_bin file")
            sys.exit(1)
    elif path.is_dir():
        scan_directory(path)
    else:
        print(f"Error: {path} is neither a file nor directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
