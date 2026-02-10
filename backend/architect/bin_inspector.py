#!/usr/bin/env python3
"""
GVE Binary Inspector - Analyze .gve_bin file structure and slot usage

Usage:
    python bin_inspector.py <file.gve_bin>
    python bin_inspector.py <directory>  # Scan all .gve_bin files
"""

import struct
import sys
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass


# Constants matching binary_writer.py
GVE_MAGIC = b"GVE1"
HEADER_SIZE = 84
HEADER_FMT = "<4s I I Q Q Q Q Q Q I I I I 8x"


@dataclass
class BinarySlot:
    """Represents a data slot in the binary"""
    name: str
    offset: int
    size: int
    count: Optional[int] = None


@dataclass
class BinaryInfo:
    """Parsed binary file information"""
    path: Path
    file_size: int
    version: int
    flags: int
    slots: List[BinarySlot]
    
    @property
    def total_data_size(self) -> int:
        return sum(slot.size for slot in self.slots)


def parse_gve_bin(path: Path) -> Optional[BinaryInfo]:
    """Parse a .gve_bin file and extract metadata"""
    try:
        with open(path, "rb") as f:
            # Read header
            header_data = f.read(HEADER_SIZE)
            if len(header_data) < HEADER_SIZE:
                return None
                
            # Unpack header
            (magic, version, flags,
             sdf_bytecode_offset, volume_offset, splat_offset,
             shell_offset, audio_offset, triplanar_offset,
             sdf_bytecode_size, volume_size, splat_count, vertex_count) = struct.unpack(HEADER_FMT, header_data)
            
            if magic != GVE_MAGIC:
                return None
            
            # Get file size
            f.seek(0, 2)
            file_size = f.tell()
            
            # Build slot list
            slots = []
            
            # SDF Bytecode
            if sdf_bytecode_offset > 0:
                slots.append(BinarySlot("SDF Bytecode", sdf_bytecode_offset, sdf_bytecode_size))
            
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
    print(f"\n📊 File Information:")
    print(f"  Total Size:    {format_size(info.file_size)}")
    print(f"  Version:       0x{info.version:08X}")
    print(f"  Header:        {HEADER_SIZE} bytes")
    print(f"  Data Payload:  {format_size(info.total_data_size)}")
    
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
        header_pct = HEADER_SIZE / info.file_size
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
