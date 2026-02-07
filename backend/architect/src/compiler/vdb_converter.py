"""
VDB and Dense Grid conversion for SDF baking.

Converts PyTorch SDF functions to:
- VDB volumes (for mesh extraction via MeshLib)
- Dense grids (for GPU raymarching in WebGPU)
"""
import struct
import torch
import numpy as np
import meshlib.mrmeshpy as mrmesh
import tempfile
import os
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class BakeResult:
    """Result of SDF baking containing both VDB and dense grid data."""
    vdb_volume: mrmesh.VdbVolume
    dense_grid: np.ndarray  # Shape: (X, Y, Z), dtype: float32
    dims: Tuple[int, int, int]
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    voxel_size: float


def bake_sdf(
    sdf_fn,
    bounds_min: tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounds_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
    voxel_size: float = 0.04
) -> BakeResult:
    """
    Bake a PyTorch SDF function into both VDB and dense grid.
    
    Returns BakeResult containing:
    - vdb_volume: For mesh extraction
    - dense_grid: For GPU raymarching (WebGPU 3D texture)
    
    PERFORMANCE: Uses batched PyTorch evaluation (50-100x faster than per-voxel).
    """
    # 1. Calculate grid dimensions
    b_min = np.array(bounds_min, dtype=np.float32)
    b_max = np.array(bounds_max, dtype=np.float32)
    extent = b_max - b_min
    res = np.ceil(extent / voxel_size).astype(np.int32)
    
    # Clamp to avoid OOM
    MAX_RES = 512
    if np.any(res > MAX_RES):
        print(f"    Warning: Resolution {res} exceeds {MAX_RES}. Clamping.", flush=True)
        scale = MAX_RES / np.max(res)
        res = (res * scale).astype(np.int32)
        voxel_size = float(np.max(extent / res))
    
    print(f"    Grid resolution: {res[0]}x{res[1]}x{res[2]} = {np.prod(res):,} voxels", flush=True)
    
    # 2. Generate ALL voxel coordinates at once (BATCHED - KEY OPTIMIZATION)
    x = torch.linspace(b_min[0], b_max[0], res[0])
    y = torch.linspace(b_min[1], b_max[1], res[1])
    z = torch.linspace(b_min[2], b_max[2], res[2])
    
    # Create meshgrid and flatten to [N, 3]
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    coords = torch.stack([
        grid_x.flatten(),
        grid_y.flatten(),
        grid_z.flatten()
    ], dim=1)  # Shape: [N, 3]
    
    # 3. Single batched SDF evaluation (50-100x FASTER than callback)
    print(f"    Evaluating {coords.shape[0]:,} voxels in single batch...", flush=True)
    with torch.no_grad():
        distances = sdf_fn(coords)  # ONE PyTorch call for ALL voxels!
    
    # 4. Reshape to 3D grid (keep original SDF sign for raymarching)
    # Note: We store the TRUE SDF (negative = inside) for raymarching
    distance_grid_sdf = distances.reshape(res[0], res[1], res[2]).cpu().numpy().astype(np.float32)
    
    # For MeshLib, we need inverted sign (positive = inside)
    distance_grid_meshlib = -distance_grid_sdf
    value_min = float(distance_grid_meshlib.min())
    value_max = float(distance_grid_meshlib.max())
    
    # 5. Create MeshLib SimpleVolumeMinMax (dense storage)
    simple_vol = mrmesh.SimpleVolumeMinMax()
    simple_vol.dims = mrmesh.Vector3i(int(res[0]), int(res[1]), int(res[2]))
    simple_vol.voxelSize = mrmesh.Vector3f(voxel_size, voxel_size, voxel_size)
    
    # X-fastest ordering for MeshLib
    flat_data = distance_grid_meshlib.flatten('F').astype(np.float32)
    simple_vol.data = mrmesh.std_vector_float(flat_data.tolist())
    simple_vol.min = value_min
    simple_vol.max = value_max
    
    total_voxels = int(res[0] * res[1] * res[2])
    print(f"    SimpleVolumeMinMax created with {total_voxels:,} voxels", flush=True)
    
    # 6. Convert SimpleVolumeMinMax to sparse VdbVolume
    print(f"    Converting to sparse VDB...", flush=True)
    vdb_volume = mrmesh.simpleVolumeToVdbVolume(simple_vol)
    print(f"    ✅ VDB conversion complete", flush=True)
    
    return BakeResult(
        vdb_volume=vdb_volume,
        dense_grid=distance_grid_sdf,  # True SDF for raymarching
        dims=(int(res[0]), int(res[1]), int(res[2])),
        bounds_min=tuple(b_min.tolist()),
        bounds_max=tuple(b_max.tolist()),
        voxel_size=voxel_size,
    )


def bake_sdf_to_vdb(
    sdf_fn,
    bounds_min: tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounds_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
    voxel_size: float = 0.04
) -> mrmesh.VdbVolume:
    """
    Bake a PyTorch SDF function into a sparse OpenVDB volume.
    
    Backwards-compatible wrapper around bake_sdf().
    """
    result = bake_sdf(sdf_fn, bounds_min, bounds_max, voxel_size)
    return result.vdb_volume

def vdb_to_bytes(grid: mrmesh.VdbVolume) -> bytes:
    """Serialize MeshLib VDB volume to .vdb bytes."""
    with tempfile.NamedTemporaryFile(suffix=".vdb", delete=False) as tmp:
        tmp_name = tmp.name
    
    try:
        mrmesh.saveVoxels(grid, tmp_name)
        with open(tmp_name, "rb") as f:
            data = f.read()
        return data
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def dense_grid_to_bytes(bake_result: BakeResult, compress: bool = True) -> bytes:
    """
    Serialize dense grid for GPU raymarching with LZ4 compression.
    
    Binary format (little-endian):
    - dims: [u32; 3]             (12 bytes) - Grid dimensions
    - bounds_min: [f32; 3]       (12 bytes) - World-space min corner  
    - bounds_max: [f32; 3]       (12 bytes) - World-space max corner
    - uncompressed_size: u32     (4 bytes)  - Original voxel data size
    - compressed_data: [u8; N]   (variable) - LZ4 compressed voxels
    
    Total header: 40 bytes (before compressed data)
    
    SDF data compresses well (smooth gradients) - expect 5-10x compression.
    """
    import lz4.block
    
    dims = bake_result.dims
    b_min = bake_result.bounds_min
    b_max = bake_result.bounds_max
    
    # Flatten grid to X-fastest order (Fortran/column-major) for GPU
    voxels_raw = bake_result.dense_grid.flatten('F').astype(np.float32).tobytes()
    uncompressed_size = len(voxels_raw)
    
    if compress:
        # LZ4 block compression - store_size=False to output raw LZ4 block (no 4-byte size prefix)
        # Rust lz4_flex expects raw block data
        compressed = lz4.block.compress(voxels_raw, mode='high_compression', store_size=False)
        ratio = uncompressed_size / len(compressed)
        print(f"    Dense grid: {dims[0]}x{dims[1]}x{dims[2]}, {uncompressed_size} -> {len(compressed)} bytes ({ratio:.1f}x compression)", flush=True)
        voxels_data = compressed
    else:
        print(f"    Dense grid: {dims[0]}x{dims[1]}x{dims[2]} = {uncompressed_size} bytes (uncompressed)", flush=True)
        voxels_data = voxels_raw
    
    # Pack header: 3x u32 dims + 3x f32 bounds_min + 3x f32 bounds_max + u32 uncompressed_size
    header = struct.pack(
        "<III fff fff I",
        dims[0], dims[1], dims[2],
        b_min[0], b_min[1], b_min[2],
        b_max[0], b_max[1], b_max[2],
        uncompressed_size,
    )
    
    return header + voxels_data
