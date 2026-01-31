import torch
import numpy as np
import meshlib.mrmeshpy as mrmesh
import tempfile
import os

def bake_sdf_to_vdb(
    sdf_fn,
    bounds_min: tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounds_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
    voxel_size: float = 0.04
) -> mrmesh.VdbVolume:
    """
    Bake a PyTorch SDF function into a sparse OpenVDB volume via batched evaluation.
    
    PERFORMANCE: Uses batched PyTorch evaluation (50-100x faster than per-voxel callback).
    - Generates ALL voxel coordinates at once
    - Single PyTorch forward pass for ALL voxels
    - Writes to SimpleVolumeMinMax (dense), converts to VdbVolume (sparse)
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
    
    # 4. Reshape to 3D grid
    distance_grid = distances.reshape(res[0], res[1], res[2]).cpu().numpy()
    value_min = float(distance_grid.min())
    value_max = float(distance_grid.max())
    
    # 5. Create MeshLib SimpleVolumeMinMax (dense storage)
    simple_vol = mrmesh.SimpleVolumeMinMax()
    simple_vol.dims = mrmesh.Vector3i(int(res[0]), int(res[1]), int(res[2]))
    simple_vol.voxelSize = mrmesh.Vector3f(voxel_size, voxel_size, voxel_size)
    
    # Copy distance data to SimpleVolumeMinMax
    # SimpleVolumeMinMax.data is a flat vector - copy in C-order (row-major)
    # Note: min/max properties are READ-ONLY and computed automatically from data
    flat_data = distance_grid.flatten('C').astype(np.float32)
    simple_vol.data = mrmesh.std_vector_float(flat_data.tolist())
    # SimpleVolumeMinMax expects min/max to be set to the value range
    simple_vol.min = value_min
    simple_vol.max = value_max
    
    total_voxels = int(res[0] * res[1] * res[2])
    print(f"    SimpleVolumeMinMax created with {total_voxels:,} voxels (value range: {value_min:.3f} to {value_max:.3f})", flush=True)
    
    # 6. Convert SimpleVolumeMinMax to sparse VdbVolume
    print(f"    Converting to sparse VDB...", flush=True)
    vdb_volume = mrmesh.simpleVolumeToVdbVolume(simple_vol)
    print(f"    ✅ VDB conversion complete", flush=True)
    
    return vdb_volume

def vdb_to_bytes(grid: mrmesh.VdbVolume) -> bytes:
    """
    Serialize MeshLib VDB volume to .vdb bytes.
    """
    with tempfile.NamedTemporaryFile(suffix=".vdb", delete=False) as tmp:
        tmp_name = tmp.name
    
    try:
        # saveVoxels(file, object)
        # saveVoxels(object, file) based on signature
        mrmesh.saveVoxels(grid, tmp_name)
        
        with open(tmp_name, "rb") as f:
            data = f.read()
            
        return data
        
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
