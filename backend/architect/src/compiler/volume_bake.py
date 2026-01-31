import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List

def bake_volume(
    sdf_graph: nn.Module, 
    resolution: int = 64, 
    bounds: Optional[Tuple[List[float], List[float]]] = None
) -> bytes:
    """
    Sample the SDF graph onto a 3D grid.
    Returns raw float32 bytes of the volume data.
    
    Args:
        sdf_graph: SDF evaluation module (may have .bounds attribute)
        resolution: Grid resolution
        bounds: Optional (min_xyz, max_xyz). If None, uses sdf_graph.bounds or defaults.
    """
    # Determine bounds
    if bounds is None:
        if hasattr(sdf_graph, 'bounds') and sdf_graph.bounds is not None:
            bounds = sdf_graph.bounds
        else:
            bounds = ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    
    min_xyz, max_xyz = bounds
    
    # Add padding (10% on each side)
    padding = 0.1
    range_xyz = [max_xyz[i] - min_xyz[i] for i in range(3)]
    min_xyz = [min_xyz[i] - range_xyz[i] * padding for i in range(3)]
    max_xyz = [max_xyz[i] + range_xyz[i] * padding for i in range(3)]
    
    # Create grid coordinates with per-axis bounds
    x = torch.linspace(min_xyz[0], max_xyz[0], resolution)
    y = torch.linspace(min_xyz[1], max_xyz[1], resolution)
    z = torch.linspace(min_xyz[2], max_xyz[2], resolution)
    
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    
    points = torch.stack([
        grid_x.flatten(),
        grid_y.flatten(),
        grid_z.flatten()
    ], dim=1)
    
    # Evaluate SDF
    with torch.no_grad():
        distances = sdf_graph(points)
        
    # Reshape back to volume
    volume = distances.reshape(resolution, resolution, resolution)
    
    # Convert to numpy float32
    vol_np = volume.cpu().numpy().astype(np.float32)
    
    return vol_np.tobytes()
