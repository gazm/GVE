import torch
import torch.nn as nn
import numpy as np
import struct
from typing import Optional, Tuple, List

def generate_shell(
    sdf_graph: nn.Module, 
    resolution: int = 64,
    bounds: Optional[Tuple[List[float], List[float]]] = None,
    volume_data: Optional[bytes] = None
) -> bytes:
    """
    Generate a mesh shell from the SDF graph using Dual Contouring.
    Returns raw bytes of the binary mesh format (NumVerts:u32, Verts:[f32;6]*N, NumIndices:u32, Indices:[u32]*M).
    
    Format:
    - Vertex: [x, y, z, nx, ny, nz] (float32 * 6)
    - Index: uint32
    
    Args:
        sdf_graph: SDF evaluation module (may have .bounds attribute)
        resolution: Grid resolution for meshing
        bounds: Optional (min_xyz, max_xyz) bounds. If None, uses sdf_graph.bounds or defaults.
        volume_data: Optional pre-computed SDF volume (reused if resolution matches)
    """
    
    # Determine bounds
    if bounds is None:
        # Try to get bounds from sdf_graph
        if hasattr(sdf_graph, 'bounds') and sdf_graph.bounds is not None:
            bounds = sdf_graph.bounds
        else:
            # Default fallback - will likely produce empty mesh for small objects
            bounds = ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    
    # Convert bounds to per-axis values with padding
    min_xyz, max_xyz = bounds
    padding = 0.1  # 10% padding on each side
    
    range_xyz = [max_xyz[i] - min_xyz[i] for i in range(3)]
    min_xyz = [min_xyz[i] - range_xyz[i] * padding for i in range(3)]
    max_xyz = [max_xyz[i] + range_xyz[i] * padding for i in range(3)]
    
    # Use the largest axis range for uniform grid spacing
    max_range = max(max_xyz[i] - min_xyz[i] for i in range(3))
    
    print(f"    [shell_gen] Bounds: min={min_xyz}, max={max_xyz}, max_range={max_range:.3f}", flush=True)
    
    # 1. Get Volume Data (SDF values at grid corners)
    # Re-evaluate since bounds may differ
    sdf_grid = _evaluate_grid_3d(sdf_graph, resolution, min_xyz, max_xyz)

    # 2. Identify Active Edges & Sign Changes
    # Edges where one end is inside (<0) and one is outside (>0)
    
    # Grid spacing per axis
    step_xyz = [(max_xyz[i] - min_xyz[i]) / (resolution - 1) for i in range(3)]
    
    # boolean grid: True if inside
    inside = sdf_grid < 0
    
    # Count inside/outside for debugging
    inside_count = np.sum(inside)
    total_count = inside.size
    print(f"    [shell_gen] Inside cells: {inside_count}/{total_count} ({100*inside_count/total_count:.1f}%)", flush=True)
    
    # Find crossings
    edges_x = np.logical_xor(inside[:-1, :, :], inside[1:, :, :])
    edges_y = np.logical_xor(inside[:, :-1, :], inside[:, 1:, :])
    edges_z = np.logical_xor(inside[:, :, :-1], inside[:, :, 1:])
    
    crossing_count = np.sum(edges_x) + np.sum(edges_y) + np.sum(edges_z)
    print(f"    [shell_gen] Edge crossings: {crossing_count}", flush=True)
    
    # 3. Place Vertices in Active Cells
    R = resolution
    cell_vertex_indices = np.full((R-1, R-1, R-1), -1, dtype=np.int32)
    vertices: List[List[float]] = []
    
    # Helper to get world pos of grid index (per-axis spacing)
    def get_pos(ix, iy, iz):
        return [
            min_xyz[0] + ix * step_xyz[0],
            min_xyz[1] + iy * step_xyz[1],
            min_xyz[2] + iz * step_xyz[2]
        ]

    # Pre-compute intersection points for all edges to speed up averaging?
    # Or just iterate cells. Iterating cells is O(N^3).
    
    # Let's iterate coordinates
    # Optimization: Use indices of active cells
    # A cell (i,j,k) uses:
    # X-edges: (i,j,k)..(i+1, j+1, k+1) variations
    # 12 edges total per cell.
    
    # 4 edges in X dir: (i, j, k), (i, j+1, k), (i, j, k+1), (i, j+1, k+1)
    # Similar for Y and Z.
    
    # Iterate all cells is simplest to implement first
    
    # We need to generate normals too. Use finite difference on SDF graph at vertex pos.
    
    count_verts = 0
    
    for i in range(R - 1):
        for j in range(R - 1):
            for k in range(R - 1):
                # Collect intersection points
                intersections = []
                
                # Check 12 edges
                # X-parallel edges
                for dy in [0, 1]:
                    for dz in [0, 1]:
                        if edges_x[i, j+dy, k+dz]:
                            # Interpolate
                            d0 = sdf_grid[i, j+dy, k+dz]
                            d1 = sdf_grid[i+1, j+dy, k+dz]
                            t = d0 / (d0 - d1)
                            p = get_pos(i + t, j+dy, k+dz)
                            intersections.append(p)

                # Y-parallel edges
                for dx in [0, 1]:
                    for dz in [0, 1]:
                        if edges_y[i+dx, j, k+dz]:
                            d0 = sdf_grid[i+dx, j, k+dz]
                            d1 = sdf_grid[i+dx, j+1, k+dz]
                            t = d0 / (d0 - d1)
                            p = get_pos(i+dx, j + t, k+dz)
                            intersections.append(p)

                # Z-parallel edges
                for dx in [0, 1]:
                    for dy in [0, 1]:
                        if edges_z[i+dx, j+dy, k]:
                            d0 = sdf_grid[i+dx, j+dy, k]
                            d1 = sdf_grid[i+dx, j+dy, k+1]
                            t = d0 / (d0 - d1)
                            p = get_pos(i+dx, j+dy, k + t)
                            intersections.append(p)
                            
                if intersections:
                    # Active cell!
                    # Average position
                    avg_p = np.mean(intersections, axis=0)
                    
                    # Compute normal
                    # (In a real engine, we'd batch this or evaluate on GPU)
                    p_tensor = torch.from_numpy(np.array([avg_p], dtype=np.float32))
                    normal = _compute_gradient(sdf_graph, p_tensor).numpy()[0]
                    
                    # Store
                    cell_vertex_indices[i, j, k] = count_verts
                    # Pos (3) + Normal (3)
                    vertices.append([avg_p[0], avg_p[1], avg_p[2], normal[0], normal[1], normal[2]])
                    count_verts += 1

    # 4. Generate Quads
    # For every active edge, output a quad connecting the 4 cells around it.
    indices = []
    
    # X-edges -> Quads in YZ plane
    # Edge (i, j, k)-(i+1, j, k) shared by cells:
    # (i, j-1, k-1), (i, j, k-1), (i, j-1, k), (i, j, k) -- Wait, indices need care.
    # Actually simpler:
    # An active edge connects Inside/Outside.
    # The quad connects the vertices of the 4 cells adjacent to the edge.
    # Cells are (i, j-1, k-1), (i, j-1, k), (i, j, k), (i, j, k-1) around x-edge? No.
    # Let's map edge (i,j,k) to the 4 cells around it.
    # X-edge at (i,j,k) [between grid node i and i+1].
    # Adjacent cells are those that share this edge segment.
    # The edge is along X. The cells are around it in YZ.
    # Cells: (i, j-1, k-1), (i, j, k-1), (i, j, k), (i, j-1, k) ?
    # Wait, the edge is a grid edge.
    # Grid edge connecting (i,j,k) and (i+1,j,k).
    # The cells that share this edge are:
    # 1. Cell (i, j-1, k-1) - Top-Right-Back corner is (i+1, j, k)? No.
    # Cell (i, j, k) starts at (i,j,k). So grid edge (i,j,k)-(i+1,j,k) is Bottom-Front-Left edge of Cell(i,j,k).
    # It is Top-Front-Left of Cell(i, j-1, k).
    # It is Bottom-Back-Left of Cell(i, j, k-1). 
    # It is Top-Back-Left of Cell(i, j-1, k-1).
    
    # So the 4 cells are:
    # C0 = (i, j-1, k-1)
    # C1 = (i, j-1, k)
    # C2 = (i, j, k)
    # C3 = (i, j, k-1)
    
    # We must ensure i,j,k are valid for these cells.
    # Also need to determine winding order based on sign of edge crossing.
    
    # NOTE: Since we only generated vertices for 0..R-2 cells, we must be careful with boundary edges.
    # We only generate quads for internal edges where all 4 cells exist.
    
    def add_quad(c0, c1, c2, c3, flip):
        i0 = cell_vertex_indices[c0]
        i1 = cell_vertex_indices[c1]
        i2 = cell_vertex_indices[c2]
        i3 = cell_vertex_indices[c3]
        
        if i0 == -1 or i1 == -1 or i2 == -1 or i3 == -1:
            return # Should not happen if edge is active? Actually, edge active implies cells active? 
                   # Yes, if edge crosses surface, then cells sharing it must have intersections (this specific edge).
        
        if flip:
            indices.extend([i0, i1, i3, i1, i2, i3]) # 0-1-2-3 quad -> 0-1-3, 1-2-3 ?
        else:
            indices.extend([i0, i3, i1, i3, i2, i1]) # Reversed
            
    # Process X-Edges
    # Range: needs neighbors j-1, k-1. So j start 1, k start 1.
    for i in range(R - 1):
        for j in range(1, R - 1):
            for k in range(1, R - 1):
                if edges_x[i, j, k]:
                    # Flip if transition is Inside -> Outside
                    # inside[i,j,k] is True means (i,j,k) is inside.
                    # if (i) is in, (i+1) is out -> Normal points +X
                    flip = inside[i, j, k] 
                    add_quad(
                        (i, j-1, k-1), # C0
                        (i, j-1, k),   # C1
                        (i, j, k),     # C2
                        (i, j, k-1),   # C3
                        flip
                    )

    # Process Y-Edges
    for i in range(1, R - 1):
        for j in range(R - 1):
            for k in range(1, R - 1):
                if edges_y[i, j, k]:
                    flip = inside[i, j, k]
                    # Cells around Y edge (i,j,k)-(i,j+1,k)
                    # Along Y. Neighbors in XZ.
                    # (i-1, j, k-1), (i, j, k-1), (i, j, k), (i-1, j, k)
                    add_quad(
                        (i-1, j, k-1),
                        (i, j, k-1),
                        (i, j, k),
                        (i-1, j, k),
                        flip
                    )

    # Process Z-Edges
    for i in range(1, R - 1):
        for j in range(1, R - 1):
            for k in range(R - 1):
                if edges_z[i, j, k]:
                    flip = inside[i, j, k]
                    # Cells around Z edge
                    # (i-1, j-1, k), (i-1, j, k), (i, j, k), (i, j-1, k)
                    add_quad(
                        (i-1, j-1, k),
                        (i-1, j, k),
                        (i, j, k),
                        (i, j-1, k),
                        flip
                    )
                    
    # Pack Data
    # NumVerts (u32), Verts... NumIndices (u32), Indices...
    
    # Flatten vertices
    verts_flat = [val for v in vertices for val in v]
    
    header_struct = struct.Struct('I') # u32
    # Verts: 6 floats each
    
    output = bytearray()
    output.extend(header_struct.pack(len(vertices)))
    output.extend(struct.pack(f'{len(verts_flat)}f', *verts_flat))
    
    output.extend(header_struct.pack(len(indices)))
    output.extend(struct.pack(f'{len(indices)}I', *indices))
    
    return bytes(output)

def _evaluate_grid(sdf_graph: nn.Module, resolution: int, bounds: Tuple[float, float]) -> np.ndarray:
    """Legacy single-range bounds evaluation."""
    min_val, max_val = bounds
    return _evaluate_grid_3d(sdf_graph, resolution, 
                             [min_val, min_val, min_val], 
                             [max_val, max_val, max_val])


def _evaluate_grid_3d(sdf_graph: nn.Module, resolution: int, 
                      min_xyz: List[float], max_xyz: List[float]) -> np.ndarray:
    """Evaluate SDF on a 3D grid with per-axis bounds."""
    x = torch.linspace(min_xyz[0], max_xyz[0], resolution)
    y = torch.linspace(min_xyz[1], max_xyz[1], resolution)
    z = torch.linspace(min_xyz[2], max_xyz[2], resolution) 
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
    
    with torch.no_grad():
        dists = sdf_graph(points)
        
    return dists.reshape(resolution, resolution, resolution).cpu().numpy()

def _compute_gradient(sdf_graph: nn.Module, p: torch.Tensor) -> torch.Tensor:
    # simple central difference
    epsilon = 0.001
    dx = torch.tensor([epsilon, 0.0, 0.0])
    dy = torch.tensor([0.0, epsilon, 0.0])
    dz = torch.tensor([0.0, 0.0, epsilon])
    
    with torch.no_grad():
        # Batch? p is [1, 3] usually here
        val_x1 = sdf_graph(p + dx)
        val_x2 = sdf_graph(p - dx)
        val_y1 = sdf_graph(p + dy)
        val_y2 = sdf_graph(p - dy)
        val_z1 = sdf_graph(p + dz)
        val_z2 = sdf_graph(p - dz)
        
        grad = torch.stack([
            val_x1 - val_x2,
            val_y1 - val_y2,
            val_z1 - val_z2
        ], dim=1) / (2.0 * epsilon)
        
        return torch.nn.functional.normalize(grad, dim=1)
