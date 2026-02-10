import meshlib.mrmeshpy as mrmesh
import numpy as np
import io
import struct

def repair_and_decimate(
    grid,
    iso_value: float = 0.0,
    target_tris: int = 1000,
    voxel_size: float | None = None,
    bounds_min: tuple[float, float, float] | None = None,
) -> bytes:
    """
    Extracts mesh from MeshLib VdbVolume or FloatGrid, repairs it, and returns binary mesh data (shell format).
    
    Args:
        grid: Either mrmesh.VdbVolume or mrmesh.FloatGrid
        iso_value: Iso-surface value for marching cubes (default 0.0)
        target_tris: Target triangle count for decimation (default 1000)
    """
    # 1. Extract FloatGrid from VdbVolume if needed
    # VdbVolume.data contains the actual FloatGrid
    float_grid = grid.data if isinstance(grid, mrmesh.VdbVolume) else grid
    print(f"    [mesh_repair] 🔍 Grid type: {type(grid).__name__}, float_grid type: {type(float_grid).__name__}", flush=True)
    
    # 2. Grid to Mesh
    # MeshLib: gridToMesh(grid: FloatGrid, settings: GridToMeshSettings) -> Mesh
    grid_settings = mrmesh.GridToMeshSettings()
    grid_settings.isoValue = iso_value
    # CRITICAL: Set voxelSize! Default is (0,0,0) which makes all vertices at origin
    if voxel_size is not None:
        grid_settings.voxelSize = mrmesh.Vector3f(voxel_size, voxel_size, voxel_size)
    mesh = mrmesh.gridToMesh(float_grid, grid_settings)
    
    initial_verts = mesh.points.size()
    initial_tris = len(mesh.topology.getAllTriVerts())
    print(f"    [mesh_repair] After gridToMesh: {initial_verts} verts, {initial_tris} tris", flush=True)
    
    
    if initial_verts == 0:
        print(f"    [mesh_repair] ⚠️ Empty mesh from gridToMesh! Returning minimal shell.", flush=True)
        return _pack_shell_data(mesh, mrmesh.VertCoords(), bounds_min)
    
    # 3. Skip self-intersection repair - it's too slow (O(n²)) and too aggressive
    # The marching cubes output is generally clean enough for rendering
    print(f"    [mesh_repair] Skipping self-intersection repair (too slow for {initial_tris} tris)", flush=True)
    
    # Fill holes trivially (simple triangulation) - this is fast
    hole_edges = mesh.topology.findHoleRepresentiveEdges()
    if hole_edges:
        print(f"    [mesh_repair] Filling {len(hole_edges)} holes...", flush=True)
        for edge in hole_edges:
            mrmesh.fillHoleTrivially(mesh, edge)
    
    post_repair_verts = mesh.points.size()
    post_repair_tris = len(mesh.topology.getAllTriVerts())
    print(f"    [mesh_repair] After hole fill: {post_repair_verts} verts, {post_repair_tris} tris", flush=True)
    
    # 4. Decimate
    # Calculate how many vertices to delete to reach target
    current_vert_count = mesh.points.size()
    if current_vert_count > target_tris:
        decimate_settings = mrmesh.DecimateSettings()
        decimate_settings.maxDeletedVertices = current_vert_count - target_tris
        decimate_settings.maxError = 0.01
        mrmesh.decimateMesh(mesh, decimate_settings)
        
        post_decimate_verts = mesh.points.size()
        post_decimate_tris = len(mesh.topology.getAllTriVerts())
        print(f"    [mesh_repair] After decimate: {post_decimate_verts} verts, {post_decimate_tris} tris", flush=True)
    
    # 5. Extract Data for Binary (Vertices/Indices)
    # Compute per-vertex normals (MeshLib returns a VertCoords vector)
    normals = mrmesh.computePerVertNormals(mesh)
    
    return _pack_shell_data(mesh, normals, bounds_min)

def _pack_shell_data(mesh: mrmesh.Mesh, normals: mrmesh.VertCoords, bounds_min: tuple[float, float, float] | None = None) -> bytes:
    """
    Pack MeshLib mesh into GVE Shell Format.
    Format:
      vertex_count: u32
      vertices: [pos(3f), normal(3f)] * count
      index_count: u32
      indices: [u32] * count
      
    Note: After decimation, MeshLib mesh has holes in vertex ID space.
    We must pack only VALID vertices and remap indices accordingly.
    
    Also: MeshLib gridToMesh outputs vertices at grid-local coords (0,0,0 origin).
    We must transform to world space by adding bounds_min offset.
    """
    # Get valid vertex IDs (decimation leaves holes in the VertId space)
    valid_verts = mesh.topology.getValidVerts()
    
    # World-space offset (gridToMesh uses (0,0,0) as origin)
    offset_x = bounds_min[0] if bounds_min else 0.0
    offset_y = bounds_min[1] if bounds_min else 0.0
    offset_z = bounds_min[2] if bounds_min else 0.0
    
    # Build old_id -> new_id mapping and collect valid vertex data
    old_to_new = {}
    vertex_data = []
    new_idx = 0
    
    # Iterate through all vertex IDs and collect only valid ones
    for old_id in range(mesh.points.size()):
        vid = mrmesh.VertId(old_id)
        if valid_verts.test(vid):
            p = mesh.points[vid]
            n = normals[vid] if normals.size() > old_id else mrmesh.Vector3f(0, 1, 0)
            
            # Transform to world space
            world_x = p.x + offset_x
            world_y = p.y + offset_y
            world_z = p.z + offset_z
            
            vertex_data.append((world_x, world_y, world_z, n.x, n.y, n.z))
            old_to_new[old_id] = new_idx
            new_idx += 1
    
    packed_vertex_count = len(vertex_data)
    print(f"    [mesh_repair] Packing {packed_vertex_count} valid vertices (from {mesh.points.size()} total IDs)", flush=True)
    
    # Write to BytesIO
    buf = io.BytesIO()
    
    # Vertex Count
    buf.write(struct.pack("<I", packed_vertex_count))
    
    # Write vertex data
    for vx, vy, vz, nx, ny, nz in vertex_data:
        buf.write(struct.pack("<ffffff", vx, vy, vz, nx, ny, nz))
    
    # Indices - remap to new packed IDs
    tri_verts = mesh.topology.getAllTriVerts()
    faces_count = len(tri_verts)
    index_count = faces_count * 3
    
    buf.write(struct.pack("<I", index_count))
    
    for i in range(faces_count):
        tri = tri_verts[i]
        # Remap old vertex IDs to new packed IDs
        new_i0 = old_to_new.get(tri[0].get(), 0)
        new_i1 = old_to_new.get(tri[1].get(), 0)
        new_i2 = old_to_new.get(tri[2].get(), 0)
        buf.write(struct.pack("<III", new_i0, new_i1, new_i2))
    
    return buf.getvalue()
