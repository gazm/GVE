import meshlib.mrmeshpy as mrmesh
import numpy as np
import io
import struct

def repair_and_decimate(
    grid,
    iso_value: float = 0.0,
    target_tris: int = 1000,
    voxel_size: float | None = None,
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
    
    # 2. Grid to Mesh
    # MeshLib: gridToMesh(grid: FloatGrid, settings: GridToMeshSettings) -> Mesh
    grid_settings = mrmesh.GridToMeshSettings()
    grid_settings.isoValue = iso_value
    mesh = mrmesh.gridToMesh(float_grid, grid_settings)
    
    # 3. Repair
    # fixSelfIntersections requires voxelSize parameter (scalar float)
    safe_voxel_size = voxel_size if voxel_size and voxel_size > 0 else 0.01
    mrmesh.fixSelfIntersections(mesh, float(safe_voxel_size))
    
    # Fill holes trivially (simple triangulation)
    # Get all hole edges and fill them
    hole_edges = mesh.topology.findHoleRepresentiveEdges()
    if hole_edges:
        for edge in hole_edges:
            mrmesh.fillHoleTrivially(mesh, edge)
    
    # 4. Decimate
    # Calculate how many vertices to delete to reach target
    current_vert_count = mesh.points.size()
    if current_vert_count > target_tris:
        decimate_settings = mrmesh.DecimateSettings()
        decimate_settings.maxDeletedVertices = current_vert_count - target_tris
        decimate_settings.maxError = 0.01
        mrmesh.decimateMesh(mesh, decimate_settings)
    
    # 5. Extract Data for Binary (Vertices/Indices)
    # Compute per-vertex normals (MeshLib returns a VertCoords vector)
    normals = mrmesh.computePerVertNormals(mesh)
    
    return _pack_shell_data(mesh, normals)

def _pack_shell_data(mesh: mrmesh.Mesh, normals: mrmesh.VertCoords) -> bytes:
    """
    Pack MeshLib mesh into GVE Shell Format.
    Format:
      vertex_count: u32
      vertices: [pos(3f), normal(3f)] * count
      index_count: u32
      indices: [u32] * count
    """
    # Extract vertices and normals
    # mrmeshpy uses separate buffers.
    # We need to iterate or use buffer access.
    
    # To handle potential large arrays, we should efficiently access.
    # assuming mesh.points and mesh.normals are accessible as vectors.
    
    # NOTE: mrmeshpy API might require explicit iteration if no memory view.
    # For generated shells (low poly < 1000 tris), iteration is fine.
    
    # mesh.topology.vertMap is BitSet of valid verts (decimation leaves holes in ID space).
    # The Python bindings here don't expose packMesh, so we write all points and
    # rely on topology to keep faces valid.
    
    vertex_count = mesh.points.size()
    
    # Vertices
    # Access: mesh.points.vec[i] or mesh.points[i]
    # Normals: mesh.normals[i]
    
    # We write to BytesIO
    buf = io.BytesIO()
    
    # Vertex Count
    buf.write(struct.pack("<I", vertex_count))
    
    for i in range(vertex_count):
        p = mesh.points[mrmesh.VertId(i)]
        n = normals[mrmesh.VertId(i)] if normals.size() > i else mrmesh.Vector3f(0, 1, 0)
        
        buf.write(struct.pack("<ffffff", p.x, p.y, p.z, n.x, n.y, n.z))
        
    # Indices
    # MeshLib provides a compact list of valid triangles
    tri_verts = mesh.topology.getAllTriVerts()
    faces_count = len(tri_verts)
    index_count = faces_count * 3
    
    buf.write(struct.pack("<I", index_count))
    
    for i in range(faces_count):
        tri = tri_verts[i]
        buf.write(struct.pack("<III", tri[0].get(), tri[1].get(), tri[2].get()))
        
    return buf.getvalue()
