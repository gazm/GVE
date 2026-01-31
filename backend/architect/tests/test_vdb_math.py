import meshlib.mrmeshpy as mrmesh
import numpy as np

print("Testing FunctionVolume...")

try:
    print("Attempting instantiation...")
    # Signature: FunctionVolume(data, dims, voxelSize)
    
    dims = mrmesh.Vector3i(10,10,10)
    v_size = mrmesh.Vector3f(0.1, 0.1, 0.1)
    
    # Callback uses Vector3i
    def sdf_callback(vi):
        # Convert voxel index to world space
        # We simulate a sphere at center (0.5, 0.5, 0.5)
        wx = vi.x * 0.1
        wy = vi.y * 0.1
        wz = vi.z * 0.1
        
        dist = np.sqrt((wx-0.5)**2 + (wy-0.5)**2 + (wz-0.5)**2) - 0.2
        return float(dist)

    print("Attempting FunctionVolume with correct signature...")
    
    # We remove 'origin' from constructor
    fv = mrmesh.FunctionVolume(data=sdf_callback, dims=dims, voxelSize=v_size)
    print("Success Instantiation")
    
    print("Converting to VDB...")
    vdb = mrmesh.functionVolumeToVdbVolume(fv)
    print("Success Conversion")
    
    if vdb.valid():
        print(f"VDB Valid! Active Voxels: {vdb.activeVoxelCount()}")
    
except TypeError as e:
    print(f"TypeError: {e}")
    with open("error_log.txt", "w") as f:
        f.write(str(e))
except Exception as e:
    print(f"Error: {e}")
