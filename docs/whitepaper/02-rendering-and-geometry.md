# 02 – Rendering and Geometry

**Source:** [System Overview](../architecture/overview.md) §3, [Rendering Pipeline](../core-systems/rendering-pipeline.md), [Splat Strategy](../improvements/splat-strategy.md)

---

## Shell and Volume Pipeline

To reach 60 FPS on mobile and mid-range hardware, the engine uses a hybrid pipeline that trades VRAM for reduced compute: proxy meshes bound raymarching, and LOD switches between math evaluation, baked textures, and splat-only rendering.

**Pass order:**

1. **Pass 1: Shell rasterization (early Z)** — A low-poly proxy mesh (from Dual Contouring in the Architect) is rasterized to populate the depth buffer. This culls pixels behind or inside objects and provides a start depth for the raymarcher, reducing steps.
2. **Pass 2: Raymarcher** — SDF evaluation with distance-based LOD (see below).
3. **Pass 3: Gaussian splat sort** — Tile-based sort and draw of splats; occluded splats are culled.

---

## SDF Geometry and LOD

Geometry is defined by Signed Distance Fields. The raymarcher selects LOD from camera distance:

| LOD | Range | Method | Use |
|-----|-------|--------|-----|
| **LOD 0** | Close | Raw math tree | Infinite resolution, crisp booleans and curves |
| **LOD 1** | Mid | Baked 3D texture | Trilinear sampling, much faster than algebraic evaluation |
| **LOD 2** | Far | Splats only | No raymarching; object treated as point cloud |

So: close-up detail stays mathematically exact; mid-range uses pre-baked volume; far range uses only splats for low cost.

---

## Surface Strategy

Traditional UV maps are replaced by **triplanar mapping**: texture is projected from three axes (X, Y, Z) and blended by surface normal. At compile time the projection is evaluated per splat and stored as a u32; runtime does not perform triplanar lookups, so surface cost is low.

Gaussian splats are the primary surface representation. They are pre-trained (densified/pruned) in the Architect so they align to the SDF surface, reducing runtime density and artifacts. For size and performance metrics (e.g. 5–29× smaller than traditional textures, ~2× faster pixel cost), see [Splat Strategy](../improvements/splat-strategy.md).

---

## Terrain Pipeline

Terrain is a global voxel volume (e.g. 2048×2048×256), from LiDAR or procedural generation. Gameplay edits (e.g. explosions) apply mathematical operations such as `Subtract(Sphere)`; a compute shader stamps the change into the volume texture. The renderer raymarches the updated volume. Because edits are baked into the data, render cost is **O(1)** with respect to the number of modifications—identical for flat or heavily cratered terrain.

---

[Back to Whitepaper](README.md) | Prev: [01 – Architecture](01-architecture.md) | Next: [03 – Compiler and Data](03-compiler-and-data.md)
