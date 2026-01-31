# Forge Editor: World Editor

**Purpose:** Level design, terrain sculpting, real-world data import, and scene composition using the SDF-based world system.

**Related Docs:**
- [AI Pipeline](../workflows/ai-pipeline.md) - Landscape Track (Track C)
- [Compiler Pipeline](../workflows/compiler-pipeline.md) - World baking

---

## **World Structure**

### Map Bricks (Chunks)

The world is divided into 8m × 8m × 8m streaming chunks:

```
┌────┬────┬────┬────┐
│ A1 │ A2 │ A3 │ A4 │  ← Row A
├────┼────┼────┼────┤
│ B1 │ B2 │ B3 │ B4 │  ← Row B
├────┼────┼────┼────┤
│ C1 │ C2 │ C3 │ C4 │  ← Row C
└────┴────┴────┴────┘
        ↑
    8m × 8m per chunk
```

**Chunk States:**
- ⬜ **Unprocessed** - Raw data, not compiled
- 🟨 **Analyzing** - AI processing terrain/props
- 🟦 **Baking** - Generating splats
- 🟩 **Finalized** - Ready for runtime

---

## **Terrain System**

### SDF Brushes

Unlike mesh terrain, brushes modify SDF density values:

```
┌─────────────────────────────────────────┐
│ Terrain Brushes                         │
├─────────────────────────────────────────┤
│ [Raise] [Lower] [Smooth] [Flatten]      │
│ [Noise] [Erode] [Paint]                 │
│                                         │
│ Brush Size: [====·····] 4.0m            │
│ Strength:   [===······] 0.5             │
│ Falloff:    [Smooth ▼]                  │
└─────────────────────────────────────────┘
```

**Brush Operations:**
- **Raise/Lower:** Add/subtract from SDF field
- **Smooth:** Gaussian blur on SDF values
- **Flatten:** Target height plane
- **Noise:** Apply Perlin/Voronoi displacement
- **Erode:** Simulate hydraulic erosion
- **Paint:** Change material ID (not geometry)

---

### Material Painting

```
┌─────────────────────────────────────────┐
│ Terrain Materials                       │
├─────────────────────────────────────────┤
│ ● Grass      [Preview]                  │
│ ○ Dirt                                  │
│ ○ Rock                                  │
│ ○ Sand                                  │
│ ○ Snow                                  │
│                                         │
│ Blend: [Hard ▼]  Opacity: [====·] 0.8   │
└─────────────────────────────────────────┘
```

Materials stored as voxel Material_ID, not textures. Splat colors derived from material at bake time.

---

## **Geo-Importer (Real-World Data)**

Import real geographic data to create levels based on actual locations.

### UI

```
┌─────────────────────────────────────────────────────────────┐
│ Import Real-World Location                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Location:                                                   │
│  [Search: Times Square, NYC          ] [🔍]                │
│                                                             │
│ Coordinates:                                                │
│  Lat: [40.7580]  Lon: [-73.9855]                           │
│  Radius: [0.5] km                                           │
│                                                             │
│ ─────────────────────────────────────────────────────────── │
│ Data Sources:                                               │
│                                                             │
│  ☑ Terrain                                                  │
│     Source: [USGS 3DEP ▼] (1m resolution, US only)         │
│     Fallback: SRTM 30m (global)                            │
│                                                             │
│  ☑ Buildings                                                │
│     Source: [OpenStreetMap ▼] (free, ODbL license)         │
│     ☑ Auto-extrude missing heights (3m/floor)              │
│     ☑ Infer materials from tags                            │
│                                                             │
│  ☐ Photorealistic 3D (Google)                              │
│     ⚠ Commercial license, pay-per-tile                     │
│                                                             │
│  ☑ Roads & Infrastructure                                  │
│     Source: OpenStreetMap                                   │
│                                                             │
│  ☐ Vegetation                                               │
│     Source: OSM + Procedural generation                     │
│                                                             │
│ ─────────────────────────────────────────────────────────── │
│ Estimated:                                                  │
│  Chunks: 16       Buildings: ~240      Size: ~45 MB        │
│                                                             │
│ [Preview Map] [Import] [Cancel]                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Data Sources

#### USGS 3DEP (Terrain - US)
- **Resolution:** 1m (where available), 10m, 30m
- **Format:** LiDAR point clouds, DEMs
- **License:** Public domain
- **API:** OpenTopography, AWS S3

```python
def import_usgs_terrain(lat, lon, radius_km):
    # Fetch DEM from USGS
    dem = usgs_api.get_dem(lat, lon, radius_km, resolution="1m")
    
    # Convert to SDF voxel grid
    terrain_sdf = voxelize_heightmap(dem, voxel_size=0.5)
    
    # Apply default terrain material
    terrain_sdf.material_id = TERRAIN_GRASS
    
    return terrain_sdf
```

#### OpenStreetMap (Buildings, Roads)
- **Coverage:** Global, ~60% of buildings
- **Data:** Footprints, heights (partial), building types
- **License:** ODbL (attribution required)
- **API:** Overpass API, OSM files

```python
def import_osm_buildings(lat, lon, radius_km):
    buildings = osm_api.query(f"""
        [out:json];
        way["building"](around:{radius_km * 1000},{lat},{lon});
        out body;
    """)
    
    for building in buildings:
        footprint = building.polygon
        
        # Get height or estimate
        height = building.tags.get('height') or \
                 building.tags.get('building:levels', 3) * 3.0
        
        # Extrude to SDF box
        building_sdf = extrude_polygon(
            footprint,
            height=height,
            base_z=terrain.get_elevation(footprint.center)
        )
        
        # Assign material from building type
        building_type = building.tags.get('building', 'yes')
        building_sdf.material = infer_material(building_type)
        
        world.add(building_sdf)
```

#### Google Maps (Premium)
- **Data:** Photorealistic 3D tiles, textured meshes
- **Coverage:** Thousands of cities
- **License:** Commercial, pay-per-use
- **Format:** OGC 3D Tiles (glTF)

```python
def import_google_3d_tiles(lat, lon, radius_km):
    # Requires API key + billing
    tiles = google_api.get_photorealistic_tiles(
        lat, lon, radius_km,
        lod=2  # Level of detail
    )
    
    # Convert mesh to SDF (Dual Contouring)
    for tile in tiles:
        mesh = tile.gltf_mesh
        sdf = mesh_to_sdf(mesh)
        
        # Extract textures for splat colors
        colors = extract_vertex_colors(mesh)
        
        world.add(sdf, colors)
```

---

### Material Inference

Auto-assign materials from OSM building tags:

```python
BUILDING_MATERIALS = {
    'residential': 'BRICK',
    'commercial': 'CONCRETE',
    'industrial': 'METAL_STEEL',
    'retail': 'GLASS',
    'church': 'LIMESTONE',
    'warehouse': 'METAL_CORRUGATED',
    'garage': 'CONCRETE',
    'yes': 'CONCRETE',  # Default
}

def infer_material(building_type: str) -> Material:
    mat_name = BUILDING_MATERIALS.get(building_type, 'CONCRETE')
    return material_library.get(mat_name)
```

---

## **Scene Composition**

### Entity Placement

```
┌─────────────────────────────────────────┐
│ Entity Library                          │
├─────────────────────────────────────────┤
│ Filter: [Props ▼]                       │
│                                         │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        │
│ │Tree │ │Crate│ │Car  │ │Lamp │        │
│ └─────┘ └─────┘ └─────┘ └─────┘        │
│                                         │
│ Drag to viewport to place              │
└─────────────────────────────────────────┘
```

**Placement Modes:**
- **Single:** Click to place one
- **Scatter:** Brush-based random placement
- **Array:** Grid/radial patterns
- **Surface:** Snap to terrain

---

### Chunk Blending

1m "halo" margins between chunks for seamless streaming:

```
    Chunk A          Blend Zone         Chunk B
┌──────────────┬───────────────────┬──────────────┐
│              │◄── 1m margin ───►│              │
│   Content    │   SmoothUnion     │   Content    │
│              │   Splat Dither    │              │
└──────────────┴───────────────────┴──────────────┘
```

---

### Artist Pass

AI-assisted polish for terrain and props:

```
┌─────────────────────────────────────────┐
│ Artist Pass (AI Polish)                 │
├─────────────────────────────────────────┤
│ Selected: Chunks B2, B3, C2, C3         │
│                                         │
│ Enhancements:                           │
│  ☑ Add erosion details                  │
│  ☑ Scatter foliage                      │
│  ☑ Add terrain variation                │
│  ☐ Weather effects (rust, moss)         │
│                                         │
│ Style Reference: [Upload Image]         │
│                                         │
│ [Preview] [Apply to Selection]          │
└─────────────────────────────────────────┘
```

---

## **Workflow Example**

### Creating NYC Block

1. **Import Location**
   - Search: "Times Square, NYC"
   - Radius: 0.5 km
   - Sources: USGS terrain, OSM buildings

2. **Auto-Process**
   - Terrain voxelized (1m resolution)
   - 240 buildings extruded from OSM
   - Materials auto-assigned (commercial → concrete)

3. **Manual Polish**
   - Smooth terrain edges
   - Add street furniture (props)
   - Paint road materials

4. **Artist Pass**
   - AI adds window details
   - Scatter debris/signage
   - Apply wear patterns

5. **Bake**
   - Generate splats for all chunks
   - Export to runtime format

**Total Time:** ~15 minutes for 0.5 km² city block

---

## **Data Source Comparison**

| Source | Terrain | Buildings | Textures | License | Cost |
|--------|---------|-----------|----------|---------|------|
| **USGS 3DEP** | ★★★★★ | ☆☆☆☆☆ | ☆☆☆☆☆ | Public Domain | Free |
| **OpenStreetMap** | ★★☆☆☆ | ★★★★☆ | ☆☆☆☆☆ | ODbL | Free |
| **Google Maps** | ★★★★☆ | ★★★★★ | ★★★★★ | Commercial | $$$ |
| **Combined** | ★★★★★ | ★★★★☆ | ★★☆☆☆ | Mixed | $ |

**Recommendation:** USGS + OSM for free tier, add Google for hero locations.

---

**Version:** 1.0  
**Last Updated:** January 25, 2026  
**Related:** [AI Pipeline](../workflows/ai-pipeline.md) | [Libraries](./forge-libraries.md)
