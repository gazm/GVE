# Forge Editor: Component Libraries

**Purpose:** Browse, manage, and reuse components across all assets. Five interconnected libraries power the card-chain workflow.

**Related Docs:**
- [Card-Chain Workflow](./forge-card-chain.md) - How libraries integrate with asset creation
- [Material Database](../data/material-database.md) - Physical material specifications
- [Texture Implementation](./texture-library-implementation.md) - Backend API details

---

## **The Five Libraries**

```
┌─────────────────────────────────────────────────────────────┐
│                    Component Libraries                       │
├─────────────┬─────────────┬─────────────┬─────────────┬─────┤
│  Geometry   │  Materials  │  Textures   │   Audio     │Recipes│
│  (SDF/CSG)  │  (Physics)  │  (PBR Maps) │(DSP Patches)│(Templates)│
└─────────────┴─────────────┴─────────────┴─────────────┴─────┘
       ↓              ↓             ↓            ↓          ↓
   Card-Chain Assembly → Combined into Final Assets
```

---

## **1. Geometry Library**

**Purpose:** Reusable 3D shapes as SDF trees (CSG components)

### Structure

```json
{
  "id": "geo_ak_receiver_001",
  "name": "AK Receiver Pattern",
  "type": "geometry",
  "tags": ["weapon", "rifle", "receiver", "ak", "military"],
  
  "sdf_tree": {
    "root_node": {
      "type": "operation",
      "op": "subtract",
      "children": [
        {"type": "primitive", "shape": "box", "params": {"size": [0.4, 0.15, 0.08]}},
        {"type": "primitive", "shape": "cylinder", "params": {"radius": 0.008, "height": 0.4}}
      ]
    }
  },
  
  "attachment_points": {
    "stock_mount": {"position": [-0.2, 0, 0], "normal": [-1, 0, 0]},
    "top_rail": {"position": [0, 0.075, 0], "normal": [0, 1, 0]},
    "magazine_well": {"position": [0.05, -0.075, 0], "normal": [0, -1, 0]}
  },
  
  "material_zones": ["receiver_body", "dust_cover", "rail_mount"],
  
  "metadata": {
    "source": "ai-generated",
    "created": "2026-01-25",
    "usage_count": 47,
    "rating": 4.5
  }
}
```

### Categories

| Category | Examples | Typical Count |
|----------|----------|---------------|
| **Weapons** | Receivers, stocks, barrels, grips | 50-100 per type |
| **Furniture** | Legs, frames, cushions, backs | 30-50 per style |
| **Vehicles** | Chassis, wheels, doors, engines | 100+ per vehicle class |
| **Architecture** | Walls, doors, windows, stairs | 200+ modular pieces |
| **Props** | Crates, barrels, signs, debris | Unlimited |

### UI Browser

```
┌─────────────────────────────────────────────────────────────┐
│ Geometry Library                              [+ New] [↑ Upload]│
├─────────────────────────────────────────────────────────────┤
│ Filter: ☑ Weapons  ☐ Furniture  ☐ Vehicles  ☐ Architecture │
│ Tags: [receiver] [rifle] [x]                   [Clear All]  │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│ │ ◇◇◇◇◇◇◇ │ │ ◇◇◇◇◇◇◇ │ │ ◇◇◇◇◇◇◇ │ │ ◇◇◇◇◇◇◇ │            │
│ │ [3D]    │ │ [3D]    │ │ [3D]    │ │ [3D]    │            │
│ │ preview │ │ preview │ │ preview │ │ preview │            │
│ │ ◇◇◇◇◇◇◇ │ │ ◇◇◇◇◇◇◇ │ │ ◇◇◇◇◇◇◇ │ │ ◇◇◇◇◇◇◇ │            │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘            │
│  AK Receiver AR15 Upper  M4 Stock   Pistol Grip            │
│  ★★★★½      ★★★★☆       ★★★★★      ★★★☆☆                 │
│  Used: 47   Used: 32    Used: 89   Used: 15                │
└─────────────────────────────────────────────────────────────┘
```

---

## **2. Materials Library**

**Purpose:** Physical specifications for realistic physics and audio synthesis

### Integration with Material Database

Materials link to verified ASTM/AMS specifications in `material-database.md`:

```json
{
  "id": "mat_steel_worn",
  "name": "Steel (Battle-Worn)",
  "base_spec": "ASTM_A36",  // Links to material-database.md
  
  "overrides": {
    "roughness": 0.6,  // Higher than spec default (0.4)
    "edge_wear": 0.7,
    "cavity_grime": 0.5
  },
  
  "audio_profile": {
    "damping_modifier": 1.2,  // 20% more damping than pristine
    "resonance_shift": -50    // Hz lower pitch
  }
}
```

### Preset Materials

**Metals:**
- Pristine Steel, Battle-Worn Steel, Rusted Steel
- Clean Aluminum, Anodized Aluminum, Scratched Aluminum
- Polished Brass, Aged Brass, Tarnished Copper

**Wood:**
- Oak (Natural, Worn, Weathered)
- Pine (Fresh, Aged)
- Walnut (Polished, Distressed)

**Stone/Concrete:**
- Clean Concrete, Cracked Concrete
- Limestone, Marble (Polished, Aged)

**Plastics:**
- ABS (Black, Gray, Colored)
- Polycarbonate, Carbon Fiber

---

## **3. Textures Library**

**Purpose:** PBR texture maps for visual detail

### Structure

```json
{
  "id": "tex_rusty_steel_001",
  "name": "Rusty Steel - Heavy Wear",
  "tags": ["metal", "rust", "worn", "realistic"],
  
  "maps": {
    "albedo": "storage/tex_001_albedo.bc7",      // 2048x2048
    "normal": "storage/tex_001_normal.bc5",
    "roughness": "storage/tex_001_rough.bc4",
    "metallic": "storage/tex_001_metal.bc4"
  },
  
  "properties": {
    "resolution": 2048,
    "triplanar_scale": 1.5,
    "color_mode": "oklab"  // For dynamic runtime effects
  },
  
  "metadata": {
    "source": "ai-generated",
    "usage_count": 47,
    "rating": 4
  }
}
```

### Adding Textures

**Three Methods:**

1. **Upload PBR Set:**
   - Drag-drop albedo, normal, roughness, metallic PNGs
   - Auto-compress to BC7/BC5/BC4
   - Auto-generate tags via vision AI

2. **AI Generate:**
   - Prompt: "Battle-damaged steel with bullet impacts"
   - Cost: ~$0.02, Time: ~5s
   - Outputs seamless tileable PBR set

3. **Extract from Asset:**
   - Right-click geometry in viewport
   - "Extract Material to Library"
   - Saves baked splat colors as texture

---

## **4. Audio Library**

**Purpose:** DSP patches for physics-driven sound synthesis

### Structure

```json
{
  "id": "aud_steel_impact_001",
  "name": "Steel Impact - Heavy",
  "tags": ["metal", "impact", "heavy", "resonant"],
  
  "patch": {
    "oscillators": [
      {"waveform": "sine", "frequency": 440, "amplitude": 1.0},
      {"waveform": "sine", "frequency": 880, "amplitude": 0.5, "modulator": 0}
    ],
    "envelope": {
      "attack_ms": 1,
      "decay_ms": 500,
      "sustain": 0.0,
      "release_ms": 200
    },
    "dsp_chain": [
      {"type": "lowpass", "cutoff": 2000},
      {"type": "reverb", "room_size": 0.3}
    ]
  },
  
  "physics_mapping": {
    "velocity_to_amplitude": "linear",
    "mass_to_pitch": "inverse_sqrt"
  }
}
```

### Preset Categories

- **Impacts:** Metal, wood, glass, stone, plastic
- **Scrapes:** Friction sounds by material pair
- **Explosions:** Debris scatter, shockwave
- **Ambient:** Wind, rain (with material interactions)

---

## **5. Recipe Library**

**Purpose:** Complete card-chain templates for instant asset creation

### Structure

```json
{
  "id": "recipe_ak47_worn",
  "name": "Standard AK-47 (Worn)",
  "tags": ["weapon", "rifle", "ak", "military", "worn"],
  
  "cards": [
    {
      "type": "geometry",
      "component": "receiver",
      "source": "library",
      "library_id": "geo_ak_receiver_001"
    },
    {
      "type": "material",
      "target": "receiver",
      "material_id": "mat_steel_worn",
      "texture_id": "tex_rusty_steel_001"
    },
    {
      "type": "geometry",
      "component": "stock",
      "source": "library",
      "library_id": "geo_ak_wood_stock"
    },
    {
      "type": "material",
      "target": "stock",
      "material_id": "mat_oak_worn",
      "texture_id": "tex_worn_oak_003"
    }
  ],
  
  "metadata": {
    "total_cost": 0,      // All from library
    "total_time": 0,      // Instant
    "usage_count": 47,
    "rating": 4.8,
    "created_by": "user_123"
  }
}
```

### Recipe Browser

```
┌─────────────────────────────────────────────────────────────┐
│ Recipe Library                           [+ New] [Import]   │
├─────────────────────────────────────────────────────────────┤
│ Filter: ☑ Weapons  ☐ Vehicles  ☐ Furniture  ☐ Characters   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Standard AK-47 (Worn)                      ★★★★★        │ │
│ │ 4 cards  •  $0  •  instant                              │ │
│ │ Tags: weapon, rifle, ak, military                       │ │
│ │ Used 47 times by 12 users                               │ │
│ │                                                         │ │
│ │ [Use Recipe] [Clone & Edit] [Preview]                   │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Sci-Fi Laser Rifle                         ★★★★☆        │ │
│ │ 6 cards  •  $0.08  •  12s (2 AI generated)              │ │
│ │ Tags: weapon, sci-fi, energy                            │ │
│ │ Used 12 times                                           │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## **Tag System**

### Categories

```yaml
asset_type:
  - weapon, vehicle, furniture, prop, architecture, character

material_class:
  - metal, wood, stone, plastic, fabric, glass, organic

condition:
  - pristine, clean, worn, damaged, rusted, weathered

style:
  - realistic, stylized, sci-fi, fantasy, military, industrial

era:
  - modern, futuristic, historical, medieval, post-apocalyptic
```

### Smart Suggestions

When user enters prompt, system suggests matching library items:

```python
def suggest_components(prompt: str) -> list[LibraryItem]:
    # Embed prompt
    embedding = embed_text(prompt)
    
    # Search all libraries by tag embeddings
    results = []
    for library in [geometry, materials, textures, audio, recipes]:
        matches = library.vector_search(
            query=embedding,
            limit=5,
            min_score=0.7
        )
        results.extend(matches)
    
    # Rank by usage + rating + relevance
    return rank_suggestions(results)
```

**Example:**
```
Prompt: "rusty AK-47"
  → Geometry: AK Receiver Pattern (★★★★½)
  → Material: Steel (Battle-Worn)
  → Texture: Rusty Steel - Heavy Wear (★★★★)
  → Recipe: Standard AK-47 (Worn) (★★★★★)
```

---

## **Library Management**

### Adding Items

```
┌─────────────────────────────────────────────────────────────┐
│ Add to Library                                              │
├─────────────────────────────────────────────────────────────┤
│ Source:                                                     │
│  ○ Upload file (OBJ, glTF, PNG)                            │
│  ○ Generate with AI                                         │
│  ● Extract from current asset                               │
│                                                             │
│ Component: [Receiver ▼]                                     │
│ Name: [AK Receiver - Custom]                                │
│ Tags: [weapon] [rifle] [receiver] [+]                       │
│                                                             │
│ [Save to Library]  [Cancel]                                 │
└─────────────────────────────────────────────────────────────┘
```

### Versioning

Each library item maintains version history:

```json
{
  "id": "geo_ak_receiver_001",
  "current_version": 3,
  "versions": [
    {"version": 1, "date": "2026-01-10", "notes": "Initial"},
    {"version": 2, "date": "2026-01-15", "notes": "Added rail mount"},
    {"version": 3, "date": "2026-01-20", "notes": "Fixed attachment points"}
  ]
}
```

### Usage Tracking

```sql
-- Track which assets use which library items
SELECT 
  l.name,
  COUNT(u.asset_id) as usage_count,
  AVG(u.rating) as avg_rating
FROM library_items l
JOIN usage_log u ON l.id = u.library_item_id
WHERE u.created_at > NOW() - INTERVAL '30 days'
GROUP BY l.id
ORDER BY usage_count DESC;
```

---

## **Cross-Library Relationships**

Libraries connect to form complete assets:

```
Recipe
  ├── Geometry Card → Geometry Library
  │     └── attachment_points
  │
  ├── Material Card → Materials Library
  │     ├── base_spec → material-database.md
  │     └── texture → Textures Library
  │
  └── Audio Card → Audio Library
        └── physics_mapping
```

**Dependency Resolution:**
```python
def resolve_recipe(recipe: Recipe) -> Asset:
    components = {}
    
    for card in recipe.cards:
        if card.type == "geometry":
            geo = geometry_library.get(card.library_id)
            components[card.component] = geo
            
        elif card.type == "material":
            mat = materials_library.get(card.material_id)
            tex = textures_library.get(card.texture_id)
            
            # Apply material to geometry component
            target = components[card.target]
            target.apply_material(mat, tex)
    
    return Asset(components)
```

---

## **MongoDB Integration**

All component libraries are stored in MongoDB, providing flexible document storage for complex SDF trees, tag-based queries, and vector search for smart suggestions.

### Database Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MongoDB Cluster                         │
├─────────────────────────────────────────────────────────────┤
│  Database: gve_forge                                        │
│                                                             │
│  Collections:                                               │
│  ├── geometry_library     (SDF trees, attachment points)   │
│  ├── materials_library    (physics specs, overrides)       │
│  ├── textures_library     (PBR map references, metadata)   │
│  ├── audio_library        (DSP patches, physics mapping)   │
│  ├── recipes_library      (card chains, templates)         │
│  ├── usage_log            (analytics, ratings)             │
│  └── embeddings_cache     (vector search, $vectorSearch)   │
└─────────────────────────────────────────────────────────────┘
```

### Collection Schemas

**Geometry Collection:**
```javascript
db.geometry_library.createIndex({ "tags": 1 });
db.geometry_library.createIndex({ "metadata.usage_count": -1 });

// Document structure
{
  "_id": ObjectId("..."),
  "id": "geo_ak_receiver_001",
  "name": "AK Receiver Pattern",
  "tags": ["weapon", "rifle", "receiver", "ak"],
  "sdf_tree": {
    "root_node": { /* CSG tree */ }
  },
  "attachment_points": { /* named positions */ },
  "material_zones": ["receiver_body", "dust_cover"],
  "embedding": [0.123, -0.456, ...],  // 768-dim for vector search
  "metadata": {
    "source": "ai-generated",
    "created": ISODate("2026-01-25"),
    "usage_count": 47,
    "rating": 4.5,
    "version": 3
  }
}
```

**Recipes Collection:**
```javascript
db.recipes_library.createIndex({ "tags": 1 });
db.recipes_library.createIndex({ "metadata.rating": -1 });

{
  "_id": ObjectId("..."),
  "id": "recipe_ak47_worn",
  "name": "Standard AK-47 (Worn)",
  "tags": ["weapon", "rifle", "ak", "military"],
  "cards": [
    {
      "type": "geometry",
      "component": "receiver",
      "source": "library",
      "library_ref": { "$ref": "geometry_library", "$id": ObjectId("...") }
    },
    {
      "type": "material",
      "target": "receiver",
      "material_ref": { "$ref": "materials_library", "$id": ObjectId("...") },
      "texture_ref": { "$ref": "textures_library", "$id": ObjectId("...") }
    }
  ],
  "metadata": {
    "total_cost": 0,
    "usage_count": 47,
    "rating": 4.8
  }
}
```

### Query Patterns

**Tag-Based Search (htmx endpoint):**
```python
# FastAPI endpoint returning htmx partial
@app.get("/api/library/search", response_class=HTMLResponse)
async def search_library(tags: list[str], library: str):
    collection = db[f"{library}_library"]
    
    results = collection.find({
        "tags": {"$all": tags}
    }).sort("metadata.usage_count", -1).limit(20)
    
    # Return htmx partial HTML
    return templates.TemplateResponse(
        "partials/library_grid.html",
        {"items": list(results)}
    )
```

**Vector Search (Smart Suggestions):**
```python
@app.get("/api/library/suggest")
async def suggest_components(prompt: str):
    embedding = embed_model.encode(prompt)
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": "embedding_index",
                "path": "embedding",
                "queryVector": embedding.tolist(),
                "numCandidates": 100,
                "limit": 10
            }
        },
        {
            "$project": {
                "id": 1, "name": 1, "tags": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    results = db.geometry_library.aggregate(pipeline)
    return templates.TemplateResponse(
        "partials/suggestions.html",
        {"suggestions": list(results)}
    )
```

**Usage Tracking:**
```python
async def log_usage(library_id: str, asset_id: str, rating: float = None):
    db.usage_log.insert_one({
        "library_item_id": library_id,
        "asset_id": asset_id,
        "rating": rating,
        "created_at": datetime.utcnow()
    })
    
    # Increment usage count atomically
    db.geometry_library.update_one(
        {"id": library_id},
        {"$inc": {"metadata.usage_count": 1}}
    )
```

### htmx + MongoDB Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (htmx)                                             │
│                                                             │
│  <input hx-get="/api/library/search"                       │
│         hx-trigger="keyup changed delay:300ms"              │
│         hx-target="#results"                                │
│         name="tags" />                                      │
│                                                             │
│  <div id="results">                                         │
│    <!-- Server returns HTML partial directly from MongoDB -->│
│  </div>                                                     │
└──────────────┬──────────────────────────────────────────────┘
               │ HTTP GET /api/library/search?tags=metal,worn
               ↓
┌─────────────────────────────────────────────────────────────┐
│  FastAPI Backend                                            │
│                                                             │
│  1. Parse tags from query params                            │
│  2. Query MongoDB: db.textures_library.find({tags: $all})   │
│  3. Render Jinja2 partial template                          │
│  4. Return HTML fragment (not JSON!)                        │
└──────────────┬──────────────────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────────────────────┐
│  MongoDB                                                    │
│  db.textures_library.find({"tags": {"$all": ["metal","worn"]}})│
└─────────────────────────────────────────────────────────────┘
```

### Indexes for Performance

```javascript
// Essential indexes for library queries
db.geometry_library.createIndex({ "tags": 1 });
db.geometry_library.createIndex({ "metadata.usage_count": -1 });
db.geometry_library.createIndex({ "metadata.rating": -1 });

// Vector search index (Atlas Search)
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 768,
        "similarity": "cosine"
      }
    }
  }
}

// Compound index for filtered sorting
db.recipes_library.createIndex({ 
  "tags": 1, 
  "metadata.rating": -1 
});
```

---

## **Benefits**

| Metric | Without Libraries | With Libraries |
|--------|------------------|----------------|
| **First Asset** | $0.12, 15s | Same |
| **Second Asset (same type)** | $0.12, 15s | $0, instant |
| **10 Variants** | $1.20, 150s | $0.12, 15s (10× faster) |
| **Library After 100 Assets** | N/A | 80%+ cached |

**ROI:** Library investment pays off after 2-3 assets of same type.

---

**Version:** 2.0  
**Last Updated:** January 26, 2026  
**Related:** [Card-Chain](./forge-card-chain.md) | [Textures](./texture-library-implementation.md) | [Materials](../data/material-database.md)
