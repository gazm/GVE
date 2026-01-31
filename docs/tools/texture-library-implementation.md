# Texture Library Implementation Guide

## Backend API Spec

### Endpoints

**GET /api/textures**
```json
{
  "filters": {
    "tags": ["metal", "worn"],  // AND logic
    "search": "rust",
    "source": "ai-generated",
    "min_rating": 3
  },
  "sort": "usage_count" | "created_date" | "rating",
  "limit": 20,
  "offset": 0
}

Response:
{
  "textures": [
    {
      "id": "tex_001",
      "name": "Rusty Steel",
      "tags": ["metal", "rust", "worn"],
      "thumbnail_url": "/storage/tex_001_thumb.webp",
      "usage_count": 47,
      "rating": 4
    }
  ],
  "total": 156
}
```

**POST /api/textures/upload**
```
Form data:
- albedo: File (PNG/JPG, max 4096x4096)
- normal: File (optional)
- roughness: File (optional)
- metallic: File (optional)
- name: String
- tags: ["metal", "worn"]

Backend processing:
1. Validate files
2. Resize to power-of-2
3. Compress to BC7/BC5/BC4
4. Generate mipmaps
5. Auto-tag via vision model
6. Store in database
```

**POST /api/textures/generate**
```json
{
  "prompt": "Battle-damaged steel with bullet impacts",
  "tags": ["metal", "damaged"],
  "resolution": 2048,
  "generate_maps": ["albedo", "normal", "roughness", "metallic"]
}

Response (async):
{
  "job_id": "gen_12345",
  "status": "processing",
  "estimated_time_sec": 8
}
```

**PUT /api/textures/:id/tags**
```json
{
  "add_tags": ["battle-worn"],
  "remove_tags": ["clean"]
}
```

**POST /api/assets/generate**
```json
{
  "prompt": "AK-47",
  "material_presets": {
    "receiver": {
      "texture_id": "tex_rusty_steel_001",
      "skip_generation": true
    },
    "stock": {
      "texture_id": "tex_worn_oak_003",
      "skip_generation": true
    }
  }
}
```

---

## Database Schema

```sql
CREATE TABLE texture_library (
    id VARCHAR(32) PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    source ENUM('procedural', 'ai-generated', 'uploaded', 'scanned'),
    created_at TIMESTAMP DEFAULT NOW(),
    user_id INT,
    
    -- Usage tracking
    usage_count INT DEFAULT 0,
    user_rating INT CHECK(user_rating >= 1 AND user_rating <= 5),
    
    -- File paths
    albedo_path VARCHAR(256),
    normal_path VARCHAR(256),
    roughness_path VARCHAR(256),
    metallic_path VARCHAR(256),
    thumbnail_path VARCHAR(256),
    
    -- Properties
    triplanar_scale FLOAT DEFAULT 1.0,
    color_mode ENUM('rgb', 'oklab') DEFAULT 'rgb',
    
    -- Searchable metadata
    tags_array TEXT[],  -- For PostgreSQL array support
    tags_vector VECTOR(768),  -- For semantic search (embedding)
    
    INDEX idx_tags (tags_array),
    INDEX idx_usage (usage_count),
    INDEX idx_rating (user_rating)
);

CREATE TABLE texture_tags (
    id SERIAL PRIMARY KEY,
    texture_id VARCHAR(32) REFERENCES texture_library(id) ON DELETE CASCADE,
    tag VARCHAR(64) NOT NULL,
    INDEX idx_tag (tag)
);

-- Track usage
CREATE TABLE texture_usage_log (
    id SERIAL PRIMARY KEY,
    texture_id VARCHAR(32) REFERENCES texture_library(id),
    asset_id VARCHAR(32),
    used_at TIMESTAMP DEFAULT NOW()
);
```

---

## Frontend Components (htmx + Rust WASM)

### Texture Browser Component

```html
<!-- Htmx dynamic filtering -->
<div id="texture-browser">
  <form hx-get="/api/textures" 
        hx-trigger="change, search" 
        hx-target="#texture-grid">
    
    <!-- Tag filters -->
    <div class="tag-filters">
      <label><input type="checkbox" name="tags[]" value="metal"> Metal</label>
      <label><input type="checkbox" name="tags[]" value="worn"> Worn</label>
      <label><input type="checkbox" name="tags[]" value="wood"> Wood</label>
      <!-- ... more tags -->
    </div>
    
    <!-- Search -->
    <input type="search" 
           name="search" 
           placeholder="Search textures..."
           hx-trigger="keyup changed delay:300ms">
  </form>
  
  <!-- Grid populated via htmx -->
  <div id="texture-grid" class="texture-grid">
    <!-- Server renders texture cards here -->
  </div>
</div>
```

### WASM Texture Preview

```rust
// Rust WASM viewport integration
pub struct TexturePreview {
    texture_id: String,
    preview_mesh: Mesh,  // Simple sphere
    material: Material,
}

impl TexturePreview {
    pub fn load_texture(&mut self, texture_id: &str) {
        // Fetch texture from server
        let texture_data = fetch_texture_compressed(texture_id);
        
        // Upload to GPU
        self.material.albedo_map = upload_to_gpu(texture_data.albedo);
        self.material.normal_map = upload_to_gpu(texture_data.normal);
        self.material.roughness_map = upload_to_gpu(texture_data.roughness);
        
        // Trigger re-render
        self.render_preview();
    }
    
    fn render_preview(&self) -> HTMLCanvasElement {
        // Render 256x256 preview
        // Uses actual engine PBR shader (perfect parity)
    }
}
```

---

## Usage Tracking

```python
def track_texture_usage(texture_id: str, asset_id: str):
    # Log usage
    db.execute(
        "INSERT INTO texture_usage_log (texture_id, asset_id) VALUES (%s, %s)",
        (texture_id, asset_id)
    )
    
    # Increment counter
    db.execute(
        "UPDATE texture_library SET usage_count = usage_count + 1 WHERE id = %s",
        (texture_id,)
    )

def get_trending_textures(days: int = 7) -> list[Texture]:
    """Get most-used textures in last N days."""
    return db.query("""
        SELECT t.*, COUNT(u.id) as recent_usage
        FROM texture_library t
        JOIN texture_usage_log u ON t.id = u.texture_id
        WHERE u.used_at > NOW() - INTERVAL '%s days'
        GROUP BY t.id
        ORDER BY recent_usage DESC
        LIMIT 10
    """, (days,))
```

---

## Recommended Tags Taxonomy

```yaml
material_type:
  - metal
  - wood
  - stone
  - plastic
  - fabric
  - organic
  - composite
  - glass
  - ceramic

condition:
  - pristine
  - clean
  - worn
  - damaged
  - battle-worn
  - rusted
  - weathered
  - corroded
  - scratched

style:
  - realistic
  - stylized
  - sci-fi
  - fantasy
  - military
  - industrial
  - natural
  - modern
  - retro

surface:
  - smooth
  - rough
  - polished
  - brushed
  - matte
  - glossy
  - anisotropic

color:
  - gray
  - brown
  - black
  - white
  - golden
  - copper
  - silver
  - red
  - blue
  - green
  - camo
```
