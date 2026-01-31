# **AI Context Guide**

Templates and examples for `.agent/modules/` context files that help AI coding assistants understand module boundaries.

**Last Updated:** January 26, 2026

---

## **Purpose**

AI context files solve three problems:

1. **Scope Confusion** - AI doesn't know what belongs in this module
2. **Dependency Hallucination** - AI invents imports that don't exist
3. **Anti-Pattern Blind Spots** - AI repeats mistakes across modules

Each `.agent/modules/{name}.md` file gives the AI a "briefing" before it starts coding.

---

## **Template Structure**

```markdown
# Module: {name}

## Scope
{One sentence: what this module does and ONLY does}

## Files
{File list with one-line purpose each}

## Public API
{What external code should import}

## Dependencies
{What this module imports and why}

## Key Types
{Important structs/classes the AI needs to know}

## Related Docs
{Links to detailed documentation}

## Anti-Patterns
{Common mistakes to avoid}

## Examples
{Short code snippets showing correct usage}
```

---

## **Complete Examples**

### Example 1: architect-compiler.md

```markdown
# Module: compiler

## Scope
Transforms DNA JSON into compiled `.gve_bin` assets. Does NOT handle database 
operations (that's librarian) or splat training (that's splat_trainer).

## Files
- `__init__.py` - Public API exports
- `math_jit.py` - Builds PyTorch SDF graph from JSON tree
- `volume_bake.py` - Evaluates SDF to 3D texture
- `shell_gen.py` - Dual Contouring mesh generation
- `binary_writer.py` - Packs all data into .gve_bin
- `queue.py` - Priority-based job scheduling

## Public API
```python
# backend/architect/src/compiler/__init__.py
from .pipeline import compile_asset, CompileRequest, CompileResult
from .queue import enqueue_compile, get_compile_status, CompilePriority

__all__ = [
    "compile_asset", "CompileRequest", "CompileResult",
    "enqueue_compile", "get_compile_status", "CompilePriority",
]
```

## Dependencies
| Import | From | Purpose |
|--------|------|---------|
| `AssetDocument` | `generated.types` | Asset schema |
| `load_asset` | `librarian` | Fetch asset from DB |
| `torch` | External | SDF evaluation |

## Key Types
```python
@dataclass
class CompileRequest:
    asset_id: ObjectId
    priority: CompilePriority = CompilePriority.NORMAL
    force_recompile: bool = False

@dataclass
class CompileResult:
    success: bool
    binary_path: Path
    compile_time_sec: float
    error: Optional[str] = None
```

## Related Docs
- [Compiler Pipeline](../../docs/workflows/compiler-pipeline.md) - Full algorithm details
- [Engine API §5](../../docs/architecture/engine-api.md#5-compiler-interface) - Interface spec
- [Data Specs §3](../../docs/data/data-specifications.md) - Binary format

## Anti-Patterns
- ❌ Never access MongoDB directly → use `librarian.load_asset()`
- ❌ Never run splat training here → that's `splat_trainer` module
- ❌ Never import from `api/` → compiler is called BY api, not vice versa
- ❌ Never block on long compiles → use `queue.enqueue_compile()` for async

## Examples

### Correct: Async compilation
```python
from compiler import enqueue_compile, CompilePriority

# Queue the job (returns immediately)
job_id = await enqueue_compile(asset_id, priority=CompilePriority.HIGH)

# Later: check status
status = await get_compile_status(job_id)
```

### Wrong: Direct MongoDB access
```python
# ❌ BAD - compiler should not know about MongoDB
from motor.motor_asyncio import AsyncIOMotorClient
client = AsyncIOMotorClient()
asset = await client.gve.assets.find_one({"_id": asset_id})

# ✅ GOOD - use librarian
from librarian import load_asset
asset = await load_asset(asset_id)
```
```

---

### Example 2: engine-renderer.md

```markdown
# Module: renderer

## Scope
wgpu rendering pipeline for hybrid volumetric + splatting. Handles ONLY 
GPU draw calls. Does NOT handle asset loading (that's loader) or physics 
(that's physics module).

## Files
- `mod.rs` - Public exports
- `pipeline.rs` - RenderPipeline initialization
- `shell_pass.rs` - Early-Z proxy mesh rasterization
- `volume_pass.rs` - SDF raymarching shader
- `splat_pass.rs` - Gaussian splat sorting and drawing
- `lod.rs` - LOD transition logic

## Public API
```rust
// engine/runtime/src/renderer/mod.rs
pub use pipeline::{Renderer, RenderConfig};
pub use lod::LodManager;

// Create renderer
pub fn create_renderer(device: &wgpu::Device, config: RenderConfig) -> Renderer;

// Per-frame
pub fn render_frame(renderer: &mut Renderer, scene: &Scene, view: &View);
```

## Dependencies
| Crate | Purpose |
|-------|---------|
| `wgpu` | GPU abstraction |
| `gve_shared` | Type definitions |
| `bytemuck` | Buffer casting |

## Key Types
```rust
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shell_pipeline: wgpu::RenderPipeline,
    volume_pipeline: wgpu::RenderPipeline,
    splat_pipeline: wgpu::RenderPipeline,
}

pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub msaa_samples: u32,
    pub max_splats: u32,
}
```

## Related Docs
- [Rendering Pipeline](../../docs/core-systems/rendering-pipeline.md) - Algorithm details
- [System Overview §3](../../docs/architecture/overview.md#3-core-rendering-architecture-shell--volume) - Architecture

## Anti-Patterns
- ❌ Never load assets in renderer → use `loader` module
- ❌ Never run physics queries → that's `physics` module
- ❌ Never allocate GPU resources per-frame → pre-allocate in `Renderer::new()`
- ❌ Never hard-code shader paths → use include_str!() or embed

## Examples

### Correct: Render loop
```rust
// In main loop
let view = camera.get_view_matrix();
let scene = scene_manager.get_visible(&view);

renderer.render_frame(&scene, &view);
```

### Wrong: Loading in render
```rust
// ❌ BAD - causes frame stutter
fn render_frame(&mut self, scene: &Scene) {
    for entity in scene.entities {
        let asset = self.load_asset(entity.asset_id);  // WRONG!
        self.draw(asset);
    }
}

// ✅ GOOD - assets pre-loaded
fn render_frame(&mut self, scene: &Scene) {
    for entity in scene.entities {
        // Asset already loaded via loader module
        let handle = entity.asset_handle;
        self.draw(handle);
    }
}
```
```

---

### Example 3: architect-librarian.md

```markdown
# Module: librarian

## Scope
MongoDB wrapper for asset and material CRUD. The ONLY module that touches 
the database. Other modules MUST go through librarian.

## Files
- `__init__.py` - Public API exports
- `assets.py` - Asset document CRUD
- `materials.py` - Material spec lookup and caching
- `cache.py` - Binary file path resolution

## Public API
```python
# backend/architect/src/librarian/__init__.py
from .assets import (
    load_asset, save_asset, delete_asset,
    list_assets, search_assets,
)
from .materials import (
    get_material, get_audio_properties, 
    resolve_impact_pair,
)
from .cache import (
    resolve_cache_path, check_cache_validity, 
    CacheStatus,
)

__all__ = [
    # Assets
    "load_asset", "save_asset", "delete_asset",
    "list_assets", "search_assets",
    # Materials
    "get_material", "get_audio_properties", "resolve_impact_pair",
    # Cache
    "resolve_cache_path", "check_cache_validity", "CacheStatus",
]
```

## Dependencies
| Import | From | Purpose |
|--------|------|---------|
| `AssetDocument`, `MaterialSpec` | `generated.types` | Schemas |
| `motor` | External | Async MongoDB driver |
| `bson.ObjectId` | External | Document IDs |

## Key Types
```python
class CacheStatus(Enum):
    VALID = "valid"      # Binary matches DB
    STALE = "stale"      # Binary outdated
    MISSING = "missing"  # No binary exists
```

## Related Docs
- [Engine API §4](../../docs/architecture/engine-api.md#4-librarian-api) - Full API spec
- [Database Architecture](../../docs/data/database-architecture.md) - MongoDB setup
- [Material Database](../../docs/data/material-database.md) - Material specs

## Anti-Patterns
- ❌ Never import MongoDB client elsewhere → ALL DB access goes through librarian
- ❌ Never cache materials in other modules → librarian handles caching
- ❌ Never construct file paths manually → use `resolve_cache_path()`
- ❌ Never use blocking I/O → all functions are async

## Examples

### Correct: Loading an asset
```python
from librarian import load_asset

asset = await load_asset(asset_id)
print(asset.name, asset.version)
```

### Wrong: Direct MongoDB access elsewhere
```python
# ❌ BAD - in compiler module
from motor.motor_asyncio import AsyncIOMotorClient
db = AsyncIOMotorClient().gve
asset = await db.assets.find_one({"_id": asset_id})

# ✅ GOOD - use librarian
from librarian import load_asset
asset = await load_asset(asset_id)
```
```

---

## **File Naming Convention**

```
.agent/modules/
├── engine-shared.md       # engine/shared/
├── engine-renderer.md     # engine/runtime/renderer/
├── engine-physics.md      # engine/runtime/physics/
├── engine-audio.md        # engine/runtime/audio/
├── engine-loader.md       # engine/runtime/loader/
├── engine-wasm.md         # engine/wasm/
├── architect-api.md       # backend/architect/src/api/
├── architect-compiler.md  # backend/architect/src/compiler/
├── architect-splat.md     # backend/architect/src/splat_trainer/
├── architect-librarian.md # backend/architect/src/librarian/
├── architect-ai.md        # backend/architect/src/ai_pipeline/
└── forge-ui.md            # tools/forge-ui/
```

---

## **When to Update Context Files**

Update the context file when:
- ✅ Adding new files to the module
- ✅ Changing the public API
- ✅ Adding new dependencies
- ✅ Discovering new anti-patterns

---

**See Also:**
- [Overview](./overview.md) - Monorepo layout, phases
- [Module Breakdowns](./module-breakdowns.md) - Detailed file tables
