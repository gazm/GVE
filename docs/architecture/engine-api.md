# **GVE-1 Engine API**

**Role:** The internal orchestration layer  
**Strategy:** Type-safe cross-layer communication with schema sync  
**Philosophy:** Rust as source of truth, Python/TypeScript derived automatically

---

## **1. Architecture Overview**

### Tri-Layer Communication

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: The Forge (Interface)                             │
│  HTML5 + htmx + Rust WASM                                   │
│  Location: /tools/forge-ui                                  │
└────────────────────────┬────────────────────────────────────┘
                         │ htmx requests → HTML partials
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: The Architect (Logic)                             │
│  Python + FastAPI + PyTorch + MongoDB                       │
│  Location: /backend/architect                               │
└────────────────────────┬────────────────────────────────────┘
                         │ Binary assets → File system cache
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: The Engine (Runtime)                              │
│  Rust + wgpu + rapier3d + cpal                              │
│  Location: /engine/runtime                                  │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Responsibility | Never Does |
|-------|----------------|------------|
| **Forge** | UI, preview, user input | Business logic, DB access |
| **Architect** | AI, compilation, DB | Real-time rendering |
| **Engine** | Simulation, rendering, audio | HTTP, DB queries |

---

## **2. Schema Generator**

### The Rosetta Stone

Rust structs are the **single source of truth** for all data types. Python and TypeScript types are auto-generated at build time.

```
┌─────────────────────────────────────────────────────────────┐
│  Rust Structs (Source of Truth)                             │
│  Location: /engine/shared/src/types.rs                      │
└────────────────────────┬────────────────────────────────────┘
                         │ cargo build (typeshare + schemars)
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
    ┌─────────┐    ┌──────────┐    ┌──────────┐
    │ Python  │    │ TypeScript│    │ JSON     │
    │ Pydantic│    │ Types    │    │ Schema   │
    └─────────┘    └──────────┘    └──────────┘
```

### Rust Definition

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typeshare::typeshare;

#[typeshare]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AssetMetadata {
    pub name: String,
    pub version: u32,
    pub category: AssetCategory,
    pub material_spec: Option<String>,  // e.g., "ASTM_A36"
}

#[typeshare]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub enum AssetCategory {
    Architecture,
    Weapon,
    Vehicle,
    Prop,
    Character,
}
```

### Generated Python (Pydantic)

```python
# Auto-generated - DO NOT EDIT
from pydantic import BaseModel
from enum import Enum
from typing import Optional

class AssetCategory(str, Enum):
    ARCHITECTURE = "Architecture"
    WEAPON = "Weapon"
    VEHICLE = "Vehicle"
    PROP = "Prop"
    CHARACTER = "Character"

class AssetMetadata(BaseModel):
    name: str
    version: int
    category: AssetCategory
    material_spec: Optional[str] = None
```

### Generated TypeScript

```typescript
// Auto-generated - DO NOT EDIT
export type AssetCategory = 
    | "Architecture" 
    | "Weapon" 
    | "Vehicle" 
    | "Prop" 
    | "Character";

export interface AssetMetadata {
    name: string;
    version: number;
    category: AssetCategory;
    material_spec?: string;
}
```

> [!IMPORTANT]
> Never manually edit generated files. All changes flow from Rust → Python/TypeScript.

---

## **3. Material Specification API**

### Material Lookup

Materials are identified by ASTM/AMS spec codes. The Librarian resolves specs to full property sets.

```rust
/// Material specification identifier
pub type MaterialSpecId = String;  // e.g., "ASTM_A36", "WOOD_OAK"

/// Full material properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialSpec {
    pub spec_id: MaterialSpecId,
    pub display_name: String,
    
    // Physical properties (for physics)
    pub density_kg_m3: f32,
    pub youngs_modulus_gpa: f32,
    pub poissons_ratio: f32,
    
    // Audio properties (for synthesis)
    pub damping_coefficient: f32,
    pub resonance_freq_min_hz: f32,
    pub resonance_freq_max_hz: f32,
    
    // Visual properties (for rendering)
    pub base_color_srgb: [u8; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub recommended_color_mode: ColorMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorMode {
    Rgb,    // Static assets
    Oklab,  // Dynamic/procedural assets
}
```

### Python Librarian API

```python
class MaterialLibrarian:
    """Material specification lookup and caching."""
    
    def get_material(self, spec_id: str) -> MaterialSpec:
        """
        Resolve material spec to full properties.
        
        Args:
            spec_id: ASTM/AMS code (e.g., "ASTM_A36")
            
        Returns:
            MaterialSpec with physics, audio, and visual properties
            
        Raises:
            MaterialNotFoundError: Unknown spec_id
        """
        if spec_id not in self._cache:
            self._cache[spec_id] = self._load_from_db(spec_id)
        return self._cache[spec_id]
    
    def get_audio_properties(self, spec_id: str) -> AudioProperties:
        """Get only audio-relevant properties."""
        mat = self.get_material(spec_id)
        return AudioProperties(
            damping=mat.damping_coefficient,
            resonance_range=(mat.resonance_freq_min_hz, mat.resonance_freq_max_hz),
        )
    
    def resolve_impact_pair(
        self, 
        spec_a: str, 
        spec_b: str,
    ) -> ImpactProperties:
        """Calculate combined properties for material collision."""
        mat_a = self.get_material(spec_a)
        mat_b = self.get_material(spec_b)
        
        return ImpactProperties(
            combined_damping=(mat_a.damping_coefficient + mat_b.damping_coefficient) / 2,
            avg_resonance_freq=(mat_a.resonance_freq_max_hz + mat_b.resonance_freq_min_hz) / 2,
        )
```

---

## **4. Librarian API**

### The Database Abstraction Layer

The Librarian wraps MongoDB and manages cache synchronization.

### CRUD Operations

```python
class AssetLibrarian:
    """Asset database abstraction layer."""
    
    async def save_asset(self, asset: AssetDocument) -> ObjectId:
        """
        Atomic save to MongoDB + trigger background compile.
        
        Flow:
        1. Increment version counter
        2. Write to MongoDB
        3. Queue compilation job
        4. Return immediately (compile runs async)
        """
        asset.version += 1
        asset.updated_at = datetime.utcnow()
        
        result = await self.db.assets.replace_one(
            {"_id": asset.id},
            asset.dict(),
            upsert=True
        )
        
        # Trigger background compilation
        await self.compiler_queue.enqueue(asset.id)
        
        return result.upserted_id or asset.id
    
    async def load_asset(self, asset_id: ObjectId) -> AssetDocument:
        """Load asset from MongoDB."""
        doc = await self.db.assets.find_one({"_id": asset_id})
        if not doc:
            raise AssetNotFoundError(asset_id)
        return AssetDocument(**doc)
    
    async def delete_asset(self, asset_id: ObjectId) -> None:
        """
        Delete asset and clean up cache files.
        
        Flow:
        1. Delete from MongoDB
        2. Remove binary files from /cache
        3. Clear from any in-memory caches
        """
        asset = await self.load_asset(asset_id)
        
        await self.db.assets.delete_one({"_id": asset_id})
        
        # Garbage collect cache files
        cache_path = self._resolve_cache_path(asset)
        if cache_path.exists():
            cache_path.unlink()
```

### Path Resolution

```python
def _resolve_cache_path(self, asset: AssetDocument) -> Path:
    """
    Calculate cache file path from metadata.
    
    Format: /cache/{type}/{category}/{subcategory}/{name}_{short_id}.gve_bin
    Example: /cache/geometry/military/bunker_heavy_01_507f1f.gve_bin
    """
    short_id = str(asset.id)[-6:]
    filename = f"{snake_case(asset.name)}_{short_id}.gve_bin"
    
    return (
        self.cache_root
        / asset.type.value
        / asset.meta.category
        / asset.meta.subcategory
        / filename
    )
```

### Cache Invalidation

```python
def check_cache_validity(self, asset_id: ObjectId) -> CacheStatus:
    """
    Compare DB version vs file version.
    
    Returns:
        CacheStatus.VALID: Binary matches DB
        CacheStatus.STALE: Binary outdated, recompile needed
        CacheStatus.MISSING: No binary exists
    """
    asset = self.load_asset(asset_id)
    cache_path = self._resolve_cache_path(asset)
    
    if not cache_path.exists():
        return CacheStatus.MISSING
    
    file_version = self._read_binary_version(cache_path)
    
    if file_version < asset.version:
        return CacheStatus.STALE
    
    return CacheStatus.VALID
```

---

## **5. Compiler Interface**

### DNA JSON → GVE Binary

The Compiler transforms JSON recipes into optimized binaries.

### Compilation Request

```python
@dataclass
class CompileRequest:
    asset_id: ObjectId
    priority: CompilePriority = CompilePriority.NORMAL
    force_recompile: bool = False
    
    # Optional overrides
    splat_iterations: Optional[int] = None  # Default: 300
    volume_resolution: Optional[int] = None  # Default: 128

class CompilePriority(IntEnum):
    LOW = 0       # Background batch
    NORMAL = 1    # User-triggered
    HIGH = 2      # Preview refresh
    IMMEDIATE = 3 # Blocking (for tests)
```

### Compilation Pipeline

```python
class AssetCompiler:
    """DNA JSON → GVE Binary compiler."""
    
    async def compile(self, request: CompileRequest) -> CompileResult:
        """
        Full compilation pipeline.
        
        Stages:
        1. Math JIT - Build PyTorch SDF graph
        2. Volume Bake - Evaluate to 3D texture
        3. Shell Gen - Dual Contouring mesh
        4. Splat Train - Optimize gaussians
        5. Indexing - Generate embeddings
        6. Write Binary - Pack .gve_bin
        """
        asset = await self.librarian.load_asset(request.asset_id)
        
        # Stage 1: Math JIT
        sdf_fn = self.math_jit.compile_tree(asset.dna.root_node)
        
        # Stage 2: Volume Bake
        volume = self.volume_baker.bake(
            sdf_fn,
            resolution=request.volume_resolution or 128
        )
        
        # Stage 3: Shell Generation
        shell_mesh = self.shell_generator.generate(sdf_fn)
        
        # Stage 4: Splat Training
        splats = self.splat_trainer.train(
            sdf_fn,
            iterations=request.splat_iterations or 300
        )
        
        # Stage 5: Semantic Indexing
        embedding = self.indexer.create_embedding(asset.dna, splats)
        
        # Stage 6: Write Binary
        binary_path = self.librarian._resolve_cache_path(asset)
        self.binary_writer.write(
            binary_path,
            volume=volume,
            shell=shell_mesh,
            splats=splats,
            audio_patch=asset.dna.audio_patch,
        )
        
        return CompileResult(
            success=True,
            binary_path=binary_path,
            compile_time_sec=elapsed,
        )
```

### Progress Callbacks

```python
class CompileProgress:
    """Progress updates for UI feedback."""
    
    stage: CompileStage
    progress: float  # 0.0 - 1.0
    message: str

class CompileStage(Enum):
    MATH_JIT = "Building SDF graph"
    VOLUME_BAKE = "Baking 3D texture"
    SHELL_GEN = "Generating mesh"
    SPLAT_TRAIN = "Training splats"
    INDEXING = "Creating embeddings"
    WRITING = "Writing binary"

# WebSocket event
async def on_compile_progress(asset_id: ObjectId, progress: CompileProgress):
    await ws_manager.broadcast(
        f"compile:{asset_id}",
        progress.dict()
    )
```

---

## **6. Runtime API**

### Asset Loading (Rust)

```rust
pub struct AssetLoader {
    cache_root: PathBuf,
    loaded_assets: HashMap<AssetId, LoadedAsset>,
}

impl AssetLoader {
    /// Load asset from cache into GPU memory.
    pub fn load_asset(&mut self, asset_id: AssetId) -> Result<AssetHandle> {
        let path = self.resolve_cache_path(asset_id)?;
        
        // Memory-map binary file
        let mmap = unsafe { Mmap::map(&File::open(&path)?)? };
        
        // Parse header
        let header = GVEBinaryHeader::from_bytes(&mmap[0..64])?;
        
        // Upload to GPU
        let sdf_texture = self.gpu.upload_texture_3d(
            &mmap[header.sdf_texture_offset as usize..],
            header.sdf_texture_size,
        )?;
        
        let splat_buffer = self.gpu.upload_buffer(
            &mmap[header.splat_data_offset as usize..],
            header.splat_count * 48,  // 48 bytes per splat
        )?;
        
        let shell_mesh = self.gpu.upload_mesh(
            &mmap[header.shell_mesh_offset as usize..],
        )?;
        
        let loaded = LoadedAsset {
            sdf_texture,
            splat_buffer,
            shell_mesh,
            audio_patch: self.parse_audio_patch(&mmap, &header),
        };
        
        let handle = self.register_asset(asset_id, loaded);
        Ok(handle)
    }
    
    /// Unload asset and free GPU memory.
    pub fn unload_asset(&mut self, handle: AssetHandle) -> Result<()> {
        let asset = self.loaded_assets.remove(&handle.id)
            .ok_or(AssetNotLoadedError)?;
        
        self.gpu.free_texture(asset.sdf_texture);
        self.gpu.free_buffer(asset.splat_buffer);
        self.gpu.free_mesh(asset.shell_mesh);
        
        Ok(())
    }
    
    /// Hot-reload asset without unloading.
    pub fn hot_reload(&mut self, handle: AssetHandle) -> Result<()> {
        let asset_id = handle.id;
        self.unload_asset(handle)?;
        self.load_asset(asset_id)?;
        Ok(())
    }
}
```

---

## **7. Event Bus**

### Cross-Layer Events

Events flow between layers via WebSocket and internal channels.

### Event Definitions

| Event | Direction | Payload |
|-------|-----------|---------|
| `asset:created` | Architect → Forge | `{asset_id, name}` |
| `asset:updated` | Architect → Forge | `{asset_id, version}` |
| `asset:deleted` | Architect → Forge | `{asset_id}` |
| `compile:started` | Architect → Forge | `{asset_id, stages}` |
| `compile:progress` | Architect → Forge | `{asset_id, stage, progress}` |
| `compile:complete` | Architect → Forge | `{asset_id, binary_path}` |
| `compile:failed` | Architect → Forge | `{asset_id, error}` |

### WebSocket Protocol

```python
# Server (FastAPI)
@app.websocket("/ws/events")
async def event_stream(websocket: WebSocket):
    await websocket.accept()
    
    async for event in event_bus.subscribe():
        await websocket.send_json({
            "type": event.type,
            "payload": event.payload,
            "timestamp": event.timestamp.isoformat(),
        })
```

```typescript
// Client (Forge)
const ws = new WebSocket("/ws/events");

ws.onmessage = (event) => {
    const { type, payload } = JSON.parse(event.data);
    
    switch (type) {
        case "compile:progress":
            updateProgressBar(payload.asset_id, payload.progress);
            break;
        case "compile:complete":
            refreshPreview(payload.asset_id);
            break;
    }
};
```

---

## **8. WASM Binary Messaging Protocol**

### The Communication Layer

The Forge (WASM) and Architect (Python) communicate via a **raw binary messaging protocol** over WebSocket. Since `.gve_bin` is a solved format, we pass it directly without JSON wrapping.

### Message Frame Format

```
┌──────────────────────────────────────────────────────────────┐
│ Header (16 bytes)                                            │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ msg_type │ asset_id │ version  │ payload  │ reserved        │
│ u8       │ u64      │ u32      │ u32      │ u8              │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│ Payload (variable)                                           │
│ Raw .gve_bin bytes or command-specific data                  │
└──────────────────────────────────────────────────────────────┘
```

### Message Types

```rust
#[repr(u8)]
enum MessageType {
    // Architect → Forge
    AssetReady     = 0x01,  // Payload: .gve_bin bytes
    AssetProgress  = 0x02,  // Payload: stage (u8) + progress (f32)
    AssetInvalid   = 0x03,  // Payload: error string (UTF-8)
    VersionBump    = 0x04,  // Payload: none (header.version is new version)
    
    // Forge → Architect
    RequestAsset   = 0x10,  // Payload: none (uses header.asset_id)
    RequestCompile = 0x11,  // Payload: DNA JSON bytes
    CancelCompile  = 0x12,  // Payload: none
}
```

### Cache Versioning (Zero Binary Bloat)

Version tracking lives in the **message header**, not the binary file:

```python
# Architect side
class AssetCache:
    """Manages asset versions without modifying binary files."""
    
    def __init__(self):
        self.version_map: dict[ObjectId, int] = {}  # In-memory version tracker
    
    async def send_asset(self, ws: WebSocket, asset_id: ObjectId):
        version = self.version_map.get(asset_id, 0)
        binary_path = self.resolve_cache_path(asset_id)
        
        # Build header
        header = struct.pack(
            "<BQI I B",  # msg_type, asset_id, version, payload_len, reserved
            MessageType.ASSET_READY,
            int(asset_id),
            version,
            binary_path.stat().st_size,
            0
        )
        
        # Stream raw binary directly
        await ws.send_bytes(header + binary_path.read_bytes())
    
    async def on_compile_complete(self, asset_id: ObjectId):
        """Bump version and notify connected clients."""
        self.version_map[asset_id] = self.version_map.get(asset_id, 0) + 1
        
        # Broadcast version bump to all Forge instances
        header = struct.pack(
            "<BQI I B",
            MessageType.VERSION_BUMP,
            int(asset_id),
            self.version_map[asset_id],
            0,  # No payload
            0
        )
        await self.broadcast(header)
```

```rust
// Forge (WASM) side
struct AssetVersionCache {
    versions: HashMap<u64, u32>,  // asset_id → version
}

impl AssetVersionCache {
    fn on_message(&mut self, header: MessageHeader, payload: &[u8]) {
        match header.msg_type {
            MessageType::VersionBump => {
                // Server notifies us version changed - request fresh asset
                self.versions.insert(header.asset_id, header.version);
                self.request_asset(header.asset_id);
            }
            MessageType::AssetReady => {
                let cached_version = self.versions.get(&header.asset_id);
                
                if cached_version.map_or(true, |v| header.version > *v) {
                    // New version - load it
                    self.load_binary(payload);
                    self.versions.insert(header.asset_id, header.version);
                }
                // else: stale message, ignore
            }
            MessageType::AssetProgress => {
                // Show compilation progress in UI
                let stage = payload[0];
                let progress = f32::from_le_bytes(payload[1..5].try_into().unwrap());
                self.update_progress_bar(header.asset_id, stage, progress);
            }
            _ => {}
        }
    }
    
    fn is_stale(&self, asset_id: u64, version: u32) -> bool {
        self.versions.get(&asset_id).map_or(false, |v| version < *v)
    }
}
```

### Preventing Stale Reads

The version-in-header approach solves cache races:

```
Timeline:
─────────────────────────────────────────────────────────────
t=0   User edits asset (triggers compile)
t=1   Architect broadcasts VERSION_BUMP (version=12)
t=2   Forge marks local cache stale, shows "Compiling..."
t=60  Compile completes
t=61  Architect sends ASSET_READY (version=12, payload=binary)
t=62  Forge loads new binary, hides progress bar
```

**Key Properties:**
- No binary format changes required
- No disk writes for version tracking
- Forge never displays stale data (shows progress instead)
- Multiple Forge instances stay synchronized

---

## **9. WASM Scene Snapshot (Hierarchy Panel)**

The viewport exposes a binary scene snapshot so the Forge UI can show the current engine scene (loaded meshes/SDFs) in the Hierarchy panel.

### `get_scene_snapshot()` (WASM export)

- **Source:** `engine/runtime` renderer pipeline → exposed via `engine/wasm` as `get_scene_snapshot()`.
- **Returns:** `Vec<u8>` (typed as `Uint8Array` in JS).

### Binary layout (little-endian)

| Offset | Type   | Description |
|--------|--------|-------------|
| 0      | `u32`  | Entry count `N` |
| 4      | …      | `N` entries, each 10 bytes (see below) |

**Per entry (10 bytes):**

| Offset (within entry) | Type  | Description |
|------------------------|-------|-------------|
| 0 | `u64` (8 bytes) | Asset ID (engine internal) |
| 8 | `u8`  | Type: `0` = mesh, `1` = SDF |
| 9 | `u8`  | Active: `0` = inactive, `1` = active (e.g. SDF currently raymarched) |

Meshes are listed first, then SDFs. The JS hierarchy builder uses `window._sceneLabels[assetId]` when set (e.g. from Add to Chain with a name) to show a friendly label instead of `Mesh <id>` / `SDF <id>`.

---

## **10. Implementation Checklist**

### Schema Generator
- [ ] Configure typeshare for Rust → TypeScript
- [ ] Configure schemars for JSON Schema generation
- [ ] Set up Pydantic model generation from JSON Schema
- [ ] Add CI check for schema drift

### Librarian
- [ ] Implement AssetLibrarian CRUD
- [ ] Implement MaterialLibrarian lookup
- [ ] Add path resolution logic
- [ ] Implement cache invalidation

### Compiler
- [ ] Build compilation queue (Redis/RabbitMQ)
- [ ] Implement progress WebSocket events
- [ ] Add priority scheduling
- [ ] Implement incremental compilation

### Runtime
- [ ] Implement AssetLoader with GPU upload
- [ ] Add hot-reload support
- [ ] Implement streaming for large assets
- [ ] Add memory pressure handling

---

**Version:** 1.0  
**Last Updated:** January 29, 2026  
**Related:** [System Overview](./overview.md) | [Database Architecture](../data/database-architecture.md) | [Compiler Pipeline](../workflows/compiler-pipeline.md)
