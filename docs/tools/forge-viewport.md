# Forge Editor: Viewport & Rendering

**Purpose:** The viewport is the actual game engine running in the browser via Rust WASM, ensuring perfect parity between editor preview and runtime.

**Related Docs:**
- [Rendering Pipeline](../engine/rendering-pipeline.md) - GPU architecture
- [Data Specifications](../data/data-specifications.md) - SDF and splat formats

---

## **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                      Browser (HTML5)                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐   ┌─────────────────────────────────┐  │
│  │   htmx UI       │   │    Rust WASM Module             │  │
│  │                 │   │                                 │  │
│  │  - Card Chain   │   │  - wgpu Renderer                │  │
│  │  - Modifiers    │   │  - Rapier3d Physics             │  │
│  │  - Libraries    │   │  - Audio Synthesis              │  │
│  │                 │◄──┤  - SDF Evaluator                │  │
│  │  [Sliders]      │   │  - Splat Renderer               │  │
│  │  [Buttons]      │   │                                 │  │
│  └────────┬────────┘   └──────────────┬──────────────────┘  │
│           │                           │                      │
│           └─────── SharedArrayBuffer ─┘                      │
│                    (Zero-Copy)                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Technologies

| Layer | Technology | Purpose |
|-------|------------|---------|
| **UI** | htmx | Server-driven hypermedia controls |
| **Bridge** | SharedArrayBuffer | Zero-copy JS ↔ WASM communication |
| **Renderer** | wgpu (WebGPU) | GPU-accelerated rendering |
| **Physics** | Rapier3d | Collision, dynamics |
| **Audio** | Web Audio API | Real-time synthesis |

---

## **Zero-Copy Parameter Updates**

Sliders update WASM memory directly without JavaScript serialization:

```rust
// Rust WASM side
#[wasm_bindgen]
pub struct ViewportState {
    // Shared with JavaScript via SharedArrayBuffer
    uniforms: SharedArrayBuffer,
}

#[wasm_bindgen]
impl ViewportState {
    pub fn get_uniform_ptr(&self) -> *mut f32 {
        self.uniforms.as_ptr() as *mut f32
    }
    
    pub fn render_frame(&mut self) {
        // Read uniforms directly from shared memory
        let twist = unsafe { *self.uniforms.offset(0) };
        let bend = unsafe { *self.uniforms.offset(1) };
        
        // Update SDF shader uniforms
        self.sdf_shader.set_twist(twist);
        self.sdf_shader.set_bend(bend);
        
        // Render at 60 FPS
        self.render();
    }
}
```

```javascript
// JavaScript side
const uniformBuffer = new Float32Array(viewportState.get_uniform_ptr(), 16);

// Slider updates write directly to shared memory
twistSlider.oninput = (e) => {
    uniformBuffer[0] = parseFloat(e.target.value);
    // No serialization! WASM sees change immediately
};
```

**Result:** 60 FPS slider scrubbing with zero latency.

---

## **Visualization Modes**

### View Cube Overlay

Every viewport now has a **view-cube overlay** in the lower-right corner (shared + dedicated). Clicking one of the faces (Front, Back, Left, Right, Top, Bottom) issues `window.snap_camera_to(...)` with the appropriate yaw/pitch while keeping the camera position constant, so you can instantly orient the camera without recalculating vectors in JS.

The overlay stays in HTML/CSS; the WASM engine handles the actual camera update internally and rerenders the next frame with the new orientation.

### Mode A: SDF Source

**Purpose:** Raw mathematical SDF visualization

```
View → Visualization Mode → SDF Source
```

**Features:**
- Infinite resolution (raymarched)
- Blend operation visualization (smooth union seams)
- Surface normals overlay
- Distance field contours

**Use Cases:**
- Checking surface continuity
- Debugging CSG operations
- Verifying material zone boundaries

---

### Mode B: Shell Mesh

**Purpose:** Optimization proxy visualization

```
View → Visualization Mode → Shell Mesh
```

**Features:**
- Low-poly mesh generated from SDF
- Wireframe overlay
- Red zones: mesh clipping into SDF
- Green zones: proper offset maintained

**Use Cases:**
- Verifying Early-Z culling mesh
- Checking LOD transitions
- Performance optimization

---

### Mode C: Splat View (Default)

**Purpose:** Final render preview

```
View → Visualization Mode → Splat View
```

**Features:**
- Gaussian splat rendering (identical to runtime)
- PBR materials applied
- LOD pyramid visualization
- Splat density heatmap

**Use Cases:**
- Final quality check
- Material preview
- "What You See Is What You Play"

---

### Mode D: Physics Debug

**Purpose:** Collision and dynamics visualization

```
View → Visualization Mode → Physics Debug
```

**Features:**
- Collision shapes (Rapier3d debug drawer)
- Joint constraints
- Center of mass markers
- Velocity vectors
- Contact points

**Use Cases:**
- Debugging physics interactions
- Tuning collision shapes
- Verifying joint limits

---

### Mode E: Audio Visualization

**Purpose:** Acoustic properties display

```
View → Visualization Mode → Audio/Resonance
```

**Features:**
- Material resonance frequencies (color-coded)
- Occlusion rays (sound propagation)
- Damping coefficients overlay
- Impact preview (click to hear)

**Use Cases:**
- Tuning material audio properties
- Verifying sound zones
- Previewing impact sounds

---

### Mode F: World Editor

**Purpose:** Level composition and terrain

```
View → Visualization Mode → World Editor
```

**Features:**
- 8m³ chunk grid overlay
- Terrain SDF brushes
- Entity placement tools
- Streaming boundary visualization

**Use Cases:**
- Level design
- Terrain sculpting
- Scene composition

---

## **Camera Controls**

| Action | Control |
|--------|---------|
| **Orbit** | Left-click drag |
| **Pan** | Middle-click drag or Shift+Left |
| **Zoom** | Scroll wheel |
| **Focus** | F key (fit selection) |
| **Reset** | Home key |
| **Fly Mode** | Right-click + WASD |

### Camera Presets

```
┌─────────────────────────────────────────┐
│ Camera: [Perspective ▼]                 │
├─────────────────────────────────────────┤
│  [Front] [Back] [Left] [Right]          │
│  [Top] [Bottom] [Iso]                   │
│                                         │
│  FOV: [60°]  Near: [0.01]  Far: [1000]  │
└─────────────────────────────────────────┘
```

---

## **Performance Metrics**

Real-time overlay (toggle with ` backtick):

```
┌─────────────────────────┐
│ FPS: 60.0 (16.6ms)      │
│ GPU: 8.2ms              │
│ CPU: 4.1ms              │
│                         │
│ Splats: 70,412          │
│ SDF Nodes: 24           │
│ Draw Calls: 3           │
│                         │
│ VRAM: 128 MB            │
│ RAM: 45 MB              │
└─────────────────────────┘
```

### Performance Targets

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| **Frame Time** | <16ms | 16-33ms | >33ms |
| **Splat Count** | <100k | 100-200k | >200k |
| **SDF Nodes** | <50 | 50-100 | >100 |
| **VRAM** | <256MB | 256-512MB | >512MB |

---

## **Gizmos & Tools**

### Transform Gizmo

```
     [T]ranslate  [R]otate  [S]cale
     
           ↑ Y
           │
           │
     ──────┼────── X →
          /│
         / │
        Z  
```

### Snapping

```
┌─────────────────────────────────────────┐
│ Snap Settings                           │
├─────────────────────────────────────────┤
│ ☑ Position: [0.1] m                     │
│ ☑ Rotation: [15] °                      │
│ ☑ Scale: [0.1]                          │
│ ☐ Snap to Grid                          │
│ ☐ Snap to Surface                       │
└─────────────────────────────────────────┘
```

---

## **Selection System**

### Selection Methods

- **Click:** Select single object
- **Shift+Click:** Add to selection
- **Ctrl+Click:** Toggle selection
- **Box Select:** Drag rectangle
- **Lasso:** Alt+Drag freeform

### Hierarchy Panel

```
┌─────────────────────────────────────────┐
│ Scene Hierarchy                         │
├─────────────────────────────────────────┤
│ ▼ AK-47                                 │
│   ├─ Receiver (geometry)                │
│   │   └─ Steel Material                 │
│   ├─ Stock (geometry)                   │
│   │   └─ Wood Material                  │
│   └─ Magazine (geometry)                │
│       └─ Polymer Material               │
└─────────────────────────────────────────┘
```

#### Tree Viewer

The hierarchy panel is implemented as a **Tree Viewer** driven by app state (chain). It is server-rendered via htmx: the same data source as the Card Chain strip (`_chain_slots` from the backend) is used to build the tree (Scene → geometry/splat/audio slots and their assets). No WASM snapshot is used; hierarchy and viewport both reflect the chain. When the chain loads or updates, the viewport is synced from the chain (each filled slot's binary is pushed to WASM). Clicking a node selects that card and updates the property editor. The tree loads with `hx-get="/api/assets/partials/tree"` and is refreshed when the chain changes (e.g. after adding/clearing a slot).

---

## **Integration with Card-Chain**

When a card changes, viewport updates in real-time:

```javascript
// Card selection triggers viewport update
function onCardChange(card) {
    if (card.type === 'geometry') {
        viewport.loadGeometry(card.library_id);
    } else if (card.type === 'material') {
        viewport.applyMaterial(card.target, card.material_id);
    }
    
    // Instant preview - no "apply" button needed
    viewport.render();
}
```

---

**Version:** 1.1  
**Last Updated:** January 26, 2026  
**Related:** [Rendering Pipeline](../engine/rendering-pipeline.md) | [Card-Chain](./forge-card-chain.md)
