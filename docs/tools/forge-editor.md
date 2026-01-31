# **GVE-1 Tooling: "The Iron Forge"**

**Architecture:** Server-Driven UI (Hypermedia) + Rust WASM Core

**Philosophy:** The Forge acts as a "Remote Control" for the backend logic and a "Window" directly into the runtime engine. By coupling htmx for state management with a compiled Rust WASM module for the viewport, we ensure zero discrepancy between the tool's preview and the final game.

---

## **Documentation Structure**

### 📦 Core Concepts
- **[Viewport & Rendering](./forge-viewport.md)** - Rust WASM core, visualization modes
- **[Card-Chain Workflow](./forge-card-chain.md)** - Modular asset assembly system
- **[Component Libraries](./forge-libraries.md)** - Geometry, materials, textures, audio, recipes

### 🎨 Asset Creation
- **[Texture System](./texture-library-implementation.md)** - Upload, generate, tag-based library
- **[Material Database](../data/material-database.md)** - Physical properties for 27+ materials
- **[AI Pipeline Integration](../workflows/ai-pipeline.md)** - How AI generates components

### 🌍 World Editor
- **[Terrain Tools](./forge-world-editor.md)** - SDF brushes, material painting, LiDAR import
- **[Level Composition](./forge-world-editor.md#scene-composition)** - Entity placement, chunks

### 🔧 Advanced Tools
- **[Property Table Editor](./forge-property-editor.md)** - Type-aware struct editing with sliders, dropdowns, color pickers
- **[Modifier Stack](./forge-modifiers.md)** - Twist, bend, round, facet operations
- **[Splat Refinement](./forge-splat-refinement.md)** - Training controls, loss graphs
- **[Audio Inspector](./forge-audio.md)** - Spectrogram, physics-driven synthesis

---

## **Quick Start**

**Create Your First Asset (Simple Chair):**

1. Click **[+ New Asset]**
2. Select Type: **Prop**
3. Enter Prompt: `"chair"`
4. Select Texture: `"oak"` *(auto-suggests Oak Wood texture)*
5. Click **[Generate]**

**Result:**
- Cost: $0.04
- Time: ~4 seconds
- Output: Dining chair with oak texture

**Next Chair:** FREE & Instant (uses library components)

---

## **System Architecture Overview**

```
┌─────────────────────────────────────────┐
│  The Forge (HTML5 + htmx + Rust WASM)  │
│  - Card-chain UI                        │
│  - Component library browser            │
│  - Live WASM viewport                   │
└──────────────┬──────────────────────────┘
               ↓ (REST API)
┌─────────────────────────────────────────┐
│  The Architect (Python + FastAPI)      │
│  - AI Pipeline orchestration            │
│  - Component library (MongoDB)          │
│  - Compiler (DNA → .gve_bin)            │
└──────────────┬──────────────────────────┘
               ↓ (.gve_bin files)
┌─────────────────────────────────────────┐
│  The Engine (Rust + wgpu + rapier3d)   │
│  - Runtime rendering                    │
│  - Physics simulation                   │
│  - Audio synthesis                      │
└─────────────────────────────────────────┘
```

---

## **Key Features**

### 🎯 **Card-Chain Assembly** → [Learn More](./forge-card-chain.md)
Build assets from reusable components instead of regenerating everything:
- **75% cost reduction** on average
- **80% faster** iteration
- **100% reusability** once components are cached

### 📚 **Component Libraries** → [Learn More](./forge-libraries.md)
Five interconnected libraries:
- **Geometry:** 3D shapes (SDF trees)
- **Materials:** Physical specs (ASTM/AMS standards)
- **Textures:** PBR maps (2K/4K, BC7 compressed)
- **Audio:** DSP patches (FM synthesis)
- **Recipes:** Complete asset templates

### 🏷️ **Tag-Based Search** → [Learn More](./forge-libraries.md#tag-system)
Smart filtering with auto-suggestions:
```
Tags: metal, worn, rust → Suggests: Rusty Steel ★★★★
Tags: wood, natural → Suggests: Oak Wood ★★★★
```

### 🤖 **AI Integration** → [Learn More](../workflows/ai-pipeline.md)
Only generate what's missing:
- Library components: $0, instant
- AI-generated: $0.02-0.05 per component
- Hybrid approach: Best of both worlds

---

## **Viewport Modes**

| Mode | Purpose | Key Features |
|------|---------|-------------|
| **SDF Source** | Raw math visualization | Infinite resolution, blend operations |
| **Shell Mesh** | Optimization check | Early-Z culling, clipping detection |
| **Splat View** | Final render preview | What you see = what players see |
| **Physics Debug** | Collision testing | Rapier3d boundaries, constraints |
| **Audio Viz** | Sound properties | Resonance frequencies, occlusion |
| **World Editor** | Level composition | Terrain, entities, chunks |

---

## **Workflow Examples**

### Example 1: Weapon (AK-47)
**Input:** Prompt "AK-47", Style "worn"

**Card Chain:**
1. Receiver geometry (library) - $0
2. Material: Rusty steel (library) - $0
3. Stock geometry (AI generate) - $0.03
4. Material: Worn oak (library) - $0

**Total:** $0.03, ~3 seconds, 75% cached

---

### Example 2: Furniture (Leather Chair)
**Input:** Prompt "chair", Texture "leather"

**Card Chain:**
1. Chair frame (library - from previous oak chair) - $0
2. Material: Leather texture (AI generate) - $0.02

**Total:** $0.02, ~2 seconds, 50% cached

---

### Example 3: Vehicle (Military Jeep)
**Input:** Recipe "Military Jeep (Worn)"

**Card Chain:** 12 cards, all from library

**Total:** $0, instant, 100% cached

---

## **Getting Started**

1. **Browse Examples** → [Recipe Library](./forge-libraries.md#recipe-library)
2. **Learn Card System** → [Card-Chain Guide](./forge-card-chain.md)
3. **Explore Materials** → [Material Database](../data/material-database.md)
4. **Understand AI** → [AI Pipeline](../workflows/ai-pipeline.md)

---

## **Technical Deep Dives**

### For Artists:
- [Card-Chain Workflow](./forge-card-chain.md) - Visual asset assembly
- [Texture Management](./texture-library-implementation.md) - Upload, generate, organize
- [World Editor Tools](./forge-world-editor.md) - Terrain sculpting, level design

### For Developers:
- [WASM Viewport](./forge-viewport.md) - Rust architecture, shared buffers
- [Backend API](./texture-library-implementation.md#backend-api) - REST endpoints
- [Compiler Integration](../workflows/compiler-pipeline.md) - DNA → Binary pipeline

### For Technical Artists:
- [Modifier Stack](./forge-modifiers.md) - Procedural operations
- [Splat Training](./forge-splat-refinement.md) - Hyperparameter tuning
- [Material System](../data/material-database.md) - PBR properties, physics

---

**Version:** 2.0  
**Last Updated:** January 25, 2026  
**Status:** ✅ Production Ready