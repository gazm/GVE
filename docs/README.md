# GVE-1 Documentation

**Generative Volumetric Engine** - A hybrid volumetric rendering engine with procedural generation, SDF-based geometry, and Gaussian Splatting.

**Last Updated:** January 26, 2026  
**Project Version:** 1.6

---

## Quick Navigation

### 🏗️ [Architecture](./architecture)
High-level system design and principles
- **[System Overview](./architecture/overview.md)** - Complete system architecture, tri-layer design, rendering pipeline
- **[Engine API](./architecture/engine-api.md)** - Internal orchestration, schema sync, Librarian DAL
- **[Codebase Structure](./architecture/codebase-structure.md)** - Module breakdown, AI-friendly development


### ⚙️ [Core Systems](./core-systems)
Major subsystems and engines
- **[Audio System](./core-systems/audio-system.md)** - Resonance Engine, procedural audio synthesis
- **[Physics System](./core-systems/physics-system.md)** - Rapier3d integration, SDF collisions
- **[Rendering Pipeline](./core-systems/rendering-pipeline.md)** - Hybrid volumetric + splatting pipeline
- **[Math Library](./core-systems/math-library.md)** - Vector/matrix math, JSON schemas


### 🔄 [Workflows](./workflows)
Process pipelines and toolchains
- **[AI Pipeline](./workflows/ai-pipeline.md)** - Multi-track AI generation system
- **[Compiler Pipeline](./workflows/compiler-pipeline.md)** - DNA → Binary compilation process
- **[LiDAR Pipeline](./workflows/lidar-pipeline.md)** - Real-world scan integration

### 📊 [Data](./data)
Data formats, schemas, and storage
- **[Data Specifications](./data/data-specifications.md)** - JSON schemas, primitives, operators
- **[Database Architecture](./data/database-architecture.md)** - MongoDB + local cache system

### 🛠️ [Tools](./tools)
Development tools and editors
- **[Forge Editor](./tools/forge-editor.md)** - Web-based WASM editor with live preview

### 📖 [Guides](./guides)
How-to documentation and tutorials
- *Getting Started Guide* (Coming Soon)
- *Asset Creation Guide* (Coming Soon)

### [Technical Whitepaper](./whitepaper/README.md)
Single-document technical overview for engineers and researchers: architecture, rendering, compiler, AI, audio/physics, runtime, and references.

---

## Core Philosophy

**"Recipe over Asset"** - GVE-1 generates content procedurally from lightweight parametric recipes rather than storing heavy static assets.

### Key Innovations

- **Signed Distance Fields (SDFs)** - Geometry defined mathematically for infinite resolution
- **Gaussian Splatting** - Surface rendering without UVs using triplanar projection
- **Physics-Driven Audio** - Synthesized sound based on material properties and collision data
- **AI-Assisted Generation** - Multi-stage LLM pipeline for intelligent asset creation
- **Real-World Integration** - LiDAR-to-SDF pipeline for environment reconstruction

---

## System Architecture at a Glance

```
┌─────────────────────────────────────────┐
│  Layer 1: The Forge (Interface)        │
│  HTML5 + htmx + Rust WASM              │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│  Layer 2: The Architect (Logic)        │
│  Python + FastAPI + PyTorch + MongoDB  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│  Layer 3: The Engine (Runtime)         │
│  Rust + wgpu + rapier3d + cpal         │
└─────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | HTML5, htmx, Rust (WASM), wgpu |
| **Backend** | Python 3.11+, FastAPI, PyTorch |
| **Database** | MongoDB (with vector search) |
| **Runtime** | Rust, wgpu, rapier3d, cpal |
| **Schema** | typeshare, schemars (cross-language) |

---

## Document Status Legend

- ✅ **Complete** - Comprehensive documentation
- 🔄 **In Progress** - Being actively developed
- 📝 **Draft** - Initial outline created
- 🔮 **Planned** - Scheduled for creation

---

## Contributing to Documentation

When adding new documentation:

1. **Choose the right folder** based on topic
2. **Use descriptive filenames** in kebab-case (e.g., `audio-system.md`)
3. **Add metadata** at the top (Last Updated, Status, Related Docs)
4. **Update this README** with links to new documents
5. **Cross-reference** related documentation

---

## Getting Help

- **New to GVE?** Start with [System Overview](./architecture/overview.md)
- **Building assets?** Check [Data Specifications](./data/data-specifications.md)
- **Setting up tools?** See [Forge Editor](./tools/forge-editor.md)
- **Integrating systems?** Review [Workflows](./workflows)

---

*For questions or suggestions about documentation, please contact the development team.*
