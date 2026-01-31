# GVE-1 Technical Whitepaper

**Generative Volumetric Engine (GVE-1)** — Recipe over Asset.

**Version:** 1.6  
**Last Updated:** January 31, 2026

---

## Summary

GVE-1 is a generative game engine that replaces the traditional polygon-mesh pipeline with a **hybrid volumetric and Gaussian-splatting** architecture. Geometry is defined by Signed Distance Fields (SDFs) from parametric recipes; surfaces are rendered via quantized splats with triplanar-style projection, and physics and audio are driven by the same mathematical primitives. The system is organized in three layers: **The Forge** (web UI + Rust WASM), **The Architect** (Python/FastAPI/PyTorch/MongoDB for AI, compilation, and storage), and **The Engine** (Rust/wgpu/rapier3d/cpal runtime). This whitepaper describes the architecture, rendering, compiler, AI workflows, audio/physics, and runtime for engineers and researchers.

---

## Whitepaper Sections

| Section | Description |
|---------|-------------|
| [01 – Architecture](01-architecture.md) | Tri-layer topology, schema generator, module map, hardware targets |
| [02 – Rendering and Geometry](02-rendering-and-geometry.md) | Shell and volume pipeline, SDF LOD, surface strategy, terrain |
| [03 – Compiler and Data](03-compiler-and-data.md) | Compiler pipeline stages, DNA data model, binary format |
| [04 – AI and Workflows](04-ai-and-workflows.md) | AI pipeline, multi-track orchestration, LiDAR and schema sync |
| [05 – Audio and Physics](05-audio-and-physics.md) | Resonance Engine, DSP path, Rapier3d and SDF colliders |
| [06 – Runtime and Tooling](06-runtime-and-tooling.md) | Engine loops, WASM and binary protocol, Forge UI |
| [07 – References](07-references.md) | Links to authoritative docs and version info |

---

## System Architecture at a Glance

```
┌─────────────────────────────────────────┐
│  Layer 1: The Forge (Interface)          │
│  HTML5 + htmx + Rust WASM                 │
└─────────────────┬─────────────────────────┘
                  │
┌─────────────────▼─────────────────────────┐
│  Layer 2: The Architect (Logic)           │
│  Python + FastAPI + PyTorch + MongoDB     │
└─────────────────┬─────────────────────────┘
                  │
┌─────────────────▼─────────────────────────┐
│  Layer 3: The Engine (Runtime)            │
│  Rust + wgpu + rapier3d + cpal            │
└───────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | HTML5, htmx, Rust (WASM), wgpu |
| **Backend** | Python 3.11+, FastAPI, PyTorch |
| **Database** | MongoDB (with vector search) |
| **Runtime** | Rust, wgpu, rapier3d, cpal |
| **Schema** | typeshare, schemars (cross-language) |

---

For full system design and implementation details, see the [main documentation](../README.md) and the section files above.
