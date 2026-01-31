# 07 – References

**Version:** 1.6  
**Last Updated:** January 31, 2026

---

## Authoritative Documentation

Links below point to the canonical docs for each area. The whitepaper summarizes them; for implementation details and API specs, use these sources.

### Architecture

- [System Overview](../architecture/overview.md) — Tri-layer design, rendering pipeline, audio, hardware targets
- [Engine API](../architecture/engine-api.md) — Tri-layer communication, schema generator, WASM binary protocol, Librarian DAL
- [Codebase Structure](../architecture/codebase-structure.md) — Monorepo layout, module breakdown, schema sync flow
- [Codebase Structure (detailed)](../architecture/codebase-structure/) — AI context guide, module breakdowns

### Core Systems

- [Rendering Pipeline](../core-systems/rendering-pipeline.md) — Shell, volume, splat passes; LOD; terrain
- [Audio System](../core-systems/audio-system.md) — Resonance Engine, FM synthesis, DSP graph
- [Physics System](../core-systems/physics-system.md) — Rapier3d, SdfShape, collision strategy
- [Math Library](../core-systems/math-library.md) — Vector/matrix math, JSON schemas
- [Destruction System](../core-systems/destruction-system.md) — Destructible geometry and related systems

### Data

- [Data Specifications](../data/data-specifications.md) — JSON schemas, primitives, operators, Game Unit, binary format
- [Database Architecture](../data/database-architecture.md) — MongoDB and local cache
- [Material Database](../data/material-database.md) — Material specs and lookup

### Workflows

- [AI Pipeline](../workflows/ai-pipeline.md) — Multi-track orchestration, agents, RAG, validation
- [Compiler Pipeline](../workflows/compiler-pipeline.md) — DNA to binary stages, Math JIT, baking, export
- [LiDAR Pipeline](../workflows/lidar-pipeline.md) — Real-world scan to SDF
- [Schema Sync](../workflows/schema-sync.md) — Rust to Python/TypeScript generation

### Tools

- [Forge Editor](../tools/forge-editor.md) — Web-based WASM editor, quick start, architecture
- [Forge Viewport](../tools/forge-viewport.md) — WASM viewport and visualization modes
- [Forge Card Chain](../tools/forge-card-chain.md) — Asset assembly workflow
- [Forge Libraries](../tools/forge-libraries.md) — Component libraries
- [Forge Property Editor](../tools/forge-property-editor.md) — Property editing
- [Forge World Editor](../tools/forge-world-editor.md) — Terrain and level composition

### Improvements and Strategy

- [Splat Strategy](../improvements/splat-strategy.md) — Gaussian splats as primary surface representation, metrics, optimization

---

## Whitepaper Index

- [Whitepaper README](README.md)
- [01 – Architecture](01-architecture.md)
- [02 – Rendering and Geometry](02-rendering-and-geometry.md)
- [03 – Compiler and Data](03-compiler-and-data.md)
- [04 – AI and Workflows](04-ai-and-workflows.md)
- [05 – Audio and Physics](05-audio-and-physics.md)
- [06 – Runtime and Tooling](06-runtime-and-tooling.md)

---

[Back to Whitepaper](README.md)
