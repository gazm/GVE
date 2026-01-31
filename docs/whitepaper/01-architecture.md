# 01 – Architecture

**Source:** [System Overview](../architecture/overview.md), [Engine API](../architecture/engine-api.md), [Codebase Structure](../architecture/codebase-structure.md)

---

## Tri-Layer Responsibilities

The system enforces a strict tri-layer separation: creation logic and heavy compute live in the Architect; the Engine assumes valid, binary-packed input and runs the simulation loop; the Forge provides the interface and runs the same engine via WASM for editor/game parity.

| Layer | Role | Stack | Location |
|-------|------|-------|----------|
| **Forge** | Interface: user intent, preview, tool interactions | HTML5, htmx, Rust WASM | `/tools/forge-ui` |
| **Architect** | Logic: AI, DB, compilation, baking | Python 3.11+, FastAPI, PyTorch, MongoDB | `/backend/architect` |
| **Engine** | Runtime: render, physics, audio | Rust, wgpu, rapier3d, cpal | `/engine/runtime` |

**Layer rules:**

- **Forge:** UI and preview only; no business logic or DB access.
- **Architect:** AI orchestration, compilation, DB; no real-time rendering.
- **Engine:** Simulation and rendering only; no HTTP or DB queries.

---

## Schema Generator (Rust as Source of Truth)

All cross-layer data types are defined once in Rust. Python (Pydantic) and TypeScript types are generated at build time via **typeshare** and **schemars**.

```
Rust structs (engine/shared/src/types/*.rs)
         │
         │ cargo build (typeshare + schemars)
         ├──────────────┬──────────────┐
         ▼              ▼              ▼
   Python Pydantic   TypeScript    JSON Schema
   (backend/         (forge-ui)    (schemas/)
    generated/)
```

Developers add or change fields in Rust; the build pipeline regenerates Python and TypeScript, eliminating manual sync and a large class of integration bugs. Generated files must not be edited by hand.

---

## Module Map

Codebase layout and AI context mapping:

| Directory | Context File | Scope |
|-----------|--------------|-------|
| `engine/shared/` | engine-shared.md | Cross-language type definitions |
| `engine/runtime/src/renderer/` | engine-renderer.md | wgpu rendering pipeline |
| `engine/wasm/` | engine-wasm.md | Browser WASM target |
| `backend/architect/src/api/` | architect-api.md | FastAPI routes |
| `backend/architect/src/compiler/` | architect-compiler.md | DNA to binary compilation |
| `backend/architect/src/librarian/` | architect-librarian.md | MongoDB wrapper |
| `backend/architect/src/ai_pipeline/` | architect-ai.md | LLM orchestration |
| `tools/forge-ui/` | forge-ui.md | htmx + WASM web UI |

**Ownership rules:**

1. **Librarian owns the database** — All MongoDB access goes through `librarian`.
2. **Shared owns the types** — Cross-language types live in `engine/shared`.
3. **Compiler owns DNA to binary** — No other module performs compilation.
4. **Renderer owns GPU draws** — No loading or physics inside the render loop.
5. **Binary protocol for WASM** — JS and WASM communicate via binary messages, not JSON.

Full path-to-context mapping: [.agent/modules/INDEX.md](../../.agent/modules/INDEX.md).

---

## Hardware Targets

| Metric | Desktop | Mobile/ARM | Notes |
|--------|---------|------------|-------|
| **Max splats** | 4,000,000+ | ~500,000 | Mobile uses shell visibility culling |
| **Terrain** | 1024³ voxel volume | 256³ voxel volume | Scales with RAM |
| **Audio** | 64 active voices (DSP) | 16 active voices (DSP) | Voice stealing on constrained devices |
| **Physics** | 5,000 rigid bodies | 500 rigid bodies | Rapier SIMD supports high counts on mobile |

---

[Back to Whitepaper](README.md) | Next: [02 – Rendering and Geometry](02-rendering-and-geometry.md)
