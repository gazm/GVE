# Architecture Documentation

High-level system design, principles, and technology decisions.

## Documents

### [System Overview](./overview.md) ✅
**Complete system architecture** covering the tri-layer design (Forge/Architect/Engine), hybrid rendering pipeline, audio architecture, physics integration, and hardware targets.

**Key Topics:**
- Tri-layered architecture (Interface/Logic/Runtime)
- Cross-cutting schema system
- Hybrid volumetric + splatting rendering
- LOD system (Math Tree / 3D Texture / Splats)
- Terrain voxel repacking
- Hardware performance targets

---

### [Engine API](./engine-api.md) ✅
**Internal orchestration layer** connecting MongoDB, Rust Engine, and JSON translation.

**Key Topics:**
- Tri-layer communication patterns
- Schema Generator (Rust → Python/TypeScript)
- Material Specification API (ASTM/AMS lookup)
- Librarian (MongoDB DAL, cache management)
- Compiler Interface (DNA JSON → GVE Binary)
- Runtime API (asset loading, hot-reload)
- Event Bus (cross-layer WebSocket events)

---

### Design Principles 📝
*(Planned)* - Core philosophy, architectural patterns, and design decisions extracted from various documents.

**Will Cover:**
- "Recipe over Asset" philosophy
- Compile-time vs runtime trade-offs
- Type safety through schema generation
- Separation of concerns
- Progressive enhancement strategies

---

### Technology Stack 📝
*(Planned)* - Technology choices, version requirements, and justifications.

**Will Cover:**
- Frontend: HTML5, htmx, Rust WASM, wgpu
- Backend: Python, FastAPI, PyTorch
- Database: MongoDB with vector search
- Runtime: Rust, rapier3d, cpal
- Dependency management
- Alternative technologies considered

---

## Related Documentation

- **Core Systems:** [/core-systems](../core-systems) - Implementation details for subsystems
- **Workflows:** [/workflows](../workflows) - Process pipelines and compilation
- **Data:** [/data](../data) - Data formats and schemas

[← Back to Documentation Home](../README.md)
