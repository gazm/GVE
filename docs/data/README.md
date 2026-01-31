# Data Documentation

Data formats, schemas, and storage systems.

## Documents

### [Data Specifications](./data-specifications.md) ✅
**Definitive guide to data structures, APIs, and schemas.**

**Key Topics:**
- Global standards (Game Unit = 1 meter)
- Meta-language: Variables ($) and Anchors (@)
- Geometry primitives (Box, Sphere, Cylinder, etc.)
- Advanced primitives (Polyhedron, Bezier, Heightmap)
- Machining macros (Bore, Slot, Chamfer, Array)
- Generative macros (Scatter, Truss)
- Domain modifiers (Twist, Bend, Noise, etc.)
- Texture & surface modifiers
- Audio modifiers
- GVE-JSON schema (intermediate format)
- GVE-BIN schema (runtime format)

---

### [Database Architecture](./database-architecture.md) ✅
**MongoDB + local cache system.**

**Key Topics:**
- Document model (DNA, Entity, RAG metadata)
- Indexing strategy (text, vector, faceted)
- File system cache hierarchy
- The Librarian module (DAL)
- Vectorization pipeline
- Cache invalidation
- API endpoints

---

### Material Library 🔮
*(Planned)* - Standard materials and PBR properties.

**Will Cover:**
- ASTM/AMS material specifications
- PBR property definitions
- Material validation rules
- Standard material library
- Custom material creation

---

### API Reference 🔮
*(Planned)* - Complete REST/WebSocket API documentation.

**Will Cover:**
- Forge ↔ Architect API
- Authentication/authorization
- Endpoint specifications
- Request/response schemas
- WebSocket real-time updates
- Rate limiting
- Error codes

---

## Related Documentation

- **Architecture:** [/architecture](../architecture) - Why these formats were chosen
- **Workflows:** [/workflows](../workflows) - How data flows through pipelines
- **Core Systems:** [/core-systems](../core-systems) - How data is consumed at runtime

[← Back to Documentation Home](../README.md)
