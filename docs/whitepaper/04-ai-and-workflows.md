# 04 – AI and Workflows

**Source:** [AI Pipeline](../workflows/ai-pipeline.md), [Compiler Pipeline](../workflows/compiler-pipeline.md), [Schema Sync](../workflows/schema-sync.md)

---

## AI Pipeline Role

The AI pipeline turns natural-language prompts into valid GVE-1 assets (`.gve_bin`). It runs in **Layer 2 (The Architect)** and feeds DNA JSON into the compiler, which produces binaries for the Engine and Forge.

**Flow:** User prompt → AI Pipeline (orchestrator, RAG, agents) → DNA JSON → Compiler Pipeline → `.gve_bin` → Engine/Forge.

---

## Orchestration

**Principle:** Split generation into small, verifiable steps instead of one large LLM call.

- **Single-responsibility agents** — Each agent has one domain (geometry, materials, terrain, audio). Reduces tokens and improves reliability; allows parallel execution where possible.
- **Constrained output** — Agents output structured JSON (or code) validated by JSON schema, not freeform text.
- **Hierarchical planning** — An orchestrator (meta-agent) uses a track router to classify intent, then runs track-specific pipelines:
  - **Track A (Matter):** Blacksmith (form) → Machinist (function) → Artist (surface) → Vision-Critic validator.
  - **Track B (Landscape):** Geologist → Terraformer → Ecologist.
  - **Track C (Audio):** Analyst (optional) → Composer → Sound Designer.
- **State management** — Generation state holds prompt, selected track, RAG context, per-stage outputs, and validation history; retries are bounded.

RAG injects API specs and examples so agents use exact constants (e.g. material IDs, primitives) instead of guessing.

---

## Other Workflows

- **LiDAR ingestion** — Raw LiDAR is converted into game-ready SDFs (e.g. chunking, halo sampling for blending). Handled in the Architect; see [LiDAR Pipeline](../workflows/lidar-pipeline.md).
- **Schema sync** — Rust structs in `engine/shared` are the source of truth; `cargo build` with typeshare and schemars regenerates Python and TypeScript. See [Schema Sync](../workflows/schema-sync.md) and [Engine API](../architecture/engine-api.md).

---

[Back to Whitepaper](README.md) | Prev: [03 – Compiler and Data](03-compiler-and-data.md) | Next: [05 – Audio and Physics](05-audio-and-physics.md)
