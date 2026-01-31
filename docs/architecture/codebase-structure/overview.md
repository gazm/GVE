# **Codebase Structure Overview**

**Role:** Development structure and module boundaries  
**Strategy:** Small modules (~300 lines) for AI-assisted development  
**Philosophy:** Clear boundaries prevent hallucinations, enable parallel work

**Version:** 1.0  
**Last Updated:** January 26, 2026

---

## **Monorepo Layout**

```
gve/
├── .agent/modules/              # AI context files
│   ├── engine-shared.md
│   ├── engine-renderer.md
│   ├── engine-physics.md
│   ├── engine-audio.md
│   ├── architect-compiler.md
│   ├── architect-librarian.md
│   ├── architect-ai.md
│   └── forge-viewport.md
│
├── engine/                      # Layer 3: Rust Runtime
│   ├── shared/                  # Source of truth for types
│   ├── runtime/                 # wgpu + rapier3d + cpal
│   └── wasm/                    # Browser build target
│
├── backend/                     # Layer 2: Python Architect
│   └── architect/
│       ├── generated/           # Auto-generated from Rust
│       └── src/
│           ├── api/             # FastAPI routes
│           ├── compiler/        # DNA → binary
│           ├── splat_trainer/   # PyTorch optimization
│           ├── librarian/       # MongoDB wrapper
│           └── ai_pipeline/     # LLM orchestration
│
├── tools/                       # Layer 1: Forge UI
│   └── forge-ui/
│       ├── static/              # HTML + CSS
│       ├── templates/           # htmx partials
│       └── js/                  # WASM wrapper
│
└── docs/                        # Documentation
```

---

## **Implementation Phases**

### Phase 1: Foundation
- [ ] Create directory structure
- [ ] Initialize `engine/shared` with type definitions
- [ ] Set up schema generation (typeshare → Python/TS)

**Related Docs:**
- [Engine API §2](../engine-api.md#2-schema-generator) - Schema Generator details
- [Data Specifications](../../data/data-specifications.md) - JSON schemas

### Phase 2: Backend Core
- [ ] Implement `librarian/` (MongoDB wrapper)
- [ ] Implement `compiler/` (DNA → binary)
- [ ] Implement `api/` (FastAPI routes)

**Related Docs:**
- [Engine API §4](../engine-api.md#4-librarian-api) - Librarian API
- [Compiler Pipeline](../../workflows/compiler-pipeline.md) - Full compilation process
- [Database Architecture](../../data/database-architecture.md) - MongoDB setup

### Phase 3: Engine Runtime
- [ ] Implement `loader/` (binary parsing)
- [ ] Implement `renderer/` (wgpu pipeline)
- [ ] Implement `physics/` (rapier3d)
- [ ] Implement `audio/` (cpal + DSP)

**Related Docs:**
- [Rendering Pipeline](../../core-systems/rendering-pipeline.md) - Shell & Volume rendering
- [Physics System](../../core-systems/physics-system.md) - Rapier3d + SDF collisions
- [Audio System](../../core-systems/audio-system.md) - Resonance Engine

### Phase 4: Forge UI
- [ ] Build htmx templates
- [ ] Implement WASM viewport wrapper
- [ ] Connect WebSocket events

**Related Docs:**
- [Forge Editor](../../tools/forge-editor.md) - UI overview
- [Forge Viewport](../../tools/forge-viewport.md) - WASM integration
- [Engine API §8](../engine-api.md#8-wasm-binary-messaging-protocol) - Binary messaging

### Phase 5: Advanced
- [ ] Implement `splat_trainer/` (PyTorch)
- [ ] Implement `ai_pipeline/` (LLM orchestration)
- [ ] Build `.agent/modules/` context files

**Related Docs:**
- [Compiler Pipeline §3](../../workflows/compiler-pipeline.md#stage-3-splat-refinement-neural-training) - Splat training
- [AI Pipeline](../../workflows/ai-pipeline.md) - LLM orchestration

---

## **Schema Sync Flow**

```
┌─────────────────────────────────────────────────────────────┐
│  engine/shared/src/types/*.rs                               │
│  (Source of Truth)                                          │
└────────────────────────┬────────────────────────────────────┘
                         │ cargo build (typeshare + schemars)
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │ Python    │   │ TypeScript│   │ JSON      │
   │ Pydantic  │   │ Types     │   │ Schema    │
   │ backend/  │   │ tools/    │   │ schemas/  │
   │ generated │   │ forge-ui  │   │           │
   └───────────┘   └───────────┘   └───────────┘
```

> [!IMPORTANT]
> Never manually edit generated files. All changes flow from Rust → Python/TypeScript.

---

## **Why ~300 Lines?**

| Benefit | Explanation |
|---------|-------------|
| **AI Context Fit** | Entire file fits in Claude/GPT context with room for imports |
| **Single Responsibility** | Each file does ONE thing well |
| **Easy Testing** | Mock one file, test another |
| **Parallel Work** | Different files = no merge conflicts |
| **Clear Ownership** | "Who owns oscillators.rs?" → Audio team |

---

**See Also:**
- [Module Breakdowns](./module-breakdowns.md) - Detailed file tables
- [AI Context Guide](./ai-context-guide.md) - Context file templates
