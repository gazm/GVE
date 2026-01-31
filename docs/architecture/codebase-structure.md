# **GVE-1 Codebase Architecture**

**Role:** Development structure and module boundaries  
**Strategy:** Small modules (~300 lines) for AI-assisted development  
**Philosophy:** Clear boundaries prevent hallucinations, enable parallel work

**Version:** 1.0  
**Last Updated:** January 26, 2026

---

## **1. Monorepo Structure**

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

## **2. Module Breakdown**

### 2.1 Engine (Rust) — 23 Files

#### shared/ (~320 lines total)
| File | Lines | Purpose |
|------|-------|---------|
| `types/mod.rs` | 20 | Re-exports |
| `types/asset.rs` | 80 | AssetMetadata, AssetCategory |
| `types/material.rs` | 100 | MaterialSpec, ColorMode |
| `types/message.rs` | 80 | MessageType, MessageHeader |
| `binary_format.rs` | 150 | GVEBinaryHeader, offsets |
| `lib.rs` | 20 | Crate root |

#### runtime/renderer/ (~1000 lines total)
| File | Lines | Purpose |
|------|-------|---------|
| `mod.rs` | 20 | Re-exports |
| `pipeline.rs` | 200 | RenderPipeline setup |
| `shell_pass.rs` | 150 | Early-Z rasterization |
| `volume_pass.rs` | 200 | Raymarching |
| `splat_pass.rs` | 250 | Gaussian splatting |
| `lod.rs` | 100 | LOD transitions |

#### runtime/physics/ (~550 lines total)
| File | Lines | Purpose |
|------|-------|---------|
| `mod.rs` | 20 | Re-exports |
| `world.rs` | 150 | Rapier world wrapper |
| `sdf_collider.rs` | 200 | SdfShape implementation |
| `events.rs` | 100 | Collision callbacks |

#### runtime/audio/ (~750 lines total)
| File | Lines | Purpose |
|------|-------|---------|
| `mod.rs` | 20 | Re-exports |
| `engine.rs` | 150 | cpal initialization |
| `dsp_graph.rs` | 250 | Node graph execution |
| `oscillators.rs` | 200 | FM synthesis |
| `effects.rs` | 150 | Filters, reverb |

#### runtime/loader/ (~650 lines total)
| File | Lines | Purpose |
|------|-------|---------|
| `mod.rs` | 20 | Re-exports |
| `cache.rs` | 150 | Path resolution |
| `binary.rs` | 200 | .gve_bin parsing |
| `gpu_upload.rs` | 150 | Texture/buffer upload |

#### wasm/ (~450 lines total)
| File | Lines | Purpose |
|------|-------|---------|
| `lib.rs` | 100 | WASM entry points |
| `bindings.rs` | 150 | JS interop |
| `message_handler.rs` | 200 | Binary protocol |

---

### 2.2 Backend (Python) — 21 Files

#### api/ (~420 lines total)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 20 | Router aggregation |
| `assets.py` | 150 | CRUD endpoints |
| `compile.py` | 100 | Compile triggers |
| `websocket.py` | 150 | Event streaming |

#### compiler/ (~870 lines total)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 20 | Public API |
| `math_jit.py` | 250 | PyTorch SDF graph |
| `volume_bake.py` | 150 | 3D texture baking |
| `shell_gen.py` | 200 | Dual Contouring |
| `binary_writer.py` | 150 | .gve_bin output |
| `queue.py` | 100 | Priority scheduling |

#### splat_trainer/ (~670 lines total)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 20 | Public API |
| `sampler.py` | 150 | Poisson disk init |
| `optimizer.py` | 250 | Zero-layer training |
| `loss.py` | 150 | Surface/normal/overlap |
| `export.py` | 100 | Oklab → RGB conversion |

#### librarian/ (~520 lines total)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 20 | Public API |
| `assets.py` | 200 | Asset CRUD |
| `materials.py` | 150 | Material lookup |
| `cache.py` | 150 | Path resolution |

#### ai_pipeline/ (~670 lines total)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 20 | Public API |
| `orchestrator.py` | 200 | Multi-track flow |
| `vision.py` | 150 | Image → DNA |
| `pal.py` | 150 | Program-aided LLM |
| `rag.py` | 150 | Retrieval search |

---

### 2.3 Forge (Web) — 10 Files

#### static/ (~450 lines total)
| File | Lines | Purpose |
|------|-------|---------|
| `index.html` | 100 | Shell + htmx |
| `css/base.css` | 150 | Design tokens |
| `css/layout.css` | 100 | Grid/flex |
| `css/components.css` | 200 | Buttons, cards |

#### templates/ (~310 lines total)
| File | Lines | Purpose |
|------|-------|---------|
| `asset_list.html` | 50 | Asset browser |
| `asset_card.html` | 80 | Single asset |
| `property_editor.html` | 150 | Field editing |
| `progress_bar.html` | 30 | Compile status |

#### js/ (~300 lines total)
| File | Lines | Purpose |
|------|-------|---------|
| `viewport.js` | 200 | WASM canvas wrapper |
| `events.js` | 100 | WebSocket handling |

---

## **3. AI Context File Template**

Each module gets a `.agent/modules/{name}.md` file:

```markdown
# Module: {name}

## Scope
One-line description of what this module does.

## Files
- `file1.py` - Purpose
- `file2.py` - Purpose

## Public API
```python
from .module import function1, function2
```

## Dependencies
- `module_a` - Why
- `module_b` - Why

## Anti-Patterns
- ❌ Never do X
- ❌ Never import Y
```

---

## **4. Implementation Phases**

### Phase 1: Foundation
- [ ] Create directory structure
- [ ] Initialize `engine/shared` with type definitions
- [ ] Set up schema generation (typeshare → Python/TS)

### Phase 2: Backend Core
- [ ] Implement `librarian/` (MongoDB wrapper)
- [ ] Implement `compiler/` (DNA → binary)
- [ ] Implement `api/` (FastAPI routes)

### Phase 3: Engine Runtime
- [ ] Implement `loader/` (binary parsing)
- [ ] Implement `renderer/` (wgpu pipeline)
- [ ] Implement `physics/` (rapier3d)
- [ ] Implement `audio/` (cpal + DSP)

### Phase 4: Forge UI
- [ ] Build htmx templates
- [ ] Implement WASM viewport wrapper
- [ ] Connect WebSocket events

### Phase 5: Advanced
- [ ] Implement `splat_trainer/` (PyTorch)
- [ ] Implement `ai_pipeline/` (LLM orchestration)
- [ ] Build `.agent/modules/` context files

---

## **5. Schema Sync Flow**

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

## **6. Why ~300 Lines?**

| Benefit | Explanation |
|---------|-------------|
| **AI Context Fit** | Entire file fits in Claude/GPT context with room for imports |
| **Single Responsibility** | Each file does ONE thing well |
| **Easy Testing** | Mock one file, test another |
| **Parallel Work** | Different files = no merge conflicts |
| **Clear Ownership** | "Who owns oscillators.rs?" → Audio team |

---

**Related Docs:**
- [System Overview](./overview.md) - High-level architecture
- [Engine API](./engine-api.md) - Tri-layer communication
- [Compiler Pipeline](../workflows/compiler-pipeline.md) - DNA → binary process
