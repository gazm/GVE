# **Module Breakdowns**

Detailed file-by-file breakdown for each layer. All modules target ~300 lines max.

**Last Updated:** January 26, 2026

---

## **Engine (Rust)**

Total: 23 files, ~3,870 lines

### shared/ (~320 lines)

Source of truth for all cross-language types.

| File | Lines | Purpose | Reference |
|------|-------|---------|-----------|
| `types/mod.rs` | 20 | Re-exports | — |
| `types/asset.rs` | 80 | AssetMetadata, AssetCategory | [Engine API §2](../engine-api.md#2-schema-generator) |
| `types/material.rs` | 100 | MaterialSpec, ColorMode | [Engine API §3](../engine-api.md#3-material-specification-api) |
| `types/message.rs` | 80 | MessageType, MessageHeader | [Engine API §8](../engine-api.md#8-wasm-binary-messaging-protocol) |
| `binary_format.rs` | 150 | GVEBinaryHeader, offsets | [Data Specs §3](../../data/data-specifications.md) |
| `lib.rs` | 20 | Crate root | — |

### runtime/renderer/ (~1,000 lines)

wgpu rendering pipeline.

| File | Lines | Purpose | Reference |
|------|-------|---------|-----------|
| `mod.rs` | 20 | Re-exports | — |
| `pipeline.rs` | 200 | RenderPipeline setup | [Rendering Pipeline](../../core-systems/rendering-pipeline.md) |
| `shell_pass.rs` | 150 | Early-Z rasterization | [Overview §3.1](../overview.md#31-the-geometry-pipeline-lod-system) |
| `volume_pass.rs` | 200 | Raymarching | [Overview §3.1](../overview.md#31-the-geometry-pipeline-lod-system) |
| `splat_pass.rs` | 250 | Gaussian splatting | [Overview §3.2](../overview.md#32-the-visual-pipeline-splat-culling) |
| `lod.rs` | 100 | LOD transitions | [Rendering Pipeline](../../core-systems/rendering-pipeline.md) |

### runtime/physics/ (~550 lines)

Rapier3d integration with SDF collision.

| File | Lines | Purpose | Reference |
|------|-------|---------|-----------|
| `mod.rs` | 20 | Re-exports | — |
| `world.rs` | 150 | Rapier world wrapper | [Physics System](../../core-systems/physics-system.md) |
| `sdf_collider.rs` | 200 | SdfShape implementation | [Physics System §SDF](../../core-systems/physics-system.md) |
| `events.rs` | 100 | Collision callbacks | [Audio System §Trigger](../../core-systems/audio-system.md) |

### runtime/audio/ (~750 lines)

cpal + DSP synthesis.

| File | Lines | Purpose | Reference |
|------|-------|---------|-----------|
| `mod.rs` | 20 | Re-exports | — |
| `engine.rs` | 150 | cpal initialization | [Audio System](../../core-systems/audio-system.md) |
| `dsp_graph.rs` | 250 | Node graph execution | [Audio System §DSP](../../core-systems/audio-system.md) |
| `oscillators.rs` | 200 | FM synthesis | [Audio System §FM](../../core-systems/audio-system.md) |
| `effects.rs` | 150 | Filters, reverb | [Audio System §FX](../../core-systems/audio-system.md) |

### runtime/loader/ (~650 lines)

Asset loading and GPU upload.

| File | Lines | Purpose | Reference |
|------|-------|---------|-----------|
| `mod.rs` | 20 | Re-exports | — |
| `cache.rs` | 150 | Path resolution | [Engine API §4](../engine-api.md#4-librarian-api) |
| `binary.rs` | 200 | .gve_bin parsing | [Data Specs §3](../../data/data-specifications.md) |
| `gpu_upload.rs` | 150 | Texture/buffer upload | [Engine API §6](../engine-api.md#6-runtime-api) |

### wasm/ (~450 lines)

Browser build target.

| File | Lines | Purpose | Reference |
|------|-------|---------|-----------|
| `lib.rs` | 100 | WASM entry points | [Forge Viewport](../../tools/forge-viewport.md) |
| `bindings.rs` | 150 | JS interop | [Forge Viewport](../../tools/forge-viewport.md) |
| `message_handler.rs` | 200 | Binary protocol | [Engine API §8](../engine-api.md#8-wasm-binary-messaging-protocol) |

---

## **Backend (Python)**

Total: 21 files, ~3,150 lines

### api/ (~420 lines)

FastAPI routes.

| File | Lines | Purpose | Reference |
|------|-------|---------|-----------|
| `__init__.py` | 20 | Router aggregation | — |
| `assets.py` | 150 | CRUD endpoints | [Engine API §4](../engine-api.md#4-librarian-api) |
| `compile.py` | 100 | Compile triggers | [Engine API §5](../engine-api.md#5-compiler-interface) |
| `websocket.py` | 150 | Event streaming | [Engine API §7](../engine-api.md#7-event-bus) |

### compiler/ (~870 lines)

DNA → binary compilation.

| File | Lines | Purpose | Reference |
|------|-------|---------|-----------|
| `__init__.py` | 20 | Public API | — |
| `math_jit.py` | 250 | PyTorch SDF graph | [Compiler §1](../../workflows/compiler-pipeline.md#stage-1-math-jit-tensor-core) |
| `volume_bake.py` | 150 | 3D texture baking | [Compiler §2.1](../../workflows/compiler-pipeline.md#21-volume-baking-math--texture) |
| `shell_gen.py` | 200 | Dual Contouring | [Compiler §2.2](../../workflows/compiler-pipeline.md#22-shell-generation-dual-contouring) |
| `binary_writer.py` | 150 | .gve_bin output | [Data Specs §3](../../data/data-specifications.md) |
| `queue.py` | 100 | Priority scheduling | [Engine API §5](../engine-api.md#5-compiler-interface) |

### splat_trainer/ (~670 lines)

PyTorch splat optimization.

| File | Lines | Purpose | Reference |
|------|-------|---------|-----------|
| `__init__.py` | 20 | Public API | — |
| `sampler.py` | 150 | Poisson disk init | [Compiler §3.1](../../workflows/compiler-pipeline.md#31-initialization-poisson-disk-sampling) |
| `optimizer.py` | 250 | Zero-layer training | [Compiler §3.2](../../workflows/compiler-pipeline.md#32-zero-layer-training-with-oklab-colors) |
| `loss.py` | 150 | Surface/normal/overlap | [Compiler §3.2](../../workflows/compiler-pipeline.md#32-zero-layer-training-with-oklab-colors) |
| `export.py` | 100 | Oklab → RGB | [Compiler §3.2](../../workflows/compiler-pipeline.md#32-zero-layer-training-with-oklab-colors) |

### librarian/ (~520 lines)

MongoDB wrapper.

| File | Lines | Purpose | Reference |
|------|-------|---------|-----------|
| `__init__.py` | 20 | Public API | — |
| `assets.py` | 200 | Asset CRUD | [Engine API §4](../engine-api.md#4-librarian-api) |
| `materials.py` | 150 | Material lookup | [Engine API §3](../engine-api.md#3-material-specification-api) |
| `cache.py` | 150 | Path resolution | [Database Architecture](../../data/database-architecture.md) |

### ai_pipeline/ (~670 lines)

LLM orchestration.

| File | Lines | Purpose | Reference |
|------|-------|---------|-----------|
| `__init__.py` | 20 | Public API | — |
| `orchestrator.py` | 200 | Multi-track flow | [AI Pipeline](../../workflows/ai-pipeline.md) |
| `vision.py` | 150 | Image → DNA | [AI Pipeline](../../workflows/ai-pipeline.md) |
| `pal.py` | 150 | Program-aided LLM | [AI Pipeline](../../workflows/ai-pipeline.md) |
| `rag.py` | 150 | Retrieval search | [Compiler §4](../../workflows/compiler-pipeline.md#stage-4-semantic-indexing-rag) |

---

## **Forge (Web)**

Total: 10 files, ~1,060 lines

### static/ (~450 lines)

HTML + CSS.

| File | Lines | Purpose | Reference |
|------|-------|---------|-----------|
| `index.html` | 100 | Shell + htmx | [Forge Editor](../../tools/forge-editor.md) |
| `css/base.css` | 150 | Design tokens | — |
| `css/layout.css` | 100 | Grid/flex | — |
| `css/components.css` | 200 | Buttons, cards | [Forge Card-Chain](../../tools/forge-card-chain.md) |

### templates/ (~310 lines)

htmx partials.

| File | Lines | Purpose | Reference |
|------|-------|---------|-----------|
| `asset_list.html` | 50 | Asset browser | [Forge Libraries](../../tools/forge-libraries.md) |
| `asset_card.html` | 80 | Single asset | [Forge Card-Chain](../../tools/forge-card-chain.md) |
| `property_editor.html` | 150 | Field editing | [Forge Property Editor](../../tools/forge-property-editor.md) |
| `progress_bar.html` | 30 | Compile status | [Engine API §7](../engine-api.md#7-event-bus) |

### js/ (~300 lines)

JavaScript wrapper.

| File | Lines | Purpose | Reference |
|------|-------|---------|-----------|
| `viewport.js` | 200 | WASM canvas wrapper | [Forge Viewport](../../tools/forge-viewport.md) |
| `events.js` | 100 | WebSocket handling | [Engine API §7](../engine-api.md#7-event-bus) |

---

**See Also:**
- [Overview](./overview.md) - Monorepo layout, phases
- [AI Context Guide](./ai-context-guide.md) - Context file templates
