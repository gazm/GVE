# 06 – Runtime and Tooling

**Source:** [Engine API](../architecture/engine-api.md), [Forge Editor](../tools/forge-editor.md), [System Overview](../architecture/overview.md)

---

## Engine Runtime

The Engine (**Layer 3**) assumes all input is valid, optimized, and binary-packed. It does not perform HTTP, DB, or file discovery; it runs the simulation loop as fast as possible.

- **Render loop** — Shell rasterization → raymarching (LOD-based) → splat sort and draw. See [02 – Rendering and Geometry](02-rendering-and-geometry.md).
- **Physics loop** — Rapier3d rigid-body step and SdfShape collision queries against math bytecode.
- **Audio loop** — DSP graph (Resonance Engine) runs on a dedicated high-priority thread for glitch-free synthesis.

Data enters as `.gve_bin` (or equivalent binary messages); the loader parses and uploads to GPU/CPU structures. No JSON is parsed in the hot path.

---

## WASM and Binary Protocol

The same Engine code is compiled to **WebAssembly** for the Forge so the browser viewport runs the real runtime, not a JavaScript stand-in. Editor and game use the same code path for rendering and physics.

**Communication:** JavaScript and WASM exchange data via a **binary protocol**, not JSON. Message headers carry type and version; payloads are packed binary. This keeps parsing cheap and avoids schema drift at the boundary. Details: [Engine API](../architecture/engine-api.md) (WASM Binary Messaging Protocol).

---

## Forge (Tooling)

The **Forge** (`/tools/forge-ui`) is the web-based editor:

- **Stack:** HTML5, htmx (server-driven UI), Rust WASM viewport.
- **Role:** Captures user intent (prompts, component choices, property edits), shows live preview via the WASM engine, and talks to the Architect over REST (and WebSocket where used). It does not own business logic or DB access; the Architect handles AI, compilation, and storage.
- **Docs:** Full tool coverage (viewport, card-chain, libraries, property editor, world editor, etc.) is in [Forge Editor](../tools/forge-editor.md) and the [Tools](../tools/README.md) section (forge-viewport, forge-card-chain, forge-libraries, forge-property-editor, forge-world-editor, etc.).

---

[Back to Whitepaper](README.md) | Prev: [05 – Audio and Physics](05-audio-and-physics.md) | Next: [07 – References](07-references.md)
