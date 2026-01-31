# **GVE-1 System Architecture**

**Version:** 1.6 (Unified Stack)

**Philosophy:** "Recipe over Asset" — Procedural generation via Signed Distance Fields (SDF) and Engineering-Grade Materials.

## **1\. Executive Summary**

GVE-1 is a generative game engine that fundamentally abandons the traditional polygon-mesh pipeline in favor of a **Hybrid Volumetric \+ Splatting** architecture. Unlike standard engines that rely on static, memory-heavy assets (FBX/PNG), GVE-1 relies on lightweight, parametric recipes.

* **Geometry:** Defined by Signed Distance Fields (SDFs) generated via AI-scripted CSG trees. This allows for infinite resolution, boolean operations (like drilling holes), and seamless blending that would be impossible or computationally prohibitive with triangle meshes.  
* **Surface:** Rendered via Quantized Gaussian Splats (no UVs) using "Smart Skinning" logic. This decoupling of geometry from texture eliminates the need for UV unwrapping—the most brittle part of procedural generation—allowing surfaces to adapt dynamically to changes in shape.  
* **Physics:** Native interaction based on math primitives (SDFs). Collisions are calculated against the mathematical truth of the object, ensuring that visual and physical boundaries are always perfectly aligned, regardless of how complex the shape becomes.  
* **Database:** MongoDB Document Store is utilized not just for storage, but for its massive scalability and semantic search capabilities, enabling the handling of millions of procedural variants with instant retrieval times.

## **2\. The Codebase Topology**

The system follows a strict Tri-Layered Separation to ensure performance, safety, and a clear distinction between creation logic and runtime execution.

### **Layer 1: The Forge (Interface)**

* **Role:** The "Remote Control." It captures user intent, visualization settings, and tool interactions, displaying high-fidelity previews.  
* **Stack:** HTML5, htmx, Rust WASM (Core Shared Runtime).  
* **Location:** /tools/forge-ui  
* **Function:** By compiling the actual engine core to WebAssembly, The Forge guarantees perfect parity between the editor and the game. It is not a JavaScript approximation; it is the engine running in a browser tab, controlled via Htmx for state management.

### **Layer 2: The Architect (Logic & Generation)**

* **Role:** The "Brain." This layer handles the heavy, non-realtime tasks: AI orchestration, database operations, and the complex mathematics of Asset Compilation.  
* **Stack:** Python 3.11+, FastAPI, PyTorch (Baking), MongoDB (Storage).  
* **Location:** /backend/architect  
* **Responsibility:**  
  * **AI Pipeline:** Orchestrates the multi-stage LLM workflow (Vision-Critic, PAL, RAG), managing context windows and prompting strategies to ensure coherent output.  
  * **Tensor Module:** Leverages GPU acceleration (via PyTorch/JAX) to perform the heavy "Baking" tasks—converting SDF math into 3D textures and running **Neural Splat Refinement** (Zero-Layer training) to snap textures to geometry.  
  * **Librarian:** Acts as the database abstraction layer, wrapping MongoDB for CRUD operations and handling the vectorization of assets for Semantic Search.  
  * **Map Ingestion:** Manages the pipeline for converting raw LiDAR data into game-ready SDFs, including the "Halo" sampling needed for seamless chunk blending.

### **Layer 3: The Engine (Runtime)**

* **Role:** The "Muscle." It assumes all incoming data is valid, optimized, and binary-packed. Its sole focus is executing the simulation loop as fast as possible.  
* **Stack:** Rust, wgpu (Graphics), rapier3d (Physics), cpal (Audio).  
* **Location:** /engine/runtime  
* **Responsibility:**  
  * **Render Loop:** Executes the hybrid pipeline: Rasterizing Shells \-\> Raymarching Volumes \-\> Sorting and Drawing Splats.  
  * **Physics Loop:** Runs the SIMD-optimized Rigid Body Solver and handles the custom SdfShape collision queries against the math bytecode.  
  * **Audio Loop:** Processes the DSP Graph (Resonance Engine) on a dedicated high-priority thread to ensure glitch-free synthesis.

### **Cross-Cutting: The Schema Generator**

* **Role:** The "Rosetta Stone."  
* **Mechanism:** Uses typeshare and schemars to automatically generate Python Pydantic models and TypeScript interfaces directly from the **Rust Structs**.  
* **Benefit:** This ensures the Engine (the ultimate Source of Truth) never desyncs from the Architect or Forge. If a developer adds a field to a Rust struct, the Python validation logic and TypeScript UI types are updated automatically during the build process, eliminating a massive class of integration bugs.

## **3\. Core Rendering Architecture: "Shell & Volume"**

To achieve the target of 60FPS on mobile and mid-range hardware, we cannot rely on brute-force raymarching. We trade VRAM capacity (which is abundant) for reduced Compute load (which is scarce) through a hybrid pipeline.

### **3.1 The Geometry Pipeline (LOD System)**

1. **Pass 1: Shell Rasterization (Early Z)**  
   * The engine draws a low-poly, tight-fitting "Proxy Mesh" (generated by the Architect via Dual Contouring).  
   * **Benefit:** This writes to the Depth Buffer, allowing the GPU to cull pixels *behind* the object or *inside* other objects before the expensive shader runs. Crucially, it provides a start depth for the raymarcher, bounding the steps significantly.  
2. **Pass 2: The Raymarcher**  
   * **LOD 0 (Close):** Evaluates the raw Math Tree (Infinite Resolution). This allows for perfect curves and crisp booleans when the camera is near.  
   * **LOD 1 (Mid):** Samples a Baked 3D Texture. This uses hardware-accelerated trilinear interpolation, which is significantly faster than solving algebraic equations.  
   * **LOD 2 (Far):** Stops raymarching entirely and renders only the Gaussian Splats, treating the object as a point cloud.

### **3.2 The Visual Pipeline (Splat Culling)**

* **Tile-Based Sort:** To handle overdraw, a Compute Shader buckets splats into 16x16 screen tiles and sorts them front-to-back.  
* **Sorting:** This allows the rasterizer to skip drawing splats that are occluded by opaque splats in front of them.  
* **Refinement:** Splats are pre-trained (Densified/Pruned) by the Architect during the compilation phase. This ensures they "snap" perfectly to the SDF surface, reducing the need for runtime density and minimizing visual artifacts.

### **3.3 The Terrain Pipeline (Voxel Repacking)**

* **Data:** The terrain is stored as a Global Voxel Volume (e.g., 2048x2048x256), derived from real-world LiDAR or procedural generators.  
* **Edit Logic:**  
  * **Gameplay:** Interactions like explosions trigger mathematical operations, such as Subtract(Sphere).  
  * **Compute:** A Compute Shader "Stamps" this subtraction directly into the Volume Texture memory.  
  * **Render:** The renderer raymarches the updated Volume. Because the change is baked into the data, the cost of rendering a cratered terrain is **O(1)**—identical to rendering a flat one—regardless of modification count.

### **3.4 Surface Skinning Strategy**

* **Logic:** We replace traditional UV maps with **Triplanar Mapping**, which projects texture data from three orthogonal axes (X, Y, Z) and blends them based on surface normal.  
* **Source:** 2D Images or AI-generated material swatches are used as the source for this projection.  
* **Baking:** To avoid running expensive triplanar blending for every pixel at runtime, the compiler calculates the final color for every Splat and stores it as a simple u32 integer. This effectively "bakes" the projection into the point cloud, removing the need for runtime texture lookups entirely.

## **4\. Audio Architecture: "The Resonance Engine"**

Instead of playing static .wav files, which consume memory and lack interactivity, the engine performs audio physically based on the object's properties.

* **Trigger:** The physics collision system provides precise Velocity and Material\_ID data upon impact.  
* **Patch:** Each .gve\_bin contains an AudioPatch struct defining FM settings (carrier/modulator ratios), ADSR envelopes, and the DSP Chain.  
* **DSP Graph:**  
  1. **Source:** An FM Oscillator (Carrier \+ Modulator) creates the tonal body, or a Granular Sampler provides texture.  
  2. **Filter:** Dynamic LowPass filters (driven by Mass) and Distortion units (driven by Material/Rust) shape the timbre.  
  3. **Mix:** The voice is routed to the Master Bus.  
  4. **Occlusion:** Calculated via raycasting through the Visual Voxel Volume, reusing the rendering data to determine if sound is muffled by walls or terrain.

## **5\. Hardware Targets**

| Metric         | Desktop GPU Target     | Mobile/ARM Target      | Notes                                                                         |
| :------------- | :--------------------- | :--------------------- | :---------------------------------------------------------------------------- |
| **Max Splats** | 4,000,000+             | \~500,000              | Desktop allows for high density; mobile relies on "Shell" visibility culling. |
| **Terrain**    | 1024³ Voxel Volume     | 256³ Voxel Volume      | Voxel resolution scales with RAM capacity.                                    |
| **Audio**      | 64 Active Voices (DSP) | 16 Active Voices (DSP) | Voice stealing manages CPU load on constrained devices.                       |
| **Physics**    | 5,000 Rigid Bodies     | 500 Rigid Bodies       | SIMD optimizations in Rapier allow high counts even on mobile CPUs.           |

