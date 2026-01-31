# Core Systems Documentation

Implementation details for major subsystems and engines.

## Documents

### [Audio System](./audio-system.md) ✅
**"The Resonance Engine"** - Physics-driven procedural audio synthesis.

**Key Topics:**
- Philosophy: "Physics, not Playback"
- The "Analogizer" pipeline (Sample → Math resynthesis)
- Hybrid source nodes (FM synthesis + granular sampling)
- Intelligent DSP routing (Baked/Bus/Per-Voice)
- Audio LOD system
- Performance budgets

---

### [Physics System](./physics-system.md) ✅
**Rapier3d integration** with custom SDF collision shapes.

**Key Topics:**
- Why Rapier3d (WASM parity, SIMD, custom shapes)
- Collision strategy hierarchy (Primitives/SdfShape/Voxel)
- Physics collision against mathematical truth (SDF)
- Debris lifecycle: Active → Sleeping → Frozen (voxel stamping)
- Multibody joints for articulated structures

---

### [Destruction System](destruction-system.md) ✅

The material-driven environment destruction pipeline.

**Key Features:**
- Material-based fracture patterns (Voronoi, Radial, Planar)
- SDF geometry modification via subtraction
- Debris generation and physics simulation
- Splat regeneration strategies (masking vs regeneration)
- Sleep-to-voxel freezing lifecycle

**Materials Database:** Concrete, Steel, Glass, Wood, Brick with yield strengths and fracture behaviors

---

### [Math Library](./math-library.md) ✅
**Vector and matrix math** for physics and SDF calculations.

**Key Topics:**
- Core types: Vec2/3/4, Quat, Mat3/4, Isometry3
- Vector operations: dot, cross, normalize, lerp
- Quaternion rotations and slerp interpolation
- JSON schema integration for DNA format
- SDF-specific: gradients, smooth min/max, domain ops
- SIMD batch operations via simba

---

### Rendering Pipeline 🔮
*(Planned)* - Detailed rendering architecture extraction from system overview.

**Will Cover:**
- Hybrid volumetric + splatting pipeline
- Shell rasterization (Early-Z pass)
- Three-tier LOD system
- Raymarching optimization
- Splat tile-based sorting
- Voxel repacking for terrain
- Surface skinning (triplanar projection)

---

### Scripting System 🔮
*(Planned)* - Gameplay logic and entity behavior.

**Will Cover:**
- Scripting language choice
- Entity Component System (ECS)
- Component definitions
- System execution order
- Behavior trees / State machines
- Event system

---

## Related Documentation

- **Architecture:** [/architecture](../architecture) - High-level system design
- **Workflows:** [/workflows](../workflows) - How these systems are fed data
- **Data:** [/data](../data) - Data formats these systems consume

[← Back to Documentation Home](../README.md)
