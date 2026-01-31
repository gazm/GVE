# **GVE-1 Data Specifications**

**Context:** The definitive guide to Data Structures, APIs, and Schemas for the Generative Volumetric Engine.

**Version:** 2.1 (Sparse VDB Support)

**Related Docs:**
- [Rendering Pipeline](../core-systems/rendering-pipeline.md) - How these structures are rendered
- [Compiler Pipeline](../workflows/compiler-pipeline.md) - How DNA JSON is compiled to binary
- [AI Pipeline](../workflows/ai-pipeline.md) - How AI generates these structures

---

## **Part 0: Global Standards (The "Game Unit")**

To ensure seamless integration between the Physics (Rapier3d), Audio (Resonance), and Rendering (WGPU) subsystems, the engine strictly enforces SI-based units internally.

* **Distance:** **1.0 GU \= 1.0 Meter**.  
  * *Implication:* Gravity is \-9.8 GU/s². The speed of sound is \~343 GU/s. AI must not use arbitrary scales (e.g., 100 units for a sword); it must use realistic meters (1.2 units for a sword) to ensure physical mass and audio resonance calculations are accurate.  
* **Grid:** **1 Chunk \= 4.0 GU**.  
  * *Context:* The "Brickmap" rendering optimization subdivides space into 4x4x4 meter regions. Keeping assets aligned to this grid where possible improves culling efficiency.  
* **Mass:** **1.0 Mass \= 1.0 Kg**.  
  * *Context:* Used by the Asset Compiler to calculate the Inertia Tensor via Monte Carlo integration (Volume × Density).

### **0.4 The Meta-Language (Parametric Logic)**

The Source JSON (dna.json) is not just a static data dump; it supports a dynamic expression layer. The Python Compiler resolves these expressions into raw floats *before* baking the Runtime Binary (.gve\_bin), keeping the engine fast while making assets smart and reactive.

#### **A. Variables ($)**

Variables allow the definition of "Gene" inputs at the root level, making the asset configurable without rewriting the geometry tree.

* **Syntax:** "$var\_name" can be used inside any numeric parameter field.  
* **Math:** Supports basic arithmetic strings evaluated during compilation: "$length \* 0.5 \+ 0.2".  
* **Scope:** Variables are global to the artifact. They allow a single "Knob" (e.g., caliber) to adjust the barrel radius, magazine size, and bore diameter simultaneously.

#### **B. Anchors (@)**

Anchors provide relative positioning based on the computed bounding box of other nodes. This eliminates the need for the AI to calculate absolute float coordinates.

* **Keywords:**  
  * @parent: The direct ancestor node in the tree.  
  * @prev: The immediate sibling defined before the current node (useful for stacks).  
  * @root: The base node of the hierarchy.  
* **Accessors:**  
  * .top / .bottom: Max/Min Y face center.  
  * .left / .right: Min/Max X face center.  
  * .front / .back: Min/Max Z face center.  
  * .center: The volumetric centroid.  
* **Example:** pos: "@parent.top" automatically snaps the child's pivot to the top face of the parent primitive, regardless of the parent's current size.

## **Part 1: The API Reference (The "Law")**

The AI **MUST** adhere to these definitions to prevent hallucinations.

### **1.1 Geometry Primitives**

These are the fundamental building blocks of the Signed Distance Field.

* Sphere(radius): The most efficient primitive (![][image1]). Used for biological forms, joints, and spherical tanks.  
* Box(size\_vec3): Defines a rectangular prism using half-extents. A \[1, 2, 1\] box is 2m wide, 4m tall, and 2m deep.  
* Cylinder(radius, height, sides):  
  * **sides Parameter:** If 0, renders a perfect mathematical cylinder. If \> 2 (e.g., 6), renders a prism (Hexagon). Crucial for "Low Poly" aesthetics or mechanical bolts.  
* Capsule(radius, height): A cylinder with hemispherical caps. Essential for organic limbs and mechanical joints because it has no sharp edges to snag physics collisions.  
* Torus(major\_r, minor\_r): A ring. major\_r is the distance from center to tube center; minor\_r is the tube radius.  
* Cone(angle, height, sides): A capped cone. Supports segmentation via sides (e.g., 4 \= Pyramid).  
* Plane(normal, dist): An infinite cutting plane. Usually used in Subtract operations to slice objects flat.

### **1.2 Geometry Primitives (Advanced)**

Specialized shapes for complex surface details and effects.

* Primitive\_Polyhedron(type, radius):  
  * **Types:** tetrahedron, octahedron, dodecahedron, icosahedron.  
  * **Use Case:** Crystalline structures, raw gems, sci-fi artifacts, or fantasy runes.  
* Primitive\_Bezier\_Tube(p0, p1, p2, radius):  
  * **Action:** Generates a Quadratic Bezier curve swept with a radius.  
  * **Use Case:** Hydraulic hoses, hanging cables, organic tentacles, wireframes.  
* Primitive\_Heightmap\_Project(image\_id, scale, height):  
  * **Action:** Extrudes a 2D heightmap image into 3D volume.  
  * **Use Case:** Embossed logos, tire treads, intricate engravings, terrain patches.  
* Primitive\_Splat\_Cloud(shape, density, falloff):  
  * **Action:** Fills a volume with semi-transparent splats. No solid surface.  
  * **Use Case:** Smoke, static gas clouds, magical auras, holograms.  
* Primitive\_Splat\_Fire(height, intensity, color\_core, color\_tip):  
  * **Action:** Generates a vertical gradient of emissive splats that animate upward in the shader.  
* Primitive\_Splat\_Foliage(type, height, radius):  
  * **Efficiency:** High. Bypasses SDF raymarching entirely. Uses soft sensor collision (![][image1] lookup).  
  * **Types:** bush, fern, grass\_clump.

### **1.3 Machining Macros**

High-level generative commands that expand into complex CSG trees during compilation.

* Machine\_Bore(target, axis, diameter, depth): Drills a cylindrical hole. Automatically sets the inner material to "Raw/Unpainted."  
* Machine\_Slot(target, axis, length, width, depth): Mills a linear slot (pill-shaped cutout).  
* Machine\_Pocket(target, axis, profile, depth): Mills a shallow recess (non-through hole).  
  * **Profiles:** "circle", "rect", or "hex" (for socket wrenches).  
* Machine\_Chamfer(target, edge\_axis, size): Slices a 45-degree edge off a corner using a Subtract(Plane).  
* Machine\_Array\_Linear(target, axis, count, spacing): Repeats the target shape count times along a vector.  
* Machine\_Array\_Radial(target, axis, count, radius): Repeats the target in a circle (e.g., bolt patterns on a hub).

### **1.4 Generative Macros**

* Gen\_Scatter\_Surface(target, greeble, count, seed):  
  * **Action:** Python runs a physics raycast against the target SDF to find surface positions/normals, then appends count instances of greeble.  
  * **Use Case:** Rivets on armor, barnacles on rocks.  
* Gen\_Structure\_Truss(start, end, width, segments):  
  * **Action:** Generates a cross-braced girder structure between two points.

### **1.5 Domain Modifiers**

Mathematical functions applied to the coordinate space of a Node *before* rendering.

* Effect\_Twist(axis, rate): Rotates space along an axis (Screw/Drill).  
* Effect\_Bend(axis, angle): Bends space along a curve.  
* Effect\_Taper(axis, scale\_min, scale\_max): Scales space linearly (Pinches one end).  
* Effect\_Shear(axis, strength): Slants the coordinate system.  
* Effect\_Mirror(axis): Applies abs(p.x) logic for instant symmetry.  
* Effect\_Noise(type, scale, amplitude): Displaces surface with 3D Perlin/Voronoi noise.  
* Effect\_Elongate(axis, length): Stretches the center of a shape without distorting the rounded corners (SDF specific operation).

**Topological Modifiers (Shape Changing):**

* Effect\_Round(radius): Inflates the shape via d \- radius. Turns a Cube into a Sphere.  
* Effect\_Facet(sides): Slices a round object (Cylinder/Sphere) into N flat radial segments.  
* Effect\_Morph(target\_shape, factor): Linearly interpolates the distance field between two shapes.  
* Effect\_Shell(thickness): Hollows the object via abs(d) \- thickness. Essential for vehicles.  
* Effect\_Lattice(thickness, spacing): Converts the volume into a wireframe grid structure.

### **1.6 Material & Surface Modifiers**

Controls for material assignment and splat generation.

#### Material Assignment

```json
{
  "material": {
    "spec_id": "ASTM_C114",  // Material from registry
    "color_mode": "rgb|oklab",  // See §1.6.1
    "base_color": "#FF5733" | [0.8, 0.2, 0.1],  // Hex or RGB array
    "metallic": 0.8,
    "roughness": 0.3,
    "emissive": [0.0, 0.5, 1.0],  // RGB additive glow
  }
}
```

#### 1.6.1 Color Mode Selection

> [!IMPORTANT]
> **Color Mode determines runtime behavior** (see `rendering-pipeline.md` §4.1)

**RGB Mode (Default):**
- Use for: Static environment props, terrain, non-dynamic objects
- Runtime: Simple sRGB→linear conversion (~10 cycles/pixel)
- Storage: RGBA8 (4 bytes packed)

**Oklab Mode:**
- Use for: Weapons (wear effects), characters (team colors), destructible materials
- Runtime: Enables smooth color interpolation without muddy transitions
- Storage: Lab8 (4 bytes packed) + Oklab→RGB matrix multiply (~30 cycles/pixel)
- Trade-off: 20 cycles slower, but essential for procedural color effects

**When to use Oklab:**
```javascript
// Weapon wear system (runtime color lerp)
color_mode: "oklab"  // Smooth rust gradient

// Character team colors (dynamic uniforms)
color_mode: "oklab"  // Clean color blending

// Damage tinting (health-based color shifts)
color_mode: "oklab"  // Perceptually linear transitions
```

**When to use RGB:**
```javascript
// Static crates, walls, rocks
color_mode: "rgb"  // Fastest rendering

// Pre-baked color (no runtime changes)
color_mode: "rgb"  // No interpolation needed
```

#### Surface Texture Modifiers

Controls for Gaussian Splat generation (see `compiler-pipeline.md` §3):

* **Texture_Pattern_Triplanar:** Projects a noise/camo pattern from 3 axes
* **Texture_Wear_Override:** Forces edge chipping (`edge_wear`) or cavity dirt (`grime`) levels
  ```json
  {"edge_wear": 0.7, "cavity_grime": 0.5}  // 0=clean, 1=maximum wear
  ```
* **Texture_Finish_Anisotropic:** Aligns splat normals to simulate brushed metal
* **Texture_Decal_Projection:** Projects a volumetric shape (sphere/box) of color onto surface
* **Texture_Emission_Pulse:** Animates brightness for sci-fi glow effects
  ```json
  {"frequency_hz": 2.0, "min_intensity": 0.2, "max_intensity": 1.0}
  ```
* **Texture_Layer_Mask:** Blends two material specs based on noise threshold

### **1.7 Audio Modifiers**

Digital Signal Processing filters applied to the synthesis chain.

* Audio\_Filter\_LowPass: Muffles high freq (Mass simulation).  
* Distortion: Adds harmonics (Rust/Corrosion simulation).  
* PitchShift: Scales samples.  
* Granular: Jitters playback for texture.  
* Reverb: Simulates internal cavity resonance.

## **Part 2: GVE-JSON (Intermediate Schema)**

**Context:** Output from AI \-\> Input to Python Compiler.

**Example: A Parametric Sword**

Notice how variables define the "Genes" of the object, and Anchors allow components to snap together logically without hard-coded coordinates.

{  
  "artifact\_name": "Procedural Sword",  
  "variables": {  
      "blade\_len": 1.2,  
      "width": 0.1,  
      "guard\_width": 0.3  
  },  
  "dna": {  
    "root\_node": {  
      "type": "operation",  
      "op": "union",  
      "children": \[  
        // 1\. The Blade (Driven by Variables)  
        {  
          "type": "primitive",  
          "shape": "box",  
          "id": "blade",  
          "params": {"size": \["$width", "$blade\_len", 0.05\]},  
          "modifiers": \[{ "cmd": "Effect\_Taper", "axis": "y", "scale\_max": 0.1 }\]  
        },  
        // 2\. The Crossguard (Snapped to Blade Bottom)  
        {  
          "type": "primitive",  
          "shape": "box",  
          "id": "guard",  
          "transform": { "pos": "@blade.bottom" },  
          "params": {"size": \["$guard\_width", 0.05, 0.1\]}  
        },  
        // 3\. The Handle (Snapped to Guard Bottom)  
        // Offset adds a visual gap or specific grip position  
        {  
          "type": "primitive",  
          "shape": "cylinder",  
          "transform": { "pos": "@guard.bottom \+ \[0, \-0.15, 0\]" },   
          "params": {"radius": 0.03, "height": 0.3},  
          "material": { "spec\_id": "ASTM\_D4181" } // Wrapped Grip  
        }  
      \]  
    }  
  }  
}

---

## **Part 2.5: Animated Characters (SDF Node-Based Animation)**

**Philosophy:** Instead of skinning individual splats, animate SDF tree nodes and keep splats relatively rigid per-node. This is simpler, faster, and perfect for stylized/mechanical characters.

### 2.5.1 Character Structure

```json
{
  "artifact_name": "Armored Soldier",
  "type": "animated_character",
  
  "skeleton": {
    "bones": [
      {"name": "Root", "parent": null, "rest_pose": {"pos": [0,0,0], "rot": [0,0,0,1]}},
      {"name": "Spine", "parent": "Root", "rest_pose": {"pos": [0,1.0,0], "rot": [0,0,0,1]}},
      {"name": "LeftShoulder", "parent": "Spine", "rest_pose": {"pos": [-0.3,1.5,0], "rot": [0,0,0,1]}},
      {"name": "LeftElbow", "parent": "LeftShoulder", "rest_pose": {"pos": [-0.3,1.1,0], "rot": [0,0,0,1]}}
      // ... more bones
    ]
  },
  
  "dna": {
    "root_node": {
      "type": "operation",
      "op": "union",
      "children": [
        // Torso (rigid body part)
        {
          "id": "torso",
          "type": "primitive",
          "shape": "box",
          "params": {"size": [0.6, 0.8, 0.4]},
          "bone_binding": "Spine",  // Binds entire node to bone
          "animation_mode": "rigid",  // No deformation
          "material": {"spec_id": "METAL_STEEL", "color_mode": "rgb"}
        },
        
        // Left upper arm (rigid)
        {
          "id": "left_upper_arm",
          "type": "primitive",
          "shape": "capsule",
          "params": {"radius": 0.1, "height": 0.4},
          "bone_binding": "LeftShoulder",
          "animation_mode": "rigid"
        },
        
        // Shoulder joint (smooth blend zone)
        {
          "id": "shoulder_joint",
          "type": "primitive",
          "shape": "sphere",
          "params": {"radius": 0.12},
          "transform": {"pos": "@left_upper_arm.top"},
          "animation_mode": "skinned",
          "bone_influences": [
            {"bone": "Spine", "weight": 0.3},
            {"bone": "LeftShoulder", "weight": 0.7}
          ]
        },
        
        // Left forearm (rigid)
        {
          "id": "left_forearm",
          "type": "primitive",
          "shape": "capsule",
          "params": {"radius": 0.08, "height": 0.35},
          "bone_binding": "LeftElbow",
          "animation_mode": "rigid"
        }
        // ... more body parts
      ]
    }
  },
  
  "animations": [
    {
      "name": "idle",
      "duration_sec": 2.0,
      "loop": true,
      "keyframes": [
        {
          "time": 0.0,
          "bone_transforms": {
            "LeftShoulder": {"pos": [0,0,0], "rot": [0,0,0,1]},
            "LeftElbow": {"pos": [0,0,0], "rot": [0,0,0,1]}
          }
        },
        {
          "time": 1.0,
          "bone_transforms": {
            "LeftShoulder": {"pos": [0,0,0], "rot": [0.1,0,0,0.995]},  // Slight rotation
            "LeftElbow": {"pos": [0,0,0], "rot": [0.05,0,0,0.999]}
          }
        }
      ]
    }
  ]
}
```

### 2.5.2 Animation Modes

**Rigid Mode (Default - 80% of splats):**
- Entire SDF node transforms as one unit
- All splats bound to the node move together
- Perfect for: armor plates, robot parts, mechanical segments
- Performance: ~0.5ms for 70k splats

**Skinned Mode (Joint Zones - 20% of splats):**
- Splats blend between multiple bones
- Smooth deformation at joints
- Auto-detected at node boundaries or manually specified
- Performance: ~0.05ms for 5k joint splats

**Total Animation Cost: ~0.55ms/character** (5× faster than traditional mesh skinning)

### 2.5.3 Automatic Joint Detection

The compiler automatically detects and skins joint areas:

```python
# Compiler pseudo-code
for node_pair in connected_nodes:
    if node_pair[0].bone != node_pair[1].bone:
        # Different bones = joint
        joint_zone = detect_blend_zone(
            node_pair[0], 
            node_pair[1], 
            blend_radius=0.1  # 10cm
        )
        
        # Auto-weight splats in blend zone
        for splat in joint_zone.splats:
            dist_a = distance(splat.pos, node_pair[0].center)
            dist_b = distance(splat.pos, node_pair[1].center)
            
            weight_a = dist_b / (dist_a + dist_b)
            weight_b = 1.0 - weight_a
            
            splat.binding = Skinned {
                bones: [node_pair[0].bone, node_pair[1].bone],
                weights: [weight_a, weight_b]
            }
```

### 2.5.4 Runtime Binary Format

```rust
struct AnimatedCharacterBinary {
    // Standard sections (from Part 3)
    header: GVEBinaryHeader,
    sdf_bytecode: SDFBytecode,
    splat_data: SplatData,
    
    // Animation-specific data
    skeleton: SkeletonData,
    node_bindings: NodeToBoneMap,
    animations: AnimationClipData,
}

struct SkeletonData {
    bone_count: u16,
    bones: [BoneDefinition; bone_count],
}

struct BoneDefinition {
    name_len: u8,
    name: [u8; name_len],
    parent_idx: u16,  // 0xFFFF = no parent (root)
    rest_pose: Transform,  // Position + Rotation (28 bytes)
}

struct NodeToBoneMap {
    mapping_count: u32,
    mappings: [NodeBinding; mapping_count],
}

enum NodeBinding {
    Rigid { 
        node_id: u32, 
        bone_idx: u16 
    },  // 6 bytes
    
    Skinned { 
        node_id: u32,
        bone_count: u8,
        bones: [u16; bone_count],
        weights: [f32; bone_count],
    },  // Variable size
}

struct AnimationClipData {
    clip_count: u16,
    clips: [AnimationClip; clip_count],
}

struct AnimationClip {
    name_len: u8,
    name: [u8; name_len],
    duration_sec: f32,
    loop_flag: u8,
    keyframe_count: u16,
    keyframes: [Keyframe; keyframe_count],
}

struct Keyframe {
    time_sec: f32,
    bone_transform_count: u16,
    transforms: [BoneTransform; bone_transform_count],
}

struct BoneTransform {
    bone_idx: u16,
    position: [f32; 3],
    rotation: [f32; 4],  // Quaternion
}
```

### 2.5.5 Performance Characteristics

**CPU Animation Cost per Frame:**
```
Rigid splats:   65,000 × (matrix multiply) = 0.46ms
Skinned splats:  5,000 × (4-bone blend)    = 0.06ms
Total:                                       0.52ms/character

Compare to traditional mesh skinning: ~2.4ms
→ 4.6× faster!
```

**Memory Overhead:**
```
Skeleton:        30 bones × 56 bytes     = 1.7KB
Node bindings:   50 mappings × 6 bytes   = 300 bytes
Animations:      5 clips × ~20KB         = 100KB
Total overhead:                            ~102KB

Base character:                            5-8MB
→ <2% overhead for animation support
```

### 2.5.6 Advantages Over Traditional

✅ **Simpler Workflow:**
- No weight painting required for 80% of character
- Auto-binding via naming convention
- Auto-detection of joint zones

✅ **Better Performance:**
- 4-5× faster than mesh skinning
- Parallel processing friendly (SIMD)
- Cache-coherent memory access

✅ **Modular Design:**
- Swap armor pieces = swap SDF nodes
- Outfit variations trivial (just different node configs)
- Team colors via Oklab mode

✅ **Perfect for:**
- Mechs and robots (rigid parts)
- Armored characters (plates + joints)
- Stylized humanoids (Fortnite-style)
- Modular character systems

### 2.5.7 Use Cases by Character Type

| Character Type | Difficulty | Animation Mode | Quality |
|---------------|-----------|----------------|----------|
| **Robots/Mechs** | Easy | 100% Rigid | Excellent |
| **Armored soldiers** | Easy | 85% Rigid + 15% Skinned | Excellent |
| **Stylized humanoids** | Moderate | 70% Rigid + 30% Skinned | Very Good |
| **Organic creatures** | Hard | 40% Rigid + 60% Skinned | Good |
| **Realistic humans** | Hard | Hybrid (mesh face + splat body) | Moderate |

### 2.5.8 Complete Workflow

**1. Asset Creation (AI → Compiler):**
```
AI generates → DNA JSON with body parts
  ↓
Compiler:
  - Parses SDF tree
  - Auto-binds nodes to bones (name matching)
  - Generates splats per node
  - Detects joint zones
  - Auto-weights joint splats
  ↓
Exports → .gve_bin with animation data
```

**2. Runtime Animation:**
```rust
fn animate_character(
    character: &Character,
    animation: &str,
    time: f32
) -> Vec<Splat> {
    // Sample animation
    let bone_transforms = character.sample_animation(animation, time);
    
    let mut output = Vec::new();
    
    // For each SDF node
    for (node_id, binding) in &character.node_bindings {
        match binding {
            Rigid { bone_idx } => {
                // Transform all splats rigidly
                let transform = bone_transforms[*bone_idx];
                for splat in &character.splats_per_node[node_id] {
                    output.push(transform_rigid(splat, transform));
                }
            }
            Skinned { bones, weights } => {
                // Blend transform for joint splats
                for splat in &character.splats_per_node[node_id] {
                    output.push(transform_skinned(
                        splat, 
                        bones, 
                        weights, 
                        &bone_transforms
                    ));
                }
            }
        }
    }
    
    output
}
```

**3. Rendering Integration:**
```
CPU: Animate character (0.52ms)
  ↓
Upload animated splats to GPU
  ↓
Render using splat pass (rendering-pipeline.md §4)
```

---

## **Part 3: GVE-BIN (Runtime Binary Format)**

**Context:** Output from Compiler → Input to Rust Engine  
**Format:** Custom binary format optimized for zero-copy GPU upload

> [!NOTE]
> The compilation process resolves all variables (`$blade_len` → `1.2`) and anchors (`@blade.bottom` → `[0, -1.2, 0]`) into static 32-bit floats. The Runtime Engine has no concept of "Variables," only raw bytecode and baked data.

### 3.1 Binary File Structure

```rust
// Header (64 bytes)
struct GVEBinaryHeader {
    magic: [u8; 4],          // "GVE1"
    version: u32,            // Format version (0x00021000 for v2.1)
    flags: u32,              // Bit flags (compression, LOD levels, etc.)
    
    // Offsets to data sections
    sdf_bytecode_offset: u64,
    volume_data_offset: u64,   // (Was sdf_texture_offset in v2.0)
    splat_data_offset: u64,
    shell_mesh_offset: u64,
    audio_patch_offset: u64,
    metadata_offset: u64,
    
    // Size fields
    sdf_bytecode_size: u32,
    sdf_texture_size: u32,
    splat_count: u32,
    vertex_count: u32,
    
    _padding: [u8; 8],
}
```

### 3.2 SDF Bytecode Section

**Purpose:** LOD 0 (math evaluation) - see `rendering-pipeline.md` §3.3

```rust
struct SDFBytecode {
    instruction_count: u32,
    instructions: [SDFInstruction; instruction_count],
}

enum SDFInstruction {
    Primitive { op: u8, params: [f32; 8] },      // 33 bytes
    BinaryOp { op: u8, left_idx: u16, right_idx: u16 },  // 5 bytes
    Modifier { op: u8, child_idx: u16, params: [f32; 4] },  // 19 bytes
}

// Example ops:
// Primitive: 0x01=Sphere, 0x02=Box, 0x03=Cylinder
// BinaryOp: 0x10=Union, 0x11=Subtract, 0x12=Intersect, 0x13=SmoothUnion
// Modifier: 0x20=Twist, 0x21=Bend, 0x22=Mirror
```

**Size:** Typical asset: 10-30 instructions ≈ 200-600 bytes

### 3.3 Volume Data Section (LOD 1)

**Purpose:** Sparse volumetric data (OpenVDB) for physics and mesh generation.

```rust
struct VolumeData {
    // Standard OpenVDB (.vdb) file content
    // Contains grid metadata, tree structure, and compressed voxel data.
    // Parsed via nanoVDB or OpenVDB library.
    data: [u8; data_size] 
}
```

**Note:** In v2.1, this replaces the dense 3D texture.

**Size:** 128³ × 2 bytes (R16) ≈ 4MB uncompressed → ~2MB with Zstd

### 3.4 Splat Data Section

**Purpose:** Gaussian splats for LOD 2 (distant rendering)

```rust
struct SplatData {
    count: u32,
    lod_pyramid: SplatLODPyramid,
    splats: [Splat; count],
}

struct SplatLODPyramid {
    lod0_count: u32,  // 70k splats (close range)
    lod1_count: u32,  // 20k splats (mid range)
    lod2_count: u32,  // 5k splats (distant)
}

#[repr(C, packed)]
struct Splat {
    position: [f32; 3],      // 12 bytes
    scale: [f32; 3],         // 12 bytes (ellipsoid radii)
    rotation: [f32; 4],      // 16 bytes (quaternion)
    color_packed: u32,       // 4 bytes (RGBA8 or Oklab8)
    flags: u8,               // 1 byte (bit 0: color_mode, bits 1-7: reserved)
    _padding: [u8; 3],       // 3 bytes (alignment)
}  // Total: 48 bytes per splat
```

**Color Packing:**
```rust
// RGB mode (flags & 0x01 == 0)
color_packed = (R << 24) | (G << 16) | (B << 8) | A

// Oklab mode (flags & 0x01 == 1)  
color_packed = (L << 24) | (a_scaled << 16) | (b_scaled << 8) | A
// where: a_scaled = (a + 0.4) / 0.8 * 255
//        b_scaled = (b + 0.4) / 0.8 * 255
```

**Size:** 70k splats × 48 bytes ≈ 3.36MB (uncompressed)

### 3.5 Shell Mesh Section

**Purpose:** Depth-only rasterization for Pass 1 (Early-Z)

```rust
struct ShellMesh {
    vertex_count: u32,
    index_count: u32,
    vertices: [ShellVertex; vertex_count],
    indices: [u16; index_count],  // or u32 if vertex_count > 65k
}

#[repr(C)]
struct ShellVertex {
    position: [f32; 3],   // 12 bytes
    normal: [f32; 3],     // 12 bytes (for backface culling)
}  // Total: 24 bytes per vertex
```

**Size:** 1000 vertices × 24 bytes ≈ 24KB

### 3.6 Audio Patch Section

**Purpose:** Physics-driven DSP configuration

```rust
struct AudioPatch {
    oscillator_count: u8,
    oscillators: [FMOscillator; oscillator_count],
    envelope: ADSREnvelope,
    dsp_chain_count: u8,
    dsp_chain: [DSPEffect; dsp_chain_count],
}

struct FMOscillator {
    waveform: u8,       // 0=Sine, 1=Square, 2=Saw, 3=Triangle
    frequency_hz: f32,
    amplitude: f32,
    modulation_idx: u8,  // Index of modulator oscillator (255=none)
    mod_amount: f32,
}

struct ADSREnvelope {
    attack_ms: f32,
    decay_ms: f32,
    sustain_level: f32,  // 0.0-1.0
    release_ms: f32,
}

struct DSPEffect {
    effect_type: u8,     // 0=LowPass, 1=Distortion, 2=Reverb, etc.
    params: [f32; 4],    // Effect-specific parameters
}
```

**Size:** Typical: <1KB

### 3.7 Metadata Section

**Purpose:** Asset manifest for engine

```rust
struct Metadata {
    name_len: u16,
    name: [u8; name_len],        // UTF-8 asset name
    
    // Physics properties
    mass_kg: f32,
    center_of_mass: [f32; 3],
    inertia_tensor: [f32; 9],    // 3×3 matrix
    
    // Rendering hints
    bounding_box_min: [f32; 3],
    bounding_box_max: [f32; 3],
    recommended_lod_distances: [f32; 3],  // LOD0, LOD1, LOD2 cutoffs
    
    // Material registry
    material_count: u16,
    materials: [MaterialRef; material_count],
}

struct MaterialRef {
    spec_id_len: u8,
    spec_id: [u8; spec_id_len],  // e.g., "ASTM_C114"
    color_mode: u8,               // 0=RGB, 1=Oklab
}
```

### 3.8 Complete Binary Layout

```
┌───────────────────────────┐ Offset 0
│ GVEBinaryHeader (64 bytes)│
├───────────────────────────┤
│ SDF Bytecode Section      │ → LOD 0 (math eval)
│  ~200-600 bytes           │
├───────────────────────────┤
│ SDF Texture Section       │ → LOD 1 (baked texture)
│  ~2-4 MB (compressed)     │
├───────────────────────────┤
│ Splat Data Section        │ → LOD 2 (splats)
│  ~3.36 MB (70k splats)    │
│  ├─ LOD0 (70k)            │
│  ├─ LOD1 (20k)            │
│  └─ LOD2 (5k)             │
├───────────────────────────┤
│ Shell Mesh Section        │ → Early-Z pass
│  ~24-50 KB                │
├───────────────────────────┤
│ Audio Patch Section       │ → DSP config
│  <1 KB                    │
├───────────────────────────┤
│ Metadata Section          │ → Manifest
│  <1 KB                    │
└───────────────────────────┘

Total: ~5-8 MB per asset (with compression)
```

### 3.9 Loading Strategy

**Streaming:**
```rust
// Load only what's needed for current LOD
if distance < 10m {
    load_sdf_bytecode();  // 600 bytes
} else if distance < 50m {
    load_sdf_texture();   // 2MB
} else {
    load_splats_lod2();   // 240KB (5k splats)
}
```

**Zero-Copy Upload:**
```rust
// Binary format designed for direct GPU upload
let splat_buffer = device.create_buffer_init(&wgpu::BufferInitDescriptor {
    label: Some("Splat Data"),
    contents: &binary.splat_section,  // No parsing, direct memcpy
    usage: wgpu::BufferUsages::STORAGE,
});
```

---

## **Part 4: Pipeline Flow**

```
User Prompt
    ↓
┌─────────────────────┐
│ AI Pipeline         │ → See: ai-pipeline.md
│  Generates DNA JSON │
└──────────┬──────────┘
           ↓ (dna.json)
┌─────────────────────┐
│ Python Compiler     │ → See: compiler-pipeline.md
│  ├─ Resolve $vars  │
│  ├─ Resolve @anchors│
│  ├─ Bake SDF texture│
│  ├─ Train splats    │
│  └─ Generate shell  │
└──────────┬──────────┘
           ↓ (.gve_bin)
┌─────────────────────┐
│ Rust Engine         │ → See: rendering-pipeline.md
│  ├─ Load binary     │
│  ├─ Upload to GPU   │
│  └─ Render (3-pass) │
└─────────────────────┘
```

---

**Status:** ✅ Complete  
**Last Updated:** January 25, 2026  
**Version:** 2.0  


[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAYCAYAAACIhL/AAAACrklEQVR4Xu2WO2hTURzGb6gBwReJeZnHSaguguCgKEIFRx0UKSKC3RXsoIIKunR1k+oUhKKgdCjo4lIchC71AYpYndxcdBEEBQervy85x557yL1NqYUI/eDHvfl/5/E/z5soWtcAqV6vHxTNZvNC6CXJGDPcaDQu8ZoNvTRl1BGVZ+hsnucrnm94nhb4Q2EFyh/AmxbFYnFz6KNstVptiNCg7VHqXeQ1Y1lWg5kgneSF7Wgqn89v9bxdNPRB4F2NvIaUELHHsE+4eNRNipAZo95rnveE5ztlid9nqU+I0Owol8tto9BTSzvqsSfo5KzlJxxycRo9w++HvG6wdFSpVFq0NWG6M/8yJUFNwFG8WdFrBTSCtpuhUqlUDgtIxs4QfCepy4q1Wq2N1HmixMPyTuoQ/1lagsx0Ae+58Affkc1+EWNcxExPfoJwzcaGYaFWq+0Nyzv1kyDKUOaBcIPvyJuBrxh7hFcpJsqMWn57CY7w+z3sCMs79Zmgtsp1ESvDyOsEPsFcoVDYIrw6MVFmUihBZv2YjR1X5732jVO/CWrQItaesUumqY1Sjrn2ZXPpFL8rl8slxdcT1OYm8C2tooQ/BouWMS++tgna473ggr06svfZR2iLyLsjlaDqqx2vSkwrTRBmIu8+1ek5T/AHz8Niqcrf5F7AFMu6Sfi+6W6RtzpsftxXvwlS5q6gzGToZTFuwGehe4hCp+CWOjcJ31/JroAGMBJ62qfE52jzS7N7Nf2y6NaY9gerd2O/JImfO7fEXCFHKHQSdkYJiXnSBXuHhidCYyWijd0wL+i/Fvqrkuku8yzLvF2Efj8isXENcrUDTZJm8QqNnxOhuZy0z6n/SFvCXV//WgOfoKR/RDeF/ryGZpJ0ONjrt2F/6K2FdKCG0r7nPZQNr67/Sn8Ad2T1YadvDLwAAAAASUVORK5CYII=>