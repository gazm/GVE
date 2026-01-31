# **GVE-1 Rendering Pipeline Architecture**

**Graphics API:** wgpu (Vulkan/Metal/DX12)  
**Strategy:** Hybrid Shell + Volume + Splat  
**Philosophy:** Trade VRAM for compute — pre-bake complexity, minimize runtime math

---

## **1. The Three-Pass System**

### Why Hybrid Rendering?

**Problem:** Pure raymarching is O(steps × complexity), prohibitively expensive for 60fps on mobile.

**Solution:** Progressive refinement across three render passes:

```
Pass 1: Shell Rasterization  → Early-Z culling, depth bounds
Pass 2: Volumetric Raymarch  → SDF evaluation (LOD-based)
Pass 3: Gaussian Splat Sort → Point cloud rendering
```

**Key Insight:** Each pass leverages GPU strengths:
- **Rasterization:** Fixed-function hardware (triangles → pixels)
- **Compute Shaders:** Parallel raymarching with depth bounds
- **Tile-Based Sorting:** Cache-friendly overdraw reduction

---

## **2. Pass 1: Shell Rasterization (Early-Z)**

### Purpose
Write depth buffer to bound raymarching steps and enable early fragment tests.

### Algorithm

```wgsl
// Vertex Shader
struct ShellVertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

@vertex
fn vs_shell(vertex: ShellVertex) -> @builtin(position) vec4<f32> {
    let mvp = uniforms.proj * uniforms.view * uniforms.model;
    return mvp * vec4(vertex.position, 1.0);
}

// Fragment Shader (minimal)
@fragment
fn fs_shell() -> @builtin(frag_depth) f32 {
    // Only write depth, no color output
    // Depth written automatically by rasterizer
    return builtin(frag_depth);
}
```

### Shell Mesh Properties

**Generation:** Dual Contouring (see `compiler-pipeline.md` Stage 2.2)

**Topology:**
- Target: 500-2000 triangles per object
- Decimation: QEM (Quadratic Error Metric) to preserve silhouette
- Tight-fitting: AABB inflation < 5% volume expansion

**Performance:**
```
Desktop GPU:
  - 10,000 objects × 1000 tris = 10M triangles
  - Cost: ~2ms @ 1080p (fixed-function rasterization)

Mobile GPU:
  - 1,000 objects × 500 tris = 500K triangles
  - Cost: ~3ms @ 720p
```

### Depth Buffer Usage

**Format:** `D32_SFLOAT` (32-bit float depth)

**Benefits:**
1. **Early-Z Test:** GPU discards pixels behind shell before fragment shader
2. **Raymarch Start Depth:** Initial t₀ for sphere tracing
3. **Occlusion Culling:** Objects behind shell don't raymarch

**Optimization:** Render front-to-back sorted by camera distance to maximize Early-Z efficiency.

---

## **3. Pass 2: Volumetric Raymarching (Compute)**

### Problem Statement
Evaluate SDF for visible pixels with adaptive LOD based on distance.

### 3.1 LOD Selection Strategy

```wgsl
fn select_lod(distance_to_camera: f32) -> u32 {
    const LOD0_RANGE: f32 = 10.0;  // 0-10m: Math evaluation
    const LOD1_RANGE: f32 = 50.0;  // 10-50m: Baked texture
    // LOD2: >50m: Skip raymarching, splats only
    
    if distance_to_camera < LOD0_RANGE {
        return 0u;  // Infinite resolution
    } else if distance_to_camera < LOD1_RANGE {
        return 1u;  // Texture lookup
    } else {
        return 2u;  // No raymarch
    }
}
```

**Rationale:**
- **LOD 0 (Math):** Crisp boolean operations, perfect curves (e.g., drilled holes in metal)
- **LOD 1 (Texture):** 10-100× faster than math, imperceptible quality loss at mid-distance
- **LOD 2 (Splats):** Zero raymarch cost, acceptable for distant objects

---

### 3.1.1 Ray Generation (WGPU NDC)

To reconstruct world-space rays from screen coordinates in WGPU:
1.  **Vertex Shader**: Generate a fullscreen triangle and pass NDC coordinates `[-1, 1]` in `out.uv`.
2.  **Fragment Shader**: Use the `inverse(proj * view)` matrix to transform NDC points on the near and far planes.

> [!IMPORTANT]
> **WGPU NDC Coordinate System:**
> - **Y-Axis**: Up (+1 is top, -1 is bottom)
> - **Depth**: [0, 1] (0 is near plane, 1 is far plane)
> - **Righthand Rule**: Matches `glam`'s `look_at_rh` and `perspective_rh`.

```wgsl
@vertex
fn vs_fullscreen(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let uv = vec2(f32((idx << 1u) & 2u), f32(idx & 2u));
    out.position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = uv * 2.0 - 1.0; // NDC [-1, 1]
    return out;
}

@fragment
fn fs_sdf(in: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = in.uv;
    
    // Near and Far points in world space
    let near = uniforms.inv_view_proj * vec4(ndc, 0.0, 1.0);
    let far = uniforms.inv_view_proj * vec4(ndc, 1.0, 1.0);
    
    let ro = near.xyz / near.w;
    let rd = normalize(far.xyz / far.w - ro);
    
    return raymarch(ro, rd);
}
```

---

### 3.2 Sphere Tracing Algorithm

```wgsl
struct RaymarchResult {
    hit: bool,
    depth: f32,
    normal: vec3<f32>,
    material_id: u32,
}

@compute @workgroup_size(8, 8, 1)
fn raymarch_sdf(
    @builtin(global_invocation_id) pixel: vec3<u32>
) {
    let uv = vec2<f32>(pixel.xy) / vec2<f32>(resolution);
    let ray = generate_camera_ray(uv);
    
    // Read shell depth as starting point
    let depth_texture_value = textureLoad(depth_buffer, pixel.xy, 0);
    var t = linearize_depth(depth_texture_value);
    
    const MAX_STEPS: u32 = 64u;
    const EPSILON: f32 = 0.001;
    const MAX_DISTANCE: f32 = 100.0;
    
    var hit = false;
    var normal = vec3<f32>(0.0);
    var material_id = 0u;
    
    for (var i = 0u; i < MAX_STEPS; i++) {
        let p = ray.origin + ray.direction * t;
        
        // LOD-based distance evaluation
        let lod = select_lod(t);
        var d: f32;
        
        if lod == 0u {
            d = evaluate_sdf_math(p);  // Bytecode VM
        } else if lod == 1u {
            d = sample_sdf_texture(p);  // Trilinear lookup
        } else {
            break;  // Skip to splats
        }
        
        // Surface hit check
        if d < EPSILON {
            hit = true;
            normal = compute_gradient(p, lod);
            material_id = lookup_material_id(p);
            break;
        }
        
        // Conservative advancement
        t += d * 0.9;  // 90% step for safety margin
        
        if t > MAX_DISTANCE {
            break;
        }
    }
    
    // Write results to G-Buffer
    if hit {
        textureStore(gbuffer_depth, pixel.xy, vec4(t));
        textureStore(gbuffer_normal, pixel.xy, vec4(normal, 1.0));
        textureStore(gbuffer_material, pixel.xy, vec4(f32(material_id)));
    }
}
```

### Gradient Calculation (Surface Normal)

**Central Differences (Robust):**
```wgsl
fn compute_gradient(p: vec3<f32>, lod: u32) -> vec3<f32> {
    const H: f32 = 0.001;
    
    let eval = if lod == 0u { evaluate_sdf_math } else { sample_sdf_texture };
    
    let dx = eval(p + vec3(H, 0.0, 0.0)) - eval(p - vec3(H, 0.0, 0.0));
    let dy = eval(p + vec3(0.0, H, 0.0)) - eval(p - vec3(0.0, H, 0.0));
    let dz = eval(p + vec3(0.0, 0.0, H)) - eval(p - vec3(0.0, 0.0, H));
    
    return normalize(vec3(dx, dy, dz) / (2.0 * H));
}
```

**Cost:** 6 SDF evaluations per hit

**Optimization:** For LOD1, use `textureGrad()` with hardware derivatives (faster).

---

### 3.3 SDF Bytecode Virtual Machine (LOD 0)

**Instruction Format:**
```rust
enum SDFInstruction {
    Primitive { op: PrimitiveOp, params: [f32; 8] },
    BinaryOp { op: BinaryOp, left_idx: u32, right_idx: u32 },
    Modifier { op: ModifierOp, child_idx: u32, params: [f32; 4] },
}

enum PrimitiveOp {
    Sphere,    // params: [cx, cy, cz, radius]
    Box,       // params: [cx, cy, cz, sx, sy, sz]
    Cylinder,  // params: [cx, cy, cz, radius, height]
    // ... others
}

enum BinaryOp {
    Union,         // min(a, b)
    Subtract,      // max(a, -b)
    Intersect,     // max(a, b)
    SmoothUnion,   // IQ polynomial smooth min
}
```

**Execution (Stack-Based):**
```wgsl
fn evaluate_sdf_math(p: vec3<f32>) -> f32 {
    var stack: array<f32, 32>;
    var stack_ptr = 0u;
    
    for (var i = 0u; i < instruction_count; i++) {
        let instr = instructions[i];
        
        switch instr.type {
            case PRIMITIVE: {
                let d = eval_primitive(instr.op, p, instr.params);
                stack[stack_ptr] = d;
                stack_ptr++;
            }
            case BINARY_OP: {
                let b = stack[stack_ptr - 1];
                let a = stack[stack_ptr - 2];
                stack_ptr -= 2;
                
                let result = apply_binary_op(instr.op, a, b);
                stack[stack_ptr] = result;
                stack_ptr++;
            }
            // ... other cases
        }
    }
    
    return stack[0];  // Final result
}
```

**Performance:**
- Typical tree: 10-30 instructions
- Per-evaluation cost: ~200 cycles (SIMD-optimized)
- Budget: 64 steps × 200 cycles = 12.8k cycles/pixel

---

### 3.4 Texture Baking (LOD 1)

**Format:** R16_SFLOAT (half-precision)  
**Resolution:** 128³ per object (adaptive)  
**Compression:** Zstd (~50% reduction)

**Sampling:**
```wgsl
@group(1) @binding(0) var sdf_volume: texture_3d<f32>;
@group(1) @binding(1) var sdf_sampler: sampler;  // Trilinear

fn sample_sdf_texture(p: vec3<f32>) -> f32 {
    // Transform world position to texture UVW
    let uvw = (p - bounds_min) / (bounds_max - bounds_min);
    
    // Hardware trilinear interpolation
    return textureSample(sdf_volume, sdf_sampler, uvw).r;
}
```

**Benefits:**
- Constant O(1) lookup regardless of tree complexity
- Hardware trilinear filtering (free on GPU)
- Gradient via `textureSampleGrad()` (analytical derivatives)

---

## **4. Pass 3: Gaussian Splat Rendering**

### 4.1 Splat Data Structure

```rust
struct Splat {
    position: Vec3,        // 12 bytes
    scale: Vec3,           // 12 bytes (ellipsoid radii)
    rotation: Quat,        // 16 bytes (quaternion)
    color_packed: u32,     // 4 bytes (packed color)
    flags: u8,             // 1 byte (bit 0: color_mode)
    _padding: [u8; 3],     // 3 bytes padding
}
// Total: 48 bytes per splat

enum ColorMode {
    RGB = 0,     // Static assets, pre-baked colors
    Oklab = 1,   // Dynamic assets, procedural effects
}
```

**Color Mode Selection:**

> [!NOTE]
> **Use RGB mode for:** Static environment props, terrain, non-dynamic objects  
> **Use Oklab mode for:** Weapons (wear/animated skins), destructible materials, character outfits

**Color Encoding:**
**RGB Mode:**
```wgsl
fn unpack_rgba(packed: u32) -> vec4<f32> {
    let r = f32((packed >> 24u) & 0xFFu) / 255.0;
    let g = f32((packed >> 16u) & 0xFFu) / 255.0;
    let b = f32((packed >> 8u) & 0xFFu) / 255.0;
    let a = f32(packed & 0xFFu) / 255.0;
    
    return vec4(r, g, b, a);
}

fn srgb_to_linear(srgb: vec3<f32>) -> vec3<f32> {
    // Fast sRGB to linear conversion
    return select(
        srgb / 12.92,
        pow((srgb + 0.055) / 1.055, vec3(2.4)),
        srgb <= vec3(0.04045)
    );
}
```

**Oklab Mode:**
```wgsl
fn unpack_oklab(packed: u32) -> vec4<f32> {
    let L = f32((packed >> 24u) & 0xFFu) / 255.0;
    let a = (f32((packed >> 16u) & 0xFFu) / 255.0) * 0.8 - 0.4;
    let b = (f32((packed >> 8u) & 0xFFu) / 255.0) * 0.8 - 0.4;
    let alpha = f32(packed & 0xFFu) / 255.0;
    
    return vec4(L, a, b, alpha);
}

fn oklab_to_linear_rgb(oklab: vec3<f32>) -> vec3<f32> {
    let L = oklab.x;
    let a = oklab.y;
    let b = oklab.z;
    
    let l_ = L + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = L - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = L - 0.0894841775 * a - 1.2914855480 * b;
    
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    
    return vec3(
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    );
}
```

### 4.5 Color Mode Usage Examples

**RGB Mode (Static Assets):**
```python
# Static environment prop
compiler.export_splats("crate.gve_bin", color_mode="rgb")
# → 70k splats × 48 bytes = 3.36MB
# → Fast rendering, no color math needed
```

**Oklab Mode (Dynamic/Procedural):**
```python
# Weapon with wear system
compiler.export_splats("weapon_ak47.gve_bin", color_mode="oklab")
# → Enables runtime: lerp(pristine_color, worn_color, wear_amount)
# → Smooth rust/dirt gradients

# Character outfit with color variations
compiler.export_splats("character_outfit.gve_bin", color_mode="oklab")
# → Enables: lerp(team_color_a, team_color_b, t)
# → No muddy interpolation
```

**Runtime Color Interpolation (WGSL):**
```wgsl
// Only works correctly with Oklab mode!
fn apply_weapon_wear(base_oklab: vec3<f32>, wear: f32) -> vec3<f32> {
    let worn_oklab = vec3(0.4, 0.05, 0.08);  // Rusty brown in Oklab
    
    // Smooth perceptual blend
    return mix(base_oklab, worn_oklab, wear);
}

// RGB mode would look muddy:
// mix(vec3(1,0,0), vec3(0,1,0), 0.5) = vec3(0.5, 0.5, 0) = ugly brown
// Oklab: smooth yellow transition
```

---

### 4.2 Tile-Based Sorting (Compute Pre-Pass)

**Problem:** Gaussian splats require front-to-back order for correct alpha blending.

**Solution:** Tile-based bucket sort

```wgsl
const TILE_SIZE: u32 = 16u;  // 16×16 pixel tiles

struct SplatKey {
    tile_id: u32,    // Which tile this splat covers
    depth: f32,      // Distance from camera
    splat_id: u32,   // Index into splat buffer
}

@compute @workgroup_size(256, 1, 1)
fn splat_sort_keys(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let splat_id = id.x;
    if splat_id >= splat_count { return; }
    
    let splat = splats[splat_id];
    
    // Project splat center to screen space
    let mvp = uniforms.proj * uniforms.view;
    let clip_pos = mvp * vec4(splat.position, 1.0);
    let ndc = clip_pos.xy / clip_pos.w;
    let screen_pos = (ndc * 0.5 + 0.5) * vec2<f32>(resolution);
    
    // Compute bounding box (conservative)
    let radius_screen = project_sphere_to_screen(splat.scale, clip_pos.w);
    let min_tile = vec2<u32>(screen_pos - radius_screen) / TILE_SIZE;
    let max_tile = vec2<u32>(screen_pos + radius_screen) / TILE_SIZE;
    
    // Emit key for each covered tile
    let depth = clip_pos.z / clip_pos.w;
    
    for (var ty = min_tile.y; ty <= max_tile.y; ty++) {
        for (var tx = min_tile.x; tx <= max_tile.x; tx++) {
            let tile_id = ty * tile_count_x + tx;
            
            // Atomic append to tile bucket
            let idx = atomicAdd(&tile_splat_counts[tile_id], 1u);
            tile_keys[tile_id][idx] = SplatKey {
                tile_id: tile_id,
                depth: depth,
                splat_id: splat_id,
            };
        }
    }
}
```

**Radix Sort Per-Tile:**
```wgsl
@compute @workgroup_size(256, 1, 1)
fn radix_sort_tile(@builtin(global_invocation_id) id: vec3<u32>) {
    let tile_id = id.x;
    let keys = tile_keys[tile_id];
    let count = tile_splat_counts[tile_id];
    
    // Radix sort by depth (front-to-back)
    radix_sort_u32(keys, count, /* sort by depth field */);
    
    tile_sorted_keys[tile_id] = keys;
}
```

---

### 4.3 Splat Rasterization (Fragment)

```wgsl
@fragment
fn fs_splat(
    @builtin(position) frag_coord: vec4<f32>,
    @location(0) splat_id: u32,
    @location(1) ellipse_uv: vec2<f32>,  // [-1, 1] quad space
) -> @location(0) vec4<f32> {
    let splat = splats[splat_id];
    
    // Gaussian falloff
    let dist_sq = dot(ellipse_uv, ellipse_uv);
    if dist_sq > 1.0 { discard; }  // Outside ellipse
    
    // Decode color based on mode
    let color_mode = splat.flags & 1u;  // Bit 0 = color mode
    var color_linear: vec3<f32>;
    var alpha_base: f32;
    
    if color_mode == 0u {  // RGB mode
        let color_srgb = unpack_rgba(splat.color_packed);
        color_linear = srgb_to_linear(color_srgb.rgb);
        alpha_base = color_srgb.a;
    } else {  // Oklab mode
        let color_oklab = unpack_oklab(splat.color_packed);
        color_linear = oklab_to_linear_rgb(color_oklab.rgb);
        alpha_base = color_oklab.a;
    }
    
    let alpha = alpha_base * exp(-0.5 * dist_sq);  // Gaussian falloff
    
    if alpha < 0.01 { discard; }  // Culling threshold
    
    return vec4(color_linear, alpha);
}
```

**Blending Mode:** Premultiplied Alpha
```
color_out = src.rgb × src.a + dst.rgb × (1 - src.a)
```

**Depth Test:** Read-only (depth written by raymarch pass)

---

### 4.4 Performance Budget

**Desktop (4M splats):**
```
Sorting:        5ms (compute)
Rasterization:  8ms (fragment shader)
────────────────────
Total:         13ms @ 60fps (22% budget)
```

**Mobile (500K splats):**
```
Sorting:        2ms
Rasterization:  6ms
────────────────────
Total:          8ms @ 60fps (48% budget)
```

**Optimization:** Frustum culling reduces splat count by ~60% in typical scenes.

---

## **5. Terrain Rendering (Voxel Volume)**

### Data Structure

**Storage:** 3D Texture (global volume)
```rust
struct TerrainVolume {
    density: Texture3D<R8_UNORM>,    // 0=air, 255=solid
    material: Texture3D<R8_UINT>,    // Material ID
    resolution: UVec3,                // e.g., 2048×2048×256
}
```

**Memory:**
- Desktop: 1024³ = 1GB (uncompressed)
- Mobile: 256³ = 16MB

---

### Raymarching Strategy

**DDA (Digital Differential Analyzer):**
```wgsl
fn raymarch_voxel_terrain(ray: Ray, max_dist: f32) -> RaymarchResult {
    var t = 0.0;
    var voxel = world_to_voxel(ray.origin);
    
    let delta = abs(1.0 / ray.direction);  // Step size per axis
    let step_sign = sign(ray.direction);
    
    var t_max = (fract(ray.origin / voxel_size) - 0.5 * step_sign + 0.5) * delta;
    
    while t < max_dist {
        // Sample voxel
        let density = textureSample(terrain_density, voxel).r;
        
        if density > 0.5 {
            // Hit solid voxel
            return RaymarchResult {
                hit: true,
                depth: t,
                normal: -step_sign,  // Voxel face normal
                material_id: textureSample(terrain_material, voxel).r,
            };
        }
        
        // Step to next voxel boundary
        if t_max.x < t_max.y && t_max.x < t_max.z {
            t = t_max.x;
            voxel.x += step_sign.x;
            t_max.x += delta.x;
        } else if t_max.y < t_max.z {
            t = t_max.y;
            voxel.y += step_sign.y;
            t_max.y += delta.y;
        } else {
            t = t_max.z;
            voxel.z += step_sign.z;
            t_max.z += delta.z;
        }
    }
    
    return RaymarchResult { hit: false };
}
```

**Complexity:** O(distance / voxel_size) — linear in voxel traversals

---

### Dynamic Modification (Explosions)

**Compute Shader Stamping:**
```wgsl
@compute @workgroup_size(8, 8, 8)
fn stamp_explosion(
    @builtin(global_invocation_id) voxel: vec3<u32>
) {
    let voxel_pos = voxel_to_world(voxel);
    let dist_to_explosion = length(voxel_pos - explosion_center);
    
    if dist_to_explosion < explosion_radius {
        // Smooth falloff
        let falloff = 1.0 - (dist_to_explosion / explosion_radius);
        let subtract_amount = falloff * 255.0;
        
        // Atomic min to subtract density
        let current_density = textureLoad(terrain_density, voxel).r;
        let new_density = max(0.0, current_density - subtract_amount);
        
        textureStore(terrain_density, voxel, vec4(new_density));
    }
}
```

**Cost:** O(crater_volume) — amortized over frames if needed

---

## **6. Lighting & Shading**

### Deferred Shading (G-Buffer)

**Outputs from Raymarch Pass:**
```rust
struct GBuffer {
    depth: Texture2D<R32_FLOAT>,
    normal: Texture2D<RGBA16_SNORM>,
    material_id: Texture2D<R16_UINT>,
    albedo: Texture2D<RGBA8_UNORM>,  // From splats (sRGB)
}
```

### PBR Material Evaluation

```wgsl
struct Material {
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    emissive: vec3<f32>,
}

fn evaluate_lighting(
    position: vec3<f32>,
    normal: vec3<f32>,
    material: Material,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    var color = vec3(0.0);
    
    // Directional light (sun)
    let light_dir = normalize(sun_direction);
    let NoL = max(dot(normal, light_dir), 0.0);
    
    // Diffuse
    let diffuse = material.base_color * NoL;
    
    // Specular (Cook-Torrance)
    let H = normalize(light_dir + view_dir);
    let NoH = max(dot(normal, H), 0.0);
    let VoH = max(dot(view_dir, H), 0.0);
    
    let D = distribution_ggx(NoH, material.roughness);
    let F = fresnel_schlick(VoH, mix(vec3(0.04), material.base_color, material.metallic));
    let G = geometry_smith(NoL, dot(normal, view_dir), material.roughness);
    
    let specular = (D * F * G) / max(4.0 * NoL * dot(normal, view_dir), 0.001);
    
    color += (diffuse + specular) * sun_color * NoL;
    
    // Ambient (IBL or constant)
    color += material.base_color * 0.03;
    
    // Emissive
    color += material.emissive;
    
    return color;
}
```

---

## **7. Complete Frame Pipeline**

```
┌─────────────────────────────────────────────────┐
│ Frame Start                                      │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│ Pass 1: Shell Rasterization                     │
│  ├─ Depth-only render                           │
│  ├─ Front-to-back sorted                        │
│  └─ Output: Depth Buffer                        │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│ Pass 2: Volumetric Raymarch (Compute)           │
│  ├─ Per-pixel dispatch (1920×1080 workgroups)   │
│  ├─ Read depth buffer (start point)             │
│  ├─ LOD selection (distance-based)              │
│  ├─ LOD0: Math VM | LOD1: Texture | LOD2: Skip  │
│  └─ Output: G-Buffer (depth, normal, material)  │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│ Pass 2.5: Splat Sort (Compute)                  │
│  ├─ Tile binning (16×16 tiles)                  │
│  ├─ Radix sort per tile (by depth)              │
│  └─ Output: Sorted splat indices                │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│ Pass 3: Splat Rasterization                     │
│  ├─ Tile-based rendering                        │
│  ├─ Gaussian alpha blending                     │
│  ├─ sRGB → Linear conversion                    │
│  └─ Output: Color Buffer (premultiplied alpha)  │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│ Pass 4: Terrain Raymarch (DDA)                  │
│  ├─ Voxel traversal                             │
│  ├─ Merged with object G-Buffer                 │
│  └─ Output: Combined G-Buffer                   │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│ Pass 5: Deferred Lighting                       │
│  ├─ Read G-Buffer                               │
│  ├─ PBR evaluation                              │
│  ├─ Shadow mapping                              │
│  └─ Output: Lit HDR buffer                      │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│ Pass 6: Post-Processing                         │
│  ├─ Tone mapping (ACES)                         │
│  ├─ Bloom                                       │
│  ├─ FXAA                                        │
│  └─ Output: Final LDR framebuffer               │
└───────────────────┬─────────────────────────────┘
                    │
                 Present
```

---

## **8. Performance Budget (1080p @ 60fps)**

| Pass | Desktop Time | Mobile Time | Notes |
|------|--------------|-------------|-------|
| **Pass 1: Shell** | 2ms | 3ms | Fixed-function rasterization |
| **Pass 2: Raymarch** | 6ms | 10ms | Compute-heavy, LOD helps |
| **Pass 2.5: Sort** | 5ms | 2ms | Radix sort, fewer splats on mobile |
| **Pass 3: Splats** | 8ms | 6ms | Alpha blending, sRGB decode |
| **Pass 4: Terrain** | 3ms | 4ms | DDA, sparse in most scenes |
| **Pass 5: Lighting** | 4ms | 6ms | PBR shading |
| **Pass 6: Post** | 2ms | 2ms | Lightweight |
| **Total** | **30ms** | **33ms** | 50% budget @ 60fps |

**Headroom:** 33% for physics, audio, game logic

---

## **9. Optimization Techniques**

### 9.1 Frustum Culling (CPU)

```rust
fn cull_objects(objects: &[SDFObject], camera: &Camera) -> Vec<&SDFObject> {
    objects.iter()
        .filter(|obj| camera.frustum.contains_aabb(&obj.bounding_box))
        .collect()
}
```

**Savings:** ~60% objects culled in typical scene

---

### 9.2 Temporal Reprojection (TAA)

**Checkerboard Rendering:** Render alternating pixels per frame, reproject previous frame

```wgsl
fn temporal_reproject(pixel: vec2<u32>, frame_idx: u32) -> bool {
    // Checkerboard pattern
    return (pixel.x + pixel.y + frame_idx) % 2u == 0u;
}
```

**Benefit:** 2× faster raymarching (50% pixels/frame)  
**Quality:** TAA blends current + history → full resolution

---

### 9.3 Variable Rate Shading (VRS)

**Concept:** Render center pixels at full rate, edges at half

```rust
let shading_rate = if distance_from_center < 0.5 {
    ShadingRate::_1x1  // Full res
} else {
    ShadingRate::_2x2  // Quarter res
};
```

**Savings:** 30-40% fragment shader cost  
**Supported:** NVIDIA RTX, AMD RDNA2, Mobile (limited)

---

## **10. Implementation Checklist**

### Core Rendering
- [ ] Implement dual contouring shell mesh generator
- [ ] Write depth-only rasterization pass (wgpu)
- [ ] Build SDF bytecode VM (WGSL)
- [ ] Implement sphere tracing compute shader
- [ ] Add LOD selection logic
- [ ] Create 3D texture baking pipeline

### Splat Rendering
- [ ] Implement tile-based splat sorting
- [ ] Write Gaussian rasterization shader
- [ ] Add sRGB to linear color conversion
- [ ] Optimize alpha blending

### Terrain
- [ ] Build DDA voxel raymarcher
- [ ] Implement compute shader stamping
- [ ] Add terrain LOD (mipmaps)

### Lighting
- [ ] Create G-Buffer layout
- [ ] Implement PBR shading
- [ ] Add shadow mapping
- [ ] Integrate IBL (environment maps)

### Optimization
- [ ] Add frustum culling
- [ ] Implement TAA
- [ ] Add VRS support (optional)
- [ ] Profile and meet performance budget

---

## **11. Future Enhancements**

**Ray-Traced Shadows:**
- Use hardware RT cores (DXR/VK_KHR_ray_tracing)
- Ray-SDF intersection for soft shadows

**Global Illumination:**
- Voxel cone tracing for diffuse GI
- Screen-space reflections for specular

**Mesh Shaders (Next-Gen):**
- Replace shell rasterization with mesh shader amplification
- Generate LOD on-GPU

**Neural Radiance Caching:**
- Train tiny MLP to cache indirect lighting
- 10× faster than voxel cone tracing
