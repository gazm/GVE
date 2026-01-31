//! WGSL shader sources for the renderer
//!
//! Contains embedded shader code for mesh and SDF raymarching pipelines.

/// SDF Raymarching Shader (Fullscreen Fragment)
///
/// Implements a stack-based VM for evaluating SDF bytecode instructions.
/// Supports primitives (sphere, box, cylinder, torus, plane) and
/// binary operations (union, subtract, intersect, smooth_union).
pub const SDF_SHADER: &str = r#"
// Uniforms for camera and SDF parameters
struct SDFUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    resolution: vec2<f32>,
    instruction_count: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: SDFUniforms;

// SDF instruction (48 bytes each, 16-byte aligned for uniform buffer)
struct SDFInstruction {
    instr_type: u32,    // 0=primitive, 1=binary, 2=modifier
    op: u32,
    operand1: u32,      // child_idx or left_idx
    operand2: u32,      // right_idx (for binary ops)
    // Use vec4 for 16-byte alignment (WebGL2 requirement)
    params0: vec4<f32>, // params[0-3]: center.xyz + param
    params1: vec4<f32>, // params[4-7]: size.xyz + param
}

// Instructions as uniform buffer (WebGL2 compatible, max 16 instructions)
struct SDFInstructionBuffer {
    data: array<SDFInstruction, 16>,
}

@group(0) @binding(1) var<uniform> instructions: SDFInstructionBuffer;

// =============================================================================
// SDF Primitives (based on Inigo Quilez / munrocket implementations)
// Reference: https://gist.github.com/munrocket/f247155fc22ecb8edf974d905c677de1
// =============================================================================

// Sphere - exact
fn sdf_sphere(p: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    return length(p - center) - radius;
}

// Box - exact
fn sdf_box(p: vec3<f32>, center: vec3<f32>, size: vec3<f32>) -> f32 {
    let q = abs(p - center) - size;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Vertical Cylinder - exact
fn sdf_cylinder(p: vec3<f32>, center: vec3<f32>, radius: f32, height: f32) -> f32 {
    let q = p - center;
    let d = abs(vec2<f32>(length(q.xz), q.y)) - vec2<f32>(radius, height);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

// Torus - exact
fn sdf_torus(p: vec3<f32>, center: vec3<f32>, major_r: f32, minor_r: f32) -> f32 {
    let q = p - center;
    let t = vec2<f32>(length(q.xz) - major_r, q.y);
    return length(t) - minor_r;
}

// Plane - exact (n must be normalized)
fn sdf_plane(p: vec3<f32>, normal: vec3<f32>, dist: f32) -> f32 {
    return dot(p, normalize(normal)) + dist;
}

// Capsule / Line - exact
fn sdf_capsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// Round Box - exact
fn sdf_round_box(p: vec3<f32>, center: vec3<f32>, size: vec3<f32>, r: f32) -> f32 {
    let q = abs(p - center) - size;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// Cone (vertical) - exact
fn sdf_cone(p: vec3<f32>, center: vec3<f32>, height: f32, radius: f32) -> f32 {
    let q = p - center;
    let c = vec2<f32>(radius / height, -1.0);
    let w = vec2<f32>(length(q.xz), q.y);
    let a = w - c * clamp(dot(w, c) / dot(c, c), 0.0, 1.0);
    let b = w - c * vec2<f32>(clamp(w.x / c.x, 0.0, 1.0), 1.0);
    let k = sign(c.y);
    let d = min(dot(a, a), dot(b, b));
    let s = max(k * (w.x * c.y - w.y * c.x), k * (w.y - c.y));
    return sqrt(d) * sign(s);
}

// =============================================================================
// Boolean Operations
// =============================================================================

// Union - exact (outside), bound (inside)
fn sdf_union(d1: f32, d2: f32) -> f32 { return min(d1, d2); }

// Subtraction - bound
fn sdf_subtract(d1: f32, d2: f32) -> f32 { return max(d1, -d2); }

// Intersection - bound
fn sdf_intersect(d1: f32, d2: f32) -> f32 { return max(d1, d2); }

// Smooth Union - bound (polynomial smooth min by IQ)
fn sdf_smooth_union(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

// Smooth Subtraction - bound
fn sdf_smooth_subtract(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (d1 + d2) / k, 0.0, 1.0);
    return mix(d1, -d2, h) + k * h * (1.0 - h);
}

// Smooth Intersection - bound  
fn sdf_smooth_intersect(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

// Evaluate SDF bytecode (stack-based VM)
fn evaluate_sdf(p: vec3<f32>) -> f32 {
    var stack: array<f32, 32>;
    var sp: u32 = 0u;
    
    let count = min(uniforms.instruction_count, 16u);  // Max 16 for WebGL2 uniform limit
    
    for (var i = 0u; i < count; i++) {
        let instr = instructions.data[i];
        
        if instr.instr_type == 0u {
            // Primitive - params0.xyz = center, params0.w/params1.xyz = size/radius
            var d: f32 = 1e10;
            let center = instr.params0.xyz;
            
            switch instr.op {
                case 0x01u: { // Sphere: center.xyz, radius
                    d = sdf_sphere(p, center, instr.params0.w);
                }
                case 0x02u: { // Box: center.xyz, size.xyz
                    let size = vec3(instr.params0.w, instr.params1.x, instr.params1.y);
                    d = sdf_box(p, center, size);
                }
                case 0x03u: { // Cylinder: center.xyz, radius, height
                    d = sdf_cylinder(p, center, instr.params0.w, instr.params1.x);
                }
                case 0x04u: { // Capsule: a.xyz, b.xyz, radius
                    let a = instr.params0.xyz;
                    let b = vec3(instr.params0.w, instr.params1.x, instr.params1.y);
                    d = sdf_capsule(p, a, b, instr.params1.z);
                }
                case 0x05u: { // Torus: center.xyz, major_r, minor_r
                    d = sdf_torus(p, center, instr.params0.w, instr.params1.x);
                }
                case 0x06u: { // Cone: center.xyz, height, radius
                    d = sdf_cone(p, center, instr.params0.w, instr.params1.x);
                }
                case 0x07u: { // Plane: normal.xyz, dist
                    d = sdf_plane(p, instr.params0.xyz, instr.params0.w);
                }
                default: {
                    d = 1e10;
                }
            }
            
            stack[sp] = d;
            sp++;
        } else if instr.instr_type == 1u {
            // Binary operation
            if sp >= 2u {
                let b = stack[sp - 1u];
                let a = stack[sp - 2u];
                sp -= 2u;
                
                var result: f32;
                switch instr.op {
                    case 0x10u: { result = sdf_union(a, b); }
                    case 0x11u: { result = sdf_subtract(a, b); }
                    case 0x12u: { result = sdf_intersect(a, b); }
                    case 0x13u: { result = sdf_smooth_union(a, b, 0.1); }
                    default: { result = a; }
                }
                
                stack[sp] = result;
                sp++;
            }
        }
        // Modifiers would go here
    }
    
    if sp > 0u {
        return stack[sp - 1u];
    }
    return 1e10;
}

// =============================================================================
// Normal Calculation (tetrahedron technique)
// =============================================================================
fn compute_normal(p: vec3<f32>) -> vec3<f32> {
    // Tetrahedron technique - 4 samples instead of 6, more accurate
    let h = 0.001;  // Match surface_dist for consistency
    let k = vec2<f32>(1.0, -1.0);
    return normalize(
        k.xyy * evaluate_sdf(p + k.xyy * h) +
        k.yyx * evaluate_sdf(p + k.yyx * h) +
        k.yxy * evaluate_sdf(p + k.yxy * h) +
        k.xxx * evaluate_sdf(p + k.xxx * h)
    );
}

// =============================================================================
// Sphere Tracing (Raymarching)
// =============================================================================
fn raymarch(ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    var t = 0.001;  // Start slightly away from camera to avoid self-intersection
    let max_dist = 100.0;
    let max_steps = 128;
    let surface_dist = 0.001;  // Hit threshold (larger = more forgiving)
    let step_factor = 0.9;     // Relaxation to prevent overshooting (<1.0)
    
    var hit = false;
    var hit_pos = vec3<f32>(0.0);
    
    for (var i = 0; i < max_steps; i++) {
        let p = ro + rd * t;
        let d = evaluate_sdf(p);
        
        // Check for surface hit (use absolute value for inside/outside)
        if d < surface_dist {
            hit = true;
            hit_pos = p;
            break;
        }
        
        // Check if we've gone too far
        if t > max_dist {
            break;
        }
        
        // Advance ray with relaxation factor to prevent overshooting
        // This is the standard fix for grazing angle artifacts
        t += d * step_factor;
    }
    
    if hit {
        let normal = compute_normal(hit_pos);
        
        // Validate normal (protect against NaN - NaN != NaN)
        var n = normal;
        let len = length(n);
        if len < 0.5 || len != len {  // len != len detects NaN
            n = vec3<f32>(0.0, 1.0, 0.0);
        } else {
            n = n / len;  // Normalize
        }
        
        // Lighting
        let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
        let n_dot_l = max(dot(n, light_dir), 0.0);
        
        // Ambient occlusion approximation
        let ao = 0.5 + 0.5 * n.y;
        
        // Diffuse + ambient lighting
        let ambient = 0.15;
        let diffuse = n_dot_l * 0.85;
        
        // Base color from normal direction (makes shape readable)
        let base_color = n * 0.5 + 0.5;
        
        // Final color
        let lit_color = base_color * (ambient + diffuse) * ao;
        
        return vec4<f32>(lit_color, 1.0);
    }
    
    // Sky gradient for miss
    let sky_t = rd.y * 0.5 + 0.5;
    let sky_color = mix(
        vec3<f32>(0.1, 0.12, 0.15),  // Horizon
        vec3<f32>(0.15, 0.2, 0.3),   // Zenith
        sky_t
    );
    return vec4<f32>(sky_color, 1.0);
}

// Fullscreen quad vertex shader
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    // Generate fullscreen triangle
    var out: VertexOutput;
    let uv = vec2(f32((vertex_idx << 1u) & 2u), f32(vertex_idx & 2u));
    out.position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = uv * 2.0 - 1.0; 
    return out;
}

@fragment
fn fs_sdf(in: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = in.uv;
    
    // Reconstruct ray direction using inverse view-projection
    // WGPU NDC is Y-up, Depth is [0, 1]
    let near_point = uniforms.inv_view_proj * vec4(ndc, 0.0, 1.0);
    let far_point = uniforms.inv_view_proj * vec4(ndc, 1.0, 1.0);
    
    let ray_origin = near_point.xyz / near_point.w;
    let ray_dir = normalize(far_point.xyz / far_point.w - ray_origin);
    
    return raymarch(ray_origin, ray_dir);
}
"#;

/// Mesh shader with MVP matrix for 3D rendering
///
/// Simple vertex + fragment shader for rendering shell meshes
/// with normal-based coloring and directional lighting.
pub const MESH_SHADER: &str = r#"
struct Uniforms {
    mvp: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvp * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    // Simple normal-based coloring (hemisphere lighting)
    out.color = in.normal * 0.5 + 0.5;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple directional lighting
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let n = normalize(in.normal);
    let diffuse = max(dot(n, light_dir), 0.0);
    let ambient = 0.2;
    let lit_color = in.color * (ambient + diffuse * 0.8);
    return vec4<f32>(lit_color, 1.0);
}
"#;
