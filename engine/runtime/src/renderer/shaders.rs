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
    view_proj: mat4x4<f32>,
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

// Z-axis Cylinder - exact (aligned with forward/barrel direction)
fn sdf_cylinder(p: vec3<f32>, center: vec3<f32>, radius: f32, height: f32) -> f32 {
    let q = p - center;
    let d = abs(vec2<f32>(length(q.xy), q.z)) - vec2<f32>(radius, height);
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

// Revolution - Y-axis solid of revolution (box cross-section approximation)
fn sdf_revolution(p: vec3<f32>, center: vec3<f32>, offset: f32, profile_w: f32, profile_h: f32) -> f32 {
    let q = p - center;
    let radial = length(q.xz) - offset;
    let d = vec2<f32>(abs(radial) - profile_w, abs(q.y) - profile_h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

// =============================================================================
// Voronoi Utilities (integer-arithmetic hash for GPU reproducibility)
// =============================================================================

fn hash33(p: vec3<f32>) -> vec3<f32> {
    // Integer-arithmetic hash: deterministic, no sin() precision issues
    var q = vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6))
    );
    return fract(sin(q) * 43758.5453);
}

fn voronoi_distance(p: vec3<f32>, cell_size: f32) -> f32 {
    let scaled = p / cell_size;
    let cell = floor(scaled);
    var min_dist = 999.0;
    
    // 27-neighbor search (unrolled for WGSL performance)
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                let neighbor = cell + vec3<f32>(f32(dx), f32(dy), f32(dz));
                let jitter = hash33(neighbor);
                let center = (neighbor + jitter) * cell_size;
                let d = length(p - center);
                min_dist = min(min_dist, d);
            }
        }
    }
    return min_dist;
}

// =============================================================================
// Fractal Primitives
// =============================================================================

// Mandelbulb - power-N 3D fractal (distance estimator)
fn sdf_mandelbulb(p: vec3<f32>, center: vec3<f32>, scale: f32, power: f32, iters: u32) -> f32 {
    let q = (p - center) / scale;
    var z = q;
    var dr = 1.0;
    var r = 0.0;
    
    for (var i = 0u; i < 12u; i++) {
        if (i >= iters) { break; }
        r = length(z);
        if (r > 2.0) { break; }
        
        // Spherical coordinates
        let theta = acos(clamp(z.z / r, -1.0, 1.0));
        let phi = atan2(z.y, z.x);
        
        dr = pow(r, power - 1.0) * power * dr + 1.0;
        
        // Power transform
        let r_pow = pow(r, power);
        let theta_n = theta * power;
        let phi_n = phi * power;
        
        z = vec3<f32>(
            sin(theta_n) * cos(phi_n),
            sin(theta_n) * sin(phi_n),
            cos(theta_n)
        ) * r_pow + q;
    }
    
    r = length(z);
    return 0.5 * log(r) * r / dr * scale;
}

// Menger Sponge - recursive cross subtraction
fn sdf_menger(p: vec3<f32>, center: vec3<f32>, scale: f32, iters: u32) -> f32 {
    let q = abs((p - center) / scale);
    var d = max(max(q.x, q.y), q.z) - 1.0;
    
    var s = 1.0;
    for (var i = 0u; i < 5u; i++) {
        if (i >= iters) { break; }
        
        let a = (q * s % vec3<f32>(2.0)) - vec3<f32>(1.0);
        s *= 3.0;
        let r = abs(1.0 - 3.0 * abs(a));
        
        let da = max(r.x, r.y);
        let db = max(r.y, r.z);
        let dc = max(r.x, r.z);
        let c = (min(min(da, db), dc) - 1.0) / s;
        
        d = max(d, c);
    }
    
    return d * scale;
}

// Julia Set (quaternion) - 3D fractal
fn sdf_julia(p: vec3<f32>, center: vec3<f32>, scale: f32, c: vec4<f32>) -> f32 {
    let q = (p - center) / scale;
    var z = vec4<f32>(q, 0.0);
    var dz = 1.0;
    
    for (var i = 0u; i < 12u; i++) {
        let r = length(z);
        if (r > 4.0) { break; }
        
        // Quaternion square: z*z + c
        let a = z.x; let b = z.y; let cc = z.z; let d = z.w;
        z = vec4<f32>(
            a*a - b*b - cc*cc - d*d + c.x,
            2.0*a*b + c.y,
            2.0*a*cc + c.z,
            2.0*a*d + c.w
        );
        dz = 2.0 * r * dz;
    }
    
    let r = length(z);
    return 0.5 * r * log(r) / dz * scale;
}

// Cone (Z-axis) - exact (aligned with forward direction)
fn sdf_cone(p: vec3<f32>, center: vec3<f32>, height: f32, radius: f32) -> f32 {
    let q = p - center;
    let c = vec2<f32>(radius / height, -1.0);
    let w = vec2<f32>(length(q.xy), q.z);
    let a = w - c * clamp(dot(w, c) / dot(c, c), 0.0, 1.0);
    let b = w - c * vec2<f32>(clamp(w.x / c.x, 0.0, 1.0), 1.0);
    let k = sign(c.y);
    let d = min(dot(a, a), dot(b, b));
    let s = max(k * (w.x * c.y - w.y * c.x), k * (w.y - c.y));
    return sqrt(d) * sign(s);
}

// Wedge - box intersected with diagonal cutting plane
// taper_axis shrinks from full-width to zero across taper_dir
// taper_axis/taper_dir: 0=X, 1=Y, 2=Z
fn sdf_wedge(p: vec3<f32>, center: vec3<f32>, size: vec3<f32>, taper_axis: u32, taper_dir: u32) -> f32 {
    let q = p - center;
    // Box SDF
    let aq = abs(q) - size;
    let box_d = length(max(aq, vec3<f32>(0.0))) + min(max(aq.x, max(aq.y, aq.z)), 0.0);
    // Diagonal plane: allowed = size_tap * (1 - t), where t ∈ [0,1] along taper_dir
    var dir_val: f32;
    var size_dir: f32;
    var tap_val: f32;
    var size_tap: f32;
    // Extract components by axis index
    if taper_dir == 0u { dir_val = q.x; size_dir = size.x; }
    else if taper_dir == 1u { dir_val = q.y; size_dir = size.y; }
    else { dir_val = q.z; size_dir = size.z; }
    if taper_axis == 0u { tap_val = q.x; size_tap = size.x; }
    else if taper_axis == 1u { tap_val = q.y; size_tap = size.y; }
    else { tap_val = q.z; size_tap = size.z; }
    let t = clamp((dir_val + size_dir) / (2.0 * size_dir + 1e-8), 0.0, 1.0);
    let allowed = size_tap * (1.0 - t);
    let plane_d = abs(tap_val) - allowed;
    return max(box_d, plane_d);
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
                case 0x08u: { // Revolution: center.xyz, offset, profile_w, profile_h
                    d = sdf_revolution(p, center, instr.params0.w, instr.params1.x, instr.params1.y);
                }
                case 0x09u: { // Mandelbulb: center.xyz, scale, power, iterations
                    d = sdf_mandelbulb(p, center, instr.params0.w, instr.params1.x, u32(instr.params1.y));
                }
                case 0x0Au: { // Menger Sponge: center.xyz, scale, iterations
                    d = sdf_menger(p, center, instr.params0.w, u32(instr.params1.x));
                }
                case 0x0Bu: { // Julia Set: center.xyz, scale, c[4]
                    let c = vec4<f32>(instr.params1.x, instr.params1.y, instr.params1.z, instr.params1.w);
                    d = sdf_julia(p, center, instr.params0.w, c);
                }
                case 0x0Cu: { // Wedge: center.xyz, size.xyz, taper_axis, taper_dir
                    let size = vec3(instr.params0.w, instr.params1.x, instr.params1.y);
                    d = sdf_wedge(p, center, size, u32(instr.params1.z), u32(instr.params1.w));
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
                    case 0x14u: { result = sdf_smooth_subtract(a, b, 0.1); }
                    case 0x15u: { result = sdf_smooth_intersect(a, b, 0.1); }
                    default: { result = a; }
                }
                
                stack[sp] = result;
                sp++;
            }
        }
        else if instr.instr_type == 2u {
            // Modifier - operates on the top of stack distance value
            if sp >= 1u {
                let child_d = stack[sp - 1u];
                sp -= 1u;
                
                var result: f32 = child_d;
                switch instr.op {
                    case 0x23u: { // Round: subtract radius from distance
                        result = child_d - instr.params0.x;
                    }
                    case 0x25u: { // Voronoi: cell_size, wall_thickness, mode
                        let cell_size = instr.params0.x;
                        let wall_thickness = instr.params0.y;
                        let mode = u32(instr.params0.z);
                        let vor_d = voronoi_distance(p, cell_size) - wall_thickness;
                        if mode == 1u {
                            result = max(child_d, vor_d);      // intersect
                        } else {
                            result = max(child_d, -vor_d);     // subtract (default)
                        }
                    }
                    default: {
                        result = child_d;
                    }
                }
                
                stack[sp] = result;
                sp++;
            }
        }
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
// Raymarch result: color + depth for frag_depth output
struct RaymarchResult {
    color: vec4<f32>,
    depth: f32,        // clip-space depth (0..1), 1.0 = far/miss
}

fn raymarch(ro: vec3<f32>, rd: vec3<f32>) -> RaymarchResult {
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
    
    var result: RaymarchResult;
    
    if hit {
        let normal = compute_normal(hit_pos);
        
        // Validate normal (protect against NaN - NaN != NaN)
        var n = normal;
        let len = length(n);
        if len < 0.5 || len != len {
            n = vec3<f32>(0.0, 1.0, 0.0);
        } else {
            n = n / len;
        }
        
        // PBR lighting (SDF preview uses normal-based color + default material)
        let base_color = n * 0.5 + 0.5;
        let metallic_val = 0.0;   // SDF preview: dielectric default
        let roughness_val = 0.5;  // SDF preview: moderate roughness

        let V = normalize(uniforms.camera_pos - hit_pos);
        let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));

        let H = normalize(V + light_dir);
        let NdotL = max(dot(n, light_dir), 0.0);
        let NdotV = max(dot(n, V), 0.001);
        let NdotH = max(dot(n, H), 0.0);
        let HdotV = max(dot(H, V), 0.0);

        // GGX specular
        let a = roughness_val * roughness_val;
        let a2 = a * a;
        let d_term = NdotH * NdotH * (a2 - 1.0) + 1.0;
        let D = a2 / (3.14159 * d_term * d_term + 0.0001);

        let r_k = roughness_val + 1.0;
        let k = (r_k * r_k) / 8.0;
        let G1v = NdotV / (NdotV * (1.0 - k) + k + 0.0001);
        let G1l = NdotL / (NdotL * (1.0 - k) + k + 0.0001);
        let G = G1v * G1l;

        let F0 = mix(vec3<f32>(0.04), base_color, metallic_val);
        let F = F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - HdotV, 0.0, 1.0), 5.0);

        let specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);
        let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic_val);
        let diffuse = kD * base_color / 3.14159;

        let light_col = vec3<f32>(1.0, 0.98, 0.95) * 2.5;
        let Lo = (diffuse + specular) * light_col * NdotL;

        // Ambient occlusion + hemisphere ambient
        let ao = 0.5 + 0.5 * n.y;
        let ambient = mix(vec3<f32>(0.06, 0.05, 0.04), vec3<f32>(0.12, 0.14, 0.18), n.y * 0.5 + 0.5) * base_color;
        let color = (Lo + ambient) * ao;

        // Tonemap (Reinhard) + gamma
        let mapped = color / (color + vec3<f32>(1.0));
        let gamma_out = pow(mapped, vec3<f32>(1.0 / 2.2));

        result.color = vec4<f32>(gamma_out, 1.0);
        
        // Project hit position to clip space for depth buffer
        let clip = uniforms.view_proj * vec4<f32>(hit_pos, 1.0);
        result.depth = clip.z / clip.w;
    } else {
        // Sky gradient for miss
        let sky_t = rd.y * 0.5 + 0.5;
        let sky_color = mix(
            vec3<f32>(0.1, 0.12, 0.15),  // Horizon
            vec3<f32>(0.15, 0.2, 0.3),   // Zenith
            sky_t
        );
        result.color = vec4<f32>(sky_color, 1.0);
        result.depth = 1.0;  // Far plane for sky
    }
    
    return result;
}

// Fullscreen quad vertex shader
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fragment output with depth
struct FragOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
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
fn fs_sdf(in: VertexOutput) -> FragOutput {
    let ndc = in.uv;
    
    // Reconstruct ray direction using inverse view-projection
    // WGPU NDC is Y-up, Depth is [0, 1]
    let near_point = uniforms.inv_view_proj * vec4(ndc, 0.0, 1.0);
    let far_point = uniforms.inv_view_proj * vec4(ndc, 1.0, 1.0);
    
    let ray_origin = near_point.xyz / near_point.w;
    let ray_dir = normalize(far_point.xyz / far_point.w - ray_origin);
    
    let result = raymarch(ray_origin, ray_dir);
    
    var out: FragOutput;
    out.color = result.color;
    out.depth = result.depth;
    return out;
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
    out.color = in.normal * 0.5 + 0.5;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let NdotL = max(dot(n, light_dir), 0.0);

    // Simple Blinn-Phong for mesh preview (no camera_pos in mesh uniforms)
    let diffuse = NdotL * 0.75;
    let ambient = mix(vec3<f32>(0.08, 0.07, 0.06), vec3<f32>(0.14, 0.16, 0.20), n.y * 0.5 + 0.5);
    let ao = 0.5 + 0.5 * n.y;
    let lit_color = (in.color * diffuse + ambient) * ao;

    // Tonemap + gamma
    let mapped = lit_color / (lit_color + vec3<f32>(1.0));
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(gamma, 1.0);
}
"#;

// Splat and Volume shaders moved to shaders_extra.rs for file size management.
// Re-export for backward compatibility.
pub use super::shaders_extra::{SPLAT_SHADER, VOLUME_SHADER};
