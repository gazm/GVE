//! Additional WGSL shader sources (Gaussian Splat + Volume Raymarching)
//!
//! Split from shaders.rs to keep files under 500 lines.
//! See shaders.rs for SDF and Mesh shaders.

/// Gaussian Splat Shader
///
/// Renders 3D Gaussians using instancing.
/// Instances provide position, scale, rotation (quat), and color.
/// Vertex shader computes the 2D covariance matrix to determine the quad bounds/orientation.
pub const SPLAT_SHADER: &str = r#"
const PI: f32 = 3.14159265359;

struct Uniforms {
    mvp: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    camera_pos: vec3<f32>,
    viewport: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct SplatInstance {
    @location(0) center: vec3<f32>,
    @location(1) scale: vec3<f32>,
    @location(2) rotation: vec4<f32>,  // quaternion (x, y, z, w)
    @location(3) color_packed: u32,    // Oklab8+A: [L, a, b, alpha]
    @location(4) mat_packed: u32,      // metallic(8) | roughness(8) | flags(8) | pad(8)
}

// Decode Oklab u8 -> linear RGB.
// Quantisation: L in [0,1]->u8, a/b in [-0.4,0.4] remapped to [0,255].
fn oklab_to_linear_rgb(L: f32, a: f32, b: f32) -> vec3<f32> {
    // Inverse of M2: Oklab -> LMS (cube-root space)
    let l_ = L + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = L - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = L - 0.0894841775 * a - 1.2914855480 * b;

    // Cube (undo cube-root)
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    // Inverse of M1: LMS -> linear sRGB
    return vec3<f32>(
         4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    );
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) normal: vec3<f32>,
    @location(4) metallic: f32,
    @location(5) roughness: f32,
}

// ── Quaternion helpers ──────────────────────────────────────────────────

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    let x2 = x + x; let y2 = y + y; let z2 = z + z;
    let xx = x * x2; let xy = x * y2; let xz = x * z2;
    let yy = y * y2; let yz = y * z2; let zz = z * z2;
    let wx = w * x2; let wy = w * y2; let wz = w * z2;
    return mat3x3<f32>(
        vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy),
        vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx),
        vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy))
    );
}

// ── PBR: Cook-Torrance BRDF ────────────────────────────────────────────

fn D_GGX(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 0.0001);
}

fn G_SchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k + 0.0001);
}

fn G_Smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    return G_SchlickGGX(NdotV, roughness) * G_SchlickGGX(NdotL, roughness);
}

fn F_Schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn pbr_lighting(
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    light_color: vec3<f32>,
) -> vec3<f32> {
    let H = normalize(V + L);
    let NdotL = max(dot(N, L), 0.0);
    let NdotV = max(dot(N, V), 0.001);
    let NdotH = max(dot(N, H), 0.0);
    let HdotV = max(dot(H, V), 0.0);

    // F0: dielectric = 0.04, metals use base_color
    let F0 = mix(vec3<f32>(0.04), base_color, metallic);

    let D = D_GGX(NdotH, roughness);
    let G = G_Smith(NdotV, NdotL, roughness);
    let F = F_Schlick(HdotV, F0);

    // Specular (Cook-Torrance)
    let spec_num = D * G * F;
    let spec_den = 4.0 * NdotV * NdotL + 0.0001;
    let specular = spec_num / spec_den;

    // Energy conservation: diffuse decreases for metals
    let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);

    // Lambertian diffuse
    let diffuse = kD * base_color / PI;

    return (diffuse + specular) * light_color * NdotL;
}

// ── Vertex Shader ──────────────────────────────────────────────────────

@vertex
fn vs_main(
    @builtin(vertex_index) v_idx: u32,
    instance: SplatInstance
) -> VertexOutput {
    var out: VertexOutput;

    // Dequantise Oklab u8 -> float, then convert to linear RGB for PBR
    let L_q   = f32(instance.color_packed & 0xFFu) / 255.0;           // L  [0, 1]
    let a_q   = f32((instance.color_packed >> 8u)  & 0xFFu) / 255.0;  // a  [0, 1] packed
    let b_q   = f32((instance.color_packed >> 16u) & 0xFFu) / 255.0;  // b  [0, 1] packed
    let alpha = f32((instance.color_packed >> 24u) & 0xFFu) / 255.0;

    // Remap a, b from [0, 1] back to [-0.4, 0.4]
    let ok_a = a_q * 0.8 - 0.4;
    let ok_b = b_q * 0.8 - 0.4;

    let linear_rgb = clamp(oklab_to_linear_rgb(L_q, ok_a, ok_b), vec3<f32>(0.0), vec3<f32>(1.0));
    out.color = vec4<f32>(linear_rgb, alpha);

    // Unpack PBR properties: metallic | roughness | flags | pad
    out.metallic  = f32(instance.mat_packed & 0xFFu) / 255.0;
    out.roughness = f32((instance.mat_packed >> 8u) & 0xFFu) / 255.0;

    // Quad UV
    let quad_x = f32(v_idx & 1u) * 2.0 - 1.0;
    let quad_y = f32((v_idx >> 1u) & 1u) * 2.0 - 1.0;
    out.uv = vec2<f32>(quad_x, quad_y);

    // Splat orientation from quaternion (local Z = normal, X/Y = tangent plane)
    let R = quat_to_mat3(instance.rotation);
    out.normal = normalize(R * vec3<f32>(0.0, 0.0, 1.0));

    // Orient quad in splat's tangent plane (attached to SDF) instead of billboarding
    let tangent_x = R[0];
    let tangent_y = R[1];
    let world_pos = instance.center
        + tangent_x * quad_x * instance.scale.x * 2.0
        + tangent_y * quad_y * instance.scale.y * 2.0;

    out.world_pos = world_pos;
    out.clip_position = uniforms.mvp * vec4<f32>(world_pos, 1.0);
    return out;
}

// ── Fragment Shader ────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) { discard; }

    let gauss_alpha = exp(-2.0 * r2) * in.color.a;
    if (gauss_alpha < 0.01) { discard; }

    let N = normalize(in.normal);
    let V = normalize(uniforms.camera_pos - in.world_pos);

    // Key light (reduced so dark materials stay dark)
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let light_col = vec3<f32>(1.0, 0.98, 0.95) * 1.2;

    var Lo = pbr_lighting(in.color.rgb, in.metallic, in.roughness, N, V, light_dir, light_col);

    // Fill light (soft, from below-left)
    let fill_dir = normalize(vec3<f32>(-0.3, -0.2, -0.5));
    let fill_col = vec3<f32>(0.3, 0.35, 0.5) * 0.35;
    Lo += pbr_lighting(in.color.rgb, in.metallic, in.roughness, N, V, fill_dir, fill_col);

    // Ambient (simple hemisphere)
    let ambient_top = vec3<f32>(0.12, 0.14, 0.18);
    let ambient_bot = vec3<f32>(0.06, 0.05, 0.04);
    let ambient = mix(ambient_bot, ambient_top, N.y * 0.5 + 0.5) * in.color.rgb;

    let color = Lo + ambient;

    // Tonemap (Reinhard) + gamma
    let mapped = color / (color + vec3<f32>(1.0));
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma, gauss_alpha);
}
"#;

/// Volume Raymarching Shader
///
/// Raymarches a 3D texture containing SDF distance values.
/// When use_triplanar is 1, samples triplanar textures at hit for base color.
pub const VOLUME_SHADER: &str = concat!(
    include_str!("shaders_sdf.wgsl"),
    r#"
struct RuntimeVolumeOp {
    op_type: u32,
    _pad0_x: u32,
    _pad0_y: u32,
    _pad0_z: u32,
    pos: vec3<f32>,
    _pad1: u32,
    params_a: vec4<f32>,
    params_b: vec4<f32>,
    aabb_min: vec3<f32>,
    _pad2: u32,
    aabb_max: vec3<f32>,
    _pad3: u32,
}

struct VolumeUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
    bounds_min: vec3<f32>,
    _pad1: f32,
    bounds_max: vec3<f32>,
    _pad2: f32,
    view_proj: mat4x4<f32>,
    use_triplanar: u32,
    triplanar_bounds_min: vec3<f32>,
    active_op_count: u32,
    triplanar_bounds_max: vec3<f32>,
    _pad4: f32,
    ops: array<RuntimeVolumeOp, 16>,
}

@group(0) @binding(0) var<uniform> uniforms: VolumeUniforms;
@group(0) @binding(1) var volume_texture: texture_3d<f32>;
@group(0) @binding(2) var volume_sampler: sampler;
@group(0) @binding(3) var triplanar_xy: texture_2d<f32>;
@group(0) @binding(4) var triplanar_xz: texture_2d<f32>;
@group(0) @binding(5) var triplanar_yz: texture_2d<f32>;
@group(0) @binding(6) var triplanar_sampler: sampler;

// ── Runtime Operation Math ──────────────────────────────────────────────

fn apply_op(d_in: f32, p: vec3<f32>, op: RuntimeVolumeOp) -> f32 {
    let p_local = p - op.pos;
    var d_op: f32 = 1000.0;

    // Primitives must match binary_format.rs PrimitiveOp enum
    switch (op.op_type) {
        case 1u: { // Sphere: [r, 0, 0, 0]
            d_op = sd_sphere(p_local, op.params_a.x);
        }
        case 2u: { // Box: [sx, sy, sz, 0]
            d_op = sd_box(p_local, op.params_a.xyz);
        }
        case 3u: { // Cylinder: [r, h, 0, 0]
            d_op = sd_cylinder(p_local, op.params_a.x, op.params_a.y);
        }
        case 4u: { // Capsule: [r, h, 0, 0]
            d_op = sd_capsule(p_local, op.params_a.x, op.params_a.y);
        }
        case 5u: { // Torus: [major, minor, 0, 0]
            d_op = sd_torus(p_local, op.params_a.x, op.params_a.y);
        }
        case 6u: { // Cone: [r, h, 0, 0]
            d_op = sd_cone(p_local, op.params_a.x, op.params_a.y);
        }
        case 7u: { // Plane: [nx, ny, nz, dist]
            d_op = sd_plane(p_local, op.params_a.xyz, op.params_a.w);
        }
        case 8u: { // Revolution: [offset, axis, 0, 0] -> axis cast to u32?
            // params: [offset, profile_w, profile_h, axis_flag]
            // Revolution modifies space, it doesn't return distance directly easily?
            // Actually it maps p -> p_profile.
            let prof = sd_revolution(p_local, op.params_a.x, u32(op.params_a.w));
            // Assume profile is box for now as runtime op?
            // Or generic revolution? For now treating as Ring/Torus-like shell
             d_op = sd_box(prof, vec3<f32>(op.params_a.y, op.params_a.z, 0.01));
        }
        case 9u: { // Mandelbulb: [scale, power, iter, 0]
            d_op = sd_mandelbulb(p_local, op.params_a.y, op.params_a.x);
        }
        case 10u: { // Menger: [scale, iter, 0, 0]
            d_op = sd_menger(p_local, op.params_a.x);
        }
        case 11u: { // Julia: [scale, c0, c1, c2, c3]
            d_op = sd_julia(p_local, op.params_b, op.params_a.x);
        }
        default: {
            d_op = 1000.0;
        }
    }
    
    // For now, all runtime ops are SUBTRACTIVE (machining)
    return max(d_in, -d_op);
}


// ── Volume Sampling ─────────────────────────────────────────────────────

// Sample SDF from 3D texture (world position -> distance).
// Space: hit_pos, bounds_min, triplanar_bounds all share same world space (no model matrix).
fn sample_sdf(world_pos: vec3<f32>) -> f32 {
    // Transform world position to UV [0,1] coordinates
    let uv = (world_pos - uniforms.bounds_min) / (uniforms.bounds_max - uniforms.bounds_min);
    
    // Clamp to valid range
    let uv_clamped = clamp(uv, vec3<f32>(0.001), vec3<f32>(0.999));
    
    var d = textureSampleLevel(volume_texture, volume_sampler, uv_clamped, 0.0).r;

    // Apply runtime operations
    if (uniforms.active_op_count > 0u) {
        for (var i = 0u; i < uniforms.active_op_count; i++) {
            let op = uniforms.ops[i];
            // AABB Culling: only evaluate expensive SDF if point is inside op bounds
            let inside = all(world_pos >= op.aabb_min) && all(world_pos <= op.aabb_max);
            if (inside) {
                d = apply_op(d, world_pos, op);
            }
        }
    }
    return d;
}

// Compute gradient (normal) via central differences
fn compute_gradient(p: vec3<f32>) -> vec3<f32> {
    let extent = uniforms.bounds_max - uniforms.bounds_min;
    let h = max(extent.x, max(extent.y, extent.z)) * 0.002; // Small step relative to bounds
    
    let dx = sample_sdf(p + vec3<f32>(h, 0.0, 0.0)) - sample_sdf(p - vec3<f32>(h, 0.0, 0.0));
    let dy = sample_sdf(p + vec3<f32>(0.0, h, 0.0)) - sample_sdf(p - vec3<f32>(0.0, h, 0.0));
    let dz = sample_sdf(p + vec3<f32>(0.0, 0.0, h)) - sample_sdf(p - vec3<f32>(0.0, 0.0, h));
    
    return normalize(vec3<f32>(dx, dy, dz));
}

// Sample triplanar textures at world position; blend by normal. Returns vec4(rgb, packed_alpha).
// Use volume bounds so UVs match hit_pos coordinate space (same as SDF).
fn sample_triplanar(p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
    let bmin = uniforms.triplanar_bounds_min;
    let bmax = uniforms.triplanar_bounds_max;
    let extent = bmax - bmin;
    let uv_xy = (p.xy - bmin.xy) / (extent.xy + vec2<f32>(0.0001));
    let uv_xz = (p.xz - bmin.xz) / (vec2<f32>(extent.x, extent.z) + vec2<f32>(0.0001));
    let uv_yz = (p.yz - bmin.yz) / (vec2<f32>(extent.y, extent.z) + vec2<f32>(0.0001));
    let uv_xy_c = clamp(uv_xy, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    let uv_xz_c = clamp(uv_xz, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    let uv_yz_c = clamp(uv_yz, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    let c_xy = textureSampleLevel(triplanar_xy, triplanar_sampler, uv_xy_c, 0.0);
    let c_xz = textureSampleLevel(triplanar_xz, triplanar_sampler, uv_xz_c, 0.0);
    let c_yz = textureSampleLevel(triplanar_yz, triplanar_sampler, uv_yz_c, 0.0);
    let w = abs(n);
    let total = w.x + w.y + w.z + 0.0001;
    let rgb = (c_xy.rgb * w.z + c_xz.rgb * w.y + c_yz.rgb * w.x) / total;
    let packed_alpha = (c_xy.a * w.z + c_xz.a * w.y + c_yz.a * w.x) / total;
    return vec4<f32>(rgb, packed_alpha);
}

// Ray-box intersection (returns t_near, t_far)
fn intersect_box(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
    let inv_rd = 1.0 / rd;
    let t1 = (uniforms.bounds_min - ro) * inv_rd;
    let t2 = (uniforms.bounds_max - ro) * inv_rd;
    
    let t_min = min(t1, t2);
    let t_max = max(t1, t2);
    
    let t_near = max(max(t_min.x, t_min.y), t_min.z);
    let t_far = min(min(t_max.x, t_max.y), t_max.z);
    
    return vec2<f32>(t_near, t_far);
}

struct VolumeHit {
    color: vec4<f32>,
    depth: f32,
}

// Sphere tracing through the volume
fn F_Schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn raymarch_volume(ro: vec3<f32>, rd: vec3<f32>) -> VolumeHit {
    // First, intersect with bounding box
    let t_bounds = intersect_box(ro, rd);
    if (t_bounds.x > t_bounds.y || t_bounds.y < 0.0) {
        // Ray misses bounding box - return sky
        let sky_t = rd.y * 0.5 + 0.5;
        let sky_color = mix(
            vec3<f32>(0.1, 0.12, 0.15),
            vec3<f32>(0.15, 0.2, 0.3),
            sky_t
        );
        return VolumeHit(
            vec4<f32>(sky_color, 1.0),
            1.0
        );
    }
    
    // Start at near intersection (or camera if inside)
    var t = max(t_bounds.x, 0.001);
    let t_max = t_bounds.y;
    
    let max_steps = 128;
    let surface_dist = 0.001;
    let step_factor = 0.9;
    
    var hit = false;
    var hit_pos = vec3<f32>(0.0);
    
    for (var i = 0; i < max_steps; i++) {
        let p = ro + rd * t;
        let d = sample_sdf(p);
        
        if (d < surface_dist) {
            hit = true;
            hit_pos = p;
            break;
        }
        
        if (t > t_max) {
            break;
        }
        
        // Adaptive step based on distance
        t += max(abs(d) * step_factor, 0.001);
    }
    
    if (hit) {
        let normal = compute_gradient(hit_pos);

        var n = normal;
        let len = length(n);
        if (len < 0.5 || len != len) {
            n = vec3<f32>(0.0, 1.0, 0.0);
        }

        var base_color: vec3<f32>;
        var roughness_val: f32 = 0.5;
        var metallic_val: f32 = 0.0;
        
        if (uniforms.use_triplanar != 0u) {
            let tri = sample_triplanar(hit_pos, n);
            base_color = tri.rgb;
            
            // Unpack 4-bit roughness and metallic from alpha
            // packed_alpha is [0..1], quantised to 255 levels.
            // val = (rough << 4) | metal
            let val = u32(tri.a * 255.0 + 0.5);
            let rough_4 = (val >> 4u) & 0x0Fu;
            let metal_4 = val & 0x0Fu;
            
            roughness_val = f32(rough_4) / 15.0;
            metallic_val = f32(metal_4) / 15.0;
            
            // If the texture was baked with old code (alpha=1.0=255 -> rough=15, metal=15),
            // this yields rough=1.0, metal=1.0. 
            // Most old bakes have alpha=1.0. This is acceptable or we could heuristic check.
            // Actually old bakes stored roughness directly in alpha. If roughness was 0.5 -> 128.
            // 128 = 1000 0000 -> rough=8 (0.53), metal=0.
            // So it actually maps decently!
            
        } else {
            base_color = n * 0.5 + 0.5;
        }
        let V = normalize(uniforms.camera_pos - hit_pos);
        let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
        let fill_dir = normalize(vec3<f32>(-0.35, 0.4, -0.25));
        let H = normalize(V + light_dir);
        let NdotL_key = max(dot(n, light_dir), 0.0);
        let NdotL_fill = max(dot(n, fill_dir), 0.0);
        let NdotV = max(dot(n, V), 0.001);
        let NdotH = max(dot(n, H), 0.0);
        let HdotV = max(dot(H, V), 0.0);
        let wrap = 0.25;
        let NdotL = (NdotL_key + wrap) / (1.0 + wrap);

        // GGX specular + energy-conserving diffuse (match mesh shader)
        let a = roughness_val * roughness_val;
        let a2 = a * a;
        let d_term = NdotH * NdotH * (a2 - 1.0) + 1.0;
        let D = a2 / (3.14159 * d_term * d_term + 0.0001);
        let r_k = roughness_val + 1.0;
        let k = (r_k * r_k) / 8.0;
        let G = (NdotV / (NdotV * (1.0 - k) + k + 0.0001))
              * (NdotL / (NdotL * (1.0 - k) + k + 0.0001));
              
        // F0: dielectric=0.04, metal=base_color
        let F0 = mix(vec3<f32>(0.04), base_color, metallic_val);
        
        let F = F_Schlick(HdotV, F0);
        
        let specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);
        
        // Energy conservation: diffuse decreases for metals and high specular
        let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic_val);
        
        let diffuse = kD * base_color / 3.14159;

        let light_col = vec3<f32>(1.0, 0.98, 0.95) * 2.0;
        let fill_col = vec3<f32>(0.6, 0.65, 0.75) * 0.5;
        let Lo = (diffuse + specular) * light_col * NdotL + diffuse * fill_col * NdotL_fill;
        // Lower ambient floor so shadows can go to black instead of grey
        let ambient = mix(vec3<f32>(0.04, 0.035, 0.03), vec3<f32>(0.14, 0.16, 0.20), n.y * 0.5 + 0.5) * base_color;
        let ao = 0.45 + 0.55 * (n.y * 0.5 + 0.5);
        var color = (Lo + ambient) * ao;
        
        let mapped = color / (color + vec3<f32>(1.0));
        let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));

        // Project hit position to clip space for depth buffer (NDC -1..1 -> 0..1)
        let clip = uniforms.view_proj * vec4<f32>(hit_pos, 1.0);
        let ndc_z = clip.z / clip.w;
        return VolumeHit(
            vec4<f32>(gamma, 1.0),
            ndc_z * 0.5 + 0.5
        );
    }
    
    // Sky gradient for miss
    let sky_t = rd.y * 0.5 + 0.5;
    let sky_color = mix(
        vec3<f32>(0.1, 0.12, 0.15),
        vec3<f32>(0.15, 0.2, 0.3),
        sky_t
    );
    return VolumeHit(
        vec4<f32>(sky_color, 1.0),
        1.0
    );
}

// Fullscreen triangle vertex shader
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let uv = vec2(f32((vertex_idx << 1u) & 2u), f32(vertex_idx & 2u));
    out.position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = uv * 2.0 - 1.0;
    return out;
}

struct FragOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_volume(in: VertexOutput) -> FragOutput {
    let ndc = in.uv;
    
    // Reconstruct ray direction using inverse view-projection
    let near_point = uniforms.inv_view_proj * vec4(ndc, 0.0, 1.0);
    let far_point = uniforms.inv_view_proj * vec4(ndc, 1.0, 1.0);
    
    let ray_origin = near_point.xyz / near_point.w;
    let ray_dir = normalize(far_point.xyz / far_point.w - ray_origin);
    
    let result = raymarch_volume(ray_origin, ray_dir);
    var out: FragOutput;
    out.color = result.color;
    out.depth = result.depth;
    return out;
}
"#);
