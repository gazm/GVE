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
    view: mat4x4<f32>,
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

    // Splat normal from quaternion (local Z axis)
    let R = quat_to_mat3(instance.rotation);
    out.normal = normalize(R * vec3<f32>(0.0, 0.0, 1.0));

    // Billboard with camera-facing quads
    let cam_right = vec3<f32>(uniforms.view[0][0], uniforms.view[1][0], uniforms.view[2][0]);
    let cam_up    = vec3<f32>(uniforms.view[0][1], uniforms.view[1][1], uniforms.view[2][1]);
    let max_scale = max(instance.scale.x, max(instance.scale.y, instance.scale.z));

    let world_pos = instance.center
        + cam_right * quad_x * max_scale * 2.0
        + cam_up    * quad_y * max_scale * 2.0;

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

    // Key light
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let light_col = vec3<f32>(1.0, 0.98, 0.95) * 2.5;

    var Lo = pbr_lighting(in.color.rgb, in.metallic, in.roughness, N, V, light_dir, light_col);

    // Fill light (soft, from below-left)
    let fill_dir = normalize(vec3<f32>(-0.3, -0.2, -0.5));
    let fill_col = vec3<f32>(0.3, 0.35, 0.5) * 0.6;
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
/// Used for visualizing baked VDB/dense grid data.
pub const VOLUME_SHADER: &str = r#"
struct VolumeUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
    bounds_min: vec3<f32>,
    _pad1: f32,
    bounds_max: vec3<f32>,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> uniforms: VolumeUniforms;
@group(0) @binding(1) var volume_texture: texture_3d<f32>;
@group(0) @binding(2) var volume_sampler: sampler;

// Sample SDF from 3D texture (world position -> distance)
fn sample_sdf(world_pos: vec3<f32>) -> f32 {
    // Transform world position to UV [0,1] coordinates
    let uv = (world_pos - uniforms.bounds_min) / (uniforms.bounds_max - uniforms.bounds_min);
    
    // Clamp to valid range
    let uv_clamped = clamp(uv, vec3<f32>(0.001), vec3<f32>(0.999));
    
    return textureSampleLevel(volume_texture, volume_sampler, uv_clamped, 0.0).r;
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

// Sphere tracing through the volume
fn raymarch_volume(ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
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
        return vec4<f32>(sky_color, 1.0);
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

        let base_color = n * 0.5 + 0.5;
        let V = normalize(uniforms.camera_pos - hit_pos);
        let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
        let H = normalize(V + light_dir);
        let NdotL = max(dot(n, light_dir), 0.0);
        let NdotV = max(dot(n, V), 0.001);
        let NdotH = max(dot(n, H), 0.0);
        let HdotV = max(dot(H, V), 0.0);

        // GGX specular (roughness = 0.5 for volume preview)
        let roughness_val = 0.5;
        let a = roughness_val * roughness_val;
        let a2 = a * a;
        let d_term = NdotH * NdotH * (a2 - 1.0) + 1.0;
        let D = a2 / (3.14159 * d_term * d_term + 0.0001);
        let r_k = roughness_val + 1.0;
        let k = (r_k * r_k) / 8.0;
        let G = (NdotV / (NdotV * (1.0 - k) + k + 0.0001))
              * (NdotL / (NdotL * (1.0 - k) + k + 0.0001));
        let F0 = vec3<f32>(0.04);
        let F = F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - HdotV, 0.0, 1.0), 5.0);
        let specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);
        let diffuse = base_color / 3.14159;

        let light_col = vec3<f32>(1.0, 0.98, 0.95) * 2.5;
        let Lo = (diffuse + specular) * light_col * NdotL;

        let ao = 0.5 + 0.5 * n.y;
        let ambient = mix(vec3<f32>(0.06, 0.05, 0.04), vec3<f32>(0.12, 0.14, 0.18), n.y * 0.5 + 0.5) * base_color;
        let color = (Lo + ambient) * ao;
        let mapped = color / (color + vec3<f32>(1.0));
        let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));

        return vec4<f32>(gamma, 1.0);
    }
    
    // Sky gradient for miss
    let sky_t = rd.y * 0.5 + 0.5;
    let sky_color = mix(
        vec3<f32>(0.1, 0.12, 0.15),
        vec3<f32>(0.15, 0.2, 0.3),
        sky_t
    );
    return vec4<f32>(sky_color, 1.0);
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

@fragment
fn fs_volume(in: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = in.uv;
    
    // Reconstruct ray direction using inverse view-projection
    let near_point = uniforms.inv_view_proj * vec4(ndc, 0.0, 1.0);
    let far_point = uniforms.inv_view_proj * vec4(ndc, 1.0, 1.0);
    
    let ray_origin = near_point.xyz / near_point.w;
    let ray_dir = normalize(far_point.xyz / far_point.w - ray_origin);
    
    return raymarch_volume(ray_origin, ray_dir);
}
"#;
