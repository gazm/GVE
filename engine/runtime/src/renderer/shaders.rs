//! WGSL shader sources for the renderer
//!
//! Contains embedded shader code for mesh and other pipelines.
//! SDF shaders have been removed in v2.3.

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
