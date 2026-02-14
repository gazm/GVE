//! Bone gizmo SDF - spheres at joints, thin capsules between parent and child.
//! Included via concat with shaders_sdf.wgsl.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct BoneGizmoUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    bone_count: u32,
    _pad: u32,
    // positions[64] = 64 * 16 bytes (vec3 + pad) = 1024
    positions: array<vec4<f32>, 64>,
}

@group(0) @binding(0)
var<uniform> uniforms: BoneGizmoUniforms;

const JOINT_RADIUS: f32 = 0.015;
const BONE_RADIUS: f32 = 0.008;
const ROOT_PARENT: u32 = 0xFFFFu;

fn sd_bone_segment(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    let q = pa - ba * h;
    return length(q) - BONE_RADIUS;
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let uv = vec2<f32>(f32((in_vertex_index << 1u) & 2u), f32(in_vertex_index & 2u));
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = uv * 2.0 - 1.0;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = vec4<f32>(in.uv.x, in.uv.y, 0.0, 1.0);
    let near_clip = uniforms.inv_view_proj * ndc;
    let near_w = 1.0 / near_clip.w;
    let ro = (near_clip * near_w).xyz;

    let ndc_far = vec4<f32>(in.uv.x, in.uv.y, 1.0, 1.0);
    let far_clip = uniforms.inv_view_proj * ndc_far;
    let far_w = 1.0 / far_clip.w;
    let rd = normalize((far_clip * far_w).xyz - ro);

    let n = min(uniforms.bone_count, 64u);
    var t: f32 = 0.0;
    let tmax = 50.0;
    let eps = 0.0003;

    for (var step = 0u; step < 80u; step++) {
        let p = ro + rd * t;
        var d: f32 = 1000.0;

        for (var i = 0u; i < n; i++) {
            let pos = uniforms.positions[i].xyz;
            d = min(d, sd_sphere(p - pos, JOINT_RADIUS));
        }

        // Segments: parent_idx stored in positions[i].w (as f32; 65535 = no parent)
        for (var i = 0u; i < n; i++) {
            let parent_f = uniforms.positions[i].w;
            if (parent_f < 65534.0) {
                let parent_idx = u32(parent_f);
                if (parent_idx < n) {
                    let a = uniforms.positions[parent_idx].xyz;
                    let b = uniforms.positions[i].xyz;
                    d = min(d, sd_bone_segment(p, a, b));
                }
            }
        }

        if (d < eps) {
            return vec4<f32>(0.2, 0.8, 1.0, 0.9);  // Cyan tint, distinct from XYZ gizmo
        }

        t += d;
        if (t > tmax) { break; }
    }

    discard;
}
