//! Gizmo SDF raymarching - cylinder + cone axes using sd_cylinder, sd_cone, sd_sphere.
//! Included via concat with shaders_sdf.wgsl in gizmos.rs.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct GizmoUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
    gizmo_pos: vec3<f32>,
    _pad1: f32,
    viewport: vec2<f32>,
    _pad2: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: GizmoUniforms;

const GIZMO_AXIS_LEN: f32 = 0.3;
const GIZMO_CENTER_RADIUS: f32 = 0.008;
const GIZMO_SHAFT_THICK: f32 = 0.006;
const GIZMO_CONE_HEIGHT: f32 = 0.03;
const GIZMO_CONE_RADIUS: f32 = 0.012;
const SHAFT_LEN: f32 = GIZMO_AXIS_LEN - GIZMO_CONE_HEIGHT;

// sd_cylinder is Y-up: radial=length(xz), axis=y, symmetric -h to +h.
// sd_cone is Y-up: base at y=0, tip at y=h.
// Swizzle p so our axis maps to their Y.

fn sd_axis_x(p: vec3<f32>) -> f32 {
    // X-axis: sd_cylinder radial=length(xz), axis=y. Our radial=length(yz), axis=x.
    let p_cyl = vec3<f32>(p.y, p.x - SHAFT_LEN * 0.5, p.z);
    let shaft = sd_cylinder(p_cyl, GIZMO_SHAFT_THICK, SHAFT_LEN * 0.5);
    let p_cone = vec3<f32>(p.y, p.x - SHAFT_LEN, p.z);
    let cone = sd_cone(p_cone, GIZMO_CONE_RADIUS, GIZMO_CONE_HEIGHT);
    return min(shaft, cone);
}

fn sd_axis_y(p: vec3<f32>) -> f32 {
    // Y-axis: sd_cylinder/sd_cone are Y-up; radial=length(xz), axis=y
    let p_cyl = vec3<f32>(p.x, p.y - SHAFT_LEN * 0.5, p.z);
    let shaft = sd_cylinder(p_cyl, GIZMO_SHAFT_THICK, SHAFT_LEN * 0.5);
    let p_cone = vec3<f32>(p.x, p.y - SHAFT_LEN, p.z);
    let cone = sd_cone(p_cone, GIZMO_CONE_RADIUS, GIZMO_CONE_HEIGHT);
    return min(shaft, cone);
}

fn sd_axis_z(p: vec3<f32>) -> f32 {
    // Z-axis: swap so our Z -> their Y; their xz = our xy
    let p_cyl = vec3<f32>(p.x, p.z - SHAFT_LEN * 0.5, p.y);
    let shaft = sd_cylinder(p_cyl, GIZMO_SHAFT_THICK, SHAFT_LEN * 0.5);
    let p_cone = vec3<f32>(p.x, p.z - SHAFT_LEN, p.y);
    let cone = sd_cone(p_cone, GIZMO_CONE_RADIUS, GIZMO_CONE_HEIGHT);
    return min(shaft, cone);
}

fn sd_gizmo(p_local: vec3<f32>) -> f32 {
    let center = sd_sphere(p_local, GIZMO_CENTER_RADIUS);
    let ax = sd_axis_x(p_local);
    let ay = sd_axis_y(p_local);
    let az = sd_axis_z(p_local);
    return min(center, min(ax, min(ay, az)));
}

// Assign one color per axis from position (not normal), so cones stay single-colored.
fn get_color(p_local: vec3<f32>) -> vec3<f32> {
    if (length(p_local) < GIZMO_CENTER_RADIUS * 1.5) {
        return vec3<f32>(0.95, 0.95, 0.95);
    }
    if (p_local.x >= p_local.y && p_local.x >= p_local.z && p_local.x > 0.001) {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    if (p_local.y >= p_local.x && p_local.y >= p_local.z && p_local.y > 0.001) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
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

    let gizmo_pos = uniforms.gizmo_pos;

    var t: f32 = 0.0;
    let tmax = 50.0;
    let eps = 0.0004;

    for (var i = 0u; i < 80u; i++) {
        let p_world = ro + rd * t;
        let p_local = p_world - gizmo_pos;
        let d = sd_gizmo(p_local);

        if (d < eps) {
            let col = get_color(p_local);
            return vec4<f32>(col, 1.0);
        }

        t += d;
        if (t > tmax) { break; }
    }

    discard;
}
