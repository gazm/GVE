struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> rotation: mat4x4<f32>;

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Fullscreen triangle to cover viewport
    let uv = vec2<f32>(f32((in_vertex_index << 1u) & 2u), f32(in_vertex_index & 2u));
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = uv * 2.0 - 1.0;
    return out;
}

fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = abs(p) - b;
    return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}

fn calcNormal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(0.001, 0.0);
    let b = vec3<f32>(0.5); // Box size
    return normalize(vec3<f32>(
        sdBox(p + e.xyy, b) - sdBox(p - e.xyy, b),
        sdBox(p + e.yxy, b) - sdBox(p - e.yxy, b),
        sdBox(p + e.yyx, b) - sdBox(p - e.yyx, b)
    ));
}

fn getFaceColor(normal: vec3<f32>) -> vec3<f32> {
    let e = 0.5;
    if (normal.x > e) { return vec3<f32>(1.0, 0.0, 0.0); } // Right (Red)
    if (normal.x < -e) { return vec3<f32>(0.5, 0.0, 0.0); } // Left (Dark Red)
    if (normal.y > e) { return vec3<f32>(0.0, 1.0, 0.0); } // Top (Green)
    if (normal.y < -e) { return vec3<f32>(0.0, 0.5, 0.0); } // Bottom (Dark Green)
    if (normal.z > e) { return vec3<f32>(0.0, 0.0, 1.0); } // Front (Blue)
    if (normal.z < -e) { return vec3<f32>(1.0, 1.0, 0.0); } // Back (Yellow)
    return vec3<f32>(1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Camera Setup (Fixed looking at 0,0,0 from 0,0,3)
    let ro = vec3<f32>(0.0, 0.0, 3.0);
    let look_at = vec3<f32>(0.0, 0.0, 0.0);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    
    let fwd = normalize(look_at - ro);
    let right = normalize(cross(fwd, up));
    let new_up = cross(right, fwd);
    
    let rd = normalize(in.uv.x * right + in.uv.y * new_up + 1.5 * fwd); 
    
    // Rotate Ray to World Space (Inverse View Rotation)
    // rotation uniform is ViewMatrix (World->View). Inverse is Transpose (for rotation matrix).
    let inv_rot = transpose(mat3x3<f32>(rotation[0].xyz, rotation[1].xyz, rotation[2].xyz));
    
    let ro_local = inv_rot * ro;
    let rd_local = inv_rot * rd;
    
    // Raymarch
    var t = 0.0;
    var tmax = 6.0;
    for (var i = 0; i < 64; i++) {
        let p = ro_local + rd_local * t;
        let d = sdBox(p, vec3<f32>(0.5));
        if (d < 0.001) {
            let n = calcNormal(p);
            let col = getFaceColor(n);
            return vec4<f32>(col, 1.0); 
        }
        t += d;
        if (t > tmax) { break; }
    }
    
    discard; 
}
