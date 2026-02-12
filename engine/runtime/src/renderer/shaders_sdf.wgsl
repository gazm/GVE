
// ── Primitives ─────────────────────────────────────────────────────────

fn sd_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_cylinder(p: vec3<f32>, r: f32, h: f32) -> f32 {
    let d = abs(vec2<f32>(length(p.xz), p.y)) - vec2<f32>(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn sd_capsule(p: vec3<f32>, r: f32, h: f32) -> f32 {
    let half_h = h * 0.5;
    let p_y = clamp(p.y, -half_h, half_h);
    return length(p - vec3<f32>(0.0, p_y, 0.0)) - r;
}

fn sd_torus(p: vec3<f32>, major_r: f32, minor_r: f32) -> f32 {
    let q = vec2<f32>(length(p.xz) - major_r, p.y);
    return length(q) - minor_r;
}

fn sd_cone(p: vec3<f32>, r: f32, h: f32) -> f32 {
    let q = vec2<f32>(length(p.xz), p.y);
    let k = r / h;
    let c = vec2<f32>(h / sqrt(h*h + r*r), r / sqrt(h*h + r*r));
    
    // Intersection with sides
    let d1 = dot(vec2<f32>(q.x, q.y - h), c);
    
    // Intersection with base
    let d2 = max(d1, -q.y);
    
    // Capped Cone logic (simplified for standard cone base at y=0, tip at y=h)
    // Adjust frame: Tip at (0,h,0), Base at (0,0,0)
    // This is approximate for a generic cone primitive
    return max(dot(vec2<f32>(c.x, c.y), vec2<f32>(q.x, q.y-h)), -h - q.y);
}

fn sd_plane(p: vec3<f32>, n: vec3<f32>, dist: f32) -> f32 {
    return dot(p, normalize(n)) + dist;
}

fn sd_wedge(p: vec3<f32>, size: vec3<f32>, taper_axis: u32, taper_dir: u32) -> f32 {
    // Simplified Wedge: Just a box for now as tapering requires complex axis logic
    // TODO: Implement proper tapering
    return sd_box(p, size);
}

fn sd_revolution(p: vec3<f32>, offset: f32, axis: u32) -> vec3<f32> {
    // Returns 2D profile coordinate p(r, h)
    var r: f32;
    var h: f32;
    // axis: 0=x, 1=y, 2=z
    if (axis == 1u) { // Y axis
        r = length(p.xz) - offset;
        h = p.y;
    } else if (axis == 0u) { // X axis
        r = length(p.yz) - offset;
        h = p.x;
    } else { // Z axis
        r = length(p.xy) - offset;
        h = p.z;
    }
    return vec3<f32>(r, h, 0.0);
}

// ── Fractals (Low Iteration) ───────────────────────────────────────────

const MAX_ITER: i32 = 4;

fn sd_mandelbulb(p: vec3<f32>, power: f32, scale: f32) -> f32 {
    var z = p / scale;
    var dr = 1.0;
    var r: f32 = 0.0;
    
    for (var i = 0; i < MAX_ITER; i++) {
        r = length(z);
        if (r > 2.0) { break; }
        
        let theta = acos(z.z / r);
        let phi = atan2(z.y, z.x);
        let dr_pow = pow(r, power - 1.0);
        dr = dr_pow * power * dr + 1.0;
        
        let zr = pow(r, power);
        let theta_n = theta * power;
        let phi_n = phi * power;
        
        z = zr * vec3<f32>(sin(theta_n) * cos(phi_n), sin(theta_n) * sin(phi_n), cos(theta_n)) + (p / scale);
    }
    return 0.5 * log(r) * r / dr * scale;
}

fn sd_menger(p: vec3<f32>, scale: f32) -> f32 {
    var z = p / scale;
    for (var i = 0; i < MAX_ITER; i++) {
        z = abs(z);
        if (z.x < z.y) { let t = z.x; z.x = z.y; z.y = t; }
        if (z.x < z.z) { let t = z.x; z.x = z.z; z.z = t; }
        if (z.y < z.z) { let t = z.y; z.y = z.z; z.z = t; }
        
        z = z * 3.0;
        z.x = z.x - 2.0;
        z.y = z.y - 2.0;
        z.z = z.z; // -0.0
        
        if (z.z > 1.0) { z.z -= 2.0; }
    }
    return (length(max(abs(z) - vec3<f32>(1.0), vec3<f32>(0.0))) - 0.0) * pow(3.0, f32(-MAX_ITER)) * scale;
}

fn sd_julia(p: vec3<f32>, c: vec4<f32>, scale: f32) -> f32 {
    var z = vec4<f32>(p / scale, 0.0);
    var dz = 1.0;
    var r2: f32;
    
    for (var i = 0; i < MAX_ITER; i++) {
        r2 = dot(z.xyz, z.xyz);
        if (r2 > 4.0) { break; }
        
        // Quaternion square + c
        // z = z*z + c
        let x = z.x; let y = z.y; let w = z.z; let k = z.w; // w is z here
        // Standard q mul q
        let nx = x*x - y*y - w*w - k*k;
        let ny = 2.0*x*y;
        let nw = 2.0*x*w;
        let nk = 2.0*x*k;
        
        z = vec4<f32>(nx, ny, nw, nk) + c;
        dz = 2.0 * sqrt(r2) * dz;
    }
    return 0.5 * sqrt(r2) * log(sqrt(r2)) / dz * scale;
}
