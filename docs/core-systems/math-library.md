# **GVE-1 Math Library**

**Library:** glam + nalgebra (via parry3d/rapier3d)  
**Strategy:** SIMD-accelerated vector math with JSON schema integration  
**Philosophy:** Mathematical precision for physics truth

---

## **1. Core Types**

### Type Overview

| Type | Rust | JSON Schema | Size | Use Case |
|------|------|-------------|------|----------|
| `Vec2` | `glam::Vec2` | `[f32; 2]` | 8 bytes | UV coords, 2D positions |
| `Vec3` | `glam::Vec3` | `[f32; 3]` | 12 bytes | 3D positions, velocities, normals |
| `Vec4` | `glam::Vec4` | `[f32; 4]` | 16 bytes | Homogeneous coords, RGBA colors |
| `Quat` | `glam::Quat` | `[f32; 4]` | 16 bytes | Rotations (xyzw order) |
| `Mat3` | `glam::Mat3` | `[f32; 9]` | 36 bytes | 3x3 transforms, inertia tensors |
| `Mat4` | `glam::Mat4` | `[f32; 16]` | 64 bytes | Full 4x4 transforms |
| `Isometry3` | `nalgebra::Isometry3<f32>` | `{pos, rot}` | 28 bytes | Position + rotation (no scale) |

---

## **2. Vector Types**

### 2.1 Vec3 - The Workhorse

Most common type in physics and SDF calculations.

#### Rust Definition
```rust
use glam::Vec3;

// Construction
let zero = Vec3::ZERO;                    // (0, 0, 0)
let one = Vec3::ONE;                      // (1, 1, 1)
let up = Vec3::Y;                         // (0, 1, 0)
let v = Vec3::new(1.0, 2.0, 3.0);
let splat = Vec3::splat(5.0);             // (5, 5, 5)

// Component access
let x = v.x;
let (x, y, z) = v.into();
```

#### JSON Schema (DNA Format)
```json
{
  "position": [1.0, 2.0, 3.0],
  "velocity": [0.0, -9.8, 0.0],
  "normal": [0.0, 1.0, 0.0]
}
```

> [!NOTE]
> All Vec3 arrays use **[x, y, z]** order in JSON. Y-up coordinate system.

---

### 2.2 Vec2

2D vector for UV coordinates and screen-space calculations.

#### Rust Definition
```rust
use glam::Vec2;

let uv = Vec2::new(0.5, 0.5);
let screen_pos = Vec2::new(1920.0, 1080.0);
```

#### JSON Schema
```json
{
  "uv_scale": [1.0, 1.0],
  "texture_offset": [0.5, 0.0]
}
```

---

### 2.3 Vec4

Homogeneous coordinates and RGBA colors.

#### Rust Definition
```rust
use glam::Vec4;

let color = Vec4::new(1.0, 0.5, 0.0, 1.0);  // RGBA orange
let homo = Vec4::new(x, y, z, 1.0);          // Homogeneous point
```

#### JSON Schema
```json
{
  "color_rgba": [1.0, 0.5, 0.0, 1.0],
  "emissive": [0.0, 0.5, 1.0, 0.0]
}
```

---

## **3. Vector Operations**

### 3.1 Arithmetic

```rust
let a = Vec3::new(1.0, 2.0, 3.0);
let b = Vec3::new(4.0, 5.0, 6.0);

// Basic arithmetic
let sum = a + b;              // (5, 7, 9)
let diff = a - b;             // (-3, -3, -3)
let scaled = a * 2.0;         // (2, 4, 6)
let divided = a / 2.0;        // (0.5, 1, 1.5)

// Component-wise
let product = a * b;          // (4, 10, 18)
let quotient = a / b;         // (0.25, 0.4, 0.5)

// Negation
let neg = -a;                 // (-1, -2, -3)
```

---

### 3.2 Dot & Cross Product

```rust
// Dot product (scalar result)
let dot = a.dot(b);           // 1*4 + 2*5 + 3*6 = 32

// Cross product (perpendicular vector)
let cross = a.cross(b);       // (-3, 6, -3)

// Usage: Surface normal from two edge vectors
let edge1 = p1 - p0;
let edge2 = p2 - p0;
let normal = edge1.cross(edge2).normalize();
```

---

### 3.3 Length & Distance

```rust
// Length (magnitude)
let len = v.length();               // √(x² + y² + z²)
let len_sq = v.length_squared();    // x² + y² + z² (faster, no sqrt)

// Distance between points
let dist = (p1 - p0).length();
let dist_sq = (p1 - p0).length_squared();

// SDF sphere: distance from center minus radius
fn sdf_sphere(p: Vec3, center: Vec3, radius: f32) -> f32 {
    (p - center).length() - radius
}
```

---

### 3.4 Normalization

```rust
// Normalize (unit length)
let unit = v.normalize();           // v / |v|

// Safe normalize (handles zero-length)
let safe = v.normalize_or_zero();   // Returns ZERO if length is ~0
let safe2 = v.try_normalize();      // Returns Option<Vec3>

// Check if normalized
let is_norm = v.is_normalized();    // |v| ≈ 1.0
```

> [!IMPORTANT]
> Always use `normalize_or_zero()` for surface normals from SDF gradients to handle degenerate cases.

---

### 3.5 Interpolation

```rust
// Linear interpolation
let mid = a.lerp(b, 0.5);           // Halfway between a and b
let quarter = a.lerp(b, 0.25);      // 25% from a toward b

// Clamped lerp (t clamped to 0..1)
let clamped = Vec3::lerp(a, b, t.clamp(0.0, 1.0));

// Smooth interpolation (for animations)
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
```

---

### 3.6 Projection & Reflection

```rust
// Project a onto b (component of a in direction of b)
let proj = a.project_onto(b);

// Reject from b (component of a perpendicular to b)
let rej = a.reject_from(b);

// Reflect across normal (for physics bouncing)
fn reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    incident - 2.0 * incident.dot(normal) * normal
}

// Usage: Ball bouncing off wall
let bounce_velocity = reflect(velocity, wall_normal);
```

---

## **4. Quaternion (Rotations)**

### 4.1 Why Quaternions?

| Method | Gimbal Lock | Interpolation | Memory |
|--------|-------------|---------------|--------|
| Euler angles | ❌ Yes | ❌ Choppy | 12 bytes |
| Rotation matrix | ✅ No | ❌ Complex | 36 bytes |
| **Quaternion** | ✅ No | ✅ Smooth (slerp) | 16 bytes |

---

### 4.2 Construction

```rust
use glam::Quat;

// Identity (no rotation)
let identity = Quat::IDENTITY;

// From axis-angle
let rot_y_90 = Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2);

// From Euler angles (radians, XYZ order)
let euler = Quat::from_euler(
    glam::EulerRot::XYZ,
    pitch,  // X rotation
    yaw,    // Y rotation
    roll,   // Z rotation
);

// Look-at rotation
let look = Quat::from_rotation_arc(Vec3::Z, direction.normalize());
```

#### JSON Schema
```json
{
  "rotation": [0.0, 0.707, 0.0, 0.707],
  "rest_pose": {
    "pos": [0.0, 1.0, 0.0],
    "rot": [0.0, 0.0, 0.0, 1.0]
  }
}
```

> [!IMPORTANT]
> Quaternion order in JSON is **[x, y, z, w]** — the scalar `w` component is LAST.

---

### 4.3 Operations

```rust
// Combine rotations (order matters!)
let combined = rot_a * rot_b;   // Apply rot_b first, then rot_a

// Rotate a vector
let rotated = quat.mul_vec3(v);
let rotated = quat * v;         // Operator shorthand

// Inverse rotation
let inv = quat.inverse();
let original = inv.mul_vec3(rotated);

// Conjugate (same as inverse for unit quaternions)
let conj = quat.conjugate();
```

---

### 4.4 Interpolation (Slerp)

```rust
// Spherical linear interpolation (smooth rotation blend)
let blended = q1.slerp(q2, t);

// Normalized linear interpolation (faster, slightly less accurate)
let nlerp = q1.lerp(q2, t).normalize();

// Usage: Animation blending
fn blend_bone_rotations(
    rest: Quat,
    target: Quat,
    weight: f32,
) -> Quat {
    rest.slerp(target, weight)
}
```

---

## **5. Transforms**

### 5.1 Isometry3 (Position + Rotation)

The standard transform type for physics bodies — no scale distortion.

#### Rust Definition (nalgebra)
```rust
use nalgebra::{Isometry3, Vector3, UnitQuaternion};

// Construction
let iso = Isometry3::new(
    Vector3::new(1.0, 2.0, 3.0),  // Translation
    Vector3::new(0.0, 1.57, 0.0), // Axis-angle rotation
);

// From components
let iso = Isometry3::from_parts(
    Translation3::new(x, y, z),
    UnitQuaternion::from_axis_angle(&Vector3::y_axis(), angle),
);

// Transform a point
let world_point = iso.transform_point(&local_point);

// Inverse transform
let local_point = iso.inverse_transform_point(&world_point);
```

#### JSON Schema
```json
{
  "transform": {
    "pos": [1.0, 2.0, 3.0],
    "rot": [0.0, 0.707, 0.0, 0.707]
  }
}
```

---

### 5.2 Mat4 (Full Transform)

For rendering and GPU upload — includes scale.

```rust
use glam::Mat4;

// From components
let transform = Mat4::from_scale_rotation_translation(
    Vec3::ONE,                    // Scale
    Quat::from_rotation_y(0.5),   // Rotation
    Vec3::new(10.0, 0.0, 5.0),    // Translation
);

// Transform point (applies translation)
let world = transform.transform_point3(local);

// Transform vector (ignores translation)
let world_dir = transform.transform_vector3(local_dir);

// Inverse
let inv = transform.inverse();
```

---

### 5.3 Mat3 (Inertia Tensor)

For physics mass distribution.

```rust
use glam::Mat3;

// Inertia tensor for solid box
fn box_inertia(mass: f32, size: Vec3) -> Mat3 {
    let factor = mass / 12.0;
    Mat3::from_diagonal(Vec3::new(
        factor * (size.y * size.y + size.z * size.z),
        factor * (size.x * size.x + size.z * size.z),
        factor * (size.x * size.x + size.y * size.y),
    ))
}
```

#### JSON Schema (Row-Major)
```json
{
  "inertia_tensor": [
    1.0, 0.0, 0.0,
    0.0, 2.0, 0.0,
    0.0, 0.0, 1.5
  ]
}
```

---

## **6. SDF-Specific Math**

### 6.1 Gradient Calculation (Surface Normal)

Central differences for numerical gradient:

```rust
fn sdf_gradient(sdf: impl Fn(Vec3) -> f32, p: Vec3) -> Vec3 {
    const H: f32 = 0.001;  // Finite difference step
    
    Vec3::new(
        sdf(p + Vec3::X * H) - sdf(p - Vec3::X * H),
        sdf(p + Vec3::Y * H) - sdf(p - Vec3::Y * H),
        sdf(p + Vec3::Z * H) - sdf(p - Vec3::Z * H),
    ) / (2.0 * H)
}

// Usage
let normal = sdf_gradient(my_sdf, surface_point).normalize_or_zero();
```

**Cost:** 6 SDF evaluations per gradient

---

### 6.2 Smooth Min/Max

For smooth CSG unions without sharp edges:

```rust
// Smooth minimum (polynomial)
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

// Smooth maximum
fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    -smooth_min(-a, -b, k)
}

// Usage: Blend two SDFs
let blended = smooth_min(sdf_sphere, sdf_box, 0.1);
```

---

### 6.3 Domain Operations

```rust
// Repeat infinitely
fn op_rep(p: Vec3, spacing: Vec3) -> Vec3 {
    (p + spacing * 0.5).rem_euclid(spacing) - spacing * 0.5
}

// Mirror across axis
fn op_mirror(p: Vec3, axis: Vec3) -> Vec3 {
    p - 2.0 * p.dot(axis).min(0.0) * axis
}

// Twist around Y axis
fn op_twist(p: Vec3, rate: f32) -> Vec3 {
    let angle = p.y * rate;
    let (s, c) = angle.sin_cos();
    Vec3::new(
        c * p.x - s * p.z,
        p.y,
        s * p.x + c * p.z,
    )
}
```

---

## **7. Utility Functions**

### 7.1 Clamping & Remapping

```rust
// Clamp to range
let clamped = value.clamp(0.0, 1.0);

// Saturate (clamp to 0..1)
fn saturate(x: f32) -> f32 { x.clamp(0.0, 1.0) }

// Remap from one range to another
fn remap(value: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    out_min + (value - in_min) / (in_max - in_min) * (out_max - out_min)
}

// Sign (returns -1, 0, or 1)
fn sign(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
}
```

---

### 7.2 Angles

```rust
// Angle between two vectors (radians)
fn angle_between(a: Vec3, b: Vec3) -> f32 {
    a.normalize().dot(b.normalize()).clamp(-1.0, 1.0).acos()
}

// Signed angle around axis
fn signed_angle(from: Vec3, to: Vec3, axis: Vec3) -> f32 {
    let cross = from.cross(to);
    let angle = angle_between(from, to);
    if cross.dot(axis) < 0.0 { -angle } else { angle }
}

// Degrees ↔ Radians
fn deg_to_rad(deg: f32) -> f32 { deg * std::f32::consts::PI / 180.0 }
fn rad_to_deg(rad: f32) -> f32 { rad * 180.0 / std::f32::consts::PI }
```

---

## **8. SIMD Considerations**

### 8.1 Automatic Vectorization

`glam` uses SIMD by default on supported platforms:

```rust
// These operations use SIMD under the hood
let dot = a.dot(b);           // Uses SSE/NEON
let cross = a.cross(b);
let normalized = v.normalize();
```

### 8.2 Batch Operations

For physics simulations with many bodies:

```rust
use simba::simd::f32x4;

// Process 4 distances at once
fn batch_sphere_sdf(
    px: f32x4, py: f32x4, pz: f32x4,  // 4 query points
    cx: f32, cy: f32, cz: f32,         // Sphere center
    radius: f32,
) -> f32x4 {
    let dx = px - f32x4::splat(cx);
    let dy = py - f32x4::splat(cy);
    let dz = pz - f32x4::splat(cz);
    
    let len_sq = dx * dx + dy * dy + dz * dz;
    len_sq.sqrt() - f32x4::splat(radius)
}
```

**Performance:** 4× throughput for SDF evaluation pipelines

---

## **9. JSON Schema Summary**

Quick reference for DNA JSON integration:

| Field | Schema | Example |
|-------|--------|---------|
| Position | `[f32; 3]` | `"pos": [1.0, 2.0, 3.0]` |
| Rotation (quat) | `[f32; 4]` (xyzw) | `"rot": [0.0, 0.707, 0.0, 0.707]` |
| Scale | `[f32; 3]` | `"scale": [1.0, 1.0, 1.0]` |
| Color (RGBA) | `[f32; 4]` | `"color": [1.0, 0.5, 0.0, 1.0]` |
| Transform | `{pos, rot}` | `"transform": {"pos": [...], "rot": [...]}` |
| Matrix3 | `[f32; 9]` | Row-major order |
| Matrix4 | `[f32; 16]` | Row-major order |

---

## **10. Implementation Checklist**

### Core Math
- [ ] Implement Vec2/Vec3/Vec4 wrappers with JSON serde
- [ ] Implement Quat with xyzw JSON order validation
- [ ] Create Isometry3 JSON schema
- [ ] Add Mat3/Mat4 serialization

### SDF Operations
- [ ] Gradient calculation (6-tap central difference)
- [ ] Smooth min/max operators
- [ ] Domain modifiers (twist, bend, mirror)

### Optimization
- [ ] SIMD batch operations via simba
- [ ] Cache-friendly array-of-structs layout
- [ ] Benchmark SDF evaluation throughput

---

**Version:** 1.0  
**Last Updated:** January 26, 2026  
**Related:** [Physics System](./physics-system.md) | [Data Specifications](../data/data-specifications.md) | [Rendering Pipeline](./rendering-pipeline.md)
