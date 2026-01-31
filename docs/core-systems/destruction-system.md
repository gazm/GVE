# Environment Destruction System - Technical Specification

**Purpose:** Define material interaction and geometry transformation during destructive events

**Date:** 2026-01-25  
**Status:** Design Document

---

## System Overview

Environment destruction integrates physics, materials, geometry, and rendering:

```
Impact → Material Lookup → Fracture Analysis → SDF Modification → 
Debris Generation → Splat Update → Physics Simulation → Voxel Freezing
```

---

## 1. Material Properties

Each material defines fracture behavior:

```rust
struct MaterialProperties {
    yield_strength: f32,      // Force to break (Newtons)
    brittleness: f32,         // 0.0 (ductile) to 1.0 (shatters)
    chunk_size_mean: f32,     // Average debris size (meters)
    chunk_size_variance: f32, // Size randomness factor
    fracture_pattern: FractureType,
}

enum FractureType {
    Voronoi,    // Concrete, stone (irregular chunks)
    Planar,     // Metal sheets (bend before breaking)
    Radial,     // Glass (spider-web cracks)
    Granular,   // Dirt, sand (many tiny pieces)
}
```

### Material Database

| Material | Yield (N) | Brittleness | Pattern | Chunk Size | Audio | VFX |
|----------|-----------|-------------|---------|------------|-------|-----|
| Concrete | 5000 | 0.8 | Voronoi | 0.2-0.5m | Deep rumble | Gray dust |
| Steel | 15000 | 0.3 | Planar | 0.5-1.0m | Metal groan | Sparks |
| Glass | 2000 | 1.0 | Radial | 0.05-0.2m | High shatter | Glint particles |
| Wood | 8000 | 0.5 | Voronoi | 0.3-0.6m | Crack/splinter | Wood chips |
| Brick | 6000 | 0.7 | Voronoi | 0.15-0.3m | Crumble | Red dust |

---

## 2. Geometry Transformation

### Impact Processing

```rust
fn handle_impact(
    object: &mut SDFObject,
    impact_point: Vec3,
    impact_force: f32,
    material: &MaterialProperties,
) {
    if impact_force < material.yield_strength {
        // Non-destructive: surface dent only
        apply_surface_dent(object, impact_point, impact_force);
        return;
    }
    
    // Destructive: fracture and debris
    apply_fracture(object, impact_point, impact_force, material);
}
```

### SDF Subtraction

Modify geometry by subtracting impact volume:

```rust
fn apply_fracture(
    object: &mut SDFObject,
    impact_point: Vec3,
    force: f32,
    material: &MaterialProperties,
) {
    // Crater size scales with force
    let crater_radius = (force / material.yield_strength).powf(0.5) * 0.3;
    
    // Create subtraction sphere
    let crater_sdf = SDF::Sphere {
        center: impact_point,
        radius: crater_radius,
    };
    
    // Modify object SDF tree
    object.sdf_tree = SDF::Subtract {
        base: Box::new(object.sdf_tree.clone()),
        subtract: Box::new(crater_sdf),
    };
    
    object.needs_recompile = true;
    
    // Generate debris chunks
    let chunks = generate_debris(object, impact_point, crater_radius, material);
    spawn_debris_physics(chunks);
}
```

---

## 3. Fracture Pattern Algorithms

### Voronoi Fracture (Concrete, Stone)

```rust
fn generate_voronoi_chunks(
    impact_point: Vec3,
    radius: f32,
    material: &MaterialProperties,
) -> Vec<DebrisChunk> {
    // Calculate chunk count from radius and material
    let volume = (4.0/3.0) * PI * radius.powi(3);
    let chunk_volume = material.chunk_size_mean.powi(3);
    let num_chunks = (volume / chunk_volume) as usize;
    
    // Generate Voronoi seeds
    let seeds = generate_random_points_in_sphere(impact_point, radius, num_chunks);
    
    // Create Voronoi cells
    let mut chunks = Vec::new();
    for seed in seeds {
        let cell_sdf = compute_voronoi_cell(seed, &seeds);
        
        chunks.push(DebrisChunk {
            sdf: cell_sdf,
            material: material.clone(),
            position: seed,
            velocity: calculate_ejection_velocity(seed, impact_point, force),
        });
    }
    
    chunks
}

fn compute_voronoi_cell(seed: Vec3, all_seeds: &[Vec3]) -> SDF {
    // Voronoi cell = intersection of half-spaces
    let mut cell = SDF::Sphere { center: seed, radius: 100.0 };  // Start large
    
    for other_seed in all_seeds {
        if *other_seed == seed { continue; }
        
        // Half-space between seed and other_seed
        let midpoint = (seed + other_seed) * 0.5;
        let normal = (seed - other_seed).normalize();
        
        let half_space = SDF::Plane { point: midpoint, normal };
        cell = SDF::Intersect {
            left: Box::new(cell),
            right: Box::new(half_space),
        };
    }
    
    cell
}
```

### Radial Fracture (Glass)

```rust
fn generate_radial_shards(
    impact_point: Vec3,
    radius: f32,
    material: &MaterialProperties,
) -> Vec<DebrisChunk> {
    let num_shards = 20;
    let mut chunks = Vec::new();
    
    for i in 0..num_shards {
        let angle = (i as f32 / num_shards as f32) * TAU;
        
        // Create triangular shard
        let shard_sdf = SDF::TriangularPrism {
            base: impact_point,
            direction: Vec3::from_angle(angle),
            width: radius * 0.1,
            length: radius,
            thickness: 0.01,  // Thin glass
        };
        
        chunks.push(DebrisChunk {
            sdf: shard_sdf,
            material: material.clone(),
            position: impact_point + Vec3::from_angle(angle) * radius * 0.5,
            velocity: Vec3::from_angle(angle) * 5.0,
        });
    }
    
    chunks
}
```

### Planar Fracture (Metal)

```rust
fn generate_planar_chunks(
    object: &SDFObject,
    impact_point: Vec3,
    force: f32,
    material: &MaterialProperties,
) -> Vec<DebrisChunk> {
    // Metal bends before breaking - create fewer, larger chunks
    let num_chunks = 3;  // Large pieces
    
    // Fracture planes oriented by stress direction
    let stress_direction = calculate_stress_direction(object, impact_point, force);
    
    let mut chunks = Vec::new();
    for i in 0..num_chunks {
        let plane_offset = (i as f32 - 1.0) * 0.5;
        
        let chunk_sdf = SDF::Intersect {
            left: Box::new(object.sdf_tree.clone()),
            right: Box::new(SDF::HalfSpace {
                point: impact_point + stress_direction * plane_offset,
                normal: stress_direction,
            }),
        };
        
        chunks.push(DebrisChunk {
            sdf: chunk_sdf,
            material: material.clone(),
            position: impact_point + stress_direction * plane_offset,
            velocity: stress_direction * 2.0,
        });
    }
    
    chunks
}
```

---

## 4. Visual Update (Splat Management)

### Option A: Masking (Fast, 1ms)

```rust
fn mask_destroyed_splats(object: &mut SDFObject, crater_aabb: AABB) {
    for splat in &mut object.splats {
        if crater_aabb.contains(splat.position) {
            let sdf_value = object.sdf_tree.evaluate(splat.position);
            
            if sdf_value > 0.0 {
                // Splat is in destroyed volume (empty space)
                splat.opacity = 0.0;
            }
        }
    }
}
```

### Option B: Regeneration (Accurate, 50ms)

```rust
fn regenerate_splats_in_crater(
    object: &mut SDFObject,
    crater_aabb: AABB,
) {
    // Remove old splats
    object.splats.retain(|s| !crater_aabb.contains(s.position));
    
    // Sample new surface
    let new_splats = poisson_disk_sample_surface(
        &object.sdf_tree,
        crater_aabb,
        density: 1000.0,  // splats/m²
    );
    
    // Quick optimization (50 iterations)
    train_splats_to_sdf(&mut new_splats, &object.sdf_tree, 50);
    
    object.splats.extend(new_splats);
}
```

### Hybrid Strategy

```rust
fn update_splats_after_destruction(
    object: &mut SDFObject,
    crater_aabb: AABB,
) {
    let affected_count = object.splats.iter()
        .filter(|s| crater_aabb.contains(s.position))
        .count();
    
    if affected_count < 1000 {
        // Small crater: regenerate immediately
        regenerate_splats_in_crater(object, crater_aabb);
    } else {
        // Large crater: mask now, regenerate later
        mask_destroyed_splats(object, crater_aabb);
        queue_regeneration(object, crater_aabb);  // Next frame
    }
}
```

---

## 5. Debris Physics

### Spawning

```rust
fn spawn_debris_physics(chunks: Vec<DebrisChunk>) -> Vec<RigidBodyHandle> {
    let mut handles = Vec::new();
    
    for chunk in chunks {
        // Create SDF collider
        let collider = ColliderBuilder::sdf_shape(chunk.sdf.clone())
            .material(chunk.material.clone())
            .build();
        
        // Create rigid body
        let rigid_body = RigidBodyBuilder::dynamic()
            .translation(chunk.position)
            .linvel(chunk.velocity)
            .angvel(random_spin())
            .build();
        
        let handle = physics_world.insert(rigid_body, collider);
        
        // Track for freezing
        debris_tracker.add(handle, DebrisState::Active);
        
        handles.push(handle);
    }
    
    handles
}
```

### Ejection Physics

```rust
fn calculate_ejection_velocity(
    chunk_pos: Vec3,
    impact_point: Vec3,
    impact_force: f32,
) -> Vec3 {
    let direction = (chunk_pos - impact_point).normalize();
    let distance = (chunk_pos - impact_point).length();
    
    // Inverse square falloff
    let speed_factor = impact_force / (1.0 + distance * distance);
    
    // Add randomness
    let random_offset = Vec3::random_unit() * 0.3;
    
    (direction + random_offset).normalize() * speed_factor.sqrt()
}
```

---

## 6. Complete Example: Bunker Destruction

```rust
// Setup
let bunker = SDFObject {
    sdf_tree: SDF::Box { size: Vec3::new(5.0, 3.0, 5.0) },
    material: CONCRETE,
    splats: load_splats("bunker.splats"),  // 150k splats
};

// Impact event
let explosion = ImpactEvent {
    point: Vec3::new(2.0, 1.5, 0.0),  // Center of wall
    force: 50000.0,  // 10× concrete yield strength
    source: ImpactSource::Explosion,
};

// Process destruction
handle_impact(&mut bunker, explosion.point, explosion.force, &CONCRETE);

// Results:
// 1. Geometry modified (SDF subtraction, crater_radius = 1.2m)
// 2. 25 Voronoi chunks generated (0.2-0.5m each)
// 3. 8000 splats masked in crater (instant)
// 4. 5000 new splats queued for generation
// 5. 25 rigid bodies spawned with physics
// 6. Audio: "concrete_explosion" at impact point
// 7. VFX: Gray dust cloud (3m radius)

// After 2 seconds:
// - Debris settles (velocities < 0.1 m/s)
// - State: Active → Sleeping → Frozen
// - 25 chunks stamped to voxel grid
// - Physics bodies removed
// - GPU cost: 20ms → 0.5ms (voxel volume)
```

---

## 7. Performance Optimizations

### Lazy Recompilation

```rust
// Don't recompile immediately
object.needs_recompile = true;

// Batch at end of frame
for object in &mut modified_objects {
    if object.needs_recompile {
        recompile_sdf_volume(&object);
        regenerate_shell_mesh(&object);
        object.needs_recompile = false;
    }
}
```

### Debris Budget

```rust
const MAX_ACTIVE_DEBRIS: usize = 500;
const MAX_FROZEN_DEBRIS: usize = 5000;

fn enforce_debris_budget() {
    if active_debris_count > MAX_ACTIVE_DEBRIS {
        // Freeze oldest 10%
        let to_freeze = active_debris
            .iter()
            .sorted_by_key(|d| d.spawn_time)
            .take(MAX_ACTIVE_DEBRIS / 10);
        
        for debris in to_freeze {
            freeze_to_voxel(debris);
        }
    }
    
    if frozen_debris_count > MAX_FROZEN_DEBRIS {
        // Remove oldest frozen debris
        remove_oldest_frozen(frozen_debris_count - MAX_FROZEN_DEBRIS);
    }
}
```

### Splat Update Strategy

| Affected Splats | Strategy | Cost | Quality |
|-----------------|----------|------|---------|
| < 1000 | Immediate regeneration | 50ms | Perfect |
| 1000-5000 | Mask + queue regen | 1ms + async | Good |
| > 5000 | Mask + progressive regen | 1ms + stream | Good |

---

## 8. Material Interaction Matrix

| Impact | Concrete | Steel | Glass | Wood |
|--------|----------|-------|-------|------|
| **Bullet (500N)** | Surface chip | Dent | Shatter | Splinter |
| **Explosion (50kN)** | Large crater | Bend + tear | Pulverize | Shatter |
| **Melee (2kN)** | Small dent | No damage | Crack | Crack |

### Audio Lookup

```rust
fn get_destruction_audio(material: &Material, force: f32) -> AudioPatch {
    let magnitude = if force < material.yield_strength * 2.0 {
        "light"
    } else if force < material.yield_strength * 10.0 {
        "medium"
    } else {
        "heavy"
    };
    
    load_audio(&format!("{}_{}_destruction", material.name, magnitude))
}
```

---

## 9. Integration Checklist

- [ ] Implement material properties database
- [ ] Build Voronoi fracture algorithm
- [ ] Build radial fracture (glass)
- [ ] Build planar fracture (metal)
- [ ] Create SDF subtraction system
- [ ] Implement splat masking
- [ ] Implement splat regeneration queue
- [ ] Build debris physics spawner
- [ ] Add ejection velocity calculations
- [ ] Integrate audio/VFX triggers
- [ ] Implement debris budget system
- [ ] Add lazy recompilation
- [ ] Profile: Target <100ms per large destruction event

---

## 10. Future Enhancements

**Procedural Repair:**
- Reverse SDF operations to "heal" geometry
- Useful for time-rewind mechanics

**Persistent Destruction:**
- Save modified SDF trees to disk
- Load destroyed state on level restart

**Networked Destruction:**
- Sync SDF operations (small data)
- Debris state shared via seeds

**Advanced Materials:**
- Composite materials (reinforced concrete)
- Temperature-dependent fracture (frozen vs melted)
- Fatigue damage (accumulate impact forces)
