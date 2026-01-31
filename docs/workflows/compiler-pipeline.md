# **GVE-1 Compiler Pipeline**

**Role:** Translating "DNA" (JSON) into "Phenotype" (Binaries)  
**Strategy:** JIT Baking with Gradient-Based Refinement  
**Philosophy:** Treat JSON recipes as source code requiring compilation, optimization, and linking

**Version:** 1.1  
**Last Updated:** January 28, 2026

**Related Docs:**
- [System Overview](../architecture/overview.md) - High-level architecture
- [Engine API](../architecture/engine-api.md) - Tri-layer orchestration & WASM protocol
- [Data Specifications](../data/data-specifications.md) - JSON schemas & binary format
- [Database Architecture](../data/database-architecture.md) - MongoDB + cache system

---

## **Architecture Context**

The Compiler Pipeline operates within **Layer 2: The Architect** of the tri-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: The Forge (Interface)                             │
│  HTML5 + htmx + Rust WASM                                   │
└────────────────────────┬────────────────────────────────────┘
                         │ htmx requests → HTML partials
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: The Architect (Logic)  ◄──── COMPILER PIPELINE    │
│  Python + FastAPI + PyTorch + MongoDB                       │
└────────────────────────┬────────────────────────────────────┘
                         │ Binary assets → File system cache
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: The Engine (Runtime)                              │
│  Rust + wgpu + rapier3d + cpal                              │
└─────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> The Compiler produces `.gve_bin` files that are streamed to the Forge via the **WASM Binary Messaging Protocol** (see `engine-api.md` §8). Version tracking lives in message headers, not the binary itself.

---

## **Stage 1: Math JIT (Tensor Core)**

### Problem Statement
Convert abstract CSG tree (JSON) into GPU-executable mathematics with automatic differentiation.

### Algorithm: Recursive PyTorch Graph Construction

```python
import torch
import torch.jit as jit

def compile_sdf_tree(json_node: dict) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Recursively compile JSON SDF tree to PyTorch computational graph.
    
    Args:
        json_node: GVE-JSON tree node
    
    Returns:
        Function: (N, 3) tensor → (N,) distance tensor
    """
    
    if json_node["type"] == "primitive":
        return compile_primitive(json_node["shape"], json_node["params"])
    
    elif json_node["type"] == "operation":
        left_fn = compile_sdf_tree(json_node["left"])
        right_fn = compile_sdf_tree(json_node["right"])
        
        if json_node["op"] == "union":
            @jit.script
            def union_op(p: torch.Tensor) -> torch.Tensor:
                return torch.min(left_fn(p), right_fn(p))
            return union_op
        
        elif json_node["op"] == "subtract":
            @jit.script
            def subtract_op(p: torch.Tensor) -> torch.Tensor:
                return torch.max(left_fn(p), -right_fn(p))
            return subtract_op
        
        elif json_node["op"] == "intersect":
            @jit.script
            def intersect_op(p: torch.Tensor) -> torch.Tensor:
                return torch.max(left_fn(p), right_fn(p))
            return intersect_op
        
        elif json_node["op"] == "smooth_union":
            # IQ Polynomial Smooth Min - C¹ continuity guaranteed
            # Source: https://iquilezles.org/articles/smin/
            k = json_node["smoothness"]
            
            @jit.script
            def smooth_union_op(p: torch.Tensor) -> torch.Tensor:
                d_left = left_fn(p)
                d_right = right_fn(p)
                
                # Interpolation factor with clamping
                h = torch.clamp(
                    0.5 + 0.5 * (d_right - d_left) / k,
                    0.0,
                    1.0
                )
                
                # Polynomial blend (continuous derivative at boundaries)
                blend = torch.lerp(d_right, d_left, h)
                
                # Subtract "bump" for smooth transition (C¹ guarantee)
                return blend - k * h * (1.0 - h)
            
            return smooth_union_op
    
    elif json_node["type"] == "modifier":
        child_fn = compile_sdf_tree(json_node["child"])
        return apply_modifier(child_fn, json_node["modifier_type"], json_node["params"])

def compile_primitive(shape: str, params: dict) -> Callable:
    """Generate SDF function for geometric primitive."""
    
    if shape == "box":
        size = torch.tensor(params["size"], dtype=torch.float32)
        
        @jit.script
        def box_sdf(p: torch.Tensor) -> torch.Tensor:
            q = torch.abs(p) - size
            outside = torch.norm(torch.max(q, torch.zeros_like(q)), dim=-1)
            inside = torch.min(torch.max(q[..., 0], torch.max(q[..., 1], q[..., 2])), torch.zeros_like(q[..., 0]))
            return outside + inside
        
        return box_sdf
    
    elif shape == "sphere":
        radius = params["radius"]
        
        @jit.script
        def sphere_sdf(p: torch.Tensor) -> torch.Tensor:
            return torch.norm(p, dim=-1) - radius
        
        return sphere_sdf
    
    elif shape == "cylinder":
        radius = params["radius"]
        height = params["height"]
        
        @jit.script
        def cylinder_sdf(p: torch.Tensor) -> torch.Tensor:
            # Infinite cylinder along Y-axis, capped at height
            d_radial = torch.norm(p[..., [0, 2]], dim=-1) - radius
            d_height = torch.abs(p[..., 1]) - height / 2.0
            
            outside = torch.norm(
                torch.max(torch.stack([d_radial, d_height], dim=-1), torch.zeros_like(p[..., :2])),
                dim=-1
            )
            inside = torch.min(torch.max(d_radial, d_height), torch.zeros_like(d_radial))
            return outside + inside
        
        return cylinder_sdf
    
    # ... additional primitives

def apply_modifier(sdf_fn: Callable, mod_type: str, params: dict) -> Callable:
    """Apply domain deformation modifier."""
    
    if mod_type == "twist":
        axis = params["axis"]  # "X", "Y", or "Z"
        rate = params["rate"]
        
        @jit.script
        def twist_sdf(p: torch.Tensor) -> torch.Tensor:
            # Twist around specified axis
            if axis == "Y":
                angle = p[..., 1] * rate
                cos_a = torch.cos(angle)
                sin_a = torch.sin(angle)
                
                # Rotate XZ plane
                p_twisted = p.clone()
                p_twisted[..., 0] = cos_a * p[..., 0] - sin_a * p[..., 2]
                p_twisted[..., 2] = sin_a * p[..., 0] + cos_a * p[..., 2]
                
                return sdf_fn(p_twisted)
            # ... other axes
        
        return twist_sdf
    
    # ... other modifiers (bend, round, etc.)
```

### Vectorization Benefits

**Traditional Loop (CPU):**
```python
# Evaluates 1 point at a time
distances = []
for point in grid_points:  # 1M points
    distances.append(evaluate_sdf(point))  # 1M function calls
# Time: ~10 seconds
```

**Vectorized (GPU):**
```python
# Evaluates all points simultaneously
grid_tensor = torch.tensor(grid_points)  # (1M, 3) shape
distances = sdf_function(grid_tensor)    # 1 CUDA kernel launch
# Time: ~50 milliseconds (200× speedup)
```

### Auto-Differentiation for Normals

```python
def compute_sdf_with_normal(sdf_fn: Callable, points: torch.Tensor) -> tuple:
    """
    Evaluate SDF and compute surface normals via autodiff.
    
    Returns:
        distances: (N,) tensor
        normals: (N, 3) tensor (gradient of SDF)
    """
    points.requires_grad_(True)
    
    # Forward pass
    distances = sdf_fn(points)
    
    # Backward pass (computes ∇f automatically)
    gradients = torch.autograd.grad(
        outputs=distances,
        inputs=points,
        grad_outputs=torch.ones_like(distances),
        create_graph=False,
    )[0]
    
    # Normalize to get unit normals
    normals = F.normalize(gradients, dim=-1)
    
    return distances, normals
```

**Performance:**
- Gradient computation: Free (PyTorch autograd)
- No manual finite differences required
- Used for lighting, physics collision normals, splat orientation

---

## **Stage 2: Optimization Baking**

### 2.1 Volume Baking: Math → OpenVDB

**Problem:** Evaluating complex SDF trees per-pixel is O(N) where N = primitive count, and dense textures waste memory on empty space.

**Solution:** Pre-compute sparse OpenVDB grid.

```python
import meshlib.mrmeshpy as mrmesh
import numpy as np

def bake_sdf_volume_vdb(
    sdf_fn: Callable,
    bounds_min: Vec3,
    bounds_max: Vec3,
    voxel_size: float = 0.04,  # 4cm
) -> mrmesh.VdbVolume:
    """
    Bake SDF into sparse OpenVDB grid using MeshLib.
    """
    # 1. Define dimensions
    extent = bounds_max - bounds_min
    res = (extent / voxel_size).ceil().astype(int)
    
    dims = mrmesh.Vector3i(res[0], res[1], res[2])
    v_size = mrmesh.Vector3f(voxel_size, voxel_size, voxel_size)
    origin = mrmesh.Vector3f(bounds_min[0], bounds_min[1], bounds_min[2])
    
    # 2. Evaluate SDF (Dense)
    # Ideally localized sparse evaluation, but dense-first for Phase 1
    dense_data = evaluate_sdf_dense(sdf_fn, bounds_min, bounds_max, res)
    
    # 3. Create SimpleVolume
    vol = mrmesh.SimpleVolume()
    vol.data = mrmesh.std_vector_float(dense_data.ravel())
    vol.dims = dims
    vol.voxelSize = v_size
    
    # 4. Convert to Sparse VDB
    # Reduces memory usage by discarding empty voxels
    vdb_vol = mrmesh.simpleVolumeToVdbVolume(mrmesh.SimpleVolumeMinMax(vol))
    
    return vdb_vol
```

**Memory Savings:**
- Sparse B+ tree only stores active surface voxels.
- **50x reduction** for large sparse objects compared to dense arrays.

### 2.2 Shell Generation: MeshLib Repair

**Problem:** Standard extraction algorithms (Marching Cubes/Dual Contouring) can produce non-manifold geometry or holes.

**Solution:** MeshLib robust repair and decimation.

```python
import meshlib.mrmeshpy as mrmesh

def generate_shell_mesh(vdb_grid: vdb.FloatGrid) -> Mesh:
    """
    Generate tight-fitting proxy mesh using MeshLib.
    """
    # 1. Convert VDB to MeshLib voxel format
    voxels = vdb_to_mr(vdb_grid)
    
    # 2. Extract Mesh (Dual Contouring equivalent)
    mesh = mrmesh.gridToMesh(voxels, iso=0.0)
    
    # 3. Robust Repair
    # Fix self-intersections, holes, and non-manifold edges
    mrmesh.fixSelfIntersections(mesh)
    mrmesh.closeHoles(mesh)
    
    # 4. Optimized Decimation
    # Reduce to target count while preserving silhouette
    mrmesh.decimateMesh(mesh, 
        mrmesh.DecimateSettings(
            maxError=0.01,
            maxVerts=1000
        )
    )
    
    return mesh
```

**Benefits:**
- **Guaranteed Manifold:** Essential for physics engines (Rapier3d) and shadows.
- **Speed:** MeshLib Decimation is parallelized and significantly faster than Python implementations.

---

## **Stage 3: Splat Refinement (Neural Training)**

### Problem Statement
Align gaussian splats to SDF surface with perceptually uniform colors.

### 3.1 Initialization: Poisson Disk Sampling

```python
def initialize_splats_poisson(
    sdf_fn: Callable,
    bounds: AABB,
    target_count: int = 100000,
    min_radius: float = 0.01,
) -> List[Splat]:
    """
    Generate initial splat distribution using Poisson disk sampling.
    Ensures even, non-overlapping coverage of isosurface.
    """
    splats = []
    candidates = []
    
    # Start with random point on surface
    initial_point = find_surface_point(sdf_fn, bounds)
    splats.append(Splat(position=initial_point))
    candidates.extend(generate_annulus_samples(initial_point, min_radius))
    
    while len(splats) < target_count and candidates:
        # Pick random candidate
        candidate = candidates.pop(random.randint(0, len(candidates) - 1))
        
        # Check minimum distance to existing splats
        if all(np.linalg.norm(candidate - s.position) >= min_radius for s in splats):
            # Project onto surface
            surface_point = project_to_surface(sdf_fn, candidate)
            
            splats.append(Splat(position=surface_point))
            candidates.extend(generate_annulus_samples(surface_point, min_radius))
    
    return splats
```

### 3.2 Zero-Layer Training with Oklab Colors

```python
import torch
import torch.nn.functional as F
from oklab import linear_rgb_to_oklab, oklab_to_linear_rgb

class SplatOptimizer:
    def __init__(self, sdf_fn: Callable, initial_splats: List[Splat]):
        self.sdf_fn = sdf_fn
        
        # Learnable parameters
        self.positions = torch.tensor(
            [s.position for s in initial_splats],
            requires_grad=True,
            dtype=torch.float32
        )
        self.scales = torch.tensor(
            [s.scale for s in initial_splats],
            requires_grad=True,
            dtype=torch.float32
        )
        self.rotations = torch.tensor(
            [s.rotation for s in initial_splats],  # Quaternions
            requires_grad=True,
            dtype=torch.float32
        )
        # Colors in Oklab (perceptually uniform)
        self.colors_oklab = torch.tensor(
            [linear_rgb_to_oklab(s.color_rgb) for s in initial_splats],
            requires_grad=True,
            dtype=torch.float32
        )
        self.opacities = torch.tensor(
            [s.opacity for s in initial_splats],
            requires_grad=True,
            dtype=torch.float32
        )
        
        self.optimizer = torch.optim.Adam([
            {'params': self.positions, 'lr': 0.01},
            {'params': self.scales, 'lr': 0.005},
            {'params': self.rotations, 'lr': 0.01},
            {'params': self.colors_oklab, 'lr': 0.01},
            {'params': self.opacities, 'lr': 0.01},
        ])
    
    def compute_loss(self):
        """Compute multi-component loss function."""
        
        # Component 1: Surface alignment loss
        sdf_values = self.sdf_fn(self.positions)
        loss_surface = (sdf_values ** 2).mean()
        
        # Component 2: Normal alignment loss
        sdf_normals = self.compute_sdf_normals(self.positions)
        splat_normals = self.quaternion_to_normal(self.rotations)
        loss_normal = (1 - (sdf_normals * splat_normals).sum(dim=-1)).mean()
        
        # Component 3: Overlap penalty
        loss_overlap = self.compute_overlap_loss()
        
        # Component 4: Oklab color regularization (prevent extreme values)
        # L should be in [0, 1], a/b in [-0.4, 0.4]
        loss_color = torch.relu(self.colors_oklab[:, 0] - 1.0).mean() + \
                     torch.relu(-self.colors_oklab[:, 0]).mean() + \
                     torch.relu(torch.abs(self.colors_oklab[:, 1:]) - 0.4).mean()
        
        # Weighted sum
        total_loss = (
            10.0 * loss_surface +
            1.0 * loss_normal +
            0.1 * loss_overlap +
            0.01 * loss_color
        )
        
        return total_loss, {
            'surface': loss_surface.item(),
            'normal': loss_normal.item(),
            'overlap': loss_overlap.item(),
            'color': loss_color.item(),
        }
    
    def compute_sdf_normals(self, positions):
        """Compute SDF gradients via autodiff."""
        positions.requires_grad_(True)
        sdf_vals = self.sdf_fn(positions)
        
        grads = torch.autograd.grad(
            sdf_vals,
            positions,
            grad_outputs=torch.ones_like(sdf_vals),
            create_graph=True,
        )[0]
        
        return F.normalize(grads, dim=-1)
    
    def compute_overlap_loss(self):
        """Penalize splat overlap to prevent clumping."""
        # Pairwise distances (N × N matrix)
        diffs = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)
        distances = torch.norm(diffs, dim=-1)
        
        # Gaussian falloff based on scale
        avg_scale = self.scales.mean(dim=-1)
        overlap = torch.exp(-distances ** 2 / (2 * avg_scale.unsqueeze(1) * avg_scale.unsqueeze(0)))
        
        # Exclude self-overlap (diagonal)
        mask = ~torch.eye(len(self.positions), dtype=torch.bool, device=self.positions.device)
        
        return overlap[mask].mean()
    
    def train(self, iterations=300):
        """Main training loop."""
        loss_history = []
        
        for i in range(iterations):
            self.optimizer.zero_grad()
            
            loss, loss_components = self.compute_loss()
            
            loss.backward()
            self.optimizer.step()
            
            loss_history.append(loss.item())
            
            # Adaptive densification/pruning every 50 iterations
            if i % 50 == 0:
                self.densify_and_prune()
                print(f"Iteration {i}: Loss={loss.item():.6f}, Splats={len(self.positions)}")
        
        return loss_history
    
    def densify_and_prune(self):
        """Split high-gradient splats, remove low-opacity splats."""
        # Compute positional gradients
        grad_norms = torch.norm(self.positions.grad, dim=-1)
        
        # Densification: split splats with high gradient (high curvature)
        high_grad_mask = grad_norms > 0.002
        if high_grad_mask.any():
            high_grad_indices = torch.where(high_grad_mask)[0]
            
            new_splats = []
            for idx in high_grad_indices:
                # Create two smaller splats offset along gradient
                offset = self.positions.grad[idx] * 0.1 * self.scales[idx].mean()
                
                new_splats.append({
                    'pos': self.positions[idx] + offset,
                    'scale': self.scales[idx] * 0.6,
                    'rot': self.rotations[idx],
                    'color': self.colors_oklab[idx],
                    'opacity': self.opacities[idx] * 0.8,
                })
                new_splats.append({
                    'pos': self.positions[idx] - offset,
                    'scale': self.scales[idx] * 0.6,
                    'rot': self.rotations[idx],
                    'color': self.colors_oklab[idx],
                    'opacity': self.opacities[idx] * 0.8,
                })
            
            # Add new splats, remove originals
            # (implementation details omitted for brevity)
        
        # Pruning: remove transparent or oversized splats
        prune_mask = (self.opacities < 0.05) | (self.scales.max(dim=-1)[0] > 1.0)
        if prune_mask.any():
            keep_mask = ~prune_mask
            self.positions = self.positions[keep_mask]
            self.scales = self.scales[keep_mask]
            # ... prune other parameters
    
    def export_splats(self, output_path: str, color_mode: str = "rgb"):
        """
        Export trained splats with configurable color mode.
        
        Args:
            color_mode: "rgb" for static assets, "oklab" for dynamic/procedural
        """
        splats_data = []
        
        for i in range(len(self.positions)):
            oklab = self.colors_oklab[i].detach().cpu().numpy()
            
            if color_mode == "rgb":
                # Convert Oklab (training space) to RGB for static assets
                rgb_linear = oklab_to_linear_rgb(oklab)
                rgb_srgb = linear_to_srgb(rgb_linear)
                
                color_packed = [
                    int(np.clip(rgb_srgb[0] * 255, 0, 255)),
                    int(np.clip(rgb_srgb[1] * 255, 0, 255)),
                    int(np.clip(rgb_srgb[2] * 255, 0, 255)),
                    int(np.clip(self.opacities[i].item() * 255, 0, 255)),
                ]
                flags = 0x00  # RGB mode
                
            else:  # "oklab" mode for dynamic assets
                # Keep Oklab for runtime color interpolation
                color_packed = [
                    int(np.clip(oklab[0] * 255, 0, 255)),  # L
                    int(np.clip((oklab[1] + 0.4) / 0.8 * 255, 0, 255)),  # a
                    int(np.clip((oklab[2] + 0.4) / 0.8 * 255, 0, 255)),  # b
                    int(np.clip(self.opacities[i].item() * 255, 0, 255)),
                ]
                flags = 0x01  # Oklab mode
            
            splats_data.append({
                'position': self.positions[i].detach().cpu().numpy(),
                'scale': self.scales[i].detach().cpu().numpy(),
                'rotation': self.rotations[i].detach().cpu().numpy(),
                'color_packed': color_packed,
                'flags': flags,
            })
        
        save_splats_binary(splats_data, output_path)
```

**Choosing Color Mode:**

```
┌─────────────────────────────────────┐
│ Does asset need runtime color      │
│ interpolation or procedural FX?    │
└────────┬───────────────────┬────────┘
         │ NO                │ YES
         ▼                   ▼
   ┌──────────┐        ┌──────────┐
   │ RGB Mode │        │ Oklab    │
   │ (Static) │        │ (Dynamic)│
   └──────────┘        └──────────┘
   Examples:           Examples:
   • Rocks             • Weapons
   • Buildings         • Characters
   • Terrain           • Damage FX
   • Props             • Team colors
```

**Asset Type Guidelines:**
- **RGB:** Static crates, walls, rocks, trees, furniture
- **Oklab:** Weapons (wear), characters (outfits), destructibles (damage)

**Character Animation:**
> [!NOTE]
> For animated characters, see `data-specifications.md` §2.5 for SDF node-based animation workflow. Most characters use 80% rigid node transforms + 20% skinned joint zones for optimal performance (0.5ms vs 2.4ms for traditional mesh skinning).

```

### Oklab Color Advantages

**Perceptual Uniformity:**
- RGB interpolation: `lerp([1,0,0], [0,1,0]) = muddy brown`
- Oklab interpolation: `lerp(red_oklab, green_oklab) = smooth yellow`

**Better for Procedural Blending:**
- Material transitions don't shift hue unexpectedly
- Gradient-based color optimization converges faster
- Artists get meaningful color sliders

### Animated Character Splats

For animated characters using **SDF Node-Based Animation** (see `data-specifications.md` §2.5), the splat training phase includes:

1. **Per-Node Splat Generation:** Splats are generated per SDF node, preserving node boundaries
2. **Rigid vs Skinned Classification:** Joint zones are auto-detected for bone blending
3. **Bone Binding Metadata:** Splat export includes `NodeBinding` data (rigid or skinned)

```python
# Extended export for animated characters
class AnimatedSplatExport:
    splats: List[Splat]               # Standard splat data
    node_bindings: List[NodeBinding]  # Per-node bone assignments
    skeleton: SkeletonData            # Bone hierarchy
```

**Performance:** 80% rigid nodes + 20% skinned joints = 0.52ms/character vs 2.4ms traditional skinning.

---

## **Stage 4: Semantic Indexing (RAG)**

### 4.1 Text Vector: Recipe Fingerprint

```python
from sentence_transformers import SentenceTransformer

def create_semantic_fingerprint(json_tree: dict) -> str:
    """Walk JSON tree and extract semantic description."""
    
    primitives = count_primitives(json_tree)
    macros = extract_macros(json_tree)
    materials = extract_materials(json_tree)
    modifiers = extract_modifiers(json_tree)
    
    description = f"""
    Primitives: {primitives['box']}×Box, {primitives['sphere']}×Sphere, {primitives['cylinder']}×Cylinder
    Macros: {', '.join(macros)}
    Materials: {', '.join(materials)}
    Modifiers: {', '.join(modifiers)}
    Complexity: {calculate_complexity_score(json_tree)}
    """
    
    return description.strip()

def embed_text_vector(description: str) -> np.ndarray:
    """Generate 384-dim embedding using Sentence Transformer."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(description)
    return embedding  # (384,)
```

### 4.2 Shape Vector: PointNet Encoding

```python
def create_shape_vector(splats: List[Splat]) -> np.ndarray:
    """Encode geometric shape using PointNet."""
    
    # Downsample splats to 1024 points (Farthest Point Sampling)
    point_cloud = farthest_point_sample(
        [s.position for s in splats],
        num_samples=1024
    )
    
    # Normalize to unit sphere
    point_cloud = normalize_point_cloud(point_cloud)
    
    # PointNet forward pass
    shape_vector = pointnet_model(point_cloud)  # (512,)
    
    return shape_vector
```

### 4.3 Retrieval-Augmented Generation

```python
def query_similar_assets(query_text: str, query_shape: Optional[np.ndarray] = None):
    """Find similar assets using hybrid text + shape search."""
    
    # Embed query
    text_vec = embed_text_vector(query_text)
    
    if query_shape is not None:
        # Fuse text and shape vectors
        fused_vec = np.concatenate([text_vec, query_shape])
        fused_vec = fused_vec / np.linalg.norm(fused_vec)
    else:
        fused_vec = text_vec
    
    # Cosine similarity search in MongoDB vector index
    results = mongodb_collection.aggregate([
        {
            '$vectorSearch': {
                'index': 'asset_vectors',
                'path': 'embedding',
                'queryVector': fused_vec.tolist(),
                'numCandidates': 100,
                'limit': 10,
            }
        },
        {
            '$project': {
                'asset_id': 1,
                'score': {'$meta': 'vectorSearchScore'},
            }
        }
    ])
    
    return list(results)
```

---

## **Performance Budget**

| Stage | CPU Time | GPU Time | Memory | Output Size |
|-------|----------|----------|--------|-------------|
| **Stage 1: Math JIT** | 50ms (graph build) | - | 100MB (graph) | Bytecode (5KB) |
| **Stage 2: Volume Bake** | - | 200ms @ 128³ | 8MB | 4MB (f16, compressed) |
| **Stage 2: Shell Gen** | 500ms (DC + decimate) | - | 20MB | 50KB (500 tris) |
| **Stage 3: Splat Train** | - | 30-60s @ 100k splats | 500MB | 2.4MB (quantized) |
| **Stage 4: Indexing** | 100ms (embedding) | - | 50MB (models) | 4KB (vectors) |
| **Total** | ~700ms | ~60s | 678MB peak | **6.5MB binary** |

**Target:** < 2 minutes compile time for typical asset

---

## **Implementation Checklist**

- [ ] Implement recursive SDF tree compiler with IQ smooth union
- [ ] Add CSE (Common Subexpression Elimination) optimization
- [ ] Create volume baking with zstd compression
- [ ] Implement Dual Contouring with QEF solver
- [ ] Build splat training loop with Oklab colors
- [ ] Add adaptive densification/pruning
- [ ] Integrate Sentence Transformer for text vectors
- [ ] Train/deploy PointNet for shape vectors
- [ ] Set up MongoDB vector search index
- [ ] Profile and optimize to performance budget
- [ ] Add incremental compilation (only re-bake changed nodes)
- [ ] Implement WASM binary messaging integration (see `engine-api.md` §8)
- [ ] Add animated character bone binding export

---

**Version:** 1.1  
**Last Updated:** January 26, 2026  
**Related:** [System Overview](../architecture/overview.md) | [Engine API](../architecture/engine-api.md) | [Data Specifications](../data/data-specifications.md)