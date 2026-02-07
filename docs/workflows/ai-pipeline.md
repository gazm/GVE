# **GVE-1 AI Pipeline**

**Role:** Intelligent asset generation via multi-agent orchestration  
**Integration:** Architect (Layer 2) → Compiler Pipeline → Engine (Layer 3)  
**Philosophy:** "Image-First Generation with Iterative Refinement"

---

## **System Overview**

The AI Pipeline transforms natural language prompts into valid GVE-1 assets (`.gve_bin`) through a **two-phase workflow**:

1. **Phase 1: Concept Generation** - Generate a 2D concept image using Gemini Nano Banana Pro
2. **User Review** - User approves, regenerates with feedback, or cancels
3. **Phase 2: 3D Generation** - Multi-agent pipeline uses concept as visual reference

Each track targets a specific domain (geometry, terrain, audio) with constrained responsibilities and validation checkpoints.

**Key Principles:**
- **Image-First**: Visual concept guides all 3D generation stages for better quality
- **User Checkpoint**: Catch misunderstandings before expensive 3D generation
- **Learning Loop**: Approved concepts become RAG examples for future generations

### Integration Points

```
User Prompt
    ↓
┌───────────────────────────────────┐
│ Phase 1: Concept Artist           │  ← Gemini 3 Pro Image Preview
│  └─ 2D Concept Image Generation   │
└──────────┬────────────────────────┘
           ↓ (User Review)
    [Approve / Regenerate / Cancel]
           ↓
┌───────────────────────────────────┐
│ Phase 2: AI Pipeline              │
│  ├─ Track Router (classifier)    │ 
│  ├─ RAG Context + Concept Image   │
│  └─ Multi-Agent Generation        │
│      (A1→A2→A3 with vision ref)   │
└──────────┬────────────────────────┘
           ↓ (DNA JSON)
┌───────────────────────────────────┐
│ Compiler Pipeline                 │ → See: compiler-pipeline.md
│  ├─ SDF Baking                    │
│  ├─ Splat Training                │
│  └─ Binary Export                 │
└──────────┬────────────────────────┘
           ↓ (.gve_bin)
┌───────────────────────────────────┐
│ GVE Engine (Rust/wgpu)            │ → See: rendering-pipeline.md
│  └─ Runtime Rendering             │
└───────────────────────────────────┘
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/generate/concept` | POST | Start concept image generation |
| `/api/generate/concept/{job_id}` | GET | Get concept status and image |
| `/api/generate/concept/{job_id}/approve` | POST | Approve and start 3D generation |
| `/api/generate/concept/{job_id}/regenerate` | POST | Regenerate with feedback |
| `/api/generate/concept/{job_id}` | DELETE | Cancel concept job |

---

## **1. Agentic Orchestration Architecture**

### Core Principles

**1. Single Responsibility Agents**
- Each agent has ONE focused task (geometry, materials, effects)
- Reduces token count, improves reliability
- Enables parallel execution where possible

**2. Constrained Output Formats**
- Agents output structured JSON/Python code, NOT freeform text
- Enforced via JSON schema validation
- Enables deterministic parsing

**3. Hierarchical Planning**

```
Orchestrator (Meta-Agent)
  │
  ├─ Stage A0: Concept Artist (gemini-3-pro-image-preview)
  │   └─ Generates 2D concept image for user approval
  │
  ├─ Track Router (Classifier)
  │
  ├─ Track A: Matter Pipeline (with concept image reference)
  │   ├─ Stage A1: Blacksmith (Form) ─ Uses concept for proportions
  │   ├─ Stage A2: Machinist (Function) ─ Uses concept for details
  │   ├─ Stage A3: Artist (Surface) ─ Uses concept for materials
  │   └─ RAG Index: Approved concept → Learning Loop
  │
  ├─ Track B: Landscape Pipeline
  │   ├─ Stage B1: Geologist
  │   ├─ Stage B2: Terraformer
  │   └─ Stage B3: Ecologist
  │
  └─ Track C: Audio Pipeline
      ├─ Stage C1: Analyst (optional)
      ├─ Stage C2: Composer
      └─ Stage C3: Sound Designer
```

**4. State Management**

```python
@dataclass
class GenerationState:
    user_prompt: str
    selected_track: str
    rag_context: dict              # API specs, material IDs, etc.
    stage_outputs: dict[str, Any]  # Accumulates results per stage
    validation_history: list[ValidationResult]
    retry_count: int = 0
    max_retries: int = 3
    concept_image_base64: str | None = None  # Visual reference for all stages
```

---

## **2. The Dispatcher (Semantic Router)**

**Entry Point:** `/api/generate`

**Responsibility:** Classify user intent and route to appropriate generation track.

### Classification Strategy

```python
def route_request(user_prompt: str) -> str:
    """
    Classify prompt into generation track.
    Options: "matter", "landscape", "audio", "composite"
    """
    # Option 1: Lightweight classifier (fine-tuned BERT)
    embedding = embed_text(user_prompt)
    track = classifier.predict(embedding)  # Target: >98% accuracy
    
    # Option 2: LLM few-shot (fallback)
    if track.confidence < 0.9:
        track = llm_classify_with_examples(user_prompt)
    
    return track
```

**Example Classifications:**
- "Make a rusty sword" → **Track A (Matter)**
- "Make a volcanic island" → **Track B (Landscape)**
- "Make a laser sound" → **Track C (Audio)**
- "Make a D-Day beach level" → **Track D (Composite)** *(orchestrates A+B+C)*

### RAG Context Injection

Once track is selected, inject relevant context:

```python
def inject_rag_context(track: str, user_prompt: str) -> dict:
    """Retrieve relevant API specs and examples."""
    
    # Hybrid search: semantic (vector) + keyword (BM25)
    rag_results = hybrid_search(
        query=user_prompt,
        collections=[f"{track}_api_spec", f"{track}_examples"],
        limit=5
    )
    
    return {
        "api_spec": rag_results["specs"],
        "examples": rag_results["examples"],
        "material_registry": get_valid_materials() if track == "matter" else None,
        "noise_configs": get_noise_presets() if track == "landscape" else None,
    }
```

**Purpose:** Agents don't guess API constants—they use exact specs from RAG.

---

## **3. Track A: Matter (3D Objects)**

**Goal:** Generate rigid body SDF geometries  
**Output:** `dna.json` → Compiler Pipeline → `.gve_bin`  
**Flow:** Form → Function → Surface

### Coordinate System Convention

All SDF primitives use a **right-handed coordinate system**:
- **Y is UP** (height)
- **Z is FORWARD** (barrel/length direction)
- **X is RIGHT** (width)

**Primitive Orientation:**
- `Cylinder`, `Capsule`, `Cone` are aligned along **Z axis** (forward)
- A weapon barrel cylinder naturally points forward without rotation
- To make a vertical grip, rotate 90° around X: `rot: [90, 0, 0]`

**Rotation (Euler XYZ, degrees):**
- `+X rotation`: Tilts forward (top → +Z)
- `-X rotation`: Tilts backward (top → -Z)
- `+Y rotation`: Rotates left (counter-clockwise from above)
- `-Y rotation`: Rotates right (clockwise from above)

### Stage A1: The Blacksmith (Form & Massing)

**Task:** Create base silhouette using **Union operations only**.

**Available Primitives:**
| Primitive | Parameters | Use Case |
|-----------|-----------|----------|
| `sphere` | `radius` | Biological forms, joints |
| `box` | `size` (half-extents [x,y,z]) | Rectangular prisms |
| `cylinder` | `radius`, `height` | Barrels, tubes (Z-aligned) |
| `capsule` | `radius`, `height` | Rounded cylinders (Z-aligned) |
| `torus` | `major_r`, `minor_r` | Rings, bands |
| `cone` | `radius`, `height` | Tapered forms (Z-aligned) |
| `wedge` | `size` (half-extents), `taper_axis`, `taper_dir` | Triangular prisms (stocks, ramps, fins) |
| `plane` | `normal`, `distance` | Ground planes, cutting |
| `revolution` | `profile`, `axis`, `offset` | Lathed forms (bowls, vases) |
| `mandelbulb` | `power`, `iterations`, `scale` | 3D fractals (alien/organic) |
| `menger` | `iterations`, `scale` | Sponge fractals (sci-fi) |
| `julia` | `c` [x,y,z,w], `iterations`, `scale` | Quaternion Julia sets |

**Available Domain Modifiers (per-node):**
| Modifier | Parameters | Effect |
|----------|-----------|--------|
| `twist` | `axis`, `rate` | Spiral along axis (rad/meter) |
| `bend` | `axis`, `angle` | Curve shape (radians) |
| `taper` | `axis`, `scale_min`, `scale_max` | Scale cross-section |
| `mirror` | `axis` | Symmetry across axis plane |
| `round` | `radius` | Bevel edges (meters) |
| `voronoi` | `cell_size`, `wall_thickness`, `mode` | 3D cellular pattern |

**System Prompt:**
```
# ROLE
You are The Blacksmith. You define the volumetric mass of 3D objects.

# TASK
Create the base geometry using CSG Union operations only.
Focus on: silhouette, proportions, major structural blocks.

# CONSTRAINTS
1. Use ONLY Union operations (no Subtract/Intersect yet)
2. Tag major blocks with lod_cutoff: 0 (always visible)
3. NO mechanical details (handles, bolts, vents)

# AVAILABLE API
{rag_context.api_spec}

# EXAMPLES
{rag_context.examples}

# OUTPUT FORMAT
{
  "sdf_tree": {
    "type": "operation",
    "op": "union",
    "children": [
      {
        "id": "unique_id",
        "type": "primitive",
        "shape": "box|sphere|cylinder|...",
        "params": {"size": [x,y,z], "radius": r, ...},
        "transform": {"pos": [x,y,z], "rot": [x_deg, y_deg, z_deg]},
        "modifiers": [{"type": "twist", "axis": "y", "rate": 2.0}],
        "lod_cutoff": 0
      }
    ]
  },
  "metadata": {
    "estimated_bounds": {"min": [...], "max": [...]},
    "primary_axis": "y"
  }
}
```

**RAG Strategy:** Query vector DB for similar assets (e.g., "tank" retrieves validated vehicle structures).

**Integration:** Output feeds into Stage A2 as immutable context.

---

### Stage A2: The Machinist (Function & Negative Space)

**Task:** Add functionality by **carving** into the mass (Subtract operations).

**Key Constraint:** Cannot delete Stage A1 nodes. Only **append** via Delta Patch.

**Available Operations:**
| Operation | Parameters | Effect |
|-----------|-----------|--------|
| `subtract` | - | Hard boolean cut |
| `smooth_subtract` | `k` (0.05-0.5) | Filleted concave edges (realistic machined cuts) |

**Available Modifiers for Subtract Geometry:**
- `voronoi`: Create honeycomb/cellular weight-reduction patterns

**System Prompt:**
```
# ROLE
You are The Machinist. You add functional features through subtraction.

# TASK
Enhance the geometry with:
- Weight reduction (hollowing, material removal)
- Mechanical features (barrels, vents, slots, bolt patterns)

# CONSTRAINTS
1. CANNOT modify Stage A1 output
2. Output ONLY new Subtract/Smooth_Subtract operations
3. Use specialized functions: Machine_Bore, Machine_Slot, Machine_Array_Radial
4. Tag features with lod_cutoff: 1 (mid-detail)
5. Use smooth_subtract with k value for filleted machined edges

# CONTEXT (READ-ONLY)
Stage A1 Output: {stage_a1_json}

# OUTPUT FORMAT
{
  "delta_patch": {
    "add_operations": [
      {
        "op": "subtract",
        "target_node_id": "box_001",
        "subtract": {"type": "primitive", "shape": "cylinder", "params": {...}},
        "lod_cutoff": 1
      },
      {
        "op": "smooth_subtract",
        "target_node_id": "box_001",
        "subtract": {"type": "primitive", "shape": "box", "params": {...}},
        "k": 0.1,
        "lod_cutoff": 1
      }
    ]
  }
}
```

**Optimization Example:**  
Instead of 8 individual cylinders for bolts:
```python
Machine_Array_Radial(count=8, radius=0.5, primitive="cylinder")
```

**Integration:** Merged with A1 → feeds A3.

---

### Stage A3: The Artist (Surface & Materials)

**Task:** Apply materials and visual style **without altering geometry**. This stage also configures **Splat Rendering** settings and **procedural textures**.

**Material Registry (27 materials, AI-friendly aliases):**
| Category | Aliases | Examples |
|----------|---------|----------|
| Metals | `METAL_STEEL`, `METAL_ALUMINUM`, `METAL_COPPER`, `METAL_TITANIUM`, `METAL_BRASS` | Weapons, machinery, aerospace |
| Stone | `CONCRETE_STANDARD`, `STONE_LIMESTONE`, `STONE_MARBLE` | Buildings, terrain |
| Wood | `WOOD_OAK`, `WOOD_PINE`, `WOOD_MAPLE` | Handles, furniture |
| Plastics | `PLASTIC_ABS`, `PLASTIC_POLYCARBONATE`, `CARBON_FIBER` | Casings, high-tech |
| Glass/Ceramic | `GLASS_CLEAR`, `CERAMIC_TILE` | Windows, pottery |
| Rubber/Textile | `RUBBER_STANDARD`, `RUBBER_SILICONE`, `TEXTILE_COTTON`, `TEXTILE_NYLON` | Grips, covers |
| Specialty | `KEVLAR_49`, `BALLISTIC_GEL` | Armor, testing |

**PBR Overrides (per-node):**
| Field | Range | Effect |
|-------|-------|--------|
| `metallic` | 0.0-1.0 | Override material registry metallic (optional) |
| `roughness` | 0.0-1.0 | Override material registry roughness (optional) |

**Texture Modifiers (per-node weathering) [aspirational -- not yet consumed by compiler]:**
| Modifier | Range | Effect |
|----------|-------|--------|
| `edge_wear` | 0.0-1.0 | Worn edges reveal underlying material |
| `cavity_grime` | 0.0-1.0 | Dirt/grime in recesses |
| `rust_amount` | 0.0-1.0 | Rust/corrosion on metals |

**Procedural Textures (noise-based pattern overlay):**
| Pattern | Effect | Best For |
|---------|--------|----------|
| `perlin` | General smooth noise | Organic surfaces, subtle variation |
| `wood_grain` | Concentric ring pattern | Natural wood grain |
| `marble` | Veined stone pattern | Marble, polished stone |
| `rust` | Patchy weathering | Realistic rust distribution |

Procedural texture params: `scale`, `intensity` (0-1), `color_variation` (0-1), `roughness_variation` (0-1), `metallic_variation` (0-1, default 0).

**System Prompt:**
```
# ROLE
You are The Artist. You define surface appearance and rendering quality.

# TASK
1. Assignment: Apply materials to existing nodes using valid Material IDs.
2. Configuration: Set global render settings for Gaussian Splatting.
3. Texturing: Optionally apply procedural texture patterns for surface detail.

# CONSTRAINTS
1. CANNOT modify geometry from A1/A2.
2. Use ONLY valid Material_IDs from the registry.
3. Configure `splat_count` based on asset complexity (Low: 10k, Med: 50k, High: 100k).

# AVAILABLE MATERIALS
{rag_context.material_registry}

# STYLE TOKEN
{user_style_token}  # e.g., "Cyberpunk", "WW2", "Industrial"

# OUTPUT FORMAT
{
  "material_config": {
    "node_001": {
      "material_id": "METAL_STEEL",
      "base_color": "#5A5A5A",
      "metallic": 0.9,
      "roughness": 0.35,
      "texture_modifiers": {
        "edge_wear": 0.3,
        "cavity_grime": 0.2,
        "rust_amount": 0.1
      },
      "procedural_texture": {
        "type": "rust",
        "scale": 4.0,
        "intensity": 0.3,
        "color_variation": 0.2,
        "roughness_variation": 0.15,
        "metallic_variation": 0.1
      }
    }
  },
  "render_settings": {
    "splat_count": 50000,
    "iterations": 300
  }
}
```

**Style Logic:**
- `Industrial` → Apply warning stripes, rust procedural texture, edge wear, high roughness
- `Cyberpunk` → Emissive panels, carbon fiber, low roughness on panels, metallic overrides
- `Organic` → SmoothUnion blending, organic materials, marble/wood textures, low metallic
- `Military` → Kevlar, steel, high cavity grime, low edge wear, high metallic
- `Steampunk` → Copper, brass, high rust_amount, wood grain textures, metallic_variation on rust

**Integration:** Combined JSON → Compiler Pipeline for baking to `.gve_bin`.

---

## **4. Track B: Landscape (Terrain)**

**Goal:** Generate voxel volume configs for map-scale environments  
**Output:** Terrain config → Voxel Baker → Runtime Volume Texture  
**Flow:** Bedrock → Features → Biomes

### Stage B1: The Geologist (Base Layer)

**Task:** Define global noise parameters for terrain height map.

**System Prompt:**
```
# TASK
Translate terrain description into Perlin/Voronoi noise parameters.

# LOGIC
Adjective → Math:
- "Jagged Mountains" → High Amplitude, High Lacunarity, Voronoi Basis
- "Rolling Hills" → Low Frequency, Perlin Basis, Low Amplitude
- "Desert Dunes" → Low Frequency, Medium Amplitude, Ridged Noise

# OUTPUT
{
  "noise_config": {
    "basis": "perlin|voronoi|simplex",
    "frequency": 0.01,
    "amplitude": 100.0,
    "lacunarity": 2.0,
    "octaves": 6,
    "seed": 12345
  }
}
```

**Integration:** Noise config → Voxel Baker generates base heightmap.

---

### Stage B2: The Terraformer (Macro Features)

**Task:** Add specific geographic features that noise misses (craters, valleys, plateaus).

**Strategy:** Use SDF primitives as global modifiers:

```
Caldera → Subtract(Cylinder, radius=500m)
Canyon → Subtract(Box, stretched along axis)
Plateau → SmoothUnion(Box, height=100m)
```

**Output:** List of global SDF modifiers applied to base terrain.

---

### Stage B3: The Ecologist (Material Biomes)

**Task:** Assign materials based on slope/height/normal rules.

**System Prompt:**
```
# TASK
"Paint" terrain materials using procedural rules.

# AVAILABLE RULES
- Rule_Slope_Mask: Assign material if slope > threshold
- Rule_Height_Mask: Assign material if height in range
- Rule_Normal_Mask: Assign material if normal.y < threshold

# EXAMPLE
If slope > 45° → ASTM_C114 (Rock)
If height < 0 → Sand (Seabed)
If slope < 10° AND height > 50 → Grass

# OUTPUT
{
  "material_rules": [
    {
      "condition": {"slope_deg": {">": 45}},
      "material_id": "ASTM_C114"
    }
  ]
}
```

**Integration:** Rules → Voxel Baker assigns material IDs per voxel.

---

## **5. Track C: Resonance (Audio)**

**Goal:** Generate physics-driven DSP patches  
**Output:** `AudioPatch` struct → Audio System (see `audio-system.md`)  
**Flow:** Analysis → Synthesis → FX

### Stage C1: The Analyst (Optional)

**Input:** `.wav` file upload  
**Task:** Extract spectral fingerprint via `librosa`

```python
def analyze_audio_sample(wav_path: str) -> dict:
    y, sr = librosa.load(wav_path)
    
    return {
        "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean(),
        "mfcc": librosa.feature.mfcc(y=y, sr=sr).mean(axis=1).tolist(),
        "attack_time_ms": detect_attack(y, sr),
        "decay_time_ms": detect_decay(y, sr),
    }
```

**Output:** Physical characteristics to guide FM synthesis.

---

### Stage C2: The Composer (Synthesis Architecture)

**Task:** Define FM oscillator topology.

**System Prompt:**
```
# TASK
Design FM synth patch matching the target sound or material.

# LOGIC
"Laser" → Sawtooth Carrier + Triangle Modulator + Fast Pitch Envelope
"Heavy Steel" → Sine Carrier + Inharmonic Ratio 1.41 + Long Decay
"Glass" → High-Frequency Carrier + Short Attack + Reverb

# OUTPUT
{
  "oscillators": {
    "carrier": {"waveform": "sine", "freq_hz": 440},
    "modulator": {"waveform": "triangle", "freq_ratio": 1.5}
  },
  "envelopes": {
    "amplitude": {"attack": 0.01, "decay": 0.5, "sustain": 0, "release": 0.2}
  }
}
```

**Integration:** Patch → Audio System for physics-driven playback.

---

### Stage C3: The Sound Designer (FX Chain)

**Task:** Add DSP effects (filters, reverb, distortion).

**System Prompt:**
```
# TASK
Add environmental effects and character.

# STYLE LOGIC
"Retro" → Add Bitcrusher + LowPass
"Cathedral" → Long Reverb (RT60 = 3s)
"Underwater" → Extreme LowPass + Chorus

# OUTPUT
{
  "dsp_chain": [
    {"type": "lowpass", "cutoff_hz": 2000, "resonance": 0.5},
    {"type": "reverb", "room_size": 0.8, "damping": 0.3}
  ]
}
```

---

## **6. Validation & Quality Assurance**

### 6.1 Vision-Critic Loop (Track A only)

**Purpose:** Ensure geometric integrity before user sees asset.

```python
def vision_critic_loop(generated_json: dict) -> dict:
    # 1. Render headless
    image = architect.render_headless(generated_json, size=512)
    
    # 2. VLM Critique
    checklist = get_checklist(asset_type="weapon")
    critique = query_vlm(
        image=image,
        prompt=f"Check for issues:\n{checklist}\nOutput JSON."
    )
    
    # 3. Auto-fix if possible
    if not critique["passed"]:
        fix_patch = generate_fix_patch(critique["errors"])
        generated_json = apply_patch(generated_json, fix_patch)
        
        #Retry once
        return vision_critic_loop(generated_json)
    
    return generated_json
```

**Checklist Example (Weapon):**
- Handle attached to blade?
- Proportions realistic?
- No floating geometry?
- Trigger guard is closed loop?

---

### 6.2 Program-Aided Language (PAL)

**Problem:** LLMs struggle with precise math (bolt circle coordinates).

**Solution:** Generate Python code, execute programmatically.

**Bad (LLM outputs raw coords):**
```json
{"bolts": [[1.0, 0.0, 0.0], [0.707, 0.707, 0.0], ...]}  // Error-prone
```

**Good (LLM outputs code):**
```python
generate_circle(count=8, radius=1.0, axis="z")
```

**Backend executes** → Perfect coordinates.

---

### 6.3 Automatic Retry with Feedback

```python
def generate_with_retry(agent: Agent, state: GenerationState) -> dict:
    for attempt in range(state.max_retries):
        output = agent.generate(state)
        validation = validate(output, schema=agent.output_schema)
        
        if validation.passed:
            # Store as positive RAG example
            vector_db.store("successful_generations", {
                "prompt": state.user_prompt,
                "output": output,
                "validation_score": validation.score
            })
            return output
        
        # Failed - add error context for retry
        state.retry_context = f"""
Previous attempt failed:
Errors: {validation.errors}
Suggestions: {validation.suggestions}
Correct these in next attempt.
"""
    
    raise ValidationFailure(f"Failed after {state.max_retries} attempts")
```

---

## **7. Prompt Engineering Best Practices**

### 7.1 Structured Template

Every agent prompt includes:
1. **Role** - "You are The Blacksmith..."
2. **Task** - Clear objective
3. **Constraints** - What NOT to do
4. **API Spec** - From RAG, not hallucinated
5. **Examples** - 3-5 few-shot (dynamically selected)
6. **Output Schema** - Strict JSON with validation
7. **Validation Rules** - Explicit requirements

### 7.2 Dynamic Few-Shot Selection

```python
def select_examples(user_prompt: str, num: int = 3) -> list[dict]:
    query_embedding = embed_text(user_prompt)
    
    # Vector search for similar successful generations
    results = vector_db.search(
        collection="successful_generations",
        query_vector=query_embedding,
        limit=num,
        filters={"validation_score": {">=": 0.9}}
    )
    
    return [{"input": r["prompt"], "output": r["output"]} for r in results]
```

**Benefit:** Most relevant examples per request → better results.

---

## **8. Cost Optimization**

### 8.1 Model Tier Selection

```python
MODEL_TIERS = {
    "simple": {"model": "gpt-4o-mini", "cost_per_1k": 0.00015},    # Router, classifier
    "moderate": {"model": "gpt-4o", "cost_per_1k": 0.0025},       # Geometry generation
    "complex": {"model": "claude-3-5-sonnet", "cost_per_1k": 0.003}, # Complex CSG
    "vision": {"model": "gpt-4-vision", "cost_per_1k": 0.01},     # Critic only
}
```

**Strategy:** Use cheapest model that can handle task.

### 8.2 Response Caching

```python
cache = Redis()  # 24hr TTL

def cached_generate(prompt: str) -> dict:
    cache_key = hashlib.sha256(prompt.encode()).hexdigest()
    
    if cached := cache.get(cache_key):
        return json.loads(cached)
    
    result = llm_generate(prompt)
    cache.setex(cache_key, 86400, json.dumps(result))
    
    return result
```

**Expected Hit Rate:** 60%+ for common requests ("rusty sword", "grass terrain").

### 8.3 Batch Processing

Process 10 requests in parallel → ~30% cost savings (shared API overhead).

---

## **9. Performance Metrics**

### Target Performance

| Metric | Target | Notes |
|--------|--------|-------|
| **First-pass success** | >95% | Asset valid without retry |
| **Validation score** | >0.9 | Quality rating (0-1) |
| **User acceptance** | >90% | User keeps generated asset |
| **Generation time (p50)** | <5s | Median latency |
| **Generation time (p95)** | <10s | 95th percentile |
| **Cost per asset** | <$0.10 | LLM API costs |
| **Cache hit rate** | >60% | Redis cache effectiveness |

### Monitoring

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "agent_generation_complete",
    agent="blacksmith",
    duration_ms=1234,
    tokens_used=890,
    validation_score=0.94,
    cost_usd=0.002
)
```

**Observability Stack:** Prometheus + Grafana

---

## **10. Implementation Checklist**

### Two-Phase Generation (NEW - Implemented Feb 2026)
- [x] Concept Artist agent (gemini-3-pro-image-preview)
- [x] `/api/generate/concept` endpoints
- [x] Concept preview UI in Forge
- [x] Approve/Regenerate/Cancel workflow
- [x] Vision-guided agents (Blacksmith, Machinist, Artist)
- [x] Concept image → RAG learning loop

### Matter Pipeline (Track A)
- [x] Stage A1: Blacksmith (Form & Massing)
- [x] Stage A2: Machinist (Function & Details)
- [x] Stage A3: Artist (Materials & Surface)
- [x] `_merge_pipeline_outputs()` integration
- [x] Concept image as visual reference for all stages

### Extended Primitives & Modifiers (Feb 2026)
- [x] Fractal primitives: Mandelbulb, Menger Sponge, Julia Set
- [x] Revolution primitive (lathe operation)
- [x] Wedge primitive (triangular prism for stocks, ramps, fins, blade edges)
- [x] Smooth subtract/intersect operations (filleted CSG)
- [x] Voronoi modifier (cellular/honeycomb patterns)
- [x] Procedural texture node (perlin, wood_grain, marble, rust)
- [x] Expanded material registry (27 materials, 17+ AI-friendly aliases)
- [x] Agent prompts updated with full compiler capabilities
- [x] DNA validator accepts new primitive shapes

### Infrastructure
- [x] MongoDB Atlas Vector Search for RAG
- [ ] Redis for response caching
- [ ] Prometheus + Grafana monitoring
- [ ] Rate limiting (per-user quotas)
- [ ] Request queue (Celery/RabbitMQ)

### AI Components
- [ ] Fine-tune track classifier (>98% accuracy)
- [x] RAG index from approved concepts (learning loop)
- [ ] Create eval dataset (100+ examples per track)
- [ ] Set up A/B testing for prompt variations

### Quality Assurance
- [x] Define validation schemas per asset type (Pydantic)
- [ ] Automated test suite (unit + integration)
- [ ] Human review queue for edge cases
- [x] Feedback loop: approved concepts → RAG examples
- [ ] Weekly hallucination rate monitoring

### Cost Management
- [ ] Per-user daily budget limits
- [ ] Model tier auto-selection
- [ ] Semantic caching with Redis
- [ ] Cost tracking dashboard
- [ ] Alerts for budget overruns (>$X/day)

---

## **11. Future Enhancements**

### Reinforcement Learning from Human Feedback (RLHF)
- Train reward model on user accept/reject decisions
- Fine-tune agents with PPO to maximize satisfaction

### Multi-Modal Input
- Accept sketch images (Vision Encoder → SDF)
- Reference photos for material/style matching
- Voice prompts for hands-free workflows

### Collaborative Multi-Agent Debates
- Multiple agents propose competing solutions
- Critic agent selects best via structured debate
- Improves creative diversity

### Automated Hyperparameter Tuning
- A/B test temperature, top_p per stage
- Bayesian optimization for quality/cost tradeoff
- Auto-adjust based on validation scores

---

## **Related Documentation**

- **[Compiler Pipeline](./compiler-pipeline.md)** - Processes AI output into runtime binaries
- **[Rendering Pipeline](../core-systems/rendering-pipeline.md)** - Renders compiled assets
- **[Audio System](../core-systems/audio-system.md)** - Physics-driven DSP patches
- **[Data Specifications](../data/data-specifications.md)** - JSON schema definitions
- **[Database Architecture](../data/database-architecture.md)** - MongoDB + RAG setup

---

**Status:** ✅ Complete  
**Last Updated:** February 6, 2026  
**Version:** 3.1 (Enhanced Primitives, Modifiers & Procedural Textures)