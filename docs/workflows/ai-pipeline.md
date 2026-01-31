# **GVE-1 AI Pipeline**

**Role:** Intelligent asset generation via multi-agent orchestration  
**Integration:** Architect (Layer 2) → Compiler Pipeline → Engine (Layer 3)  
**Philosophy:** "Incremental Assembly with Active Verification"

---

## **System Overview**

The AI Pipeline transforms natural language prompts into valid GVE-1 assets (`.gve_bin`) through specialized AI agents working in parallel tracks. Each track targets a specific domain (geometry, terrain, audio) with constrained responsibilities and validation checkpoints.

**Key Principle:** Break complex generation into simple, verifiable steps rather than one monolithic LLM call.

### Integration Points

```
User Prompt
    ↓
┌───────────────────────────────────┐
│ AI Pipeline (Architect/Python)    │
│  ├─ Track Router (classifier)    │ 
│  ├─ RAG Context Injection         │
│  └─ Multi-Agent Generation        │
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
  ├─ Track Router (Classifier)
  │
  ├─ Track A: Matter Pipeline
  │   ├─ Stage A1: Blacksmith (Form)
  │   ├─ Stage A2: Machinist (Function)
  │   ├─ Stage A3: Artist (Surface)
  │   └─ Validator: Vision-Critic
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

### Stage A1: The Blacksmith (Form & Massing)

**Task:** Create base silhouette using **Union operations only**.

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
    "operations": [
      {"op": "union", "left": {...}, "right": {...}}
    ],
    "primitives": [
      {"type": "box", "params": {"size": [x,y,z], "center": [x,y,z]}}
    ]
  },
  "metadata": {
    "estimated_bounds": {"min": [...], "max": [...]},
    "primary_lod": 0
  }
}
```

**RAG Strategy:** Query vector DB for similar assets (e.g., "tank" retrieves validated vehicle structures).

**Integration:** Output feeds into Stage A2 as immutable context.

---

### Stage A2: The Machinist (Function & Negative Space)

**Task:** Add functionality by **carving** into the mass (Subtract operations).

**Key Constraint:** Cannot delete Stage A1 nodes. Only **append** via Delta Patch.

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
2. Output ONLY new Subtract operations
3. Use specialized functions: Machine_Bore, Machine_Slot, Machine_Array_Radial
4. Tag features with lod_cutoff: 1 (mid-detail)

# CONTEXT (READ-ONLY)
Stage A1 Output: {stage_a1_json}

# OUTPUT FORMAT
{
  "delta_patch": {
    "add_operations": [
      {
        "op": "subtract",
        "target_node_id": "box_001",
        "subtract": {"type": "cylinder", ...},
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

###Stage A3: The Artist (Surface & Materials)

**Task:** Apply materials and visual style **without altering geometry**.

**System Prompt:**
```
# ROLE
You are The Artist. You define surface appearance.

# TASK
Apply materials and texture modifiers based on style token.

# CONSTRAINTS
1. CANNOT modify geometry from A1/A2
2. Use ONLY valid Material_IDs from ASTM/AMS registry
3. Apply triplanar texture modifiers (see rendering-pipeline.md §3.4)

# AVAILABLE MATERIALS
{rag_context.material_registry}

# STYLE TOKEN
{user_style_token}  # e.g., "Cyberpunk", "WW2", "Industrial"

# OUTPUT FORMAT
{
  "material_config": {
    "node_001": {
      "material_id": "ASTM_C114",  # Concrete
      "texture_modifiers": {
        "decal_projection": "warning_stripes",
        "wear_amount": 0.6
      },
      "color_mode": "rgb"  # or "oklab" for procedural
    }
  }
}
```

**Style Logic:**
- `Industrial` → Apply warning stripes, rust
- `Cyberpunk` → Emissive panels, holographic effects
- `Organic` → SmoothUnion blending, organic materials

**Integration:** Combined JSON → Compiler Pipeline for baking.

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

##**  8. Cost Optimization**

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

### Infrastructure
- [ ] Vector DB (Qdrant/Pinecone) for RAG
- [ ] Redis for response caching
- [ ] Prometheus + Grafana monitoring
- [ ] Rate limiting (per-user quotas)
- [ ] Request queue (Celery/RabbitMQ)

### AI Components
- [ ] Fine-tune track classifier (>98% accuracy)
- [ ] Build RAG index from API docs + examples
- [ ] Implement Vision Critic (GPT-4V integration)
- [ ] Create eval dataset (100+ examples per track)
- [ ] Set up A/B testing for prompt variations

### Quality Assurance
- [ ] Define validation schemas per asset type
- [ ] Automated test suite (unit + integration)
- [ ] Human review queue for edge cases
- [ ] Feedback loop: accepted assets → RAG examples
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
**Last Updated:** January 25, 2026  
**Version:** 2.0 (Agentic Architecture)