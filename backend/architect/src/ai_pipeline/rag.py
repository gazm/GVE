# backend/architect/src/ai_pipeline/rag.py
"""
RAG (Retrieval-Augmented Generation) for AI Pipeline.

Uses MongoDB Atlas Vector Search for semantic similarity.
Embeds text with sentence-transformers for query vectors.

Concept Image Learning Loop:
- When users approve concepts, they're indexed for future retrieval
- Text embeddings enable semantic search ("find concepts like X")
- Approved concepts become few-shot examples for similar prompts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel

# Lazy load embedding model to avoid startup cost
_embedding_model = None


def _get_embedding_model():
    """Lazy load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            # Wait for torch preloader before importing sentence-transformers
            from src.torch_preloader import preloader
            if not preloader.ensure_loaded():
                print("⚠️ Torch unavailable, using mock embeddings")
                _embedding_model = "mock"
                return _embedding_model
            
            # Patch missing torch.distributed.is_initialized for ROCm compatibility
            # Some sentence-transformers versions call this during model loading
            import torch
            if not hasattr(torch.distributed, 'is_initialized'):
                torch.distributed.is_initialized = lambda: False
            
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Loaded embedding model: all-MiniLM-L6-v2")
        except ImportError:
            print("⚠️ sentence-transformers not installed, using mock embeddings")
            _embedding_model = "mock"
        except Exception as e:
            print(f"⚠️ Embedding model failed: {e}, using mock embeddings")
            _embedding_model = "mock"
    return _embedding_model


class RagResult(BaseModel):
    """Result from RAG semantic search."""
    
    asset_id: str
    name: str
    score: float
    dna: dict[str, Any] | None = None
    semantic_desc: str | None = None


@dataclass
class RagContext:
    """Context injected into agents from RAG."""
    
    api_spec: dict[str, Any]
    examples: list[dict[str, Any]]
    material_registry: dict[str, Any] | None = None
    noise_configs: dict[str, Any] | None = None


def create_embedding(text: str) -> list[float]:
    """
    Generate embedding vector for text.
    
    Uses all-MiniLM-L6-v2 (384 dimensions) for efficient similarity.
    """
    model = _get_embedding_model()
    
    if model == "mock":
        # Return random embedding for testing without model
        return list(np.random.randn(384).astype(float))
    
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


async def semantic_search(
    query: str,
    limit: int = 5,
    category_filter: str | None = None,
) -> list[RagResult]:
    """
    Search for similar assets using MongoDB vector search.
    
    Combines semantic (vector) search with optional faceted filters.
    """
    # Generate query embedding
    query_embedding = create_embedding(query)
    
    # Import librarian for DB access
    from ..librarian import vector_search
    
    # Build filter if provided
    pre_filter = {}
    if category_filter:
        pre_filter["meta.category"] = category_filter
    
    # Execute vector search
    results = await vector_search(
        query_embedding=query_embedding,
        limit=limit,
        pre_filter=pre_filter if pre_filter else None,
    )
    
    # Convert to RagResult objects
    return [
        RagResult(
            asset_id=str(doc.get("_id", "")),
            name=doc.get("name", "unknown"),
            score=doc.get("score", 0.0),
            dna=doc.get("dna"),
            semantic_desc=doc.get("rag", {}).get("semantic_desc"),
        )
        for doc in results
    ]


async def index_asset(asset_id: str, dna: dict[str, Any]) -> None:
    """
    Index an asset for RAG retrieval.
    
    Generates text embedding from DNA and stores in MongoDB.
    Called automatically by librarian on asset save.
    """
    # Generate semantic description from DNA
    semantic_desc = _dna_to_description(dna)
    
    # Create embedding
    embedding = create_embedding(semantic_desc)
    
    # Update asset document with RAG data
    from ..librarian import update_asset_rag
    
    await update_asset_rag(
        asset_id=asset_id,
        embedding=embedding,
        semantic_desc=semantic_desc,
    )


# =============================================================================
# Concept Image RAG (Learning Loop)
# =============================================================================

class ConceptRagResult(BaseModel):
    """Result from concept image search."""
    
    asset_id: str
    prompt: str
    score: float
    concept_image: str | None = None  # Base64 image data
    dna: dict[str, Any] | None = None  # Linked 3D output


async def index_concept_image(
    asset_id: str,
    prompt: str,
    concept_image_base64: str,
    dna: dict[str, Any] | None = None,
) -> None:
    """
    Index an approved concept image for RAG retrieval.
    
    Called when a user approves a concept and 3D generation completes.
    Creates a text embedding from the prompt for semantic search.
    
    Args:
        asset_id: ID of the associated asset
        prompt: Original user prompt
        concept_image_base64: Base64-encoded concept image
        dna: Final DNA output (optional, links concept to successful 3D)
    """
    print(f"📚 [RAG] Indexing concept image for asset {asset_id}")
    
    # Create text embedding from prompt
    embedding = create_embedding(prompt)
    
    # Store concept data in MongoDB
    from ..librarian import store_concept_rag
    
    await store_concept_rag(
        asset_id=asset_id,
        prompt=prompt,
        embedding=embedding,
        concept_image=concept_image_base64,
        dna=dna,
    )
    
    print(f"✅ [RAG] Concept indexed for asset {asset_id}")


async def search_similar_concepts(
    query: str,
    limit: int = 3,
) -> list[ConceptRagResult]:
    """
    Search for similar approved concepts using semantic search.
    
    Returns concept images that were approved for similar prompts.
    These serve as visual examples for new generations.
    
    Args:
        query: User's prompt to find similar concepts for
        limit: Maximum number of results
        
    Returns:
        List of similar concept results with images and linked DNA
    """
    # Generate query embedding
    query_embedding = create_embedding(query)
    
    # Search concepts collection
    from ..librarian import search_concepts
    
    try:
        results = await search_concepts(
            query_embedding=query_embedding,
            limit=limit,
        )
        
        return [
            ConceptRagResult(
                asset_id=str(doc.get("asset_id", "")),
                prompt=doc.get("prompt", ""),
                score=doc.get("score", 0.0),
                concept_image=doc.get("concept_image"),
                dna=doc.get("dna"),
            )
            for doc in results
        ]
    except Exception as e:
        print(f"⚠️ [RAG] Concept search failed: {e}")
        return []


async def inject_rag_context_with_concepts(
    track: str,
    user_prompt: str,
) -> dict[str, Any]:
    """
    Inject RAG context including similar concept images.
    
    Enhanced version of inject_rag_context that also retrieves
    similar approved concepts for visual guidance.
    
    Args:
        track: Generation track (matter, landscape, audio)
        user_prompt: User's asset description
        
    Returns:
        Context dict with api_spec, examples, and concept_examples
    """
    # Get base context
    context = await inject_rag_context(track, user_prompt)
    
    # Add similar concept images
    try:
        similar_concepts = await search_similar_concepts(user_prompt, limit=2)
        context["concept_examples"] = [
            {
                "prompt": c.prompt,
                "concept_image": c.concept_image,
                "dna": c.dna,
            }
            for c in similar_concepts
            if c.concept_image
        ]
        
        if context["concept_examples"]:
            print(f"📷 [RAG] Found {len(context['concept_examples'])} similar concepts")
    except Exception as e:
        print(f"⚠️ [RAG] Failed to get concept examples: {e}")
        context["concept_examples"] = []
    
    return context


def _dna_to_description(dna: dict[str, Any]) -> str:
    """
    Convert DNA JSON to natural language description for embedding.
    
    Extracts key features: shapes, materials, operations, modifiers.
    """
    parts = []
    
    # Extract primitives
    primitives = _extract_primitives(dna.get("root_node", dna))
    if primitives:
        parts.append(f"Shapes: {', '.join(primitives)}")
    
    # Extract materials
    materials = dna.get("materials", {})
    if materials:
        mat_names = [m.get("material_id", "unknown") for m in materials.values()]
        parts.append(f"Materials: {', '.join(set(mat_names))}")
    
    # Extract operations
    ops = _extract_operations(dna.get("root_node", dna))
    if ops:
        parts.append(f"Operations: {', '.join(set(ops))}")
    
    # Extract modifiers
    mods = _extract_modifiers(dna.get("root_node", dna))
    if mods:
        parts.append(f"Modifiers: {', '.join(set(mods))}")
    
    return ". ".join(parts) if parts else "Generated 3D asset"


def _extract_primitives(node: dict, primitives: list | None = None) -> list[str]:
    """Recursively extract primitive shapes from SDF tree."""
    if primitives is None:
        primitives = []
    
    if node.get("type") == "primitive":
        shape = node.get("shape", "unknown")
        primitives.append(shape)
    
    for child in node.get("children") or []:
        _extract_primitives(child, primitives)
    
    return primitives


def _extract_operations(node: dict, ops: list | None = None) -> list[str]:
    """Recursively extract operations from SDF tree."""
    if ops is None:
        ops = []
    
    if node.get("type") == "operation":
        op = node.get("op", "unknown")
        ops.append(op)
    
    for child in node.get("children") or []:
        _extract_operations(child, ops)
    
    return ops


def _extract_modifiers(node: dict, mods: list | None = None) -> list[str]:
    """Recursively extract modifier types from SDF tree."""
    if mods is None:
        mods = []
    
    # Check if this node has modifiers
    modifiers = node.get("modifiers") or []
    if modifiers:
        for mod in modifiers:
            mod_type = mod.get("type", "unknown")
            mods.append(mod_type)
    
    for child in node.get("children") or []:
        _extract_modifiers(child, mods)
    
    return mods


async def inject_rag_context(track: str, user_prompt: str) -> dict[str, Any]:
    """
    Inject relevant RAG context for a generation request.
    
    Retrieves API specs, similar examples, and domain-specific data.
    """
    context: dict[str, Any] = {
        "api_spec": _get_api_spec(track),
        "examples": [],
        "material_registry": None,
        "noise_configs": None,
    }
    
    # Search for similar successful generations
    try:
        similar = await semantic_search(user_prompt, limit=3)
        context["examples"] = [
            {"prompt": r.semantic_desc, "dna": r.dna}
            for r in similar
            if r.dna
        ]
    except Exception as e:
        print(f"⚠️ RAG search failed (continuing without examples): {e}")
    
    # Add track-specific context
    if track in ("matter", "MATTER"):
        context["material_registry"] = _get_material_registry()
    elif track in ("landscape", "LANDSCAPE"):
        context["noise_configs"] = _get_noise_configs()
    
    return context


def _get_api_spec(track: str) -> dict[str, Any]:
    """Get API specification for track (from docs or hardcoded)."""
    # MVP: Hardcoded spec from data-specifications.md
    return {
        "primitives": [
            "Sphere(radius)",
            "Box(size_vec3)",
            "Cylinder(radius, height, sides)",
            "Capsule(radius, height)",
            "Torus(major_r, minor_r)",
            "Cone(radius, height, sides)",
            "Plane(normal, distance)",
            "Revolution(profile, axis, offset) - Lathe: spin a 2D profile around axis",
            "Mandelbulb(power, iterations, scale) - 3D fractal, compute-heavy",
            "Menger(iterations, scale) - Menger sponge fractal",
            "Julia(c=[x,y,z,w], iterations, scale) - Quaternion Julia set",
        ],
        "operations": [
            "union",
            "subtract",
            "intersect",
            "smooth_union(k)",
            "smooth_subtract(k) - filleted concave edges",
            "smooth_intersect(k) - filleted convex edges",
        ],
        "modifiers": {
            "twist": {"params": ["axis", "rate"], "desc": "Spiral along axis (rate = rad/meter)"},
            "bend": {"params": ["axis", "angle"], "desc": "Curve shape (angle in radians)"},
            "taper": {"params": ["axis", "scale_min", "scale_max"], "desc": "Scale cross-section"},
            "mirror": {"params": ["axis"], "desc": "Symmetry across axis plane"},
            "round": {"params": ["radius"], "desc": "Bevel edges (radius in meters)"},
            "voronoi": {
                "params": ["cell_size", "wall_thickness", "mode"],
                "desc": "3D Voronoi cellular pattern (mode: subtract or intersect)",
            },
        },
        "modifier_examples": [
            '{"type": "twist", "axis": "y", "rate": 3.14}',
            '{"type": "bend", "axis": "x", "angle": 0.5}',
            '{"type": "taper", "axis": "y", "scale_min": 0.2, "scale_max": 1.0}',
            '{"type": "mirror", "axis": "x"}',
            '{"type": "round", "radius": 0.02}',
            '{"type": "voronoi", "cell_size": 0.2, "wall_thickness": 0.02, "mode": "subtract"}',
        ],
        "texture_patterns": {
            "perlin": {
                "params": ["scale", "intensity", "color_variation", "roughness_variation"],
                "desc": "General smooth noise (FBM-based)",
            },
            "wood_grain": {
                "params": ["scale", "intensity", "color_variation", "roughness_variation"],
                "desc": "Concentric ring pattern for wood",
            },
            "marble": {
                "params": ["scale", "intensity", "color_variation", "roughness_variation"],
                "desc": "Veined stone pattern (sine + turbulence)",
            },
            "rust": {
                "params": ["scale", "intensity", "color_variation", "roughness_variation"],
                "desc": "Patchy weathering/corrosion (voronoi + perlin blend)",
            },
        },
        "texture_pattern_example": (
            '{"type": "wood_grain", "scale": 10.0, "intensity": 0.3, '
            '"color_variation": 0.2, "roughness_variation": 0.1}'
        ),
        "units": {
            "distance": "1.0 = 1 meter",
            "rotation": "quaternion [x, y, z, w]",
        },
    }


def _get_material_registry() -> dict[str, Any]:
    """Get available materials for Matter track from MaterialLibrarian."""
    from ..librarian import get_registry_for_rag
    return get_registry_for_rag()


def _get_noise_configs() -> dict[str, Any]:
    """Get noise presets for Landscape track."""
    return {
        "mountains": {
            "basis": "voronoi",
            "frequency": 0.005,
            "amplitude": 200.0,
            "lacunarity": 2.0,
            "octaves": 8,
        },
        "hills": {
            "basis": "perlin",
            "frequency": 0.01,
            "amplitude": 50.0,
            "lacunarity": 2.0,
            "octaves": 4,
        },
        "desert": {
            "basis": "simplex",
            "frequency": 0.02,
            "amplitude": 20.0,
            "lacunarity": 1.8,
            "octaves": 3,
        },
    }
