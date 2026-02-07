# backend/architect/src/ai_pipeline/__init__.py
"""
AI Pipeline Module - LLM orchestration for asset generation.

Handles ONLY AI inference and DNA generation. Does NOT handle
compilation (that's compiler) or database storage (that's librarian).

Two-Phase Generation Flow:
1. generate_concept_image() - Generate 2D concept (fast, ~3s)
2. User reviews and approves concept
3. generate_asset_with_concept() - Generate 3D with concept reference

Stage Previews:
- Pass on_stage_complete callback to generate_asset_with_concept()
- Callback receives (stage_name, preview_binary_bytes) after each stage
"""

from .orchestrator import (
    generate_asset,
    generate_asset_with_concept,
    GenerateRequest,
    GenerateResult,
    GenerationState,
    StageCompleteCallback,
)
from .rag import (
    semantic_search,
    index_asset,
    create_embedding,
    RagResult,
    # Concept RAG
    index_concept_image,
    search_similar_concepts,
    inject_rag_context_with_concepts,
    ConceptRagResult,
)
from .concept_artist import (
    generate_concept_image,
    regenerate_concept_with_feedback,
    get_available_styles,
    ConceptResult,
)

__all__ = [
    # Orchestration
    "generate_asset",
    "generate_asset_with_concept",
    "GenerateRequest",
    "GenerateResult",
    "GenerationState",
    "StageCompleteCallback",
    # Concept Artist
    "generate_concept_image",
    "regenerate_concept_with_feedback",
    "get_available_styles",
    "ConceptResult",
    # RAG
    "semantic_search",
    "index_asset",
    "create_embedding",
    "RagResult",
    # Concept RAG (Learning Loop)
    "index_concept_image",
    "search_similar_concepts",
    "inject_rag_context_with_concepts",
    "ConceptRagResult",
]
