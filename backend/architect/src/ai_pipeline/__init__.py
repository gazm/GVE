# backend/architect/src/ai_pipeline/__init__.py
"""
AI Pipeline Module - LLM orchestration for asset generation.

Handles ONLY AI inference and DNA generation. Does NOT handle
compilation (that's compiler) or database storage (that's librarian).
"""

from .orchestrator import (
    generate_asset,
    GenerateRequest,
    GenerateResult,
    GenerationState,
)
from .rag import (
    semantic_search,
    index_asset,
    create_embedding,
    RagResult,
)

__all__ = [
    # Orchestration
    "generate_asset",
    "GenerateRequest",
    "GenerateResult",
    "GenerationState",
    # RAG
    "semantic_search",
    "index_asset",
    "create_embedding",
    "RagResult",
]
