# backend/architect/src/ai_pipeline/track_matter.py
"""
Track A: Matter Pipeline - 3D Object Generation

Flow: Blacksmith (Form) → Machinist (Function) → Artist (Surface)

This module re-exports all components from the split files for backward compatibility.
"""

from __future__ import annotations

# Re-export schemas
from .track_matter_schemas import (
    PrimitiveParams,
    Transform,
    Modifier,
    SDFNode,
    SDFRootNode,
    BlacksmithOutput,
    SubtractPrimitive,
    MachinistDeltaPatch,
    MachinistDeltaPatchList,
    MachinistOutput,
    MaterialConfig,
    ArtistOutput,
)

# Re-export agents
from .track_matter_agents import (
    BlacksmithAgent,
    MachinistAgent,
    ArtistAgent,
)

# Re-export pipeline
from .track_matter_pipeline import (
    execute_matter_pipeline,
    StageCompleteCallback,
)

__all__ = [
    # Schemas
    "PrimitiveParams",
    "Transform",
    "Modifier",
    "SDFNode",
    "SDFRootNode",
    "BlacksmithOutput",
    "SubtractPrimitive",
    "MachinistDeltaPatch",
    "MachinistDeltaPatchList",
    "MachinistOutput",
    "MaterialConfig",
    "ArtistOutput",
    # Agents
    "BlacksmithAgent",
    "MachinistAgent",
    "ArtistAgent",
    # Pipeline
    "execute_matter_pipeline",
    "StageCompleteCallback",
]
