"""
SDF Graph Builder - PyTorch-based SDF evaluation

This module re-exports all components from the split files for backward compatibility.
"""

# Re-export nodes
from .math_jit_nodes import (
    SphereNode,
    BoxNode,
    CylinderNode,
    TorusNode,
    ConeNode,
    CapsuleNode,
    PlaneNode,
    UnionNode,
)

# Re-export modifiers
from .math_jit_modifiers import (
    TransformNode,
    TwistModifier,
    BendModifier,
    TaperModifier,
    MirrorModifier,
    RoundModifier,
    build_modifier,
    AXIS_INDEX,
)

# Re-export builder
from .math_jit_builder import (
    SdfGraph,
    build_node,
    build_sdf_graph,
)

__all__ = [
    # Nodes
    "SphereNode",
    "BoxNode",
    "CylinderNode",
    "TorusNode",
    "ConeNode",
    "CapsuleNode",
    "PlaneNode",
    "UnionNode",
    # Modifiers
    "TransformNode",
    "TwistModifier",
    "BendModifier",
    "TaperModifier",
    "MirrorModifier",
    "RoundModifier",
    "build_modifier",
    "AXIS_INDEX",
    # Builder
    "SdfGraph",
    "build_node",
    "build_sdf_graph",
]
