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
    WedgeNode,
    RevolutionNode,
    UnionNode,
    SubtractNode,
    IntersectNode,
    SmoothUnionNode,
    SmoothSubtractNode,
    SmoothIntersectNode,
    MandelbulbNode,
    MengerSpongeNode,
    JuliaSetNode,
)

# Re-export modifiers
from .math_jit_modifiers import (
    TransformNode,
    TwistModifier,
    BendModifier,
    TaperModifier,
    MirrorModifier,
    RoundModifier,
    VoronoiModifier,
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
    # Primitives
    "SphereNode",
    "BoxNode",
    "CylinderNode",
    "TorusNode",
    "ConeNode",
    "CapsuleNode",
    "PlaneNode",
    "WedgeNode",
    "RevolutionNode",
    # CSG Operations
    "UnionNode",
    "SubtractNode",
    "IntersectNode",
    "SmoothUnionNode",
    "SmoothSubtractNode",
    "SmoothIntersectNode",
    # Fractals
    "MandelbulbNode",
    "MengerSpongeNode",
    "JuliaSetNode",
    # Modifiers
    "TransformNode",
    "TwistModifier",
    "BendModifier",
    "TaperModifier",
    "MirrorModifier",
    "RoundModifier",
    "VoronoiModifier",
    "build_modifier",
    "AXIS_INDEX",
    # Builder
    "SdfGraph",
    "build_node",
    "build_sdf_graph",
]
