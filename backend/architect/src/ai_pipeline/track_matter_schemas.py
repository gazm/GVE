# backend/architect/src/ai_pipeline/track_matter_schemas.py
"""
Track A: Matter Pipeline - Pydantic Schemas

Output schemas for the Matter generation pipeline stages.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Output Schemas (Pydantic models for structured JSON output)
# =============================================================================

class PrimitiveParams(BaseModel):
    """
    Parameters for a geometry primitive.
    
    Each primitive uses a subset of these fields:
    - Sphere: radius
    - Box: size (half-extents [x, y, z])
    - Cylinder: radius, height, sides (0=smooth)
    - Capsule: radius, height
    - Torus: major_r, minor_r
    - Cone: radius, height, angle (optional), sides
    - Plane: normal, distance
    """
    # Common params
    size: list[float] | None = None       # Box half-extents [x, y, z]
    radius: float | None = None           # Sphere, Cylinder, Capsule, Cone
    height: float | None = None           # Cylinder, Capsule, Cone
    sides: int | None = None              # Cylinder, Cone (0=smooth)
    
    # Torus params
    major_r: float | None = None          # Torus major radius
    minor_r: float | None = None          # Torus minor radius
    
    # Cone params
    angle: float | None = None            # Cone angle (alternative to radius)
    
    # Plane params
    normal: list[float] | None = None     # Plane normal vector [x, y, z]
    distance: float | None = None         # Plane distance from origin


class Transform(BaseModel):
    """Transform for positioning a node in 3D space."""
    pos: list[float] = [0.0, 0.0, 0.0]              # Position [x, y, z]
    rot: list[float] = [0.0, 0.0, 0.0, 1.0]         # Rotation quaternion [x, y, z, w]
    scale: list[float] | None = None                 # Optional scale [x, y, z]


class Modifier(BaseModel):
    """
    Domain modifier that warps space before SDF evaluation.
    
    Available modifier types:
    - twist: Rotate points around axis proportional to position. {"type": "twist", "axis": "y", "rate": 1.0}
    - bend: Bend the shape around an axis. {"type": "bend", "axis": "x", "angle": 0.5}
    - taper: Scale cross-section along axis. {"type": "taper", "axis": "y", "scale_min": 0.5, "scale_max": 1.0}
    - mirror: Mirror across axis plane for symmetry. {"type": "mirror", "axis": "x"}
    - round: Bevel/round edges. {"type": "round", "radius": 0.02}
    """
    type: str = Field(..., description="Modifier type: twist, bend, taper, mirror, round")
    axis: str | None = Field(None, description="Axis: x, y, or z")
    rate: float | None = Field(None, description="Twist rate in radians per meter")
    angle: float | None = Field(None, description="Bend angle in radians")
    scale_min: float | None = Field(None, description="Taper scale at negative end")
    scale_max: float | None = Field(None, description="Taper scale at positive end")
    radius: float | None = Field(None, description="Rounding radius in meters")


class SDFNode(BaseModel):
    """
    A node in the SDF tree.
    
    Can be either:
    - primitive: Geometry shape (sphere, box, cylinder, etc.)
    - operation: CSG operation combining children (union, subtract, intersect)
    
    Modifiers warp the shape's space before SDF evaluation:
    - twist: Spiral effect (rate = radians per meter along axis)
    - bend: Curve the shape (angle in radians)
    - taper: Scale from thick to thin along axis
    - mirror: Symmetry across axis plane
    - round: Bevel/smooth edges
    """
    id: str = Field(..., description="Unique identifier for this node")
    type: str = Field(..., pattern="^(primitive|operation)$")
    
    # Primitive fields
    shape: str | None = Field(None, description="Primitive shape: box, sphere, cylinder, capsule, torus, cone, plane")
    params: PrimitiveParams | None = Field(None, description="Shape-specific parameters")
    
    # Operation fields
    op: str | None = Field(None, description="CSG operation: union, subtract, intersect, smooth_union")
    children: list["SDFNode"] | None = Field(None, description="Child nodes for operations")
    
    # Common fields
    transform: Transform | dict[str, Any] | None = Field(None, description="Position/rotation/scale")
    lod_cutoff: int = Field(0, description="LOD level at which this node disappears (0=always visible)")
    modifiers: list[dict[str, Any]] | None = Field(
        None, 
        description="Domain modifiers applied in order: twist, bend, taper, mirror, round"
    )


class SDFRootNode(BaseModel):
    """Root node of SDF tree - enforces structure to prevent invalid children."""
    type: Literal["operation"] = "operation"
    op: Literal["union"] = "union"
    children: list[SDFNode] = Field(
        ...,
        min_length=1,
        description="Array of SDF node objects. Each child MUST be a complete object with id, type, shape, params, etc. NOT strings."
    )


class BlacksmithOutput(BaseModel):
    """Output from Stage A1: The Blacksmith."""
    
    sdf_tree: SDFRootNode = Field(
        ...,
        description="Root node of SDF tree using Union operations only. Children must be objects, not strings."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Bounds, LOD hints, etc."
    )


class MachinistDeltaPatch(BaseModel):
    """A single modification from the Machinist."""
    op: str = Field(default="subtract")
    target_node_id: str
    subtract: dict[str, Any]
    lod_cutoff: int = 1


class MachinistOutput(BaseModel):
    """Output from Stage A2: The Machinist."""
    
    delta_patch: dict[str, list[MachinistDeltaPatch]] = Field(
        ...,
        description="Operations to add (cannot modify A1 output)"
    )


class MaterialConfig(BaseModel):
    """Material configuration for a node."""
    material_id: str
    color_mode: str = "rgb"
    texture_modifiers: dict[str, Any] | None = None
    base_color: str | list[float] | None = None


class ArtistOutput(BaseModel):
    """Output from Stage A3: The Artist."""
    
    material_config: dict[str, MaterialConfig] = Field(
        ...,
        description="Material assignments per node ID"
    )
