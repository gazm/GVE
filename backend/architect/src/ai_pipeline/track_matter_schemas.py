# backend/architect/src/ai_pipeline/track_matter_schemas.py
"""
Track A: Matter Pipeline - Pydantic Schemas

Output schemas for the Matter generation pipeline stages.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


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
    - Revolution: profile (child node), axis, offset
    - Mandelbulb: power, iterations, scale
    - Menger: iterations, scale
    - Julia: c (quaternion [x,y,z,w]), iterations, scale
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
    
    # Wedge params
    taper_axis: str | None = None         # Axis that tapers to zero: "x", "y", "z"
    taper_dir: str | None = None          # Axis along which taper progresses: "x", "y", "z"
    
    # Revolution params
    profile: dict[str, Any] | None = None  # Child primitive node for lathe profile
    axis: str | None = None                # Revolution axis: "x", "y", "z"
    offset: float | None = None            # Distance from revolution axis
    
    # Fractal params (Mandelbulb, Menger, Julia)
    power: float | None = None             # Mandelbulb power (default 8.0)
    iterations: int | None = None          # Fractal iteration count (max 12)
    scale: float | None = None             # Fractal scale factor
    c: list[float] | None = None           # Julia set quaternion seed [x, y, z, w]


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
    - voronoi: 3D cellular pattern. {"type": "voronoi", "cell_size": 0.2, "wall_thickness": 0.02, "mode": "subtract"}
    """
    type: str = Field(..., description="Modifier type: twist, bend, taper, mirror, round, voronoi")
    axis: str | None = Field(None, description="Axis: x, y, or z")
    rate: float | None = Field(None, description="Twist rate in radians per meter")
    angle: float | None = Field(None, description="Bend angle in radians")
    scale_min: float | None = Field(None, description="Taper scale at negative end")
    scale_max: float | None = Field(None, description="Taper scale at positive end")
    radius: float | None = Field(None, description="Rounding radius in meters")
    # Voronoi params
    cell_size: float | None = Field(None, description="Voronoi cell size in meters")
    wall_thickness: float | None = Field(None, description="Voronoi wall thickness in meters")
    mode: str | None = Field(None, description="Voronoi mode: 'subtract' (holes) or 'intersect' (walls only)")


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
    - voronoi: 3D cellular pattern (cell_size, wall_thickness, mode)
    """
    id: str = Field(..., description="Unique identifier for this node")
    type: str = Field(..., pattern="^(primitive|operation)$")
    
    # Primitive fields
    shape: str | None = Field(
        None,
        description="Primitive shape: box, sphere, cylinder, capsule, torus, cone, plane, wedge, revolution, mandelbulb, menger, julia",
    )
    params: PrimitiveParams | None = Field(None, description="Shape-specific parameters")
    
    # Operation fields
    op: str | None = Field(
        None,
        description="CSG operation: union, subtract, intersect, smooth_union, smooth_subtract, smooth_intersect",
    )
    children: list["SDFNode"] | None = Field(None, description="Child nodes for operations")
    k: float | None = Field(None, description="Smoothing factor for smooth operations (0.05-0.5)")
    
    # Common fields
    transform: Transform | dict[str, Any] | None = Field(None, description="Position/rotation/scale")
    lod_cutoff: int = Field(0, description="LOD level at which this node disappears (0=always visible)")
    modifiers: list[dict[str, Any]] | None = Field(
        None, 
        description="Domain modifiers applied in order: twist, bend, taper, mirror, round, voronoi"
    )
    procedural_texture: dict[str, Any] | None = Field(
        None,
        description="Procedural texture pattern: {type, scale, intensity, color_variation, roughness_variation}",
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
    
    reasoning: str | None = Field(
        None,
        description="Reasoning/CoT string explaining the structural analysis"
    )
    sdf_tree: SDFRootNode = Field(
        ...,
        description="Root node of SDF tree using Union operations only. Children must be objects, not strings."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Bounds, LOD hints, etc."
    )


class SubtractPrimitive(BaseModel):
    """
    The geometry to subtract from a target node.
    
    MUST be a dictionary with type, shape, and params - NOT a string!
    """
    type: Literal["primitive"] = Field(
        default="primitive",
        description="Always 'primitive' for subtract operations"
    )
    shape: str = Field(
        ...,
        description="Shape to subtract: cylinder, box, sphere, etc."
    )
    params: PrimitiveParams = Field(
        ...,
        description="Shape-specific parameters (radius, height, size, etc.)"
    )
    transform: Transform | None = Field(
        None,
        description="Optional position/rotation for the subtract geometry"
    )


class MachinistDeltaPatch(BaseModel):
    """
    A single modification from the Machinist.
    
    Each patch subtracts geometry from an existing node.
    Supports hard subtract or smooth_subtract (filleted edges).
    """
    op: str = Field(
        default="subtract",
        description="Operation type: 'subtract' (hard cut) or 'smooth_subtract' (filleted edges)",
    )
    target_node_id: str = Field(
        ...,
        description="ID of the node from Stage A1 to modify"
    )
    subtract: SubtractPrimitive = Field(
        ...,
        description="The geometry to subtract - MUST be an object with type, shape, params"
    )
    k: float | None = Field(
        None,
        description="Smoothing factor for smooth_subtract (0.05-0.5). Only used when op='smooth_subtract'",
    )
    lod_cutoff: int = Field(
        default=1,
        description="LOD level at which this detail disappears (1 = mid-detail)"
    )


class MachinistDeltaPatchList(BaseModel):
    """Wrapper for the list of delta patches with explicit key."""
    add_operations: list[MachinistDeltaPatch] = Field(
        default_factory=list,
        description="List of subtract operations to add to the geometry"
    )


class MachinistOutput(BaseModel):
    """Output from Stage A2: The Machinist."""
    
    delta_patch: MachinistDeltaPatchList = Field(
        ...,
        description="Delta patch containing add_operations array"
    )


class MaterialConfig(BaseModel):
    """Material configuration for a node.

    Note: ``texture_modifiers`` (edge_wear, cavity_grime, rust_amount) are
    captured here for intent but are **not yet consumed** by the compiler.
    They will be wired up as SDF post-processing in a future update.
    """
    material_id: str
    color_mode: str = Field(
        "oklab",
        description="Vestigial -- compiler always produces Oklab. Retained for future use.",
    )
    base_color: str | list[float] | None = None
    metallic: float | None = Field(None, ge=0.0, le=1.0, description="PBR metallic override 0.0-1.0")
    roughness: float | None = Field(None, ge=0.0, le=1.0, description="PBR roughness override 0.0-1.0")
    procedural_texture: dict[str, Any] | None = Field(
        None,
        description=(
            "Noise-based pattern overlay: {type, scale, intensity, "
            "color_variation, roughness_variation, metallic_variation}"
        ),
    )
    texture_modifiers: dict[str, Any] | None = Field(
        None,
        description="Aspirational weathering (not yet consumed by compiler): edge_wear, cavity_grime, rust_amount",
    )
    
    @field_validator('texture_modifiers', mode='before')
    @classmethod
    def convert_list_to_dict(cls, v):
        """
        Convert flat list ['key1', val1, 'key2', val2] to dict.
        
        LLMs sometimes output texture_modifiers as a flat list instead of dict.
        This validator auto-converts to the expected format.
        """
        if isinstance(v, list):
            result = {}
            for i in range(0, len(v) - 1, 2):
                if isinstance(v[i], str):
                    result[v[i]] = v[i + 1]
            return result if result else None
        return v


class ArtistOutput(BaseModel):
    """Output from Stage A3: The Artist."""
    
    material_config: dict[str, MaterialConfig] = Field(
        ...,
        description="Material assignments per node ID"
    )
