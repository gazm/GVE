# backend/architect/src/ai_pipeline/track_matter_schemas.py
"""
Track A: Matter Pipeline - Pydantic Schemas

Output schemas for the Matter generation pipeline stages.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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
    """Transform for positioning a node in 3D space.

    rot accepts either Euler degrees (3 elements) or quaternion (4 elements):
    - Euler: [x_deg, y_deg, z_deg] — compiler converts to quaternion (XYZ order).
    - Quaternion: [x, y, z, w] — used as-is.
    """
    pos: list[float] = [0.0, 0.0, 0.0]              # Position [x, y, z]
    rot: list[float] = [0.0, 0.0, 0.0, 1.0]         # Euler [x_deg,y_deg,z_deg] or quat [x,y,z,w]
    scale: list[float] | None = None                 # Optional scale [x, y, z]


class Modifier(BaseModel):
    """
    Domain modifier that warps space before SDF evaluation.

    Available modifier types:
    - twist: Rotate points around axis proportional to position. {"type": "twist", "axis": "y", "rate": 1.0}
    - bend: Bend the shape around an axis. {"type": "bend", "axis": "x", "angle": 0.5}
    - taper: Scale cross-section along axis. {"type": "taper", "axis": "y", "scale_min": 0.5, "scale_max": 1.0}
    - mirror: Mirror across axis plane for symmetry. {"type": "mirror", "axis": "x"}
    - round: Bevel/round edges (curved). {"type": "round", "radius": 0.02}
    - chamfer: Flat 45-degree bevel on edges. {"type": "chamfer", "width": 0.01}
    - voronoi: 3D cellular pattern. {"type": "voronoi", "cell_size": 0.2, "wall_thickness": 0.02, "mode": "subtract"}
    """
    type: str = Field(..., description="Modifier type: twist, bend, taper, mirror, round, chamfer, voronoi")
    axis: str | None = Field(None, description="Axis: x, y, or z")
    rate: float | None = Field(None, description="Twist rate in radians per meter")
    angle: float | None = Field(None, description="Bend angle in radians")
    scale_min: float | None = Field(None, description="Taper scale at negative end")
    scale_max: float | None = Field(None, description="Taper scale at positive end")
    radius: float | None = Field(None, description="Rounding radius in meters")
    width: float | None = Field(None, description="Chamfer bevel width in meters")
    # Voronoi params
    cell_size: float | None = Field(None, description="Voronoi cell size in meters")
    wall_thickness: float | None = Field(None, description="Voronoi wall thickness in meters")
    mode: str | None = Field(None, description="Voronoi mode: 'subtract' (holes) or 'intersect' (walls only)")


# =============================================================================
# Skeletal Animation Schemas (hybrid: AI names/bindings, math-derived poses)
# =============================================================================


class BoneInfluence(BaseModel):
    """Single bone influence for skinned joint blending."""
    bone: str = Field(..., description="Bone name")
    weight: float = Field(..., ge=0.0, le=1.0, description="Influence weight 0–1")


class BoneDefinition(BaseModel):
    """Bone definition with name and optional parent hint. Rest poses derived by compiler from geometry."""
    name: str = Field(..., description="Bone name for binding reference")
    parent: str | None = Field(None, description="Parent bone name; null = root")


class SkeletonData(BaseModel):
    """Skeleton structure—names and hierarchy. Rest poses supplied by compiler from node geometry."""
    bones: list[BoneDefinition] = Field(..., min_length=1, description="Bone definitions")


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
        description="Domain modifiers applied in order: twist, bend, taper, mirror, round, chamfer, voronoi.",
    )
    procedural_texture: dict[str, Any] | None = Field(
        None,
        description="Procedural texture pattern: {type, scale, intensity, color_variation, roughness_variation}",
    )
    # Skeletal binding (for animated_character, animated_weapon)
    bone_binding: str | None = Field(
        None,
        description="Bone name for rigid binding; node transforms with this bone",
    )
    animation_mode: str | None = Field(
        None,
        pattern="^(rigid|skinned)$",
        description="rigid = whole node moves with bone; skinned = blend via bone_influences",
    )
    bone_influences: list[BoneInfluence] | None = Field(
        None,
        description="For skinned joints: list of bone + weight for blending",
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


class Connection(BaseModel):
    """Part relationship for assembly coherence. LLM-friendly connection vocabulary."""
    type: str = Field(
        ...,
        description="SEATS_IN, MOUNTS_ON, ALIGNED_WITH, FASTENED_BY, ATTACHED_TO, REMOVABLE",
    )
    child_id: str | None = Field(None, description="Child part node ID (for SEATS_IN, MOUNTS_ON, ATTACHED_TO)")
    parent_id: str | None = Field(None, description="Parent part node ID")
    part_ids: list[str] | None = Field(None, description="Part IDs for FASTENED_BY (multiple parts)")
    part_id: str | None = Field(None, description="Single part ID for REMOVABLE")
    interface: str | None = Field(None, description="Interface hint: well, rails, etc.")


# =============================================================================
# Semantic Assembly Schemas (Blacksmith V3 — CAD mate constraints)
# =============================================================================

# Cardinal faces only — no compound/percentile faces
VALID_CARDINAL_FACES = {"top", "bottom", "front", "back", "left", "right"}

# Alignment options: center or cardinal direction for flush edge
VALID_CONSTRAINT_ALIGNS = {
    "center",
    "top", "bottom", "front", "back", "left", "right",
}


class PartDefinition(BaseModel):
    """A single part in the assembly — shape + size, NO position or rotation.

    The assembly resolver computes transforms from AssemblyDirectives.
    """
    id: str = Field(..., description="Unique part ID (e.g. 'barrel_001', 'grip_001')")
    shape: str = Field(
        ...,
        description="Primitive shape: box, sphere, cylinder, capsule, torus, cone, wedge, revolution, mandelbulb, menger, julia",
    )
    params: PrimitiveParams = Field(..., description="Shape-specific parameters (radius, height, size, etc.)")
    role: str = Field(
        ...,
        description="Semantic role: body, frame, barrel, grip, guard, blade, handle, pommel, leg, seat, band, lid, etc.",
    )
    lod_cutoff: int = Field(0, description="LOD level at which this part disappears (0=always visible)")
    modifiers: list[Modifier] | None = Field(
        None,
        description="Domain modifiers: twist, bend, taper, mirror, round, chamfer, voronoi",
    )
    # Skeletal binding (optional — assembly resolver also derives bindings from InterfaceBone)
    bone_binding: str | None = Field(
        None,
        description="Bone name for rigid binding (node transforms with this bone)",
    )
    animation_mode: str | None = Field(
        None,
        pattern="^(rigid|skinned)$",
        description="rigid = whole node moves with bone; skinned = blend via bone_influences",
    )
    bone_influences: list[BoneInfluence] | None = Field(
        None,
        description="For skinned joints: list of bone + weight for blending",
    )


class AssemblyConstraint(BaseModel):
    """CAD-style mate constraint: explicit parent_face/child_face contact with positive overlap.

    The resolver computes a mating rotation from the two face normals, applies
    tilt, then positions the child so the specified faces are in contact with
    the given overlap depth.

    First entry must have parent=null (root part placed at world origin).
    """
    part_id: str = Field(..., description="Part being placed (must match a PartDefinition.id)")
    parent: str | None = Field(
        None,
        description="Parent part ID. null = root part, placed at world origin.",
    )
    parent_face: str = Field(
        "top",
        description="Cardinal face on the PARENT where child attaches: top, bottom, front, back, left, right.",
    )
    child_face: str = Field(
        "bottom",
        description=(
            "Cardinal face on the CHILD that contacts the parent: top, bottom, front, back, left, right. "
            "For cylinder/capsule/cone: front(Z+) and back(Z-) are the circular END CAPS."
        ),
    )
    align: str = Field(
        "center",
        description=(
            "Alignment on the contact plane: center (default), or cardinal direction to flush "
            "child edge with parent edge (top, bottom, front, back, left, right)."
        ),
    )
    overlap: float = Field(
        0.0,
        ge=0.0,
        description="Overlap depth in meters. 0 = faces touch. Positive = child embeds into parent.",
    )
    tilt_axis: str | None = Field(
        None,
        description="Optional tilt axis: x, y, or z. For angled parts (e.g. grip tilted backward).",
    )
    tilt_degrees: float = Field(
        0.0,
        description="Tilt angle in degrees. Only used when tilt_axis is set.",
    )

    @model_validator(mode="after")
    def _validate_constraint(self) -> "AssemblyConstraint":
        """Validate face/align values are from the allowed cardinal vocabulary."""
        if self.parent is not None:
            if self.parent_face not in VALID_CARDINAL_FACES:
                raise ValueError(
                    f"Invalid parent_face '{self.parent_face}'. "
                    f"Must be one of: {', '.join(sorted(VALID_CARDINAL_FACES))}"
                )
            if self.child_face not in VALID_CARDINAL_FACES:
                raise ValueError(
                    f"Invalid child_face '{self.child_face}'. "
                    f"Must be one of: {', '.join(sorted(VALID_CARDINAL_FACES))}"
                )
        if self.align not in VALID_CONSTRAINT_ALIGNS:
            raise ValueError(
                f"Invalid align '{self.align}'. "
                f"Must be one of: {', '.join(sorted(VALID_CONSTRAINT_ALIGNS))}"
            )
        if self.tilt_axis is not None and self.tilt_axis not in ("x", "y", "z"):
            raise ValueError(f"tilt_axis must be 'x', 'y', or 'z', got '{self.tilt_axis}'")
        return self


# Legacy alias for backward compat
AssemblyDirective = AssemblyConstraint


class InterfaceBone(BaseModel):
    """A skeleton bone placed at the interface between connected parts.

    The assembly resolver computes the bone's world-space position from the
    anchor points of the referenced parts. Bones defined at interfaces enable
    natural joint placement without coordinate math.
    """
    bone: str = Field(..., description="Bone name (e.g. 'Slide', 'Trigger', 'Frame')")
    parent: str | None = Field(None, description="Parent bone name. None = root bone.")
    at_interface: list[str] = Field(
        ...,
        min_length=1,
        max_length=2,
        description=(
            "1-2 anchor references for bone placement. "
            "Format: 'part_id.face' (e.g. 'frame_001.top', 'slide_001.bottom'). "
            "With 2 refs, bone is placed at midpoint between them."
        ),
    )


class BlacksmithOutput(BaseModel):
    """Output from Stage A1: The Blacksmith (Semantic Assembly).

    The LLM outputs parts (shapes+sizes), assembly directives (how parts connect),
    and optional skeleton bones (at part interfaces). NO transforms or rotations.
    The assembly resolver converts this into a positioned SDFRootNode.
    """

    reasoning: str | None = Field(
        None,
        description="Reasoning/CoT explaining the structural analysis and part breakdown",
    )
    parts: list[PartDefinition] = Field(
        ...,
        min_length=1,
        description="List of parts to build. Each has shape, size, and role — NO position.",
    )
    assembly: list[AssemblyConstraint] = Field(
        ...,
        min_length=1,
        description=(
            "CAD-style mate constraints. "
            "First entry should have parent=null (root part at origin). "
            "Subsequent entries mate child_face to parent_face with overlap."
        ),
    )
    skeleton: list[InterfaceBone] | None = Field(
        None,
        description="Optional bones for animated assets. Each bone is placed at a part interface.",
    )
    connections: list[Connection] | None = Field(
        None,
        description="Part relationships (SEATS_IN, MOUNTS_ON, etc.) for Machinist context.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Bounds, LOD hints, etc.",
    )

    @model_validator(mode="after")
    def _validate_assembly_graph(self) -> "BlacksmithOutput":
        """Validate that assembly references are consistent with parts."""
        part_ids = {p.id for p in self.parts}
        # Check for duplicate part IDs
        if len(part_ids) != len(self.parts):
            seen: set[str] = set()
            for p in self.parts:
                if p.id in seen:
                    raise ValueError(f"Duplicate part ID: '{p.id}'")
                seen.add(p.id)
        # Validate assembly references
        root_count = 0
        assembled_ids: set[str] = set()
        for constraint in self.assembly:
            if constraint.part_id not in part_ids:
                raise ValueError(
                    f"Assembly references unknown part '{constraint.part_id}'. "
                    f"Valid IDs: {', '.join(sorted(part_ids))}"
                )
            if constraint.parent is None:
                root_count += 1
            elif constraint.parent not in part_ids:
                raise ValueError(
                    f"Assembly parent references unknown part '{constraint.parent}'. "
                    f"Valid IDs: {', '.join(sorted(part_ids))}"
                )
            assembled_ids.add(constraint.part_id)
        if root_count == 0:
            raise ValueError("Assembly must have at least one root part (parent=null)")
        if root_count > 1:
            raise ValueError(f"Assembly has {root_count} root parts; exactly 1 is required")
        # Warn about unassembled parts (all parts should appear in assembly)
        missing = part_ids - assembled_ids
        if missing:
            raise ValueError(
                f"Parts not in assembly graph: {', '.join(sorted(missing))}. "
                "Every part must have an assembly constraint."
            )
        # Validate skeleton interface references
        if self.skeleton:
            for bone in self.skeleton:
                for ref in bone.at_interface:
                    if "." not in ref:
                        raise ValueError(
                            f"Bone '{bone.bone}' interface ref '{ref}' must be 'part_id.face' format"
                        )
                    ref_part, ref_face = ref.rsplit(".", 1)
                    if ref_part not in part_ids:
                        raise ValueError(
                            f"Bone '{bone.bone}' references unknown part '{ref_part}'"
                        )
        return self


# =============================================================================
# Legacy Blacksmith Output (kept for migration — will be removed after testing)
# =============================================================================



# NOTE: Removed ~140 lines of dead connection positioning helpers:
# _estimate_local_aabb, _transform_aabb, _aabb_volume, _aabb_intersection_volume,
# _aabb_center, _point_inside_aabb, _auto_suggest_connections
# (assembly_resolver handles all part placement now)


def _validate_euler_rotations(root: "SDFRootNode") -> None:
    def traverse(node: "SDFNode") -> None:
        t = node.transform
        if t and isinstance(t, (Transform, dict)):
            transform = t if isinstance(t, dict) else t.model_dump()
            rot = transform.get("rot")
            if isinstance(rot, list) and len(rot) != 3:
                raise ValueError("Blacksmith rot must be Euler degrees [x,y,z] (no quaternions)")
        for child in node.children or []:
            traverse(child)

    for child in root.children:
        traverse(child)


class BlacksmithOutputLegacy(BaseModel):
    """Legacy output from Stage A1 (absolute positions + Euler rotations). Kept for migration."""
    
    reasoning: str | None = Field(
        None,
        description="Reasoning/CoT string explaining the structural analysis"
    )
    sdf_tree: SDFRootNode = Field(
        ...,
        description="Root node of SDF tree using Union operations only. Children must be objects, not strings."
    )
    skeleton: SkeletonData | None = Field(
        None,
        description="Optional skeleton for animated_character or animated_weapon; rest poses derived by compiler",
    )
    connections: list[Connection] | None = Field(
        None,
        description="Part relationships (SEATS_IN, MOUNTS_ON, etc.) for Machinist context. Optional metadata.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Bounds, LOD hints, etc."
    )

    @model_validator(mode="after")
    def _validate_rotations(self) -> "BlacksmithOutputLegacy":
        if self.sdf_tree:
            _validate_euler_rotations(self.sdf_tree)
        return self


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


class AddPrimitive(BaseModel):
    """
    The geometry to add (union) onto a target node.
    
    Same structure as SubtractPrimitive. Used for hardware (bolts, rivets).
    """
    type: Literal["primitive"] = Field(
        default="primitive",
        description="Always 'primitive' for add operations"
    )
    shape: str = Field(
        ...,
        description="Shape to add: cylinder, box, sphere, torus, capsule, etc."
    )
    params: PrimitiveParams = Field(
        ...,
        description="Shape-specific parameters (radius, height, size, etc.)"
    )
    transform: Transform | None = Field(
        None,
        description="Optional position/rotation for the add geometry in world space"
    )


class MachinistDeltaPatch(BaseModel):
    """
    A single modification from the Machinist.
    
    Supports subtract/intersect (carving/masking) or add (hardware).
    Use subtract for bores/slots, intersect for trim/mask ops,
    add for bolts/rivets. Exactly one of subtract or add required based on op.
    """
    op: str = Field(
        default="subtract",
        description="'subtract' | 'smooth_subtract' | 'intersect' | 'smooth_intersect' | 'add'",
    )
    target_node_id: str = Field(
        ...,
        description="ID of the node from Stage A1 to modify"
    )
    subtract: SubtractPrimitive | None = Field(
        None,
        description="Geometry to subtract/intersect - required when op is subtract/smooth_subtract/intersect/smooth_intersect",
    )
    add: AddPrimitive | None = Field(
        None,
        description="Geometry to add (union) - required when op is add",
    )
    k: float | None = Field(
        None,
        description="Smoothing factor for smooth_subtract/smooth_intersect (0.05-0.5)",
    )
    lod_cutoff: int = Field(
        default=1,
        description="LOD level at which this detail disappears (1 = mid-detail)"
    )

    # Ops that require the 'subtract' geometry field
    _SUBTRACT_OPS = frozenset({"subtract", "smooth_subtract", "intersect", "smooth_intersect"})

    @model_validator(mode="after")
    def validate_subtract_or_add(self) -> "MachinistDeltaPatch":
        op = (self.op or "subtract").lower()
        if op in self._SUBTRACT_OPS:
            if self.subtract is None:
                raise ValueError(f"subtract field required when op is {op}")
        elif op == "add":
            if self.add is None:
                raise ValueError("add field required when op is add")
        return self


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
    consumed by the compiler as a compile-time attribute modifier pass.
    Optional ``finish_id`` applies named visual overrides (e.g. black_oxide)
    on top of the material; explicit base_color/roughness/metallic override finish.
    """
    material_id: str
    finish_id: str | None = Field(
        None,
        description="Named finish override (e.g. black_oxide). Applied on top of material; explicit base_color/roughness/metallic override finish.",
    )
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
        description="Per-node weathering: edge_wear, cavity_grime, rust_amount",
    )

    @field_validator('procedural_texture', mode='before')
    @classmethod
    def coerce_procedural_texture_to_dict(cls, v: Any) -> dict[str, Any] | None:
        """Parse JSON string to dict when LLM returns nested object as string."""
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    @field_validator('texture_modifiers', mode='before')
    @classmethod
    def coerce_texture_modifiers_to_dict(cls, v: Any) -> dict[str, Any] | None:
        """
        Coerce to dict when LLM outputs JSON string or flat list.
        LLMs sometimes output texture_modifiers as a JSON string or flat list
        instead of dict; this validator normalizes to the expected format.
        """
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return None
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
