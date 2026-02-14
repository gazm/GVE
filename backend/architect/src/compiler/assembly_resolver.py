# backend/architect/src/compiler/assembly_resolver.py
"""
Assembly Resolver — CAD-style mate constraints to positioned SDF trees.

Takes the Blacksmith's PartDefinition + AssemblyConstraint output (no coordinates)
and deterministically computes world-space transforms for every part using:
  1. Mating rotation: auto-computed from parent_face + child_face normals
  2. Overlap positioning: always-positive, child embeds into parent
  3. Cardinal alignment: flush edges or center on the contact plane
  4. Hierarchical rotation inheritance

Also places skeleton bones at part interfaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Face normals (cardinal only — no compound faces)
# =============================================================================

_FACE_NORMALS: dict[str, list[float]] = {
    "top":    [0.0, 1.0, 0.0],
    "bottom": [0.0, -1.0, 0.0],
    "front":  [0.0, 0.0, 1.0],
    "back":   [0.0, 0.0, -1.0],
    "right":  [1.0, 0.0, 0.0],
    "left":   [-1.0, 0.0, 0.0],
}


# =============================================================================
# AABB computation (local space, before rotation)
# =============================================================================

def _local_aabb(shape: str, params: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Compute local-space AABB for a primitive from its params.

    Returns (bmin, bmax) centered at origin. Cylinder/capsule/cone height along Z.
    """
    shape = shape.lower()
    p = params or {}

    if shape == "box" or shape == "wedge":
        size = p.get("size", [0.1, 0.1, 0.1])
        if isinstance(size, (int, float)):
            size = [size, size, size]
        h = [float(x) for x in size[:3]]
        return ([-h[0], -h[1], -h[2]], [h[0], h[1], h[2]])
    if shape == "sphere":
        r = float(p.get("radius", 0.1))
        return ([-r, -r, -r], [r, r, r])
    if shape in ("cylinder", "capsule", "cone"):
        r = float(p.get("radius", 0.1))
        h = float(p.get("height", 0.2))
        half = h / 2.0
        # Height along Z in engine
        return ([-r, -r, -half], [r, r, half])
    if shape == "torus":
        major = float(p.get("major_r", 0.2))
        minor = float(p.get("minor_r", 0.05))
        ext = major + minor
        return ([-ext, -ext, -minor], [ext, ext, minor])
    if shape == "plane":
        return ([-10.0, -10.0, -0.01], [10.0, 10.0, 0.01])
    # Fallback for fractals etc
    s = float(p.get("scale", 0.1))
    return ([-s, -s, -s], [s, s, s])


def _rotated_aabb(
    bmin: list[float], bmax: list[float], euler_deg: list[float]
) -> tuple[list[float], list[float]]:
    """Rotate local AABB corners by Euler XYZ (degrees) and recompute axis-aligned bounds."""
    if all(abs(e) < 1e-6 for e in euler_deg):
        return (list(bmin), list(bmax))

    mat = _euler_to_matrix(euler_deg)

    corners = [
        [bmin[0], bmin[1], bmin[2]], [bmax[0], bmin[1], bmin[2]],
        [bmin[0], bmax[1], bmin[2]], [bmax[0], bmax[1], bmin[2]],
        [bmin[0], bmin[1], bmax[2]], [bmax[0], bmin[1], bmax[2]],
        [bmin[0], bmax[1], bmax[2]], [bmax[0], bmax[1], bmax[2]],
    ]
    transformed = []
    for c in corners:
        transformed.append([
            mat[0][0] * c[0] + mat[0][1] * c[1] + mat[0][2] * c[2],
            mat[1][0] * c[0] + mat[1][1] * c[1] + mat[1][2] * c[2],
            mat[2][0] * c[0] + mat[2][1] * c[1] + mat[2][2] * c[2],
        ])
    out_min = [min(t[i] for t in transformed) for i in range(3)]
    out_max = [max(t[i] for t in transformed) for i in range(3)]
    return (out_min, out_max)


# =============================================================================
# Cardinal face anchor (center of a cardinal face on an AABB)
# =============================================================================

def _face_center(
    aabb_min: list[float], aabb_max: list[float], face: str
) -> list[float]:
    """Compute the center point of a cardinal face on an AABB."""
    cx = (aabb_min[0] + aabb_max[0]) / 2.0
    cy = (aabb_min[1] + aabb_max[1]) / 2.0
    cz = (aabb_min[2] + aabb_max[2]) / 2.0

    face = face.lower().strip()
    if face == "top":
        return [cx, aabb_max[1], cz]
    if face == "bottom":
        return [cx, aabb_min[1], cz]
    if face == "front":
        return [cx, cy, aabb_max[2]]
    if face == "back":
        return [cx, cy, aabb_min[2]]
    if face == "right":
        return [aabb_max[0], cy, cz]
    if face == "left":
        return [aabb_min[0], cy, cz]
    # Fallback: center
    return [cx, cy, cz]


# =============================================================================
# Rotation helpers (Euler <-> Matrix, composition, point rotation)
# =============================================================================

Mat3 = list[list[float]]


def _euler_to_matrix(euler_deg: list[float]) -> Mat3:
    """Convert Euler XYZ (degrees) to a 3x3 rotation matrix."""
    rx, ry, rz = [math.radians(float(e)) for e in euler_deg[:3]]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return [
        [cy * cz, -cy * sz, sy],
        [cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy],
        [sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy],
    ]


def _matrix_multiply(a: Mat3, b: Mat3) -> Mat3:
    """Multiply two 3x3 matrices: result = A * B."""
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _matrix_to_euler(m: Mat3) -> list[float]:
    """Extract Euler XYZ (degrees) from a 3x3 rotation matrix.

    Handles gimbal lock at +/-90 deg Y rotation.
    """
    sy = m[0][2]
    sy = max(-1.0, min(1.0, sy))

    if abs(sy) < 0.99999:
        ry = math.asin(sy)
        rx = math.atan2(-m[1][2], m[2][2])
        rz = math.atan2(-m[0][1], m[0][0])
    else:
        ry = math.copysign(math.pi / 2.0, sy)
        rx = math.atan2(m[2][1], m[1][1])
        rz = 0.0

    return [math.degrees(rx), math.degrees(ry), math.degrees(rz)]


def _compose_rotations(parent_euler: list[float], child_euler: list[float]) -> list[float]:
    """Compose two Euler XYZ rotations: world = parent * child."""
    if all(abs(e) < 1e-6 for e in parent_euler):
        return list(child_euler)
    if all(abs(e) < 1e-6 for e in child_euler):
        return list(parent_euler)
    parent_mat = _euler_to_matrix(parent_euler)
    child_mat = _euler_to_matrix(child_euler)
    world_mat = _matrix_multiply(parent_mat, child_mat)
    return _matrix_to_euler(world_mat)


def _rotate_point(point: list[float], euler_deg: list[float]) -> list[float]:
    """Rotate a single point by Euler XYZ (degrees)."""
    if all(abs(e) < 1e-6 for e in euler_deg):
        return list(point)
    mat = _euler_to_matrix(euler_deg)
    x, y, z = point
    return [
        mat[0][0] * x + mat[0][1] * y + mat[0][2] * z,
        mat[1][0] * x + mat[1][1] * y + mat[1][2] * z,
        mat[2][0] * x + mat[2][1] * y + mat[2][2] * z,
    ]


# =============================================================================
# Mating rotation: auto-compute rotation from parent_face + child_face normals
# =============================================================================

def _mating_rotation(parent_face: str, child_face: str) -> list[float]:
    """Compute the Euler rotation that mates child_face to parent_face.

    The mating rotation R maps the child's face outward normal to the
    negative of the parent's face outward normal, so the two faces
    oppose each other (child face points INTO parent).

    For axis-aligned cardinal faces this always produces clean 0/90/180
    degree rotations with no numerical drift.

    Args:
        parent_face: Cardinal face on parent (top/bottom/front/back/left/right).
        child_face: Cardinal face on child that contacts parent.

    Returns:
        Euler XYZ degrees [rx, ry, rz] for the mating rotation.
    """
    pn = _FACE_NORMALS.get(parent_face, [0.0, 1.0, 0.0])
    cn = _FACE_NORMALS.get(child_face, [0.0, -1.0, 0.0])

    # Target: child's face normal should point toward parent = -parent_normal
    # So we need R such that R * cn = -pn
    target = [-pn[0], -pn[1], -pn[2]]

    # If cn already equals target, identity rotation
    if _vec_close(cn, target):
        return [0.0, 0.0, 0.0]

    # If cn is opposite to target (same as pn), 180-degree rotation
    if _vec_close(cn, pn):
        # Pick rotation axis perpendicular to cn
        # For axis-aligned normals, pick the simplest perpendicular axis
        ax, ay, az = abs(cn[0]), abs(cn[1]), abs(cn[2])
        if az > 0.5:
            # cn is along Z, rotate 180 around Y
            return [0.0, 180.0, 0.0]
        if ay > 0.5:
            # cn is along Y, rotate 180 around Z
            return [0.0, 0.0, 180.0]
        # cn is along X, rotate 180 around Y
        return [0.0, 180.0, 0.0]

    # General case: axis-angle from cn to target
    # Cross product = rotation axis, dot product = cos(angle)
    cross = [
        cn[1] * target[2] - cn[2] * target[1],
        cn[2] * target[0] - cn[0] * target[2],
        cn[0] * target[1] - cn[1] * target[0],
    ]
    dot = cn[0] * target[0] + cn[1] * target[1] + cn[2] * target[2]
    angle = math.acos(max(-1.0, min(1.0, dot)))

    # Normalize cross product
    mag = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
    if mag < 1e-8:
        return [0.0, 0.0, 0.0]
    axis = [cross[0] / mag, cross[1] / mag, cross[2] / mag]

    # Axis-angle to rotation matrix, then to Euler
    mat = _axis_angle_to_matrix(axis, angle)
    euler = _matrix_to_euler(mat)

    # Snap to clean angles (0, 90, 180, -90) to avoid float drift
    return [_snap_angle(e) for e in euler]


def _vec_close(a: list[float], b: list[float], tol: float = 1e-6) -> bool:
    """Check if two 3D vectors are approximately equal."""
    return all(abs(a[i] - b[i]) < tol for i in range(3))


def _axis_angle_to_matrix(axis: list[float], angle: float) -> Mat3:
    """Convert axis-angle to 3x3 rotation matrix (Rodrigues' formula)."""
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return [
        [t * x * x + c,     t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c,     t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def _snap_angle(deg: float, tol: float = 0.5) -> float:
    """Snap an angle to the nearest clean value (0, 90, -90, 180, -180)."""
    for target in [0.0, 90.0, -90.0, 180.0, -180.0, 270.0, -270.0]:
        if abs(deg - target) < tol:
            return target
    return round(deg, 4)


def _apply_tilt(
    base_euler: list[float], tilt_axis: str | None, tilt_degrees: float
) -> list[float]:
    """Apply tilt rotation on top of base mating rotation via matrix composition.

    Uses proper matrix multiplication for correctness at all angles.
    """
    if not tilt_axis or abs(tilt_degrees) < 1e-6:
        return list(base_euler)

    axis_map = {"x": 0, "y": 1, "z": 2}
    idx = axis_map.get(tilt_axis.lower())
    if idx is None:
        return list(base_euler)

    tilt_euler = [0.0, 0.0, 0.0]
    tilt_euler[idx] = tilt_degrees
    return _compose_rotations(base_euler, tilt_euler)


# =============================================================================
# Resolved part data
# =============================================================================

@dataclass
class ResolvedPart:
    """A part with computed world-space transform."""
    id: str
    shape: str
    params: dict[str, Any]
    pos: list[float]        # World-space position [x, y, z]
    rot: list[float]        # Euler XYZ degrees for SDF tree output
    world_rot: list[float]  # Accumulated world rotation (for child inheritance)
    lod_cutoff: int = 0
    modifiers: list[dict[str, Any]] = field(default_factory=list)
    bone_binding: str | None = None
    animation_mode: str | None = None
    bone_influences: list[dict[str, Any]] | None = None
    local_aabb_min: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    local_aabb_max: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    aabb_min: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    aabb_max: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class ResolvedBone:
    """A skeleton bone with computed world-space position."""
    name: str
    parent: str | None
    position: list[float]


@dataclass
class ResolvedAssembly:
    """Full resolved assembly — positioned parts, skeleton, connections."""
    parts: list[ResolvedPart]
    bones: list[ResolvedBone]
    connections: list[dict[str, Any]]
    metadata: dict[str, Any]


# =============================================================================
# Cardinal alignment
# =============================================================================

def _apply_cardinal_alignment(
    pos: list[float],
    align: str,
    parent_face: str,
    parent_world_min: list[float],
    parent_world_max: list[float],
    child_rot_min: list[float],
    child_rot_max: list[float],
) -> list[float]:
    """Align child on the contact plane using cardinal direction.

    For a given contact face, alignment operates on the perpendicular axes:
      - parent_face top/bottom: free axes X(0) and Z(2)
      - parent_face front/back: free axes X(0) and Y(1)
      - parent_face left/right: free axes Y(1) and Z(2)
    """
    if align == "center":
        return pos

    result = list(pos)

    # Map alignment direction to axis index and which edge (min or max)
    align_map: dict[str, tuple[int, str]] = {
        "top":    (1, "max"),
        "bottom": (1, "min"),
        "front":  (2, "max"),
        "back":   (2, "min"),
        "right":  (0, "max"),
        "left":   (0, "min"),
    }

    if align not in align_map:
        return pos

    axis, edge = align_map[align]

    child_half = (child_rot_max[axis] - child_rot_min[axis]) / 2.0
    if edge == "max":
        # Flush child's max edge with parent's max edge on this axis
        result[axis] = parent_world_max[axis] - child_half
    else:
        # Flush child's min edge with parent's min edge on this axis
        result[axis] = parent_world_min[axis] + child_half

    return result


# =============================================================================
# Main resolver
# =============================================================================

def resolve_assembly(blacksmith_output: dict[str, Any]) -> ResolvedAssembly:
    """Resolve CAD-style assembly constraints into positioned parts.

    Args:
        blacksmith_output: BlacksmithOutput.model_dump() dict with
            parts, assembly, skeleton, connections, metadata.

    Returns:
        ResolvedAssembly with world-space transforms for every part.
    """
    parts_list: list[dict[str, Any]] = blacksmith_output.get("parts", [])
    assembly_list: list[dict[str, Any]] = blacksmith_output.get("assembly", [])
    skeleton_list: list[dict[str, Any]] = blacksmith_output.get("skeleton") or []
    connections: list[dict[str, Any]] = blacksmith_output.get("connections") or []
    metadata: dict[str, Any] = blacksmith_output.get("metadata") or {}

    # Build parts lookup
    parts_by_id: dict[str, dict[str, Any]] = {}
    for p in parts_list:
        parts_by_id[p["id"]] = p

    # Resolve each part's transform (hierarchical — children inherit parent rotation)
    resolved: dict[str, ResolvedPart] = {}

    for constraint in assembly_list:
        part_id: str = constraint["part_id"]
        parent_id: str | None = constraint.get("parent") or constraint.get("attach_to")
        parent_face: str = constraint.get("parent_face", "top")
        child_face: str = constraint.get("child_face", "bottom")
        align: str = constraint.get("align", "center")
        overlap: float = float(constraint.get("overlap", 0.0))
        tilt_axis: str | None = constraint.get("tilt_axis")
        tilt_degrees: float = float(constraint.get("tilt_degrees", 0.0))

        part_def = parts_by_id.get(part_id)
        if not part_def:
            _safe_print(f"  [resolver] Unknown part '{part_id}', skipping")
            continue

        shape = part_def.get("shape", "box")
        params = part_def.get("params") or {}

        # 1) Compute local AABB (before any rotation)
        local_min, local_max = _local_aabb(shape, params)

        if parent_id is None:
            # ── Root part: placed at origin ──
            pos = [0.0, 0.0, 0.0]
            world_rot = [0.0, 0.0, 0.0]
            rot_min, rot_max = list(local_min), list(local_max)

            _safe_print(
                f"    {part_id}: ROOT at origin ({shape})"
            )
        else:
            parent = resolved.get(parent_id)
            if not parent:
                _safe_print(
                    f"  [resolver] Parent '{parent_id}' not yet resolved "
                    f"for '{part_id}', placing at origin"
                )
                pos = [0.0, 0.0, 0.0]
                world_rot = [0.0, 0.0, 0.0]
                rot_min, rot_max = list(local_min), list(local_max)
            else:
                # ── CAD mate constraint resolution ──

                # 2) Compute mating rotation from face normals
                mating_rot = _mating_rotation(parent_face, child_face)

                # 3) Apply tilt on top of mating rotation
                local_rot = _apply_tilt(mating_rot, tilt_axis, tilt_degrees)

                # 4) Compose with parent world rotation
                world_rot = _compose_rotations(parent.world_rot, local_rot)

                # 5) Parent anchor: center of parent_face on parent's world AABB
                parent_anchor_local = _face_center(
                    parent.local_aabb_min, parent.local_aabb_max, parent_face
                )
                parent_anchor_world = _rotate_point(parent_anchor_local, parent.world_rot)
                parent_anchor = [
                    parent_anchor_world[i] + parent.pos[i] for i in range(3)
                ]

                # 6) Child anchor: center of child_face on child's rotated AABB
                child_rot_min, child_rot_max = _rotated_aabb(local_min, local_max, world_rot)
                child_anchor_local = _face_center(child_rot_min, child_rot_max, child_face)

                # 7) Position: align anchors
                pos = [
                    parent_anchor[i] - child_anchor_local[i] for i in range(3)
                ]

                # 8) Apply overlap: push child INTO parent along -parent_face normal
                if overlap > 1e-7:
                    pn = _FACE_NORMALS.get(parent_face, [0.0, 1.0, 0.0])
                    # Rotate parent normal by parent world rotation
                    pn_world = _rotate_point(pn, parent.world_rot)
                    # Move child inward (opposite to parent outward normal)
                    pos = [pos[i] - pn_world[i] * overlap for i in range(3)]

                # 9) Apply cardinal alignment
                parent_rot_min, parent_rot_max = _rotated_aabb(
                    parent.local_aabb_min, parent.local_aabb_max, parent.world_rot
                )
                parent_world_min = [parent_rot_min[i] + parent.pos[i] for i in range(3)]
                parent_world_max = [parent_rot_max[i] + parent.pos[i] for i in range(3)]

                rot_min, rot_max = child_rot_min, child_rot_max

                pos = _apply_cardinal_alignment(
                    pos, align, parent_face,
                    parent_world_min, parent_world_max,
                    rot_min, rot_max,
                )

                # Debug log
                _safe_print(
                    f"    {part_id} -> {parent_id}\n"
                    f"      mate: parent_face={parent_face} child_face={child_face}\n"
                    f"      mating_rot={[round(v, 1) for v in mating_rot]} "
                    f"tilt={tilt_degrees} on {tilt_axis or '-'}\n"
                    f"      world_rot={[round(v, 1) for v in world_rot]} "
                    f"overlap={overlap:.4f} align={align}\n"
                    f"      pos={[round(v, 4) for v in pos]}"
                )

        # Compute world AABB
        world_min = [rot_min[i] + pos[i] for i in range(3)]
        world_max = [rot_max[i] + pos[i] for i in range(3)]

        rp = ResolvedPart(
            id=part_id,
            shape=shape,
            params=params,
            pos=pos,
            rot=world_rot,
            world_rot=list(world_rot),
            lod_cutoff=part_def.get("lod_cutoff", 0),
            modifiers=[m if isinstance(m, dict) else m.model_dump() if hasattr(m, "model_dump") else {}
                       for m in (part_def.get("modifiers") or [])],
            bone_binding=part_def.get("bone_binding"),
            animation_mode=part_def.get("animation_mode"),
            bone_influences=part_def.get("bone_influences"),
            local_aabb_min=list(local_min),
            local_aabb_max=list(local_max),
            aabb_min=world_min,
            aabb_max=world_max,
        )
        resolved[part_id] = rp

    # ── Resolve skeleton bones at interfaces ──
    bones: list[ResolvedBone] = []
    for bone_def in skeleton_list:
        bone_name = bone_def.get("bone", "")
        parent_bone = bone_def.get("parent")
        at_interface: list[str] = bone_def.get("at_interface", [])

        positions: list[list[float]] = []
        for ref in at_interface:
            if "." not in ref:
                continue
            ref_part_id, ref_face = ref.rsplit(".", 1)
            rp = resolved.get(ref_part_id)
            if rp:
                anchor = _face_center(rp.aabb_min, rp.aabb_max, ref_face)
                positions.append(anchor)

        if positions:
            n = len(positions)
            bone_pos = [
                sum(p[i] for p in positions) / n
                for i in range(3)
            ]
        else:
            bone_pos = [0.0, 0.0, 0.0]

        bones.append(ResolvedBone(name=bone_name, parent=parent_bone, position=bone_pos))

    resolved_parts = list(resolved.values())

    # Compute global estimated_bounds from all parts
    if resolved_parts:
        global_min = [
            min(rp.aabb_min[i] for rp in resolved_parts) for i in range(3)
        ]
        global_max = [
            max(rp.aabb_max[i] for rp in resolved_parts) for i in range(3)
        ]
        metadata["estimated_bounds"] = {
            "min": [round(v, 6) for v in global_min],
            "max": [round(v, 6) for v in global_max],
        }

    _log_assembly(resolved_parts, bones)

    return ResolvedAssembly(
        parts=resolved_parts,
        bones=bones,
        connections=connections,
        metadata=metadata,
    )


# =============================================================================
# Utilities
# =============================================================================

def _safe_print(msg: str) -> None:
    """Print with fallback for Windows cp1252 encoding."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


def _log_assembly(parts: list[ResolvedPart], bones: list[ResolvedBone]) -> None:
    """Print assembly resolution summary."""
    _safe_print(f"  [assembly_resolver] Resolved {len(parts)} parts:")
    for rp in parts:
        pos_str = f"[{rp.pos[0]:.4f}, {rp.pos[1]:.4f}, {rp.pos[2]:.4f}]"
        rot_str = f"[{rp.rot[0]:.1f}, {rp.rot[1]:.1f}, {rp.rot[2]:.1f}]"
        _safe_print(f"    {rp.id}: pos={pos_str} rot={rot_str} ({rp.shape})")
    if bones:
        _safe_print(f"  [assembly_resolver] Placed {len(bones)} bones:")
        for b in bones:
            pos_str = f"[{b.position[0]:.4f}, {b.position[1]:.4f}, {b.position[2]:.4f}]"
            _safe_print(f"    {b.name}: pos={pos_str} (parent={b.parent})")


# =============================================================================
# Conversion to SDF tree (for Machinist/Artist compatibility)
# =============================================================================

def resolved_to_sdf_tree(assembly: ResolvedAssembly) -> dict[str, Any]:
    """Convert a ResolvedAssembly into the SDFRootNode dict format.

    Produces the same structure that the legacy Blacksmith output used,
    so Machinist/Artist stages work without modification.
    """
    children: list[dict[str, Any]] = []
    for rp in assembly.parts:
        node: dict[str, Any] = {
            "id": rp.id,
            "type": "primitive",
            "shape": rp.shape,
            "params": rp.params,
            "transform": {
                "pos": list(rp.pos),
                "rot": list(rp.rot),
            },
            "lod_cutoff": rp.lod_cutoff,
        }
        if rp.modifiers:
            node["modifiers"] = rp.modifiers
        if rp.bone_binding:
            node["bone_binding"] = rp.bone_binding
        if rp.animation_mode:
            node["animation_mode"] = rp.animation_mode
        if rp.bone_influences:
            node["bone_influences"] = rp.bone_influences
        children.append(node)

    sdf_tree = {
        "type": "operation",
        "op": "union",
        "children": children,
    }

    return sdf_tree


def resolved_to_skeleton(assembly: ResolvedAssembly) -> dict[str, Any] | None:
    """Convert resolved bones to SkeletonData dict format."""
    if not assembly.bones:
        return None
    return {
        "bones": [
            {"name": b.name, "parent": b.parent}
            for b in assembly.bones
        ],
    }


def resolved_to_dna(assembly: ResolvedAssembly) -> dict[str, Any]:
    """Convert a ResolvedAssembly into a full DNA dict.

    The DNA dict is the format consumed by the pipeline's _build_intermediate_dna,
    _merge_pipeline_outputs, and ultimately the compiler.
    """
    dna: dict[str, Any] = {
        "root_node": resolved_to_sdf_tree(assembly),
        "metadata": assembly.metadata,
    }

    skeleton = resolved_to_skeleton(assembly)
    if skeleton:
        dna["skeleton"] = skeleton

    if assembly.connections:
        dna["connections"] = assembly.connections

    return dna
