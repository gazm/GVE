"""
Skeleton extraction for hybrid bones: AI names/bindings + math-derived rest poses.

Derives rest poses and hierarchy from node geometry (centers, AABBs).
Merges with optional AI skeleton (names, parent hints).
Extracts node-to-bone bindings for SKEL chunk.
"""

from __future__ import annotations

from typing import Any

from .math_jit_builder import collect_node_info

# Identity quaternion [x, y, z, w]
IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]


def _node_id_to_u32(node_id: str) -> int:
    """Stable FNV-1a 32-bit hash for node_id. Engine must use same scheme."""
    h = 2166136261
    for c in node_id.encode("utf-8"):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def derive_skeleton_from_nodes(dna: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Derive skeleton bones from node geometry.
    One bone per node: center = position, identity rotation.
    Parent = nearest node by proximity (or root for centroid).
    """
    infos = collect_node_info(dna)
    if not infos:
        return []

    # Build one bone per node
    bones: list[dict[str, Any]] = []
    for i, info in enumerate(infos):
        center = info["center"]
        name = f"bone_{i}"  # Procedural name; AI override supplies semantic names
        # Parent: root (0) for first; else nearest by center distance
        parent_idx = 0
        if i > 0:
            min_d2 = float("inf")
            best_p = 0
            for j in range(i):
                cj = infos[j]["center"]
                d2 = sum((center[k] - cj[k]) ** 2 for k in range(3))
                if d2 < min_d2:
                    min_d2 = d2
                    best_p = j
            parent_idx = best_p
        bones.append({
            "name": name,
            "parent_idx": parent_idx,
            "rest_pos": center,
            "rest_rot": IDENTITY_QUAT.copy(),
        })
    return bones


def merge_ai_skeleton(
    ai_bones: list[dict[str, Any]] | None,
    derived_bones: list[dict[str, Any]],
    node_infos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge AI skeleton names with derived rest poses.
    If ai_bones present: use AI names and parent hints; fill rest_pose from derived geometry.
    If no AI: use derived as-is.
    """
    if not derived_bones:
        return []
    if not ai_bones:
        return derived_bones

    name_to_idx: dict[str, int] = {}
    for i, b in enumerate(ai_bones):
        name_to_idx[b.get("name", f"bone_{i}")] = i

    # Resolve parent names to indices
    def parent_name_to_idx(name: str | None) -> int:
        if not name:
            return 0xFFFF  # No parent (root)
        return name_to_idx.get(name, 0xFFFF)

    merged: list[dict[str, Any]] = []
    for i, ai_b in enumerate(ai_bones):
        name = ai_b.get("name", f"bone_{i}")
        parent_name = ai_b.get("parent")
        parent_idx = parent_name_to_idx(parent_name)
        if parent_idx == 0xFFFF and i > 0:
            parent_idx = 0  # Fallback to root

        # Rest pose from derived: match by order or nearest node center
        rest_pos = [0.0, 0.0, 0.0]
        rest_rot = IDENTITY_QUAT.copy()
        if i < len(derived_bones):
            rest_pos = derived_bones[i]["rest_pos"]
            rest_rot = derived_bones[i]["rest_rot"]
        elif node_infos:
            # Use centroid of all nodes as fallback for extra AI bones
            n = len(node_infos)
            rest_pos = [
                sum(inf["center"][0] for inf in node_infos) / n,
                sum(inf["center"][1] for inf in node_infos) / n,
                sum(inf["center"][2] for inf in node_infos) / n,
            ]

        merged.append({
            "name": name,
            "parent_idx": parent_idx if parent_idx != 0xFFFF else 0xFFFF,
            "rest_pos": rest_pos,
            "rest_rot": rest_rot,
        })
    return merged


def extract_node_bindings(
    dna: dict[str, Any],
    bone_name_to_idx: dict[str, int],
) -> list[dict[str, Any]]:
    """
    Walk SDF tree; for each node with id:
    - bone_binding -> RigidBinding(node_id_hash, bone_idx)
    - bone_influences -> SkinnedBinding(node_id_hash, bones[], weights[])
    """
    bindings: list[dict[str, Any]] = []

    def traverse(node: dict[str, Any]) -> None:
        node_id = node.get("id")
        if not node_id:
            for c in node.get("children") or []:
                traverse(c)
            return
        binding = node.get("bone_binding")
        influences = node.get("bone_influences")
        if binding and binding in bone_name_to_idx:
            bindings.append({
                "kind": "rigid",
                "node_id": _node_id_to_u32(node_id),
                "bone_idx": bone_name_to_idx[binding],
            })
        elif influences:
            bones_arr: list[int] = []
            weights_arr: list[float] = []
            for inf in influences:
                bn = inf.get("bone") if isinstance(inf, dict) else getattr(inf, "bone", None)
                wt = inf.get("weight") if isinstance(inf, dict) else getattr(inf, "weight", 0.0)
                if bn and bn in bone_name_to_idx:
                    bones_arr.append(bone_name_to_idx[bn])
                    weights_arr.append(float(wt))
            if bones_arr and abs(sum(weights_arr) - 1.0) < 0.01:  # Normalize
                total = sum(weights_arr)
                if total > 0:
                    weights_arr = [w / total for w in weights_arr]
                bindings.append({
                    "kind": "skinned",
                    "node_id": _node_id_to_u32(node_id),
                    "bones": bones_arr,
                    "weights": weights_arr,
                })
        for c in node.get("children") or []:
            traverse(c)

    root = dna.get("root_node") or dna.get("nodes")
    if isinstance(root, dict):
        traverse(root)
    elif isinstance(root, list):
        for n in root:
            traverse(n)
    return bindings


def has_skeleton_or_bindings(dna: dict[str, Any]) -> bool:
    """True if dna has skeleton or any node with bone_binding/bone_influences."""
    bones = dna.get("skeleton") or {}
    if isinstance(bones, dict) and bones.get("bones"):
        return True

    def check(node: dict) -> bool:
        if node.get("bone_binding") or node.get("bone_influences"):
            return True
        for c in node.get("children") or []:
            if check(c):
                return True
        return False

    root = dna.get("root_node") or dna.get("nodes")
    if isinstance(root, dict):
        return check(root)
    if isinstance(root, list):
        return any(check(n) for n in root)
    return False


def build_skeleton_and_bindings(dna: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Full pipeline: derive + merge + extract.
    Returns (bones, bindings) ready for SKEL serialization.
    """
    node_infos = collect_node_info(dna)
    derived = derive_skeleton_from_nodes(dna)
    ai_bones = None
    skeleton = dna.get("skeleton", {})
    if isinstance(skeleton, dict) and skeleton.get("bones"):
        ai_bones = skeleton["bones"]
    bones = merge_ai_skeleton(ai_bones, derived, node_infos)
    name_to_idx = {b["name"]: i for i, b in enumerate(bones)}
    bindings = extract_node_bindings(dna, name_to_idx)
    return bones, bindings
