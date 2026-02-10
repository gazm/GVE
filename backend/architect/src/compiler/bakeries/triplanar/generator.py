"""
Triplanar Baker - Surface sample SDF + attributes, bake to triplanar.

When texture_mode is "procedural_triplanar", the compiler uses this path instead
of Gaussian splat training: sample the SDF surface, get material+procedural
attributes at each point via query_attributes, then bake into triplanar textures.
No optimization step; uses existing procedural libraries only.

Reference: docs/workflows/compiler-pipeline.md (texture mode: procedural_triplanar)
"""

from __future__ import annotations

import copy
import numpy as np
import torch
from typing import Tuple, List, Optional, Any

# Adjusted imports for new location src/compiler/bakeries/triplanar/
from ...splat_rasterizer import (
    TriplanarTextures,
    bake_triplanar_textures_oklab,
    bake_triplanar_from_voxel_oklab,
    pack_triplanar_textures,
    SplatBakeMode,
)
from ....librarian.finishes import get_finish

# Chunk size for batch query_attributes on surface voxels
_VOXEL_ATTRS_CHUNK = 50_000

# Default attrs when no material: mid-gray Oklab, 0 metallic, 0.5 roughness
_DEFAULT_ATTRS = np.array([0.627, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)


def _check_black_oxide_resolution() -> None:
    """Verify black_oxide finish resolves to dark Oklab. Logs warning if not."""
    from ...math_jit_builder import _resolve_material

    finish = get_finish("black_oxide")
    if not finish:
        return
    explicit_color = finish.get("base_color")  # e.g. "#0a0a0a"
    oklab, _, _ = _resolve_material("METAL_STEEL", explicit_color)
    L = oklab[0]
    if L > 0.15:
        print(
            f"      [voxel-triplanar] ⚠️ black_oxide base_color={explicit_color!r} -> L={L:.3f} (expected <0.15)",
            flush=True,
        )
    else:
        print(f"      [voxel-triplanar] ✓ black_oxide -> L={L:.3f} (dark OK)", flush=True)


def _resolve_attrs_from_dna_materials(
    dna: dict,
    node_bounds_list: List[Tuple[str, List[float], List[float]]],
) -> np.ndarray:
    """
    Resolve dna.materials to (n_nodes, 5) attrs [L, a, b, metallic, roughness].

    For each node_id in node_bounds_list, looks up dna["materials"].get(node_id);
    resolves base_color (from entry or finish_id), roughness, metallic; converts
    color to Oklab. Nodes not in materials get _DEFAULT_ATTRS.
    """
    from ...math_jit_builder import _parse_srgb_color, _resolve_material

    materials = dna.get("materials") or {}
    if not isinstance(materials, dict):
        materials = {}
    n_nodes = len(node_bounds_list)
    out = np.tile(_DEFAULT_ATTRS, (n_nodes, 1))

    # Verify black_oxide resolves to dark Oklab (material resolution check)
    _check_black_oxide_resolution()

    mat_keys = set(materials.keys())
    node_ids = [nb[0] for nb in node_bounds_list]
    missing = [nid for nid in node_ids if nid not in mat_keys]
    extra = [k for k in mat_keys if k not in node_ids]
    if missing or extra:
        print(
            f"      [voxel-triplanar] ⚠️ material key mismatch: "
            f"nodes_not_in_materials={missing[:5]}{'...' if len(missing) > 5 else ''} "
            f"materials_not_in_nodes={extra[:5]}{'...' if len(extra) > 5 else ''}",
            flush=True,
        )
    for j in range(n_nodes):
        node_id = node_bounds_list[j][0]
        cfg = materials.get(node_id)
        if not cfg:
            continue
        if hasattr(cfg, "model_dump"):
            cfg = cfg.model_dump(exclude_none=True)
        elif not isinstance(cfg, dict):
            cfg = vars(cfg) if hasattr(cfg, "__dict__") else {}
        finish = get_finish(cfg.get("finish_id")) if cfg.get("finish_id") else None
        explicit_color = cfg.get("base_color")
        if explicit_color is None and finish:
            explicit_color = finish.get("base_color")
        material_id = cfg.get("material_id", 0)
        oklab_list, metallic, roughness = _resolve_material(material_id, explicit_color)
        if cfg.get("roughness") is not None:
            roughness = float(cfg["roughness"])
        elif finish and finish.get("roughness") is not None:
            roughness = float(finish["roughness"])
        if cfg.get("metallic") is not None:
            metallic = float(cfg["metallic"])
        elif finish and finish.get("metallic") is not None:
            metallic = float(finish["metallic"])
        out[j, :3] = oklab_list
        out[j, 3] = metallic
        out[j, 4] = roughness
        # Diagnostic: log when black/dark material resolves to unexpectedly light (L > 0.4)
        L = oklab_list[0]
        finish_id = cfg.get("finish_id")
        is_expected_dark = (
            finish_id in ("black_oxide", "painted_black")
            or (explicit_color and str(explicit_color).strip().lower().startswith("#0"))
        )
        if is_expected_dark and L > 0.4:
            print(
                f"      [voxel-triplanar] ⚠️ node={node_id!r} expected dark but L={L:.3f} "
                f"(finish={finish_id!r} base_color={explicit_color!r})",
                flush=True,
            )
    return out


def _assign_voxels_to_nodes(
    positions_np: np.ndarray,
    node_bounds_list: List[Tuple[str, List[float], List[float]]],
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Assign each voxel (row index) to a node index. Returns (n_voxels,) int array:
    node_index in [0, len(node_bounds_list)), or -1 if no node contains the point.
    First node whose AABB contains the point wins; if none, -1.
    """
    n = positions_np.shape[0]
    node_idx = np.full(n, -1, dtype=np.int32)
    if not node_bounds_list:
        return node_idx
    bmins = np.array([bmin for _, bmin, _ in node_bounds_list], dtype=np.float32)
    bmaxs = np.array([bmax for _, _, bmax in node_bounds_list], dtype=np.float32)
    for j in range(len(node_bounds_list)):
        inside = np.all(positions_np >= bmins[j] - tol, axis=1) & np.all(positions_np <= bmaxs[j] + tol, axis=1)
        # Only assign where still unassigned
        unassigned = node_idx < 0
        node_idx[unassigned & inside] = j
    return node_idx


def _assign_outside_voxels_to_nearest_node(
    positions_np: np.ndarray,
    node_bounds_list: List[Tuple[str, List[float], List[float]]],
    node_idx: np.ndarray,
) -> None:
    """
    Assign voxels with node_idx == -1 to the nearest node by Euclidean distance
    to AABB (closest point on box boundary). Modifies node_idx in place.
    """
    outside_mask = node_idx < 0
    n_outside = int(np.sum(outside_mask))
    if n_outside == 0 or not node_bounds_list:
        return
    outside_indices = np.where(outside_mask)[0]
    outside_positions = positions_np[outside_indices]
    n_nodes = len(node_bounds_list)
    bmins = np.array([bmin for _, bmin, _ in node_bounds_list], dtype=np.float32)
    bmaxs = np.array([bmax for _, _, bmax in node_bounds_list], dtype=np.float32)
    dist = np.zeros((n_outside, n_nodes), dtype=np.float32)
    for j in range(n_nodes):
        closest = np.clip(outside_positions, bmins[j], bmaxs[j])
        dist[:, j] = np.linalg.norm(outside_positions - closest, axis=1)
    nearest = np.argmin(dist, axis=1)
    node_idx[outside_indices] = nearest
    print(
        f"      [voxel-triplanar] nearest-AABB: {n_outside} outside voxels -> assigned to nearest node",
        flush=True,
    )


def _apply_per_node_average(attrs_np: np.ndarray, node_idx: np.ndarray) -> None:
    """
    Replace each voxel's attributes with the mean (L,a,b, metallic, roughness) of its node.
    Modifies attrs_np in place. Voxels with node_idx < 0 are left unchanged.
    """
    n_nodes = node_idx.max() + 1 if node_idx.size and node_idx.max() >= 0 else 0
    if n_nodes <= 0:
        return
    for j in range(n_nodes):
        mask = node_idx == j
        if not np.any(mask):
            continue
        mean_attrs = np.mean(attrs_np[mask], axis=0)
        attrs_np[mask] = mean_attrs


def _bake_triplanar_from_voxels(
    bake_result: Any,
    sdf_graph: Any,
    resolution: int,
    surface_tolerance: float,
    dna: Optional[dict] = None,
    device: str = "cpu",
    mode: str = "gaussian",
) -> bytes:
    """
    Color surface voxels via query_attributes and bake into triplanar.
    Uses the existing dense SDF grid; no surface sampling or splat training.
    """
    dense_grid = bake_result.dense_grid  # (nx, ny, nz) float32
    nx, ny, nz = bake_result.dims
    b_min = np.array(bake_result.bounds_min, dtype=np.float32)
    b_max = np.array(bake_result.bounds_max, dtype=np.float32)
    extent = b_max - b_min

    # Surface voxels: |SDF| < tolerance
    mask = np.abs(dense_grid) < surface_tolerance
    if not np.any(mask):
        # Fallback: use a slightly larger tolerance
        surface_tolerance = min(surface_tolerance * 2.0, np.abs(dense_grid).max() * 0.1)
        mask = np.abs(dense_grid) < surface_tolerance
    ii, jj, kk = np.where(mask)
    n_surface = len(ii)
    print(f"      [voxel-triplanar] {n_surface} surface voxels (|SDF| < {surface_tolerance:.6f})", flush=True)

    # World positions: grid index -> world (same as vdb_converter linspace)
    def idx_to_world(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
        x = b_min[0] + (i.astype(np.float32) / max(nx - 1, 1)) * extent[0]
        y = b_min[1] + (j.astype(np.float32) / max(ny - 1, 1)) * extent[1]
        z = b_min[2] + (k.astype(np.float32) / max(nz - 1, 1)) * extent[2]
        return np.stack([x, y, z], axis=1)

    positions_np = idx_to_world(ii, jj, kk)  # (N, 3)

    attrs_fn = sdf_graph if hasattr(sdf_graph, "query_attributes") else None
    if attrs_fn is None:
        default = np.array([[0.627, 0.0, 0.0, 0.0, 0.5]], dtype=np.float32)
        attrs_np = np.tile(default, (n_surface, 1))
    else:
        # Batch query_attributes (CPU or CUDA). On CUDA, TextureModifierNode no-ops so edge_wear/cavity_grime/rust are skipped.
        use_cuda = device == "cuda" and torch.cuda.is_available()
        if use_cuda and hasattr(sdf_graph, "to"):
            attrs_fn = sdf_graph.to("cuda")
        else:
            attrs_fn = sdf_graph
        attrs_list = []
        for start in range(0, n_surface, _VOXEL_ATTRS_CHUNK):
            end = min(start + _VOXEL_ATTRS_CHUNK, n_surface)
            pts = torch.from_numpy(positions_np[start:end]).float()
            if use_cuda:
                pts = pts.cuda()
            with torch.no_grad():
                attrs = attrs_fn.query_attributes(pts)  # (chunk, 5)
            attrs_list.append(attrs.cpu().numpy())
        attrs_np = np.concatenate(attrs_list, axis=0)
        # Diagnostic: are we getting varied attributes per voxel?
        if n_surface >= 2:
            first_row = attrs_np[0]
            same = np.allclose(attrs_np, first_row, atol=1e-5)
            if same:
                print(f"      [voxel-triplanar] ⚠️ query_attributes constant (all voxels same L,a,b)", flush=True)
            else:
                uniq = np.unique(attrs_np[:, :3].round(decimals=4), axis=0)
                print(f"      [voxel-triplanar] query_attributes vary: ~{len(uniq)} distinct (L,a,b)", flush=True)

        # Per-node materials: when dna.materials exists, blend material base with query_attributes
        # so edge_wear, cavity_grime, rust and procedural_texture from the graph are baked.
        if dna is not None:
            from ...math_jit_builder import collect_node_bounds
            node_bounds_list = collect_node_bounds(dna)
            if node_bounds_list and (dna.get("materials") or {}):
                node_idx = _assign_voxels_to_nodes(positions_np, node_bounds_list)
                _assign_outside_voxels_to_nearest_node(positions_np, node_bounds_list, node_idx)
                n_assigned = int(np.sum(node_idx >= 0))
                n_nodes = len(node_bounds_list)
                node_attrs = _resolve_attrs_from_dna_materials(dna, node_bounds_list)
                
                # Check if sampled attributes are just the default gray (no procedural noise/texture)
                # _DEFAULT_ATTRS = [0.627, 0.0, 0.0, 0.0, 0.5]
                # We check the mean of the sampled batch to see if it's close to default.
                mean_sampled = np.mean(attrs_np, axis=0)
                is_default_gray = (
                    abs(mean_sampled[0] - 0.627) < 0.01 and 
                    abs(mean_sampled[1]) < 0.01 and 
                    abs(mean_sampled[2]) < 0.01
                )
                
                # If we have meaningful procedural variation (rust, grime, edge wear), we blend it.
                # If it's just the default background from an empty graph, we skip blending to preserve the material.
                if is_default_gray:
                    # Check for variation (e.g. maybe mean is gray but there is noise?)
                    # If std dev is low, it's flat gray.
                    std_sampled = np.std(attrs_np[:, 0])
                    if std_sampled < 0.001:
                        query_blend = 0.0
                        print(f"      [voxel-triplanar] Procedural attributes are default flat gray -> Using pure DNA material", flush=True)
                    else:
                        query_blend = 0.5
                        print(f"      [voxel-triplanar] Procedural attributes have variation -> Blending 50%", flush=True)
                else:
                    query_blend = 0.5
                    print(f"      [voxel-triplanar] Procedural attributes are colored -> Blending 50%", flush=True)

                for j in range(n_nodes):
                    mask = node_idx == j
                    if np.any(mask):
                        base = node_attrs[j : j + 1]  # (1, 5)
                        sampled = attrs_np[mask]      # (M, 5)
                        # If query_blend is 0, we just use base. 
                        # If query_blend > 0, we mix.
                        if query_blend <= 1e-5:
                             attrs_np[mask] = base
                        else:
                            attrs_np[mask] = np.clip(
                                base * (1.0 - query_blend) + sampled * query_blend,
                                0.0,
                                1.0,
                            )
                print(
                    f"      [voxel-triplanar] dna.materials: {n_assigned}/{n_surface} voxels in {n_nodes} nodes (blend={query_blend:.0%} query)",
                    flush=True,
                )
            elif node_bounds_list:
                node_idx = _assign_voxels_to_nodes(positions_np, node_bounds_list)
                _assign_outside_voxels_to_nearest_node(positions_np, node_bounds_list, node_idx)
                n_assigned = int(np.sum(node_idx >= 0))
                n_nodes = len(node_bounds_list)
                _apply_per_node_average(attrs_np, node_idx)
                print(f"      [voxel-triplanar] Per-node average: {n_assigned}/{n_surface} voxels in {n_nodes} nodes", flush=True)

    # attrs_np: (N, 5) [L, a, b, metallic, roughness] in Oklab. Bake blends in Oklab, outputs sRGB.
    use_cuda = device == "cuda" and torch.cuda.is_available()
    print(
        f"      [voxel-triplanar] Baking {n_surface} voxel Oklab {'on CUDA' if use_cuda else ''} to 3×{resolution}×{resolution}...",
        flush=True,
    )
    bake_mode = SplatBakeMode(mode) if mode in ("gaussian", "point") else SplatBakeMode.GAUSSIAN
    textures = bake_triplanar_from_voxel_oklab(
        positions_np, attrs_np, b_min, b_max, resolution=resolution, device=device, mode=bake_mode
    )
    return pack_triplanar_textures(textures)


def bake_procedural_triplanar(
    sdf_graph: Any,
    bounds: Tuple[List[float], List[float]],
    resolution: int = 512,
    target_sample_count: int = 40000,
    device: str = "cpu",
    bake_result: Optional[Any] = None,
    dna: Optional[dict] = None,
    mode: str = "gaussian",
) -> bytes:
    """
    Bake procedural colors into triplanar bytes.

    When bake_result is provided (dense SDF grid from pipeline), colors surface
    voxels via query_attributes and bakes to triplanar (fast, no surface sampling).
    Otherwise falls back to batched surface sampling + Gaussian splat bake.

    Args:
        sdf_graph: SDF module with query_attributes(x) -> (N, 5).
        bounds: (bounds_min, bounds_max) world-space AABB.
        resolution: Triplanar texture resolution (square).
        target_sample_count: Used only when bake_result is None (surface-sample path).
        device: Used only when bake_result is None.
        bake_result: Optional BakeResult from vdb_converter.bake_sdf; when set, use voxel path.
        dna: Optional DNA for material blending.
        mode: Splat bake mode ("gaussian" or "point").

    Returns:
        Packed triplanar binary (TRI1) for .gve_bin embedding.
    """
    if bake_result is not None:
        surface_tolerance = max(bake_result.voxel_size * 0.5, 1e-6)
        return _bake_triplanar_from_voxels(
            bake_result, sdf_graph, resolution, surface_tolerance, dna=dna, device=device, mode=mode
        )

    # Adjusted import for splat trainer
    from ..splat.trainer import initialize_splats_batched

    sdf_fn = sdf_graph
    attrs_fn = sdf_graph if hasattr(sdf_graph, "query_attributes") else None
    if device == "cuda" and hasattr(sdf_graph, "to"):
        # Keep original on CPU for attribute queries; use a GPU copy for batched SDF eval.
        # (Moving the same module to CPU after .to(cuda) would leave sdf_fn on CPU and break device consistency.)
        sdf_fn = copy.deepcopy(sdf_graph).to(device)
        attrs_fn = sdf_graph  # unchanged on CPU; initialize_splats_batched will pass positions.cpu() for attrs

    positions, attrs, avg_spacing = initialize_splats_batched(
        sdf_fn,
        bounds,
        target_count=target_sample_count,
        min_radius=0.02,
        device=device,
        attrs_fn=attrs_fn,
    )

    # attrs: (N, 5) float [L, a, b, metallic, roughness]. Bake in Oklab, output sRGB.
    attrs_np = attrs.detach().cpu().numpy().astype(np.float32)
    pos_np = positions.detach().cpu().numpy().astype(np.float32)
    scale_val = max(avg_spacing * 0.5, 1e-4)
    scales = np.full((len(pos_np), 3), scale_val, dtype=np.float32)
    bmin = np.array(bounds[0], dtype=np.float32)
    bmax = np.array(bounds[1], dtype=np.float32)
    
    bake_mode = SplatBakeMode(mode) if mode in ("gaussian", "point") else SplatBakeMode.GAUSSIAN

    textures = bake_triplanar_textures_oklab(
        pos_np, attrs_np, scales, bmin, bmax, resolution=resolution, mode=bake_mode
    )
    return pack_triplanar_textures(textures)
