"""SDF Graph Builder - Constructs PyTorch SDF graphs from DNA JSON."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

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
    PrimitiveNode, # Re-exporting for typing if needed, though mostly unused directly now
    GeometryNode,
    MaterialNode,
)
from .math_jit_modifiers import (
    TransformNode,
    build_modifier,
)
from .math_jit_noise import ProceduralTextureNode, TextureModifierNode
from ..librarian.materials import get_material
from ..librarian.finishes import get_finish
from .oklab import srgb_to_oklab


class SdfGraph(nn.Module):
    """
    PyTorch module that evaluates an SDF graph.
    """
    def __init__(self, root_node: nn.Module, bounds: Optional[Tuple[List[float], List[float]]] = None):
        super().__init__()
        self.root_node = root_node
        self.bounds = bounds  # (min_xyz, max_xyz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate SDF at points x.
        x: [N, 3] tensor
        Returns: [N] distances
        """
        res = self.root_node(x)
        if isinstance(res, tuple):
            return res[0]
        return res

    def query_attributes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get material attributes for points x.

        Args:
            x: [N, 3] world positions.

        Returns:
            [N, 5] tensor: [oklab_L, oklab_a, oklab_b, metallic, roughness]
        """
        res = self.root_node(x)
        if isinstance(res, tuple):
            return res[1]
        
        # Fallback for pure geometry root: return default orange
        # TODO: Should likely match _DEFAULT_FALLBACK_COLOR in math_jit_nodes
        orange = torch.tensor([1.0, 0.4, 0.0, 0.0, 0.5], device=x.device)
        return orange.unsqueeze(0).expand(x.shape[0], 5)


def _apply_modifiers_and_transform(node: nn.Module, node_data: Dict) -> nn.Module:
    """
    Apply modifiers, procedural textures, and transform to a node.
    
    Order: Base SDF → Modifiers → Procedural Texture → Texture Modifiers → Transform
    The transform is applied last so it positions the already-modified shape.
    """
    result = node
    
    # 1. Apply modifiers in order (if any)
    modifiers = node_data.get("modifiers")
    if modifiers:
        for mod_data in modifiers:
            result = build_modifier(result, mod_data)
    
    # 2. Apply procedural texture (if any) — accepts both AI and legacy field names
    tex_pattern = node_data.get("procedural_texture") or node_data.get("texture_pattern")
    if tex_pattern and isinstance(tex_pattern, dict):
        result = ProceduralTextureNode(
            child=result,
            pattern=tex_pattern.get("type", "perlin"),
            scale=float(tex_pattern.get("scale", 5.0)),
            intensity=float(tex_pattern.get("intensity", 0.3)),
            color_variation=float(tex_pattern.get("color_variation", 0.2)),
            roughness_variation=float(tex_pattern.get("roughness_variation", 0.1)),
            metallic_variation=float(tex_pattern.get("metallic_variation", 0.0)),
        )
    
    # 3. Apply texture modifiers (edge wear, grime, rust) if present
    tex_mod = node_data.get("texture_modifiers")
    if tex_mod and isinstance(tex_mod, dict):
        result = TextureModifierNode(
            child=result,
            edge_wear=_safe_float(tex_mod.get("edge_wear"), 0.0),
            cavity_grime=_safe_float(tex_mod.get("cavity_grime"), 0.0),
            rust_amount=_safe_float(tex_mod.get("rust_amount"), 0.0),
        )
    
    # 4. Apply transform (if non-identity)
    transform = node_data.get("transform")
    if transform:
        pos = transform.get("pos", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
        rot = transform.get("rot", [0.0, 0.0, 0.0, 1.0]) or [0.0, 0.0, 0.0, 1.0]

        # Handle Euler Angles (3 elements) -> Quaternion conversion
        if len(rot) == 3:
            import numpy as np
            # Convert degrees to radians
            r_rad = [np.deg2rad(a) for a in rot]
            
            # Euler to Quaternion (XYZ order)
            cx = np.cos(r_rad[0] * 0.5)
            sx = np.sin(r_rad[0] * 0.5)
            cy = np.cos(r_rad[1] * 0.5)
            sy = np.sin(r_rad[1] * 0.5)
            cz = np.cos(r_rad[2] * 0.5)
            sz = np.sin(r_rad[2] * 0.5)
            
            qw = cx * cy * cz + sx * sy * sz
            qx = sx * cy * cz - cx * sy * sz
            qy = cx * sy * cz + sx * cy * sz
            qz = cx * cy * sz - sx * sy * cz
            
            rot = [qx, qy, qz, qw]
        
        is_identity = (
            abs(pos[0]) < 1e-6 and abs(pos[1]) < 1e-6 and abs(pos[2]) < 1e-6 and
            abs(rot[0]) < 1e-6 and abs(rot[1]) < 1e-6 and abs(rot[2]) < 1e-6 and abs(rot[3] - 1.0) < 1e-6
        )
        
        if not is_identity:
            result = TransformNode(result, pos=pos, rot=rot)
    
    return result


def _parse_srgb_color(value: Optional[Union[List[float], str]]) -> Optional[List[float]]:
    """Parse sRGB color from hex or list into 0-1 floats."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("#"):
            s = s[1:]
        if len(s) in (6, 8):
            try:
                r = int(s[0:2], 16)
                g = int(s[2:4], 16)
                b = int(s[4:6], 16)
                return [r / 255.0, g / 255.0, b / 255.0]
            except ValueError:
                return None
        return None
    if isinstance(value, list) and len(value) >= 3:
        try:
            rgb = [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
        max_val = max(rgb)
        if max_val > 1.0:
            return [v / 255.0 for v in rgb]
        return rgb
    return None


def _safe_float(value: Optional[object], default: float = 0.0) -> float:
    """Coerce to float with a safe fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_material(
    material_id: Union[int, str],
    explicit_color: Optional[Union[List[float], str]] = None,
) -> Tuple[List[float], float, float]:
    """Resolve material ID to ``(oklab_color, metallic, roughness)``.

    Priority: explicit sRGB color > material registry lookup > default gray.
    Single conversion: sRGB -> Oklab via ``srgb_to_oklab``.
    """
    default_color = [0.627, 0.0, 0.0]  # approx mid-gray in Oklab

    # 1. Use explicit color if available (sRGB [0-1])
    parsed_color = _parse_srgb_color(explicit_color)
    if parsed_color is not None:
        try:
            srgb = torch.tensor(parsed_color[:3], dtype=torch.float32)
            oklab = srgb_to_oklab(srgb.unsqueeze(0))[0].tolist()
            return oklab, 0.0, 0.5
        except Exception:
            pass  # Fallback to material ID

    # 2. Look up material from registry
    spec_name = str(material_id)
    if spec_name == "0":
        return default_color, 0.0, 0.5

    try:
        mat = get_material(spec_name)
        srgb = torch.tensor(mat.base_color, dtype=torch.float32)
        oklab = srgb_to_oklab(srgb.unsqueeze(0))[0].tolist()
        return oklab, mat.metallic, mat.roughness
    except Exception:
        return default_color, 0.0, 0.5


def build_node(node_data: Dict) -> nn.Module:
    """Build a node from either legacy or AI-generated format."""
    node_type = node_data.get("type")
    
    # Handle AI-generated format: {"type": "primitive", "shape": "sphere", "params": {...}}
    if node_type == "primitive":
        shape = node_data.get("shape", "sphere").lower()
        raw_params = node_data.get("params", {}) or {}
        # Strip operation-only keys that sometimes appear inside params (errant DNA from LLM)
        _operation_keys = {"op", "children", "k"}
        params = {k: v for k, v in raw_params.items() if k not in _operation_keys}

        # Resolve material properties (color + PBR)
        # Check node_data, params.color, and params.base_color (AI-gen)
        # We need to decide if we wrap in MaterialNode or not.
        
        explicit_color = (
            node_data.get("color")
            or params.get("color")
            or params.get("base_color")
        )
        raw_mat = node_data.get("material_id", 0)
        
        # Check if we assume a default material if none provided?
        # Current logic: always resolve something (default gray if missing)
        color, metallic, roughness = _resolve_material(raw_mat, explicit_color)
        
        # Allow params to override PBR if provided explicitly
        if "metallic" in params: metallic = float(params["metallic"])
        if "roughness" in params: roughness = float(params["roughness"])
        
        base_node = None
        
        if shape == "sphere":
            radius = params.get("radius") or params.get("r") or params.get("size")
            if radius is None: radius = 0.1
            if isinstance(radius, list): radius = radius[0]
            base_node = SphereNode(radius=float(radius))
            
        elif shape == "box":
            size = params.get("size")
            if size is None:
                w = params.get("width", 1.0) or 1.0
                h = params.get("height", 1.0) or 1.0
                d = params.get("depth", 1.0) or 1.0
                size = [float(w), float(h), float(d)]
            elif not isinstance(size, list):
                size = [float(size)] * 3
            else:
                size = [float(s) for s in size]
            base_node = BoxNode(size=size)
            
        elif shape == "cylinder":
            radius = params.get("radius") or params.get("r") or 0.1
            height = params.get("height") or params.get("h") or 0.2
            base_node = CylinderNode(radius=float(radius), height=float(height))
            
        elif shape == "plane":
            normal = params.get("normal", [0.0, 1.0, 0.0])
            distance = params.get("distance", 0.0)
            base_node = PlaneNode(normal=normal, distance=float(distance))
            
        elif shape == "capsule":
            radius = params.get("radius") or 0.05
            height = params.get("height") or 0.1
            base_node = CapsuleNode(radius=float(radius), height=float(height))
            
        elif shape == "torus":
            major_r = params.get("major_r") or 0.1
            minor_r = params.get("minor_r") or 0.02
            base_node = TorusNode(major_r=float(major_r), minor_r=float(minor_r))
            
        elif shape == "cone":
            radius = params.get("radius") or 0.1
            height = params.get("height") or 0.2
            base_node = ConeNode(radius=float(radius), height=float(height))
            
        elif shape == "wedge":
            size = params.get("size")
            if size is None:
                size = [1.0, 1.0, 1.0]
            elif not isinstance(size, list):
                size = [float(size)] * 3
            else:
                size = [float(s) for s in size]
            taper_axis = str(params.get("taper_axis", "y")).lower()
            taper_dir = str(params.get("taper_dir", "z")).lower()
            base_node = WedgeNode(size=size, taper_axis=taper_axis, taper_dir=taper_dir)

        elif shape == "revolution":
            profile_data = params.get("profile")
            if profile_data:
                profile_child = build_node(profile_data)
                # Unwrap if it was wrapped in a material (revolution only cares about geometry profile)
                if isinstance(profile_child, MaterialNode):
                    profile_child = profile_child.child
            else:
                profile_child = BoxNode(size=[0.1, 0.2, 0.01])
            axis = params.get("axis", "y")
            offset = float(params.get("offset", 0.0))
            base_node = RevolutionNode(profile_child, axis=axis, offset=offset)
        
        elif shape == "mandelbulb":
            base_node = MandelbulbNode(
                power=float(params.get("power", 8.0)),
                iterations=int(params.get("iterations", 8)),
                scale=float(params.get("scale", 1.0)),
            )
        
        elif shape == "menger":
            base_node = MengerSpongeNode(
                iterations=int(params.get("iterations", 3)),
                scale=float(params.get("scale", 1.0)),
            )
        
        elif shape == "julia":
            base_node = JuliaSetNode(
                c=params.get("c", [0.3, 0.5, 0.2, 0.1]),
                iterations=int(params.get("iterations", 8)),
                scale=float(params.get("scale", 1.0)),
            )
            
        else:
            base_node = SphereNode(radius=0.01)
        
        # Apply Material Wrapper
        # We always apply it currently because _resolve_material returns a default. 
        # Ideally we only apply if user asked for it, but for compatibility we attach the resolved material.
        base_node = MaterialNode(base_node, color=color, metallic=metallic, roughness=roughness)
        
        return _apply_modifiers_and_transform(base_node, node_data)
    
    # Handle AI-generated operation nodes
    elif node_type == "operation":
        op = node_data.get("op", "union").lower()
        children = [build_node(c) for c in node_data.get("children", [])]
        
        if op == "union":
            return UnionNode(children)
        elif op == "subtract":
            return SubtractNode(children)
        elif op == "intersect":
            return IntersectNode(children)
        elif op == "smooth_union":
            k = float(node_data.get("smoothness", node_data.get("k", 0.5)))
            if len(children) <= 2:
                return SmoothUnionNode(children, k=k)
            # Fold to binary tree
            acc = children[0]
            for c in children[1:]:
                acc = SmoothUnionNode([acc, c], k=k)
            return acc
        elif op == "smooth_subtract":
            k = float(node_data.get("smoothness", node_data.get("k", 0.5)))
            return SmoothSubtractNode(children, k=k)
        elif op == "smooth_intersect":
            k = float(node_data.get("smoothness", node_data.get("k", 0.5)))
            if len(children) <= 2:
                return SmoothIntersectNode(children, k=k)
            acc = children[0]
            for c in children[1:]:
                acc = SmoothIntersectNode([acc, c], k=k)
            return acc
        elif children:
            return UnionNode(children)
        else:
            return SphereNode(radius=0.0)
    
    # Unknown node type
    print(f"⚠️ Unknown node type: {node_type}, handling as empty", flush=True)
    return MaterialNode(SphereNode(radius=0.0))


def _find_parent_and_index(node: Dict, target_id: str, parent: Optional[Dict] = None, index_in_parent: Optional[int] = None) -> Tuple[Optional[Dict], Optional[int]]:
    """Find (parent, child_index) of the node with id == target_id. Returns (None, None) if not found."""
    if node.get("id") == target_id:
        return (parent, index_in_parent)
    children = node.get("children") or node.get("nodes") or []
    for i, child in enumerate(children):
        p, idx = _find_parent_and_index(child, target_id, node, i)
        if p is not None:
            return (p, idx)
    return (None, None)


def _inject_machining_patches(dna: Dict) -> None:
    """
    Inject A2 Machinist patches into the SDF tree.
    Supports subtract (carving) and add (hardware) operations.
    """
    patches = dna.get("machining_patches") or []
    if not patches:
        return
    root = dna.get("root_node")
    if not root:
        return
    applied = 0
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        target_id = patch.get("target_node_id")
        if not target_id:
            continue
        op = (patch.get("op") or "subtract").lower()

        geom = patch.get("subtract") if op in ("subtract", "smooth_subtract") else patch.get("add")
        if not geom or not isinstance(geom, dict):
            continue

        parent, idx = _find_parent_and_index(root, target_id)
        if parent is None or idx is None:
            print(f"      [prepare_dna] ⚠️ Machinist patch target not found: {target_id}", flush=True)
            continue
        child_list = parent.get("children") or parent.get("nodes") or []
        child = child_list[idx]
        geom_node: Dict = {
            "type": "primitive",
            "shape": geom.get("shape", "box"),
            "params": geom.get("params") or geom.get("param") or {},
            "transform": geom.get("transform"),
        }
        if op == "add":
            new_op = {"type": "operation", "op": "union", "children": [child, geom_node]}
        else:
            new_op = {"type": "operation", "op": op, "children": [child, geom_node]}
            if op == "smooth_subtract" and patch.get("k") is not None:
                new_op["k"] = float(patch["k"])
        key = "children" if "children" in parent else "nodes"
        parent[key][idx] = new_op
        applied += 1
    if applied:
        print(f"      [prepare_dna] 🔧 Injected {applied} Machinist patches", flush=True)


def _prepare_dna(dna: Dict) -> Dict:
    """Merge ``dna["materials"]`` and ``dna["machining_patches"]`` before building."""
    materials = dna.get("materials", {})
    n_mats = len(materials) if isinstance(materials, dict) else 0
    if n_mats:
        print(f"      [prepare_dna] 📦 materials: {n_mats} entries", flush=True)
    else:
        return dna

    inject_count = [0]  # mutable so inner fn can update

    def _inject(node: Dict) -> None:
        node_id = node.get("id")
        if node_id and node_id in materials:
            cfg = materials[node_id]
            if hasattr(cfg, "model_dump"):
                cfg = cfg.model_dump(exclude_none=True)
            elif not isinstance(cfg, dict):
                cfg = vars(cfg)

            params = node.setdefault("params", {})
            finish_id = cfg.get("finish_id")
            if finish_id:
                finish = get_finish(finish_id)
                if finish:
                    if finish.get("base_color") is not None and "color" not in params:
                        params["color"] = finish["base_color"]
                    for key in ("roughness", "metallic"):
                        if finish.get(key) is not None:
                            params[key] = finish[key]
            
            if "base_color" in cfg and cfg["base_color"] is not None:
                params["color"] = cfg["base_color"]
            if "material_id" in cfg:
                node["material_id"] = cfg["material_id"]
            if "procedural_texture" in cfg and cfg["procedural_texture"]:
                node["procedural_texture"] = cfg["procedural_texture"]
            if "texture_modifiers" in cfg and cfg["texture_modifiers"]:
                node["texture_modifiers"] = cfg["texture_modifiers"]
            for key in ("metallic", "roughness"):
                if key in cfg and cfg[key] is not None:
                    params[key] = cfg[key]
            inject_count[0] += 1

        for child in node.get("children") or node.get("nodes") or []:
            _inject(child)

    if "root_node" in dna:
        _inject(dna["root_node"])
    elif "nodes" in dna:
        for n in dna["nodes"]:
            _inject(n)
    if n_mats and inject_count[0] != n_mats:
        print(f"      [prepare_dna] ⚠️ Injected {inject_count[0]} nodes (materials has {n_mats} keys)", flush=True)
    return dna


def collect_node_bounds(dna: Dict) -> List[Tuple[str, List[float], List[float]]]:
    """Collect world-space AABB (bmin, bmax) for every node that has an "id"."""
    infos = collect_node_info(dna)
    return [(n["id"], n["bmin"], n["bmax"]) for n in infos]


def collect_node_info(dna: Dict) -> List[Dict]:
    """Collect node id, path, center, bmin, bmax for every node with an "id".
    Used by property editor and viewport for node selection and gizmo placement.
    """
    out: List[Dict] = []

    def traverse(node: Dict, path_prefix: str, child_key: str, idx: int) -> None:
        path = f"{path_prefix}.{child_key}.{idx}" if path_prefix else f"{child_key}.{idx}"
        node_id = node.get("id")
        if node_id:
            try:
                built = build_node(node)
                if hasattr(built, "compute_bounds"):
                    b_min, b_max = built.compute_bounds()
                    b_min = b_min.detach().cpu().tolist()
                    b_max = b_max.detach().cpu().tolist()
                    center = [(b_min[i] + b_max[i]) / 2 for i in range(3)]
                    out.append({
                        "id": node_id,
                        "path": path,
                        "center": center,
                        "bmin": b_min,
                        "bmax": b_max,
                    })
            except Exception as e:
                print(f"    ⚠️ collect_node_info skip {node_id}: {e}", flush=True)
        for i, child in enumerate(node.get("children") or []):
            traverse(child, path, "children", i)

    if "root_node" in dna:
        root = dna["root_node"]
        root_id = root.get("id")
        if root_id:
            try:
                built = build_node(root)
                if hasattr(built, "compute_bounds"):
                    b_min, b_max = built.compute_bounds()
                    b_min = b_min.detach().cpu().tolist()
                    b_max = b_max.detach().cpu().tolist()
                    center = [(b_min[i] + b_max[i]) / 2 for i in range(3)]
                    out.append({
                        "id": root_id,
                        "path": "root_node",
                        "center": center,
                        "bmin": b_min,
                        "bmax": b_max,
                    })
            except Exception as e:
                print(f"    ⚠️ collect_node_info skip root {root_id}: {e}", flush=True)
        for i, child in enumerate(root.get("children") or []):
            traverse(child, "root_node", "children", i)
    elif "nodes" in dna:
        for i, n in enumerate(dna["nodes"]):
            traverse(n, "", "nodes", i)
    return out


def build_sdf_graph(dna: Dict) -> SdfGraph:
    """Convert DNA JSON to PyTorch SDF evaluation graph."""
    _inject_machining_patches(dna)
    _prepare_dna(dna)

    bounds = None
    metadata = dna.get("metadata", {})
    if metadata and "estimated_bounds" in metadata:
        eb = metadata["estimated_bounds"]
        bounds = (eb.get("min", [-1, -1, -1]), eb.get("max", [1, 1, 1]))
    
    root = None
    if "root_node" in dna:
        root = build_node(dna["root_node"])
    elif "nodes" in dna:
        children = [build_node(n) for n in dna["nodes"]]
        root = UnionNode(children) if len(children) > 1 else children[0]
    else:
        root = build_node(dna)
        
    # Auto-calculate bounds if missing
    if bounds is None and hasattr(root, "compute_bounds"):
        try:
            b_min, b_max = root.compute_bounds()
            bounds = (b_min.cpu().tolist(), b_max.cpu().tolist())
            print(f"    📐 Auto-calculated bounds: {bounds}", flush=True)
        except Exception as e:
            print(f"    ⚠️ Failed to calculate bounds: {e}", flush=True)
            bounds = ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])

    return SdfGraph(root, bounds=bounds)
