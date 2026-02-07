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
    PrimitiveNode,
)
from .math_jit_modifiers import (
    TransformNode,
    build_modifier,
)
from .math_jit_noise import ProceduralTextureNode
from ..librarian.materials import get_material
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
        dist, _ = self.root_node(x)
        return dist

    def query_attributes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get material attributes for points x.

        Args:
            x: [N, 3] world positions.

        Returns:
            [N, 5] tensor: [oklab_L, oklab_a, oklab_b, metallic, roughness]
        """
        _, attrs = self.root_node(x)
        return attrs


def _apply_modifiers_and_transform(node: nn.Module, node_data: Dict) -> nn.Module:
    """
    Apply modifiers, procedural textures, and transform to a node.
    
    Order: Base SDF → Modifiers → Procedural Texture → Transform
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
        )
    
    # 2. Apply transform (if non-identity)
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


def _resolve_material(material_id: Union[int, str], explicit_color: Optional[List[float]] = None) -> Tuple[List[float], float, float]:
    """Resolve material ID to ``(oklab_color, metallic, roughness)``.

    Priority: explicit sRGB color > material registry lookup > default gray.
    Single conversion: sRGB -> Oklab via ``srgb_to_oklab``.
    """
    default_color = [0.627, 0.0, 0.0]  # approx mid-gray in Oklab

    # 1. Use explicit color if available (sRGB [0-1])
    if explicit_color is not None and len(explicit_color) >= 3:
        try:
            srgb = torch.tensor(explicit_color[:3], dtype=torch.float32)
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
        params = node_data.get("params", {}) or {}
        
        # Get material properties (color + PBR)
        # Check both node_data (legacy/top-level) and params (AI-gen) for color
        explicit_color = node_data.get("color") or params.get("color")
        raw_mat = node_data.get("material_id", 0)
        
        color, metallic, roughness = _resolve_material(raw_mat, explicit_color)
        
        # Allow params to override PBR if provided explicitly
        if "metallic" in params: metallic = float(params["metallic"])
        if "roughness" in params: roughness = float(params["roughness"])
        
        base_node = None
        
        if shape == "sphere":
            radius = params.get("radius") or params.get("r") or params.get("size")
            if radius is None: radius = 0.1
            if isinstance(radius, list): radius = radius[0]
            base_node = SphereNode(radius=float(radius), color=color, metallic=metallic, roughness=roughness)
            
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
            base_node = BoxNode(size=size, color=color, metallic=metallic, roughness=roughness)
            
        elif shape == "cylinder":
            radius = params.get("radius") or params.get("r") or 0.1
            height = params.get("height") or params.get("h") or 0.2
            base_node = CylinderNode(radius=float(radius), height=float(height), color=color, metallic=metallic, roughness=roughness)
            
        elif shape == "plane":
            normal = params.get("normal", [0.0, 1.0, 0.0])
            distance = params.get("distance", 0.0)
            base_node = PlaneNode(normal=normal, distance=float(distance), color=color, metallic=metallic, roughness=roughness)
            
        elif shape == "capsule":
            radius = params.get("radius") or 0.05
            height = params.get("height") or 0.1
            base_node = CapsuleNode(radius=float(radius), height=float(height), color=color, metallic=metallic, roughness=roughness)
            
        elif shape == "torus":
            major_r = params.get("major_r") or 0.1
            minor_r = params.get("minor_r") or 0.02
            base_node = TorusNode(major_r=float(major_r), minor_r=float(minor_r), color=color, metallic=metallic, roughness=roughness)
            
        elif shape == "cone":
            radius = params.get("radius") or 0.1
            height = params.get("height") or 0.2
            base_node = ConeNode(radius=float(radius), height=float(height), color=color, metallic=metallic, roughness=roughness)
            
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
            base_node = WedgeNode(
                size=size, taper_axis=taper_axis, taper_dir=taper_dir,
                color=color, metallic=metallic, roughness=roughness,
            )

        elif shape == "revolution":
            profile_data = params.get("profile")
            if profile_data:
                profile_child = build_node(profile_data)
            else:
                profile_child = BoxNode(size=[0.1, 0.2, 0.01], color=color, metallic=metallic, roughness=roughness)
            axis = params.get("axis", "y")
            offset = float(params.get("offset", 0.0))
            base_node = RevolutionNode(profile_child, axis=axis, offset=offset)
        
        elif shape == "mandelbulb":
            base_node = MandelbulbNode(
                power=float(params.get("power", 8.0)),
                iterations=int(params.get("iterations", 8)),
                scale=float(params.get("scale", 1.0)),
                color=color, metallic=metallic, roughness=roughness,
            )
        
        elif shape == "menger":
            base_node = MengerSpongeNode(
                iterations=int(params.get("iterations", 3)),
                scale=float(params.get("scale", 1.0)),
                color=color, metallic=metallic, roughness=roughness,
            )
        
        elif shape == "julia":
            base_node = JuliaSetNode(
                c=params.get("c", [0.3, 0.5, 0.2, 0.1]),
                iterations=int(params.get("iterations", 8)),
                scale=float(params.get("scale", 1.0)),
                color=color, metallic=metallic, roughness=roughness,
            )
            
        else:
            base_node = SphereNode(radius=0.01, color=color, metallic=metallic, roughness=roughness)
        
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
            return SmoothUnionNode(children, k=k)
        elif op == "smooth_subtract":
            k = float(node_data.get("smoothness", node_data.get("k", 0.5)))
            return SmoothSubtractNode(children, k=k)
        elif op == "smooth_intersect":
            k = float(node_data.get("smoothness", node_data.get("k", 0.5)))
            return SmoothIntersectNode(children, k=k)
        elif children:
            return UnionNode(children)
        else:
            return SphereNode(radius=0.0)
    
    # Legacy format
    elif node_type == "sphere":
        return SphereNode(radius=node_data.get("radius", 1.0))
    elif node_type == "box":
        return BoxNode(size=node_data.get("size", [1.0, 1.0, 1.0]))
    elif node_type == "union":
        children = [build_node(c) for c in node_data.get("children", [])]
        return UnionNode(children)
        
    return SphereNode(radius=0.0)


def _prepare_dna(dna: Dict) -> Dict:
    """Merge ``dna["materials"]`` into each node's data before building.

    The AI pipeline stores material assignments in a top-level dict keyed
    by node ID.  This function pushes those values into the node dicts so
    ``build_node`` can read them without any special-casing.  Also normalises
    the ``procedural_texture`` field name (AI) vs ``texture_pattern`` (legacy).
    """
    materials = dna.get("materials", {})
    if not materials:
        return dna

    def _inject(node: Dict) -> None:
        node_id = node.get("id")
        if node_id and node_id in materials:
            cfg = materials[node_id]
            # Support both raw dicts and Pydantic models
            if hasattr(cfg, "model_dump"):
                cfg = cfg.model_dump(exclude_none=True)
            elif not isinstance(cfg, dict):
                cfg = vars(cfg)

            # Map AI fields -> compiler fields
            if "base_color" in cfg and "color" not in (node.get("params") or {}):
                node.setdefault("params", {})["color"] = cfg["base_color"]
            if "material_id" in cfg:
                node["material_id"] = cfg["material_id"]
            if "procedural_texture" in cfg and cfg["procedural_texture"]:
                node["procedural_texture"] = cfg["procedural_texture"]
            for key in ("metallic", "roughness"):
                if key in cfg and cfg[key] is not None:
                    node.setdefault("params", {})[key] = cfg[key]

        # Recurse into children
        for child in node.get("children") or []:
            _inject(child)

    if "root_node" in dna:
        _inject(dna["root_node"])
    elif "nodes" in dna:
        for n in dna["nodes"]:
            _inject(n)
    return dna


def build_sdf_graph(dna: Dict) -> SdfGraph:
    """Convert DNA JSON to PyTorch SDF evaluation graph."""
    # Merge AI-assigned materials into node dicts first
    _prepare_dna(dna)

    bounds = None
    metadata = dna.get("metadata", {})
    if metadata and "estimated_bounds" in metadata:
        eb = metadata["estimated_bounds"]
        bounds = (eb.get("min", [-1, -1, -1]), eb.get("max", [1, 1, 1]))
    
    if "root_node" in dna:
        root = build_node(dna["root_node"])
        return SdfGraph(root, bounds=bounds)
    
    if "nodes" in dna:
        children = [build_node(n) for n in dna["nodes"]]
        root = UnionNode(children) if len(children) > 1 else children[0]
        return SdfGraph(root, bounds=bounds)
    
    root = build_node(dna)
    return SdfGraph(root, bounds=bounds)
