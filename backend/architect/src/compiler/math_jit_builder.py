"""SDF Graph Builder - Constructs PyTorch SDF graphs from DNA JSON."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

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
from .math_jit_modifiers import (
    TransformNode,
    build_modifier,
)


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
        return self.root_node(x)


def _apply_modifiers_and_transform(node: nn.Module, node_data: Dict) -> nn.Module:
    """
    Apply modifiers and transform to a node.
    
    Order: Base SDF → Modifiers (in order) → Transform
    The transform is applied last so it positions the already-modified shape.
    """
    result = node
    
    # 1. Apply modifiers in order (if any)
    modifiers = node_data.get("modifiers")
    if modifiers:
        for mod_data in modifiers:
            result = build_modifier(result, mod_data)
    
    # 2. Apply transform (if non-identity)
    transform = node_data.get("transform")
    if transform:
        pos = transform.get("pos", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
        rot = transform.get("rot", [0.0, 0.0, 0.0, 1.0]) or [0.0, 0.0, 0.0, 1.0]
        
        is_identity = (
            abs(pos[0]) < 1e-6 and abs(pos[1]) < 1e-6 and abs(pos[2]) < 1e-6 and
            abs(rot[0]) < 1e-6 and abs(rot[1]) < 1e-6 and abs(rot[2]) < 1e-6 and abs(rot[3] - 1.0) < 1e-6
        )
        
        if not is_identity:
            result = TransformNode(result, pos=pos, rot=rot)
    
    return result


def build_node(node_data: Dict) -> nn.Module:
    """Build a node from either legacy or AI-generated format."""
    node_type = node_data.get("type")
    
    # Handle AI-generated format: {"type": "primitive", "shape": "sphere", "params": {...}}
    if node_type == "primitive":
        shape = node_data.get("shape", "sphere").lower()
        params = node_data.get("params", {}) or {}  # Handle None
        
        base_node = None
        
        if shape == "sphere":
            radius = params.get("radius") or params.get("r") or params.get("size")
            if radius is None:
                print(f"    ⚠️ Sphere missing radius, using 0.1m default", flush=True)
                radius = 0.1
            if isinstance(radius, list):
                radius = radius[0]
            base_node = SphereNode(radius=float(radius))
            
        elif shape == "box":
            size = params.get("size") or params.get("dimensions") or params.get("extents")
            if size is None:
                w = params.get("width", 1.0) or 1.0
                h = params.get("height", 1.0) or 1.0
                d = params.get("depth", 1.0) or 1.0
                size = [float(w), float(h), float(d)]
            elif not isinstance(size, list):
                size = [float(size), float(size), float(size)]
            else:
                size = [float(s) for s in size]
            base_node = BoxNode(size=size)
            
        elif shape == "cylinder":
            radius = params.get("radius") or params.get("r")
            height = params.get("height") or params.get("h")
            if radius is None:
                print(f"    ⚠️ Cylinder missing radius, using 0.1m default", flush=True)
                radius = 0.1
            if height is None:
                print(f"    ⚠️ Cylinder missing height, using 0.2m default", flush=True)
                height = 0.2
            base_node = CylinderNode(
                radius=float(radius),
                height=float(height)
            )
            
        elif shape == "plane":
            normal = params.get("normal", [0.0, 1.0, 0.0]) or [0.0, 1.0, 0.0]
            distance = params.get("distance", 0.0) or 0.0
            base_node = PlaneNode(normal=normal, distance=float(distance))
            
        elif shape == "capsule":
            radius = params.get("radius") or params.get("r")
            height = params.get("height") or params.get("h")
            if radius is None:
                radius = 0.05
            if height is None:
                height = 0.1
            base_node = CapsuleNode(radius=float(radius), height=float(height))
            
        elif shape == "torus":
            major_r = params.get("major_r") or params.get("radius") or 0.1
            minor_r = params.get("minor_r") or 0.02
            base_node = TorusNode(major_r=float(major_r), minor_r=float(minor_r))
            
        elif shape == "cone":
            radius = params.get("radius") or params.get("r") or 0.1
            height = params.get("height") or params.get("h") or 0.2
            base_node = ConeNode(radius=float(radius), height=float(height))
            
        else:
            print(f"    ⚠️ Unknown shape '{shape}', using tiny sphere fallback", flush=True)
            base_node = SphereNode(radius=0.01)  # Tiny 1cm sphere, not 1m!
        
        # Apply modifiers and transform
        return _apply_modifiers_and_transform(base_node, node_data)
    
    # Handle AI-generated operation nodes: {"type": "operation", "op": "union", "children": [...]}
    elif node_type == "operation":
        op = node_data.get("op", "union").lower()
        children = [build_node(c) for c in node_data.get("children", [])]
        
        if op == "union" and children:
            return UnionNode(children)
        elif children:
            return UnionNode(children)  # Default to union for other ops
        else:
            return SphereNode(radius=0.0)
    
    # Legacy format: {"type": "sphere", "radius": 1.0}
    elif node_type == "sphere":
        return SphereNode(radius=node_data.get("radius", 1.0))
        
    elif node_type == "box":
        return BoxNode(size=node_data.get("size", [1.0, 1.0, 1.0]))

    elif node_type == "cylinder":
        return CylinderNode(
            radius=node_data.get("radius", 1.0),
            height=node_data.get("height", 2.0)
        )

    elif node_type == "plane":
        return PlaneNode(
            normal=node_data.get("normal", [0.0, 1.0, 0.0]),
            distance=node_data.get("distance", 0.0)
        )
        
    elif node_type == "union":
        children = [build_node(c) for c in node_data.get("children", [])]
        return UnionNode(children)
        
    # Fallback to empty space (large distance)
    return SphereNode(radius=0.0) 


def build_sdf_graph(dna: Dict) -> SdfGraph:
    """
    Convert DNA JSON to PyTorch SDF evaluation graph.
    
    Supports both legacy format and AI-generated format:
    - Legacy: {"nodes": [...]} or {"type": "sphere", ...}
    - AI: {"root_node": {"type": "operation", "op": "union", "children": [...]}}
    
    Also extracts bounds from metadata if available.
    """
    # Extract bounds from metadata
    bounds = None
    metadata = dna.get("metadata", {})
    if metadata and "estimated_bounds" in metadata:
        eb = metadata["estimated_bounds"]
        bounds = (eb.get("min", [-1, -1, -1]), eb.get("max", [1, 1, 1]))
    
    # AI-generated format with root_node
    if "root_node" in dna:
        root = build_node(dna["root_node"])
        return SdfGraph(root, bounds=bounds)
    
    # Legacy format with nodes array
    if "nodes" in dna:
        children = [build_node(n) for n in dna["nodes"]]
        if len(children) == 1:
            root = children[0]
        else:
            root = UnionNode(children)
        return SdfGraph(root, bounds=bounds)
    
    # Assume root is a single node description
    root = build_node(dna)
    return SdfGraph(root, bounds=bounds)
