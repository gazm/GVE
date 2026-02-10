"""
SDF Graph Serializer - Converts PyTorch SDF graph to GVE Binary Bytecode.

See engine/shared/src/binary_format.rs for the binary specification.
"""
import struct
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
import logging

from .math_jit_builder import SdfGraph
from .math_jit_nodes import (
    PrimitiveNode, SphereNode, BoxNode, CylinderNode, CapsuleNode,
    TorusNode, ConeNode, PlaneNode, RevolutionNode, 
    MandelbulbNode, MengerSpongeNode, JuliaSetNode, WedgeNode,
    UnionNode, SubtractNode, IntersectNode,
    SmoothUnionNode, SmoothSubtractNode, SmoothIntersectNode
)
from .math_jit_modifiers import (
    TransformNode, TwistModifier, BendModifier, MirrorModifier, 
    RoundModifier, VoronoiModifier, TaperModifier
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants & Enums (Must match engine/shared/src/binary_format.rs)
# =============================================================================

# Instruction Types
TYPE_PRIMITIVE = 0
TYPE_BINARY_OP = 1
TYPE_MODIFIER = 2

# Primitive Ops
OP_SPHERE       = 0x01
OP_BOX          = 0x02
OP_CYLINDER     = 0x03
OP_CAPSULE      = 0x04
OP_TORUS        = 0x05
OP_CONE         = 0x06
OP_PLANE        = 0x07
OP_REVOLUTION   = 0x08
OP_MANDELBULB   = 0x09
OP_MENGER       = 0x0A
OP_JULIA        = 0x0B
OP_WEDGE        = 0x0C

# Binary Ops
OP_UNION            = 0x10
OP_SUBTRACT         = 0x11
OP_INTERSECT        = 0x12
OP_SMOOTH_UNION     = 0x13
OP_SMOOTH_SUBTRACT  = 0x14
OP_SMOOTH_INTERSECT = 0x15

# Modifier Ops
OP_TWIST    = 0x20
OP_BEND     = 0x21
OP_MIRROR   = 0x22
OP_ROUND    = 0x23
OP_ELONGATE = 0x24  # Not implemented in math_jit yet
OP_VORONOI  = 0x25

# Struct Formats
# SDFInstruction (40 bytes): type(u8), op(u8), op1(u16), op2(u16), res(u16), params(8*f32)
INSTR_FMT = "< B B H H H 8f"

# Mapping for string axes
AXIS_MAP = {"x": 0, "y": 1, "z": 2}


class BytecodeBuilder:
    def __init__(self):
        self.instructions: List[bytes] = []
        # Map python object id to instruction index to handle shared nodes (DAG)
        self.node_map: Dict[int, int] = {} 
        self.bounds_min = [-1.0, -1.0, -1.0]
        self.bounds_max = [1.0, 1.0, 1.0]

    def _emit(self, type_id: int, op_id: int, 
              op1: int = 0, op2: int = 0, 
              params: List[float] = None) -> int:
        """Emit an instruction and return its index."""
        if params is None:
            params = [0.0] * 8
        
        # Pad params to 8 floats
        if len(params) < 8:
            params = params + [0.0] * (8 - len(params))
        elif len(params) > 8:
            params = params[:8]  # Truncate if too many (shouldn't happen)
            
        index = len(self.instructions)
        
        # Pack instruction
        # type(B), op(B), op1(H), op2(H), res(H), params(8f)
        instr_bytes = struct.pack(
            INSTR_FMT,
            type_id,
            op_id,
            op1,
            op2,
            0, # Reserved
            *params
        )
        self.instructions.append(instr_bytes)
        return index

    def visit(self, node: nn.Module, transform: Optional[Dict] = None) -> int:
        """
        Recursively visit nodes and generate bytecode.
        
        Args:
            node: The node to visit
            transform: Accumulated (translation, rotation) to apply to primitives
        """
        # Node deduplication (DAG support)
        # Note: If we are pushing transforms down, we can't easily deduplicate 
        # because the same node might be visited with different transforms.
        # So we only deduplicate if no transform is active, OR we assume
        # the graph is a tree (standard for CSG).
        # For now, let's strictly treat it as a tree if transforms are involved.
        
        # Handle TransformNode wrapper (unwrap and accumulate)
        if isinstance(node, TransformNode):
            # Extract translation/rotation
            t = node.translation.tolist() # [x, y, z]
            q = node.quat.tolist()        # [x, y, z, w]
            
            # Warn about rotation if significant
            if abs(q[3]) < 0.999: # w < ~1.0 means rotation exists
                logger.warning(f"SDF Serializer: Rotation info lost for node {type(node.child)} (Engine limitation)")
                
            # Accumulate translation
            # If we already have a translation, add this one.
            # transform is {"pos": [x,y,z]}
            current_pos = transform["pos"] if transform else [0.0, 0.0, 0.0]
            new_pos = [current_pos[0] + t[0], current_pos[1] + t[1], current_pos[2] + t[2]]
            
            # Recurse with updated transform
            return self.visit(node.child, transform={"pos": new_pos})

        # Unwrap attribute-only nodes (MaterialNode, ProceduralTextureNode, TextureModifierNode)
        # These affect volume baking and splat training, but are invisible to real-time SDF bytecode.
        if type(node).__name__ in ("MaterialNode", "ProceduralTextureNode", "TextureModifierNode"):
            return self.visit(node.child, transform)

        # Base case: Primitive
        if isinstance(node, (SphereNode, BoxNode, CylinderNode, CapsuleNode, 
                            TorusNode, ConeNode, PlaneNode, RevolutionNode, 
                            MandelbulbNode, MengerSpongeNode, JuliaSetNode, WedgeNode)):
            return self._visit_primitive(node, transform)
            
        # Modifiers
        elif isinstance(node, (TwistModifier, BendModifier, MirrorModifier, 
                               RoundModifier, VoronoiModifier, TaperModifier)):
            # Modifiers operate on local space. If there's a pending transform,
            # we effectively have T(Module(Child)).
            # The engine executes Modifiers as: val = Child(Modifier(p))
            # Wait, engine execution order:
            #   Instruction X: Mod(Child)
            #   Run Child(p') where p' = transform_p(p)
            # 
            # If we have a Translation T pending from a parent TransformNode:
            # We want T(Mod(Child)).
            # Engine doesn't support T on Mods.
            # We must apply T to the CHILD of the mod?
            # No, T(Mod(Child)) != Mod(T(Child)) usually.
            # Example: Move(Twist(C)).
            #   p_local = p - move
            #   p_twisted = twist(p_local)
            #   dist = C(p_twisted)
            # 
            # If we bake 'move' into C, we get Twist(Move(C)) which is different.
            # 
            # CRITICAL: We CANNOT simply push translation through modifiers.
            # If a modifier is present, we must DROP the transform and warn,
            # OR logic is broken.
            # 
            # EXCEPTION: If the transform is Identity (or close), proceed.
            if transform and any(abs(x) > 1e-5 for x in transform["pos"]):
                 logger.warning(f"SDF Serializer: Dropping translation {transform['pos']} on Modifier {type(node)} (Engine limitation: cannot properties of transform non-primitive)")
            
            return self._visit_modifier(node)

        # CSG / Binary Ops
        elif isinstance(node, (UnionNode, SubtractNode, IntersectNode,
                               SmoothUnionNode, SmoothSubtractNode, SmoothIntersectNode)):
            # Transform applies to the result of the Operation: T(Op(A, B))
            # Distributive property: T(Union(A, B)) = Union(T(A), T(B))
            # Valid for min/max/smin operations.
            # So we CAN push translation to children!
            
            left = self.visit(node.child_nodes[0], transform)
            
            # Handle variadic children (binary ops are strictly pairs in engine)
            # A u B u C -> Union(A, Union(B, C))
            for i in range(1, len(node.child_nodes)):
                right = self.visit(node.child_nodes[i], transform)
                left = self._visit_binary_op(node, left, right)
            
            return left
            
        else:
            logger.warning(f"Unknown node type: {type(node)}")
            # Fallback to empty sphere
            return self._emit(TYPE_PRIMITIVE, OP_SPHERE, params=[0,0,0, 0.0])

    def _visit_primitive(self, node: PrimitiveNode, transform: Optional[Dict]) -> int:
        params = [0.0] * 8
        op_code = OP_SPHERE
        
        # Apply baked translation
        cx, cy, cz = 0.0, 0.0, 0.0
        if transform:
            cx, cy, cz = transform.get("pos", [0.0, 0.0, 0.0])
            
        # Helper to set center
        def set_center(idx_start=0):
            params[idx_start] = cx
            params[idx_start+1] = cy
            params[idx_start+2] = cz

        if isinstance(node, SphereNode):
            op_code = OP_SPHERE
            set_center(0)
            params[3] = node.radius
            
        elif isinstance(node, BoxNode):
            op_code = OP_BOX
            set_center(0)
            # node.b is half-size (extent)
            s = node.b.tolist()
            params[3] = s[0]
            params[4] = s[1]
            params[5] = s[2]
            
        elif isinstance(node, CylinderNode):
            op_code = OP_CYLINDER
            set_center(0)
            params[3] = node.radius
            params[4] = node.height
            
        elif isinstance(node, CapsuleNode):
            op_code = OP_CAPSULE
            set_center(0)
            params[3] = node.radius
            params[4] = node.half_height * 2.0 # Engine expects full height? 
            # Checking binary_format.rs... 'height'. Usually means full height.
            # math_jit Capsule uses half_height logic internally but exposed as height param?
            # math_jit: __init__(height), self.half = height/2. 
            # So pass full height.
            
        elif isinstance(node, TorusNode):
            op_code = OP_TORUS
            set_center(0)
            params[3] = node.major_r
            params[4] = node.minor_r
            
        elif isinstance(node, ConeNode):
            op_code = OP_CONE
            set_center(0)
            # params: angle? 
            # binary_format.rs says: [cx, cy, cz, angle, height]
            # math_jit ConeNode uses radius, height.
            # angle = atan(radius / height)
            import math
            angle = math.atan2(node.radius, node.height)
            params[3] = angle
            params[4] = node.height
            
        elif isinstance(node, PlaneNode):
            op_code = OP_PLANE
            # Plane: nx, ny, nz, dist.
            # Transform affects dist? 
            # T(Plane(n, d)) -> x.n + d.
            # New x' = x - t.
            # (x-t).n + d = x.n - t.n + d.
            # So new_dist = d - dot(t, n)
            n = node.n.tolist()
            d = node.distance
            
            # Baking translation
            if transform:
                tx, ty, tz = transform.get("pos", [0.0, 0.0, 0.0])
                dot = tx*n[0] + ty*n[1] + tz*n[2]
                d -= dot
            
            params[0] = n[0]
            params[1] = n[1]
            params[2] = n[2]
            params[3] = d
            
        elif isinstance(node, RevolutionNode):
            op_code = OP_REVOLUTION
            set_center(0)
            params[3] = node.offset
            # Explicit profile dims not stored in node?
            # RevolutionNode children are implicit. 
            # Engine Revolution currently just supports a generic Torus-like rev?
            # binary_format.rs: [cx, cy, cz, offset, profile_w, profile_h, axis_flag]
            # This seems to assume a Box profile?
            # If math_jit node.child is generic, we can't fully map it.
            # Fallback to Torus if child is Circle?
            # For now, just defaults.
            params[4] = 0.1 # w
            params[5] = 0.1 # h
            params[6] = 0.0 # axis (0=x, 1=y, 2=z, need mapping)
            if node.axis == 'y': params[6] = 1.0
            elif node.axis == 'z': params[6] = 2.0
            
        elif isinstance(node, MandelbulbNode):
            op_code = OP_MANDELBULB
            set_center(0)
            params[3] = node.scale
            params[4] = node.power
            params[5] = float(node.iterations)
            
        elif isinstance(node, MengerSpongeNode):
            op_code = OP_MENGER
            set_center(0)
            params[3] = node.scale
            params[4] = float(node.iterations)
            
        elif isinstance(node, JuliaSetNode):
            op_code = OP_JULIA
            set_center(0)
            params[3] = node.scale
            # c quaternion
            c = node.c.tolist()
            params[4] = c[0]
            params[5] = c[1]
            params[6] = c[2]
            params[7] = c[3]
            
        elif isinstance(node, WedgeNode):
            op_code = OP_WEDGE
            set_center(0)
            s = node.b.tolist()
            params[3] = s[0]
            params[4] = s[1]
            params[5] = s[2]
            params[6] = float(node.taper_idx) # axis
            params[7] = float(node.dir_idx)   # dir

        return self._emit(TYPE_PRIMITIVE, op_code, params=params)

    def _visit_modifier(self, node: nn.Module) -> int:
        child_idx = self.visit(node.child, transform=None)
        
        op_code = 0
        params = [0.0] * 8
        
        if isinstance(node, TwistModifier):
            op_code = OP_TWIST
            params[0] = float(node.axis_idx)
            params[1] = node.rate
            
        elif isinstance(node, BendModifier):
            op_code = OP_BEND
            params[0] = float(node.axis_idx)
            params[1] = node.k # Angle/rate
            
        elif isinstance(node, MirrorModifier):
            op_code = OP_MIRROR
            params[0] = float(node.axis_idx)
            
        elif isinstance(node, RoundModifier):
            op_code = OP_ROUND
            params[0] = node.radius
            
        elif isinstance(node, VoronoiModifier):
            op_code = OP_VORONOI
            params[0] = node.cell_size
            params[1] = node.wall_thickness
            params[2] = 1.0 if node.mode == "intersect" else 0.0
            
        elif isinstance(node, TaperModifier):
            # Engine bytecode has no Taper op (binary_format.rs ModifierOp).
            logger.warning(
                "SDF Serializer: Taper modifier not supported by engine bytecode; skipping. "
                "Shape will render without taper effect."
            )
            return child_idx

        return self._emit(TYPE_MODIFIER, op_code, op1=child_idx, params=params)

    def _visit_binary_op(self, node: nn.Module, left_idx: int, right_idx: int) -> int:
        op_code = 0
        
        if isinstance(node, UnionNode): op_code = OP_UNION
        elif isinstance(node, SubtractNode): op_code = OP_SUBTRACT
        elif isinstance(node, IntersectNode): op_code = OP_INTERSECT
        elif isinstance(node, SmoothUnionNode): op_code = OP_SMOOTH_UNION
        elif isinstance(node, SmoothSubtractNode): op_code = OP_SMOOTH_SUBTRACT
        elif isinstance(node, SmoothIntersectNode): op_code = OP_SMOOTH_INTERSECT
        
        # Smooth ops have 'k' param.
        params = [0.0] * 8
        if hasattr(node, 'k'):
             params[0] = node.k
             
        return self._emit(TYPE_BINARY_OP, op_code, op1=left_idx, op2=right_idx, params=params)

    def to_bytes(self) -> bytes:
        """Finalize and return complete bytecode buffer with header."""
        body = b"".join(self.instructions)
        
        # Bytecode Header: count(u32), min(3f), max(3f), reserved(u32) = 32 bytes
        # Must match engine/shared/src/binary_format.rs SDFBytecodeHeader
        header = struct.pack(
            "< I 3f 3f I",
            len(self.instructions),
            self.bounds_min[0], self.bounds_min[1], self.bounds_min[2],
            self.bounds_max[0], self.bounds_max[1], self.bounds_max[2],
            0
        )
        
        return header + body


def serialize_sdf_graph(sdf_graph: SdfGraph) -> bytes:
    """
    Serialize an SdfGraph to GVE Bytecode format.
    """
    builder = BytecodeBuilder()
    
    # Set bounds from graph
    if sdf_graph.bounds:
        # Flatten tuple of lists
        b_min, b_max = sdf_graph.bounds
        builder.bounds_min = b_min
        builder.bounds_max = b_max
    
    # Visit root
    builder.visit(sdf_graph.root_node, transform=None)
    
    return builder.to_bytes()
