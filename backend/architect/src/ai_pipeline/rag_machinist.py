# backend/architect/src/ai_pipeline/rag_machinist.py
"""
Machinist-specific RAG guidance for Matter track generation.

Covers subtract operations (bores, slots, cutouts) and additive hardware
(hex bolts, rivets, washers). Uses connection context from A1 when available.
"""

from __future__ import annotations


MACHINIST_GUIDANCE = """
# MACHINIST GUIDANCE

## Operation Limits & Priority
**Limits per target_node_id:** Up to 2 subtract + 3 add operations per part.

**Priority order (generate ONLY what the part needs, in this order):**
1. Primary functional cavity (barrel bore, chamber, main hollow) -- 1 subtract
2. Secondary cutout (slot, port, vent, trigger guard) -- 1 subtract
3. Structural hardware (load-bearing bolts at junctions) -- 1-2 add
4. Detail hardware (decorative rivets, screws) -- 1 add

If a part is simple or decorative (e.g. band, ring, pommel), return empty add_operations.

## Subtract Operations
- **subtract**: Hard boolean cut. Best for clean through-holes and slots.
- **smooth_subtract**: Filleted cut with rounded edges (k: 0.05-0.15). Realistic CNC finish.
- **intersect**: Mask/trim -- keeps only the intersection region. Useful for clamping a bore to one region.
- **smooth_intersect**: Soft intersection with fillet (k: 0.05-0.15). Rounded trim edges.

Rules:
- Bore depth must not exceed part thickness.
- Slot alignment: match part orientation (barrel bore along Z, grip grooves along Y).

## Subtract Modifiers (apply to the subtract primitive)
Modifiers warp the subtract shape for advanced machining:

| Modifier | Effect | Example |
|----------|--------|---------|
| round | Rounded/filleted edges on the cut | "modifiers": [{"type": "round", "radius": 0.005}] |
| chamfer | Flat 45-degree bevel on bore entry | "modifiers": [{"type": "chamfer", "width": 0.003}] |
| twist | Rifling grooves (spiral cut) | "modifiers": [{"type": "twist", "axis": "z", "rate": 3.0}] |
| mirror | Symmetric cuts on both sides | "modifiers": [{"type": "mirror", "axis": "x"}] |
| taper | Countersunk holes (wide at entry) | "modifiers": [{"type": "taper", "axis": "y", "scale_min": 0.5, "scale_max": 1.0}] |
| voronoi | Honeycomb/cellular weight reduction | "modifiers": [{"type": "voronoi", "cell_size": 0.1, "wall_thickness": 0.02, "mode": "subtract"}] |

## Advanced Subtract Shapes
- **revolution** as subtract: Lathe-cut profiles for O-ring channels, complex grooves.
  Example: subtract a revolution shape to cut a circular groove around a cylinder.
  "subtract": {"type": "primitive", "shape": "revolution", "params": {"profile": {"shape": "box", "params": {"size": [0.002, 0.003, 0.002]}}, "axis": "y", "offset": 0.03}}

## Add Operations (Hardware)
Use "op": "add" with "add" field (not "subtract") to attach hardware.

| Hardware | Primitive | Key params |
|----------|-----------|------------|
| Hex bolt head | cylinder | sides=6, radius 0.003-0.008 m, height 0.002-0.005 m |
| Bolt shank | cylinder | sides=0, smaller radius than head, height 0.005-0.02 m |
| Rivet | cylinder or capsule | radius 0.0015-0.003 m, short height |
| Washer | torus | major_r=outer, minor_r=thickness |
| Round screw head | cylinder or sphere | radius 0.002-0.006 m |

**Bolt assembly:** 2-3 add ops per bolt: (1) head (cylinder sides=6), (2) shank (cylinder sides=0).
Position with add.transform.pos in world space. Shank perpendicular to surface: rot [90,0,0] for Z-up cylinder.

## Array Patterns (manual positioning)
For evenly-spaced features, compute positions explicitly:

**Radial bolt pattern** (N bolts in a circle of radius R centered at [cx, cy, cz]):
  pos_i = [cx + R*cos(i*360/N), cy, cz + R*sin(i*360/N)]  for i in 0..N-1
  Example: 4 bolts at R=0.03 around a flange at [0, 0.05, 0]:
    bolt_0: pos [0.03, 0.05, 0.0], bolt_1: pos [0.0, 0.05, 0.03],
    bolt_2: pos [-0.03, 0.05, 0.0], bolt_3: pos [0.0, 0.05, -0.03]

**Linear slot array** (N slots spaced d apart along axis):
  pos_i = base_pos + i * d * axis_dir  for i in 0..N-1
  Example: 3 vent slots 0.02 apart along Z:
    slot_0: pos [0, 0.01, -0.02], slot_1: pos [0, 0.01, 0.0], slot_2: pos [0, 0.01, 0.02]

## Connection-Driven Bolt Placement
When A1 outputs "connections", use them to place hardware at junctions:

- **FASTENED_BY** between parts: Place bolts at the contact face.
  Example: FASTENED_BY between "frame_001" and "slide_001"
  Output: 2 hex bolts at frame/slide junction
    bolt_1: {"op": "add", "target_node_id": "frame_001", "add": {"type": "primitive", "shape": "cylinder", "params": {"radius": 0.004, "height": 0.003, "sides": 6}, "transform": {"pos": [0.02, 0.05, 0.01]}}, "lod_cutoff": 1}
    bolt_2: same shape, offset +0.02 along contact face axis

- **SEATS_IN**: Parent may need a cavity (subtract) where child nests.
- **MOUNTS_ON**: Interface face may need machining (flatten, bore for pins).
"""


def get_machinist_guidance(user_prompt: str = "") -> str:
    """
    Return Machinist guidance for injection into the agent prompt.

    Args:
        user_prompt: Reserved for future semantic filtering; currently ignored.

    Returns:
        Markdown-formatted guidance block.
    """
    return MACHINIST_GUIDANCE.strip()
