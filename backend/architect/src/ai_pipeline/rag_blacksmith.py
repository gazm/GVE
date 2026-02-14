# backend/architect/src/ai_pipeline/rag_blacksmith.py
"""
Blacksmith-specific RAG guidance for Matter track generation.

CAD mate constraint guidance: shape selection, sizing, modifiers, assembly rules.
No orientation/rotation rules — the assembly resolver handles those.
"""

from __future__ import annotations


BLACKSMITH_GUIDANCE = """
# ADDITIONAL GUIDANCE

## Shape Selection
- **Barrel body:** Prefer one cylinder for the body. Avoid splitting into two tapered halves — they create a sharp crease at the bilge.
- **Rings/wraps:** Use torus (major_r, minor_r) for bands that wrap around a body — not thin cylinder discs.
- **Thin tapered edges:** Use Wedge primitive for blades, fins, ramps. Avoid box with sub-centimeter thickness — it vanishes during voxelization.

## Primitive Coordinate Systems
- **Box/Sphere/Wedge**: top=Y+, bottom=Y-, front=Z+, back=Z-, right=X+, left=X-.
- **Cylinder/Capsule/Cone**: height along Z. front(Z+) and back(Z-) are the circular END CAPS.
- **Torus**: ring in XY plane, hole along Z.

**CRITICAL for cylinders:** child_face is which face CONTACTS the parent, NOT which way the part extends.
- Barrel extending FORWARD from slide: parent_face="front", child_face="back" (back end cap Z- touches slide).
  WRONG: child_face="front" would flip the barrel 180 deg backward!
- Column standing UPWARD from table: parent_face="top", child_face="back" (back end cap Z- rests on table).
Rule: the child EXTENDS AWAY from its child_face. child_face="back" means it extends forward (Z+).

## Modifier Usage — Shape the Silhouette
Use modifiers to match the real silhouette. A tapered box is far better than a plain box.
- **Taper** narrows a shape along an axis. Use it for any part that gets thinner at one end:
  - Rifle/shotgun stocks: box + taper along Z (scale_min: 0.4–0.6)
  - Chair/table legs: cylinder + taper along height (scale_min: 0.3–0.5)
  - Bottle necks, vases, boat hulls: cylinder + taper
  - Combine taper + round for smooth organic profiles
  - Limits: scale_min >= 0.15 so the narrow end stays visible. Don't taper spheres.
- **Bend** curves a shape along an axis (angle 0.3–0.8 rad). Use for curved handles, boat keels.
- **Round** curved bevel on edges. Use on organic/smooth parts (radius 0.005–0.02 m).
- **Chamfer** flat 45-degree bevel on edges — like a milling chamfer cut. Use on hard-surface/mechanical
  parts: receiver flats, bolt carriers, metal brackets, tool bodies (width 0.005–0.015 m).
  - Chamfer vs Round: chamfer = flat machined bevel, round = smooth curved bevel.
  - Firearms receivers, metal boxes, industrial parts → chamfer.
  - Wooden stocks, organic shapes, handles → round.
- **Twist** spirals along an axis (rate = rad/m). Use for drill bits, decorative columns.
- **Mirror** creates symmetry across an axis plane.
- **Modifier limit:** Use 1–3 modifiers per node. Avoid stacking 4+ modifiers — causes fragmentation.

## CAD Mate Constraint Rules
- **parent_face / child_face**: always cardinal (top, bottom, front, back, left, right). No compound faces.
- **overlap**: always >= 0. Use 0 for parts that just touch. Use positive values for parts embedded inside another.
  - Barrel inside slide: overlap = 60–80% of barrel length
  - Magazine in grip: overlap = 0.003 (press fit)
  - Slide on frame: overlap = 0.002 (tight fit)
- **align**: center is default. Use cardinal directions to flush edges:
  - Grip at back of frame: align="back"
  - Trigger guard at front of frame: align="front"
  - Forearm at top of stock: align="top"

## Assembly Structure
- **Minimum size:** Smallest half-extent >= 0.008 m. For draft preview, prefer >= 0.012 m.
- **Root part:** The root part should be the largest or most central piece. All other parts attach to it or to each other — no orphaned parts.
"""


def get_blacksmith_guidance(user_prompt: str = "") -> str:
    """Return generalized Blacksmith guidance for the CAD constraint prompt.

    Args:
        user_prompt: Reserved for future semantic filtering; currently ignored.

    Returns:
        Markdown-formatted guidance block.
    """
    return BLACKSMITH_GUIDANCE.strip()
