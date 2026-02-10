# backend/architect/src/ai_pipeline/rag_blacksmith.py
"""
Blacksmith-specific RAG guidance for Matter track generation.

Injects generalized guidance (orientation, modifiers, sizing) into the
Blacksmith agent context to reduce common errors. Rules apply across object types.
"""

from __future__ import annotations


BLACKSMITH_GUIDANCE = """
# ADDITIONAL GUIDANCE

- **Orientation (CRITICAL for barrels):** Cylinders are Z-axis aligned by default. For vertical structures (barrels, columns, grips, lids) you MUST use rot: [90, 0, 0]. Without it, cylinders render as horizontal tubes—barrels look wrong. Body and lid cylinders for a vertical barrel: both need rot: [90, 0, 0].
- **Barrel body:** Prefer one cylinder for the body with rot: [90, 0, 0]. Avoid splitting into two tapered halves—they create a sharp crease at the bilge.
- **Rings/wraps:** Use torus (major_r, minor_r) for bands that wrap around a body—not thin cylinder discs. Torus ring lies in XZ plane (hole along Y); for vertical barrel bands no rotation needed.
- **Thin tapered edges:** Use Wedge primitive for blades, fins, ramps. Avoid box+taper; the tapered end can vanish.
- **Modifier limit:** Use 1–3 modifiers per node. Avoid stacking twist+voronoi+taper+bend—causes fragmentation.
- **Taper:** scale_min ≥ 0.15 so tapered end stays ≥ 8 mm. Taper is JIT/splat only (engine skips it). Don't taper spheres.
- **Bend vs twist:** Twist = spiral along axis. Bend = curve. Bend angle: keep moderate (0.3–0.8 rad).
- **Minimum size:** Smallest half-extent ≥ 0.008 m. For draft preview, prefer ≥ 0.012 m.
- **Part positions:** Keep unioned parts reasonably close; large Z offsets can cause artifacts.
"""


def get_blacksmith_guidance(user_prompt: str = "") -> str:
    """
    Return generalized Blacksmith guidance for injection into the agent prompt.

    Args:
        user_prompt: Reserved for future semantic filtering; currently ignored.

    Returns:
        Markdown-formatted guidance block.
    """
    return BLACKSMITH_GUIDANCE.strip()
