# backend/architect/src/ai_pipeline/rag_artist.py
"""
Artist-specific RAG guidance for Matter track generation.

Material-part mapping hints, finish selection by asset category,
and procedural texture recommendations.
"""

from __future__ import annotations


ARTIST_GUIDANCE = """
# ARTIST GUIDANCE

**Material-part mapping (default hints):**
- grip, handle, stock → WOOD_OAK, RUBBER_STANDARD, or METAL_STEEL
- slide, barrel, frame → METAL_STEEL, METAL_ALUMINUM, METAL_TITANIUM
- lens, optic, sight → GLASS_CLEAR, PLASTIC_POLYCARBONATE
- band, ring, fitting → METAL_STEEL, METAL_BRASS
- body, hull, casing → Match asset category (metal for weapons, wood for barrels)

**Finish selection:**
- Black/matte metal: finish_id "black_oxide"
- Polished metal: finish_id "polished"
- Painted: finish_id "painted_black" or base_color override

**Procedural textures by surface:**
- Metal: perlin (subtle variation), rust (weathered)
- Wood: wood_grain
- Stone/marble: marble
- Worn metal: rust with moderate intensity
"""


def get_artist_guidance(user_prompt: str = "") -> str:
    """
    Return Artist guidance for injection into the agent prompt.

    Args:
        user_prompt: Reserved for future semantic filtering; currently ignored.

    Returns:
        Markdown-formatted guidance block.
    """
    return ARTIST_GUIDANCE.strip()
