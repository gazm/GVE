# backend/architect/src/ai_pipeline/rag_spatial.py
"""
Spatial awareness and connection logic for Matter track generation.

Provides category-aware spatial guidance (reference dimensions, part adjacency,
connection templates) to improve assembly coherence and part placement.

Uses CAD mate constraint vocabulary (parent_face/child_face/overlap/align).
"""

from __future__ import annotations

# =============================================================================
# World coordinate guidance (for Machinist agent — still uses world-space)
# =============================================================================

WCS_GUIDANCE = """
# WORLD COORDINATE SYSTEM (Right-Hand Rule)
- **Y is UP** (height)
- **Z is FORWARD** (length / barrel direction)
- **X is RIGHT** (width)
""".strip()


# Category keywords for lightweight classification
WEAPON_KEYWORDS = (
    "pistol", "gun", "rifle", "weapon", "revolver", "smg",
    "firearm", "handgun", "carbine", "shotgun", "magazine",
)
TOOL_KEYWORDS = (
    "hammer", "wrench", "screwdriver", "pliers", "tool",
    "drill", "saw", "chisel", "mallet",
)
FURNITURE_KEYWORDS = (
    "chair", "table", "desk", "cabinet", "shelf",
    "stool", "bench", "furniture",
)
BARREL_KEYWORDS = (
    "barrel", "cask", "keg", "drum", "vat",
)
SWORD_KEYWORDS = (
    "sword", "blade", "dagger", "knife", "machete", "saber",
)


def _detect_category(user_prompt: str) -> str:
    """Detect asset category from user prompt for targeted guidance."""
    lower = user_prompt.lower()
    if any(kw in lower for kw in WEAPON_KEYWORDS):
        return "weapon"
    if any(kw in lower for kw in SWORD_KEYWORDS):
        return "sword"
    if any(kw in lower for kw in TOOL_KEYWORDS):
        return "tool"
    if any(kw in lower for kw in FURNITURE_KEYWORDS):
        return "furniture"
    if any(kw in lower for kw in BARREL_KEYWORDS):
        return "barrel"
    return "generic"


def _get_weapon_guidance() -> str:
    return """
# ASSEMBLY PATTERNS (Pistol / Firearm)

**Reference dimensions:** Pistol total length ~0.18-0.22 m.

**Part roles and constraints:**
- frame: root part (parent: null). The central body everything attaches to.
- slide: parent_face="top", child_face="bottom", overlap=0.002. Sits on frame.
- barrel: parent_face="front" on slide, child_face="back" (cylinder end cap). overlap=0.06 (most of barrel inside slide).
- grip: parent_face="bottom" on frame, child_face="top", align="back", tilt_axis="x", tilt_degrees=15.
- magazine: parent_face="bottom" on grip, child_face="top", overlap=0.003. SEATS_IN grip.
- trigger_guard: parent_face="bottom" on frame, child_face="top", align="front".

**Connection patterns:**
- slide -> MOUNTS_ON frame (interface: rails)
- magazine -> SEATS_IN grip (interface: well)
- magazine -> REMOVABLE

**Skeleton (animated weapon):**
- Frame bone at frame.top
- Slide bone at frame.top <-> slide.bottom interface
"""


def _get_sword_guidance() -> str:
    return """
# ASSEMBLY PATTERNS (Sword / Blade)

**Reference dimensions:** Sword total ~0.8-1.5 m.

**Part roles and constraints:**
- guard: root part (parent: null). The crossguard at the junction.
- blade: parent_face="front" on guard, child_face="back". Extends forward from guard.
- handle: parent_face="back" on guard, child_face="front". Cylinder handle extends backward.
- pommel: parent_face="back" on handle, child_face="front". Sphere at the end.

**All parts share the Z axis. Assembly is linear: pommel <- handle <- guard -> blade.**
"""


def _get_barrel_guidance() -> str:
    return """
# ASSEMBLY PATTERNS (Barrel / Cask)

**Reference dimensions:** Barrel ~0.5-1.0 m tall; vertical.

**Part roles and constraints:**
- barrel_body: root part (parent: null). Cylinder body.
- bands: torus rings. Attach using parent_face="front" or "back" with overlap to position along body.
  - Top band: parent_face="front", child_face="front", overlap=0.23 (near top end).
  - Middle band: parent_face="front", child_face="front", overlap=0.50 (center of body).
  - Bottom band: parent_face="back", child_face="back", overlap=0.23 (near bottom end).
- lid: parent_face="front" on body, child_face="back". Cylinder cap on top.
"""


def _get_tool_guidance() -> str:
    return """
# ASSEMBLY PATTERNS (Tool)

**Reference dimensions:** Hammer head ~0.1 m; wrench jaw ~0.02-0.05 m.

**Part roles and constraints:**
- handle: root part (parent: null).
- head: parent_face="top" or "front" on handle, child_face="bottom" or "back". overlap=0 (touching).
- Moving parts (pliers jaw): use bone_binding for animation.
"""


def _get_furniture_guidance() -> str:
    return """
# ASSEMBLY PATTERNS (Furniture)

**Reference dimensions:** Chair seat ~0.45 m high; table top ~0.75 m.

**Part roles and constraints:**
- seat/tabletop: root part (parent: null).
- legs: parent_face="bottom" on seat, child_face="top". Use align="front"/"back" for front/back legs.
  For a cylinder leg standing downward: child_face="back" (cylinder end cap contacts seat bottom).
- backrest: parent_face="top" on seat, child_face="bottom", align="back".
- Use mirror modifier for symmetry instead of duplicating parts.
"""


def _get_generic_guidance() -> str:
    return """
# ASSEMBLY PATTERNS

**General rules:**
- The largest or most central part should be the root (parent: null).
- Parts should form a connected assembly graph — no orphaned parts.
- overlap=0 for parts that just touch. Positive overlap for parts embedded inside another (SEATS_IN).
- parent_face and child_face must be cardinal only: top, bottom, front, back, left, right.
- Detached/removable parts: attach at their logical position, mark with REMOVABLE connection.
"""


def get_spatial_guidance(user_prompt: str = "") -> str:
    """Return category-aware assembly guidance for injection into Blacksmith prompt.

    Args:
        user_prompt: User's asset description for category detection.

    Returns:
        Markdown-formatted assembly pattern block with CAD constraint vocabulary.
    """
    category = _detect_category(user_prompt)
    if category == "weapon":
        return _get_weapon_guidance().strip()
    if category == "sword":
        return _get_sword_guidance().strip()
    if category == "barrel":
        return _get_barrel_guidance().strip()
    if category == "tool":
        return _get_tool_guidance().strip()
    if category == "furniture":
        return _get_furniture_guidance().strip()
    return _get_generic_guidance().strip()
