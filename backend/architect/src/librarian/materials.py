"""
Material Librarian - Full material specification registry.

Loads all materials from docs/data/material-database.md with physical,
audio, and visual (PBR) properties. Single source of truth for both
the compiler pipeline and AI RAG context.

Base colors are stored as sRGB [0-1] matching the hex values in the
material database doc. The compiler converts to linear RGB -> Oklab.
"""

from typing import Dict, List, Optional, Tuple

from generated.types import MaterialSpec, AudioProperties, ColorMode


def _hex_to_srgb(hex_color: str) -> list[float]:
    """Convert '#RRGGBB' hex string to sRGB [0-1] float list."""
    h = hex_color.lstrip("#")
    return [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]


# =============================================================================
# Material Database (from docs/data/material-database.md v1.1)
# =============================================================================

_MATERIAL_DB: Dict[str, MaterialSpec] = {
    # ── Metals ───────────────────────────────────────────────────────────
    "ASTM_A36": MaterialSpec(
        name="Structural Steel",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#B0B0B0"),
        metallic=0.9,
        roughness=0.4,
        audio_properties=AudioProperties(
            material_name="metal_heavy", resonance=0.8, damping=0.008
        ),
    ),
    "AMS_4911": MaterialSpec(
        name="Aluminum 6061-T6",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#D3D3D3"),
        metallic=0.95,
        roughness=0.3,
        audio_properties=AudioProperties(
            material_name="metal_light", resonance=0.85, damping=0.007
        ),
    ),
    "ASTM_B152": MaterialSpec(
        name="Copper C110",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#B87333"),
        metallic=0.98,
        roughness=0.2,
        audio_properties=AudioProperties(
            material_name="metal_bell", resonance=0.9, damping=0.01
        ),
    ),
    "ASTM_B265_Grade5": MaterialSpec(
        name="Titanium Ti-6Al-4V",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#C0C0C0"),
        metallic=0.92,
        roughness=0.35,
        audio_properties=AudioProperties(
            material_name="metal_crisp", resonance=0.88, damping=0.005
        ),
    ),
    "ASTM_B127": MaterialSpec(
        name="Brass C260",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#D4AF37"),
        metallic=0.97,
        roughness=0.25,
        audio_properties=AudioProperties(
            material_name="metal_warm", resonance=0.82, damping=0.012
        ),
    ),
    # ── Stone & Concrete ─────────────────────────────────────────────────
    "ASTM_C33": MaterialSpec(
        name="Concrete Mix",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#808080"),
        metallic=0.0,
        roughness=0.9,
        audio_properties=AudioProperties(
            material_name="stone_dull", resonance=0.15, damping=0.05
        ),
    ),
    "ASTM_C568": MaterialSpec(
        name="Limestone",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#E8D5B7"),
        metallic=0.0,
        roughness=0.7,
        audio_properties=AudioProperties(
            material_name="stone_muted", resonance=0.25, damping=0.03
        ),
    ),
    "ASTM_C503": MaterialSpec(
        name="Marble",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#F5F5DC"),
        metallic=0.0,
        roughness=0.2,
        audio_properties=AudioProperties(
            material_name="stone_clear", resonance=0.35, damping=0.025
        ),
    ),
    # ── Wood ─────────────────────────────────────────────────────────────
    "WOOD_OAK": MaterialSpec(
        name="Red Oak",
        color_mode=ColorMode.Texture,
        base_color=_hex_to_srgb("#C19A6B"),
        metallic=0.0,
        roughness=0.6,
        audio_properties=AudioProperties(
            material_name="wood_solid", resonance=0.4, damping=0.01
        ),
    ),
    "WOOD_PINE": MaterialSpec(
        name="Douglas Fir",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#E3C194"),
        metallic=0.0,
        roughness=0.7,
        audio_properties=AudioProperties(
            material_name="wood_light", resonance=0.35, damping=0.008
        ),
    ),
    "WOOD_MAPLE": MaterialSpec(
        name="Hard Maple",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#F4E7D3"),
        metallic=0.0,
        roughness=0.5,
        audio_properties=AudioProperties(
            material_name="wood_bright", resonance=0.45, damping=0.009
        ),
    ),
    # ── Plastics & Composites ────────────────────────────────────────────
    "ASTM_D4181": MaterialSpec(
        name="ABS Plastic",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#E0E0E0"),
        metallic=0.0,
        roughness=0.4,
        audio_properties=AudioProperties(
            material_name="plastic_hollow", resonance=0.2, damping=0.04
        ),
    ),
    "ASTM_D638": MaterialSpec(
        name="Polycarbonate",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#F0F0F0"),
        metallic=0.0,
        roughness=0.1,
        audio_properties=AudioProperties(
            material_name="plastic_muted", resonance=0.18, damping=0.06
        ),
    ),
    "ASTM_D7031": MaterialSpec(
        name="Carbon Fiber Composite",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#1A1A1A"),
        metallic=0.2,
        roughness=0.3,
        audio_properties=AudioProperties(
            material_name="composite_tap", resonance=0.3, damping=0.02
        ),
    ),
    # ── Glass & Ceramics ─────────────────────────────────────────────────
    "ASTM_C1036": MaterialSpec(
        name="Float Glass",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#E8F4F8"),
        metallic=0.0,
        roughness=0.01,
        audio_properties=AudioProperties(
            material_name="glass_crystal", resonance=0.95, damping=0.003
        ),
    ),
    "ASTM_C373": MaterialSpec(
        name="Ceramic Tile",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#FFFFFF"),
        metallic=0.0,
        roughness=0.15,
        audio_properties=AudioProperties(
            material_name="ceramic_sharp", resonance=0.7, damping=0.004
        ),
    ),
    # ── Rubber & Elastomers ──────────────────────────────────────────────
    "ASTM_D2000": MaterialSpec(
        name="Neoprene Rubber",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#2B2B2B"),
        metallic=0.0,
        roughness=0.8,
        audio_properties=AudioProperties(
            material_name="rubber_thud", resonance=0.05, damping=0.15
        ),
    ),
    "ASTM_D412": MaterialSpec(
        name="Silicone Rubber",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#F5F5DC"),
        metallic=0.0,
        roughness=0.6,
        audio_properties=AudioProperties(
            material_name="rubber_silent", resonance=0.02, damping=0.20
        ),
    ),
    # ── Textiles ─────────────────────────────────────────────────────────
    "TEXTILE_COTTON": MaterialSpec(
        name="Canvas Cotton",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#F5F5DC"),
        metallic=0.0,
        roughness=0.9,
        audio_properties=AudioProperties(
            material_name="fabric_soft", resonance=0.08, damping=0.08
        ),
    ),
    "TEXTILE_NYLON": MaterialSpec(
        name="Ripstop Nylon",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#EBEBEB"),
        metallic=0.0,
        roughness=0.4,
        audio_properties=AudioProperties(
            material_name="fabric_crisp", resonance=0.12, damping=0.10
        ),
    ),
    # ── Special ──────────────────────────────────────────────────────────
    "KEVLAR_49": MaterialSpec(
        name="Aramid Fiber",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#FFD700"),
        metallic=0.0,
        roughness=0.6,
        audio_properties=AudioProperties(
            material_name="composite_dense", resonance=0.25, damping=0.03
        ),
    ),
    "BALLISTIC_GEL": MaterialSpec(
        name="10% Gelatin",
        color_mode=ColorMode.Solid,
        base_color=_hex_to_srgb("#FFE4B5"),
        metallic=0.0,
        roughness=0.05,
        audio_properties=AudioProperties(
            material_name="gel_wet", resonance=0.01, damping=0.50
        ),
    ),
}

# =============================================================================
# AI-friendly aliases (RAG names -> canonical ASTM IDs)
# =============================================================================

_ALIASES: Dict[str, str] = {
    "METAL_STEEL": "ASTM_A36",
    "METAL_ALUMINUM": "AMS_4911",
    "METAL_COPPER": "ASTM_B152",
    "METAL_TITANIUM": "ASTM_B265_Grade5",
    "METAL_BRASS": "ASTM_B127",
    "METAL_IRON": "ASTM_A36",  # AI often says iron, map to steel
    "CONCRETE_STANDARD": "ASTM_C33",
    "STONE_LIMESTONE": "ASTM_C568",
    "STONE_MARBLE": "ASTM_C503",
    "STONE_GRANITE": "ASTM_C568",  # closest match
    "PLASTIC_ABS": "ASTM_D4181",
    "PLASTIC_POLYCARBONATE": "ASTM_D638",
    "CARBON_FIBER": "ASTM_D7031",
    "GLASS_CLEAR": "ASTM_C1036",
    "CERAMIC_TILE": "ASTM_C373",
    "RUBBER_NEOPRENE": "ASTM_D2000",
    "RUBBER_SILICONE": "ASTM_D412",
    "RUBBER_STANDARD": "ASTM_D2000",
}

# Category grouping for RAG display
_CATEGORIES: Dict[str, list[str]] = {
    "metals": ["ASTM_A36", "AMS_4911", "ASTM_B152", "ASTM_B265_Grade5", "ASTM_B127"],
    "stone": ["ASTM_C33", "ASTM_C568", "ASTM_C503"],
    "wood": ["WOOD_OAK", "WOOD_PINE", "WOOD_MAPLE"],
    "plastics": ["ASTM_D4181", "ASTM_D638", "ASTM_D7031"],
    "glass_ceramics": ["ASTM_C1036", "ASTM_C373"],
    "rubber": ["ASTM_D2000", "ASTM_D412"],
    "textiles": ["TEXTILE_COTTON", "TEXTILE_NYLON"],
    "special": ["KEVLAR_49", "BALLISTIC_GEL"],
}


# =============================================================================
# MaterialLibrarian
# =============================================================================

class MaterialLibrarian:
    """Read-only material specification lookup with alias support."""

    def __init__(self) -> None:
        self._materials: Dict[str, MaterialSpec] = dict(_MATERIAL_DB)
        self._aliases: Dict[str, str] = dict(_ALIASES)

    def _resolve_id(self, spec_id: str) -> str:
        """Resolve aliases to canonical ID."""
        return self._aliases.get(spec_id, spec_id)

    def get_material(self, spec_id: str) -> MaterialSpec:
        """Get material spec by canonical ID or alias."""
        canonical = self._resolve_id(spec_id)
        if canonical not in self._materials:
            raise KeyError(f"Material spec not found: {spec_id} (resolved: {canonical})")
        return self._materials[canonical]

    def get_audio_properties(self, spec_id: str) -> AudioProperties:
        """Get audio properties for a material."""
        return self.get_material(spec_id).audio_properties

    def resolve_impact_pair(self, spec_a: str, spec_b: str) -> Tuple[float, float]:
        """Calculate (combined_resonance, impact_hardness) for two materials."""
        mat_a = self.get_material(spec_a)
        mat_b = self.get_material(spec_b)
        avg_resonance = (
            mat_a.audio_properties.resonance + mat_b.audio_properties.resonance
        ) / 2.0
        return (avg_resonance, 1.0)  # Hardness placeholder

    def list_materials(self) -> List[str]:
        """List all canonical material IDs."""
        return list(self._materials.keys())

    def list_aliases(self) -> Dict[str, str]:
        """Return alias -> canonical ID mapping."""
        return dict(self._aliases)

    def get_registry_for_rag(self) -> Dict[str, Dict[str, Dict[str, object]]]:
        """
        Build grouped material registry for AI RAG context injection.

        Returns a category-grouped dict with visual properties the AI
        needs when selecting materials for DNA generation.
        """
        registry: Dict[str, Dict[str, Dict[str, object]]] = {}
        for category, ids in _CATEGORIES.items():
            group: Dict[str, Dict[str, object]] = {}
            for mat_id in ids:
                mat = self._materials[mat_id]
                # Convert sRGB floats back to hex for AI readability
                r, g, b = mat.base_color
                hex_color = f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
                group[mat_id] = {
                    "name": mat.name,
                    "color": hex_color,
                    "metallic": mat.metallic,
                    "roughness": mat.roughness,
                }
            registry[category] = group
        return registry


# =============================================================================
# Module-level convenience functions
# =============================================================================

_librarian = MaterialLibrarian()

get_material = _librarian.get_material
get_audio_properties = _librarian.get_audio_properties
resolve_impact_pair = _librarian.resolve_impact_pair
list_materials = _librarian.list_materials
get_registry_for_rag = _librarian.get_registry_for_rag
