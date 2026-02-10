"""
Finish registry – named visual overrides applied on top of materials.

Finishes (e.g. black_oxide, polished) supply base_color, roughness, metallic
overrides so the AI can output "METAL_STEEL + black_oxide" instead of guessing
hex/roughness. Underlying material remains for semantics (audio, physics).
"""

from typing import Any, Dict

# finish_id -> override dict: base_color (hex), roughness, metallic (all optional)
_FINISH_REGISTRY: Dict[str, Dict[str, Any]] = {
    "black_oxide": {
        "description": "Matte black conversion coating on steel. Use for firearms, tools, machinery.",
        "base_color": "#0a0a0a",
        "roughness": 0.72,
        "metallic": 0.88,
    },
    "polished": {
        "description": "High-gloss polished surface. Use for mirrors, chrome, jewelry.",
        "roughness": 0.15,
        "metallic": None,  # keep material default
    },
    "brushed": {
        "description": "Brushed metal look, subtle anisotropy. Use for appliances, trim.",
        "roughness": 0.45,
        "metallic": None,
    },
    "painted_black": {
        "description": "Flat black paint over any substrate. Use for tactical, matte black parts.",
        "base_color": "#0c0c0c",
        "roughness": 0.7,
        "metallic": 0.0,
    },
    "painted": {
        "description": "General painted surface. Set base_color in material_config for a specific color; omit for default grey.",
        "base_color": "#6b6b6b",
        "roughness": 0.2,
        "metallic": 0.0,
    },
}


def get_finish(finish_id: str) -> Dict[str, Any] | None:
    """Return finish overrides for the given finish_id, or None if unknown."""
    return _FINISH_REGISTRY.get(finish_id)


def get_finish_registry_for_rag() -> Dict[str, Dict[str, Any]]:
    """
    Build finish registry for AI RAG context (Artist prompt).

    Returns id -> { description, base_color?, roughness?, metallic? }
    so the LLM can choose finish_id instead of guessing values.
    """
    return {
        fid: {
            "description": spec.get("description", ""),
            **{k: v for k, v in spec.items() if k != "description" and v is not None},
        }
        for fid, spec in _FINISH_REGISTRY.items()
    }
