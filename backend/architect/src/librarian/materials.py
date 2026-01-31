from typing import Dict, Optional, Tuple

from generated.types import MaterialSpec, AudioProperties, ColorMode

class MaterialLibrarian:
    """Read-only material specification lookup."""
    
    def __init__(self):
        # In a real app, this might load from a JSON file or DB.
        # For prototype, we hardcode some specs.
        self._materials: Dict[str, MaterialSpec] = {
            "ASTM_A36": MaterialSpec(
                name="Steel A36",
                color_mode=ColorMode.Solid,
                base_color=[0.5, 0.5, 0.55],
                audio_properties=AudioProperties(
                    material_name="metal_heavy",
                    resonance=0.8,
                )
            ),
            "WOOD_OAK": MaterialSpec(
                name="Oak Wood",
                color_mode=ColorMode.Texture,
                base_color=[0.6, 0.4, 0.2],
                audio_properties=AudioProperties(
                    material_name="wood_solid",
                    resonance=0.4,
                )
            )
        }

    def get_material(self, spec_id: str) -> MaterialSpec:
        if spec_id not in self._materials:
            raise KeyError(f"Material spec not found: {spec_id}")
        return self._materials[spec_id]

    def get_audio_properties(self, spec_id: str) -> AudioProperties:
        mat = self.get_material(spec_id)
        return mat.audio_properties

    def resolve_impact_pair(self, spec_a: str, spec_b: str) -> Tuple[float, float]:
        """
        Calculate (combined_resonance, impact_hardness) for two materials.
        """
        mat_a = self.get_material(spec_a)
        mat_b = self.get_material(spec_b)
        
        # Simple averaging logic for prototype
        avg_resonance = (mat_a.audio_properties.resonance + mat_b.audio_properties.resonance) / 2.0
        
        return (avg_resonance, 1.0) # Hardness placeholder

_librarian = MaterialLibrarian()

get_material = _librarian.get_material
get_audio_properties = _librarian.get_audio_properties
resolve_impact_pair = _librarian.resolve_impact_pair
