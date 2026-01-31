use serde::{Serialize, Deserialize};
#[cfg(feature = "schema_gen")]
use schemars::JsonSchema;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "schema_gen", derive(JsonSchema))]
pub enum ColorMode {
    Solid,
    Gradient,
    Texture,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "schema_gen", derive(JsonSchema))]
pub struct AudioProperties {
    pub material_name: String,
    pub resonance: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "schema_gen", derive(JsonSchema))]
pub struct MaterialSpec {

    pub name: String,
    pub color_mode: ColorMode,
    pub base_color: [f32; 3],
    pub audio_properties: AudioProperties,
}
