use serde::{Serialize, Deserialize};
#[cfg(feature = "schema_gen")]
use schemars::JsonSchema;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "schema_gen", derive(JsonSchema))]
pub enum AssetCategory {
    Primitive,
    Prop,
    Character,
    Environment,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "schema_gen", derive(JsonSchema))]
pub struct AssetSettings {
    pub resolution: u32,
    pub lod_count: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "schema_gen", derive(JsonSchema))]
pub struct AssetMetadata {

    pub id: String,
    pub name: String,
    pub category: AssetCategory,
    pub version: u32,
    pub tags: Vec<String>,
    pub settings: AssetSettings,
}
