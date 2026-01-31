use serde::{Serialize, Deserialize};
#[cfg(feature = "schema_gen")]
use schemars::JsonSchema;
use bytemuck::{Pod, Zeroable};

#[repr(u8)]
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "schema_gen", derive(JsonSchema))]
pub enum MessageType {
    // Architect → Forge
    AssetReady     = 0x01,  // Payload: .gve_bin bytes
    AssetProgress  = 0x02,  // Payload: stage (u8) + progress (f32)
    AssetInvalid   = 0x03,  // Payload: error string (UTF-8)
    VersionBump    = 0x04,  // Payload: none
    
    // Viewport Interactions
    UpdateCamera   = 0x20,  // Payload: [pos_x, pos_y, pos_z, yaw, pitch] (20 bytes)
    
    // Forge → Architect
    RequestAsset   = 0x10,
    RequestCompile = 0x11,
    CancelCompile  = 0x12,

    // Legacy/Generic
    Ping           = 0xFF,
}

unsafe impl Pod for MessageType {}
unsafe impl Zeroable for MessageType {}

#[repr(C, packed)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct MessageHeader {
    pub msg_type: MessageType,
    pub asset_id: u64,
    pub version: u32,
    pub payload_size: u32,
    pub reserved: u8,
}

