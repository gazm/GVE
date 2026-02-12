pub mod types;
pub mod binary_format;

pub use types::{
    AssetMetadata, AssetCategory,
    MaterialSpec, ColorMode,
    MessageType, MessageHeader,
};
pub use binary_format::{
    GVEBinaryHeader, ShellVertex, Splat, GVE_MAGIC,
    // GVE 3.0 chunk-based format
    GVE3Header, ChunkHeader, GVE3_MAGIC, chunk_id, align_to_16, padding_for_16,
    // SDF types (legacy)
    SDFInstruction, SDFBytecodeHeader,
    PrimitiveOp, BinaryOp, ModifierOp, SDFInstructionType,
};
