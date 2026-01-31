pub mod types;
pub mod binary_format;

pub use types::{
    AssetMetadata, AssetCategory,
    MaterialSpec, ColorMode,
    MessageType, MessageHeader,
};
pub use binary_format::{
    GVEBinaryHeader, ShellVertex, Splat, GVE_MAGIC,
    // SDF types
    SDFInstruction, SDFBytecodeHeader,
    PrimitiveOp, BinaryOp, ModifierOp, SDFInstructionType,
};
