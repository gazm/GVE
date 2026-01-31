use bytemuck::{Pod, Zeroable};

pub const GVE_MAGIC: &[u8; 4] = b"GVE1";

/// GVE Binary Header (84 bytes)
/// See docs/data/data-specifications.md Part 3.1
///
/// Layout (packed, little-endian):
/// - magic: [u8; 4]           @ 0   "GVE1"
/// - version: u32             @ 4   Format version (0x00020000 for v2.0)
/// - flags: u32               @ 8   Bit flags
/// - sdf_bytecode_offset: u64 @ 12  Offset to SDF bytecode section
/// - volume_data_offset: u64  @ 20  Offset to volume data (VDB)
/// - splat_data_offset: u64   @ 28  Offset to gaussian splat data
/// - shell_mesh_offset: u64   @ 36  Offset to shell mesh
/// - audio_patch_offset: u64  @ 44  Offset to audio patch
/// - metadata_offset: u64     @ 52  Offset to JSON metadata
/// - sdf_bytecode_size: u32   @ 60  Size of SDF bytecode in bytes
/// - volume_size: u32         @ 64  Size of volume data in bytes
/// - splat_count: u32         @ 68  Number of splats
/// - vertex_count: u32        @ 72  Number of shell vertices
/// - _padding: [u8; 8]        @ 76  Reserved for future use
/// - Total: 84 bytes
#[repr(C, packed)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GVEBinaryHeader {
    pub magic: [u8; 4],              // "GVE1"
    pub version: u32,                // Format version (0x00021000 for v2.1)
    pub flags: u32,                  // Bit flags (compression, LOD levels, etc.)
    pub sdf_bytecode_offset: u64,
    pub volume_data_offset: u64,     // (Was sdf_texture_offset) VDB/NanoVDB data
    pub splat_data_offset: u64,
    pub shell_mesh_offset: u64,
    pub audio_patch_offset: u64,
    pub metadata_offset: u64,
    pub sdf_bytecode_size: u32,
    pub volume_size: u32,            // (Was sdf_texture_size) Size od VDB data
    pub splat_count: u32,
    pub vertex_count: u32,
    pub _padding: [u8; 8],
}

// Compile-time size assertion - catches Python/Rust drift
const _: () = assert!(std::mem::size_of::<GVEBinaryHeader>() == 84);

/// Shell mesh vertex (24 bytes) - position + normal
/// Used for Early-Z pass and basic mesh rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ShellVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Gaussian splat data (48 bytes per splat)
/// See docs/data/data-specifications.md Part 3.4
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Splat {
    pub position: [f32; 3],      // 12 bytes
    pub scale: [f32; 3],         // 12 bytes (ellipsoid radii)
    pub rotation: [f32; 4],      // 16 bytes (quaternion)
    pub color_packed: u32,       // 4 bytes (RGBA8 or Oklab8)
    pub flags: u8,               // 1 byte (bit 0: color_mode)
    pub _padding: [u8; 3],       // 3 bytes (alignment)
}

// ============================================================================
// SDF Bytecode Types (See docs/data/data-specifications.md Part 3.2)
// ============================================================================

/// SDF Primitive operations
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PrimitiveOp {
    Sphere = 0x01,      // params: [cx, cy, cz, radius, 0, 0, 0, 0]
    Box = 0x02,         // params: [cx, cy, cz, sx, sy, sz, 0, 0]
    Cylinder = 0x03,    // params: [cx, cy, cz, radius, height, 0, 0, 0]
    Capsule = 0x04,     // params: [cx, cy, cz, radius, height, 0, 0, 0]
    Torus = 0x05,       // params: [cx, cy, cz, major_r, minor_r, 0, 0, 0]
    Cone = 0x06,        // params: [cx, cy, cz, angle, height, 0, 0, 0]
    Plane = 0x07,       // params: [nx, ny, nz, dist, 0, 0, 0, 0]
}

/// SDF Binary operations
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Union = 0x10,           // min(a, b)
    Subtract = 0x11,        // max(a, -b)
    Intersect = 0x12,       // max(a, b)
    SmoothUnion = 0x13,     // IQ polynomial smooth min
}

/// SDF Modifier operations
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ModifierOp {
    Twist = 0x20,       // params: [axis, rate, 0, 0]
    Bend = 0x21,        // params: [axis, angle, 0, 0]
    Mirror = 0x22,      // params: [axis, 0, 0, 0]
    Round = 0x23,       // params: [radius, 0, 0, 0]
    Elongate = 0x24,    // params: [axis, length, 0, 0]
}

/// SDF Instruction types
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SDFInstructionType {
    Primitive = 0,
    BinaryOp = 1,
    Modifier = 2,
}

/// SDF Primitive instruction (36 bytes)
/// op(1) + params(32) + padding(3)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SDFPrimitive {
    pub op: u8,             // PrimitiveOp
    pub _pad: [u8; 3],
    pub params: [f32; 8],   // 32 bytes
}

/// SDF Binary operation (8 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SDFBinaryOp {
    pub op: u8,             // BinaryOp
    pub _pad: u8,
    pub left_idx: u16,
    pub right_idx: u16,
    pub _pad2: u16,
}

/// SDF Modifier (24 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SDFModifier {
    pub op: u8,             // ModifierOp
    pub _pad: u8,
    pub child_idx: u16,
    pub params: [f32; 4],   // 16 bytes
    pub _pad2: u32,
}

/// Generic SDF instruction container (40 bytes - aligned)
/// Type determines which variant to interpret
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SDFInstruction {
    pub instr_type: u8,     // SDFInstructionType
    pub op: u8,             // Actual operation code
    pub operand1: u16,      // child_idx or left_idx
    pub operand2: u16,      // right_idx (for binary ops)
    pub _reserved: u16,
    pub params: [f32; 8],   // 32 bytes (only used by primitives/modifiers)
}

/// SDF Bytecode section header
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SDFBytecodeHeader {
    pub instruction_count: u32,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub _reserved: u32,
}
