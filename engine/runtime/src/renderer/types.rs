//! Type definitions for the renderer module
//!
//! Contains all public structs used by the renderer including
//! uniforms, loaded assets, and configuration.

use bytemuck::{Pod, Zeroable};

use shared::{ShellVertex, Splat};

// ============================================================================
// Vertex Layout
// ============================================================================

/// Vertex attributes for ShellVertex (position + normal)
pub const SHELL_VERTEX_ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
    0 => Float32x3,  // position
    1 => Float32x3,  // normal
];

/// Returns the vertex buffer layout for ShellVertex
pub fn shell_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<ShellVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &SHELL_VERTEX_ATTRIBS,
    }
}

/// Vertex attributes for Splat (instance data)
/// Splat struct: pos(12), scale(12), rot(16), color(4), metallic(1), roughness(1), flags(1), pad(1) = 48 bytes
/// At offset 44 we pack metallic|roughness|flags|pad as a u32 and unpack in shader.
pub const SPLAT_INSTANCE_ATTRIBS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
    0 => Float32x3,  // center (offset 0)
    1 => Float32x3,  // scale (offset 12)
    2 => Float32x4,  // rotation (offset 24)
    3 => Uint32,     // color_packed (offset 40)
    4 => Uint32,     // packed: metallic(8) | roughness(8) | flags(8) | pad(8) (offset 44)
];

/// Returns the vertex buffer layout for Splat instances
pub fn splat_instance_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Splat>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &SPLAT_INSTANCE_ATTRIBS,
    }
}

// ============================================================================
// Uniform Structs
// ============================================================================

/// MVP matrix uniform for mesh rendering (64 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Uniforms {
    pub mvp: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],         // Added specific view matrix for billboarding
    pub camera_pos: [f32; 3],        // Added camera pos
    pub viewport: [f32; 2],          // Added viewport dims
    pub _pad: [u32; 3],              // Alignment
}

/// SDF raymarching uniforms (160 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SDFUniforms {
    pub inv_view_proj: [[f32; 4]; 4],   // 64 bytes
    pub view_proj: [[f32; 4]; 4],       // 64 bytes (for frag_depth output)
    pub camera_pos: [f32; 3],            // 12 bytes
    pub time: f32,                        // 4 bytes
    pub resolution: [f32; 2],             // 8 bytes
    pub instruction_count: u32,           // 4 bytes
    pub _pad: u32,                        // 4 bytes (alignment)
}

/// GPU-friendly SDF instruction (48 bytes, 16-byte aligned)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GPUSDFInstruction {
    pub instr_type: u32,
    pub op: u32,
    pub operand1: u32,
    pub operand2: u32,
    pub params0: [f32; 4],  // 16-byte aligned for WebGL2
    pub params1: [f32; 4],
}

impl Default for GPUSDFInstruction {
    fn default() -> Self {
        Self {
            instr_type: 0,
            op: 0,
            operand1: 0,
            operand2: 0,
            params0: [0.0; 4],
            params1: [0.0; 4],
        }
    }
}

// ============================================================================
// Loaded Asset Structs
// ============================================================================

/// A loaded mesh ready for rendering
pub struct LoadedMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: Option<wgpu::Buffer>,
    pub vertex_count: u32,
    pub index_count: u32,
    pub use_indices: bool,
    pub index_format: wgpu::IndexFormat,  // Uint16 or Uint32
}

/// Loaded SDF asset with bytecode and cached bind group
pub struct LoadedSDF {
    pub instruction_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,  // Cached to avoid per-frame allocation
    pub instruction_count: u32,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

/// Loaded Splat asset
pub struct LoadedSplat {
    pub instance_buffer: wgpu::Buffer,
    pub instance_count: u32,
}

/// Loaded dense volume for raymarching
pub struct LoadedVolume {
    pub texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub bind_group: wgpu::BindGroup,
    pub dims: [u32; 3],
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

/// Volume raymarching uniforms (96 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct VolumeUniforms {
    pub inv_view_proj: [[f32; 4]; 4],  // 64 bytes
    pub camera_pos: [f32; 3],           // 12 bytes
    pub _pad0: f32,                     // 4 bytes (alignment)
    pub bounds_min: [f32; 3],           // 12 bytes
    pub _pad1: f32,                     // 4 bytes
    pub bounds_max: [f32; 3],           // 12 bytes
    pub _pad2: f32,                     // 4 bytes
}

// ============================================================================
// Configuration
// ============================================================================

/// Renderer configuration
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub surface_format: wgpu::TextureFormat,
}
