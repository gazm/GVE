//! Type definitions for the renderer module
//!
//! Contains all public structs used by the renderer including
//! uniforms, loaded assets, and configuration.

use bytemuck::{Pod, Zeroable};
use shared::ShellVertex;

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

// ============================================================================
// Uniform Structs
// ============================================================================

/// MVP matrix uniform for mesh rendering (64 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Uniforms {
    pub mvp: [[f32; 4]; 4],
}

/// SDF raymarching uniforms (80 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SDFUniforms {
    pub inv_view_proj: [[f32; 4]; 4],   // 64 bytes
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

// ============================================================================
// Configuration
// ============================================================================

/// Renderer configuration
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub surface_format: wgpu::TextureFormat,
}
