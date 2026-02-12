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
    pub view_inv: [[f32; 4]; 4],     // Inverse view matrix for billboarding (camera->world)
    pub camera_pos: [f32; 3],        // Added camera pos
    pub viewport: [f32; 2],          // Added viewport dims
    pub _pad: [u32; 3],              // Alignment
}

// SDF Uniforms & Instructions REMOVED (v2.3)

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

// LoadedSDF REMOVED (v2.3)

/// Loaded Splat asset
pub struct LoadedSplat {
    pub instance_buffer: wgpu::Buffer,
    pub instance_count: u32,
}

/// Loaded dense volume for raymarching (optionally with triplanar textures for color)
pub struct LoadedVolume {
    pub texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub bind_group: wgpu::BindGroup,
    pub dims: [u32; 3],
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    /// When true, volume shader samples triplanar textures for base_color instead of normals
    pub has_triplanar: bool,
    pub triplanar_bounds_min: [f32; 3],
    pub triplanar_bounds_max: [f32; 3],
    pub patches: Vec<RuntimeVolumeOp>,
}

/// Runtime volume operation (matches shared::RuntimeVolumeOp)
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RuntimeVolumeOp {
    pub op_type: u32,
    pub _pad0: [u32; 3],
    pub pos: [f32; 3],
    pub _pad1: u32,
    pub params: [f32; 8],
    pub aabb_min: [f32; 3],
    pub _pad2: u32,
    pub aabb_max: [f32; 3],
    pub _pad3: u32,
}

/// Volume raymarching uniforms (includes triplanar flag and bounds when used).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct VolumeUniforms {
    pub inv_view_proj: [[f32; 4]; 4],  // 64 bytes
    pub camera_pos: [f32; 3],          // 12 bytes
    pub _pad0: f32,                     // 4 bytes
    pub bounds_min: [f32; 3],          // 12 bytes
    pub _pad1: f32,                     // 4 bytes
    pub bounds_max: [f32; 3],           // 12 bytes
    pub _pad2: f32,                     // 4 bytes
    pub view_proj: [[f32; 4]; 4],      // 64 bytes
    pub use_triplanar: u32,             // 4 bytes  (offset 176)
    pub _pad_tri: [f32; 3],             // 12 bytes (alignment padding for WGSL vec3 16-byte alignment)
    pub triplanar_bounds_min: [f32; 3], // 12 bytes (offset 192 — matches WGSL)
    pub active_op_count: u32,           // 4 bytes  (offset 204)
    pub triplanar_bounds_max: [f32; 3], // 12 bytes (offset 208 — matches WGSL)
    pub _pad4: f32,                     // 4 bytes
    pub ops: [RuntimeVolumeOp; 16],     // 16 * 96 = 1536 bytes
}

/// WGSL expects uniform block size rounded to 16 bytes.
/// Old size was 1504. New size is 224 + 1536 = 1760.
pub const VOLUME_UNIFORM_BUFFER_SIZE: u64 = 1760;

// ============================================================================
// Configuration
// ============================================================================

/// Renderer configuration
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub surface_format: wgpu::TextureFormat,
}

// ============================================================================
// ViewMode
// ============================================================================

/// Rendering viewmode for SDF + Splat hybrid rendering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewMode {
    /// SDF raymarching only (normal-based coloring)
    Sdf,
    /// Gaussian splats only
    Splat,
    /// SDF + splats rendered on top (depth-tested overlay)
    #[default]
    SdfOverlay,
}

impl ViewMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "sdf" => ViewMode::Sdf,
            "splat" => ViewMode::Splat,
            "sdf_overlay" => ViewMode::SdfOverlay,
            "sdf_textured" => ViewMode::SdfOverlay,
            _ => ViewMode::SdfOverlay,
        }
    }
}

/// Debug state snapshot for UI overlay
#[derive(Debug, Clone)]
pub struct DebugState {
    pub view_mode: String,
    // pub active_sdf: Option<u64>, // Removed

    pub active_splat: Option<u64>,
    pub active_volume: Option<u64>,
    pub camera_pos: [f32; 3],
    pub camera_yaw: f32,
    pub camera_pitch: f32,
}


