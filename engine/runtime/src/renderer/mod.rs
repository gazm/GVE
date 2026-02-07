//! Renderer module - wgpu rendering pipeline
//!
//! # Module Structure
//!
//! - `pipeline` - Core Renderer struct and rendering logic
//! - `shaders` - SDF + Mesh WGSL shader sources
//! - `shaders_extra` - Splat + Volume WGSL shader sources
//! - `types` - Type definitions (uniforms, loaded assets, config)
//! - `loaders` - Binary parsing for .gve_bin files
//! - `test_geometry` - Debug meshes and SDFs for testing
//! - `math` - Camera and projection matrix utilities
//! - `lod` - LOD transition management
//!
//! # Example
//!
//! ```ignore
//! let renderer = create_renderer(device, queue, config);
//! renderer.load_geometry(asset_id, &binary_data);
//! renderer.render_to_view(&view);
//! ```

pub mod pipeline;
pub mod shaders;
pub mod shaders_extra;
pub mod types;
pub mod loaders;
pub mod test_geometry;
pub mod math;
pub mod lod;
pub mod view_cube;
pub mod gizmos;

// Re-export public API
pub use pipeline::Renderer;
pub use types::{LoadedMesh, LoadedSDF, RenderConfig, Uniforms, SDFUniforms, GPUSDFInstruction};
pub use lod::LodManager;
pub use math::{calculate_mvp, calculate_sdf_camera};

/// Placeholder for scene data
pub struct Scene;

/// Placeholder for view data  
pub struct View;

/// Create a new renderer instance
pub fn create_renderer(device: wgpu::Device, queue: wgpu::Queue, config: RenderConfig) -> Renderer {
    Renderer::new(device, queue, config)
}

/// Execute frame rendering (scene-based API - future)
pub fn render_frame(renderer: &mut Renderer, _scene: &Scene, _view: &View) {
    renderer.render_stub();
}
