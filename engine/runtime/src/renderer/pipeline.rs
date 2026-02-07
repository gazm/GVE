//! Core renderer implementation
//!
//! Contains the Renderer struct and rendering logic for both
//! mesh rasterization and SDF raymarching.

use std::collections::HashMap;
use wgpu::util::DeviceExt;
use glam::Mat4;

use crate::renderer::shaders::{MESH_SHADER, SDF_SHADER, SPLAT_SHADER, VOLUME_SHADER};
use crate::renderer::types::{
    shell_vertex_layout, splat_instance_layout, LoadedMesh, LoadedSDF, LoadedSplat, LoadedVolume,
    RenderConfig, SDFUniforms, Uniforms, VolumeUniforms,
};
use crate::renderer::loaders::load_geometry_from_binary;
use crate::renderer::math::calculate_manual_camera;

use crate::renderer::test_geometry::{create_test_cube, create_test_sdf};
use crate::renderer::view_cube::{ViewCube, CubeFace};
use crate::renderer::gizmos::AxesGizmo;

// Re-export public types
pub use crate::renderer::types::{LoadedMesh as LoadedMeshType, RenderConfig as RenderConfigType, Uniforms as UniformsType};

/// Depth texture format used by SDF and splat passes
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Create a depth texture view for the given dimensions
fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Buffer"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Main renderer with mesh and SDF pipelines
pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub width: u32,
    pub height: u32,
    #[allow(dead_code)]  // Stored for future pipeline recreation
    surface_format: wgpu::TextureFormat,
    
    // Shared depth buffer (SDF writes, splats test)
    depth_texture_view: wgpu::TextureView,
    
    // Mesh rendering
    mesh_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    loaded_meshes: HashMap<u64, LoadedMesh>,
    default_mesh: LoadedMesh,
    
    // SDF raymarching
    sdf_pipeline: wgpu::RenderPipeline,
    sdf_uniform_buffer: wgpu::Buffer,
    sdf_bind_group_layout: wgpu::BindGroupLayout,
    loaded_sdfs: HashMap<u64, LoadedSDF>,
    active_sdf: Option<u64>,

    // Splat rendering
    splat_pipeline: wgpu::RenderPipeline,
    loaded_splats: HashMap<u64, LoadedSplat>,
    active_splat: Option<u64>,

    // Volume raymarching (dense grid)
    volume_pipeline: wgpu::RenderPipeline,
    volume_uniform_buffer: wgpu::Buffer,
    volume_bind_group_layout: wgpu::BindGroupLayout,
    loaded_volumes: HashMap<u64, LoadedVolume>,
    active_volume: Option<u64>,
    
    // Camera & Gizmos
    camera_pos: [f32; 3],
    camera_yaw: f32,
    camera_pitch: f32,
    view_cube: ViewCube,
    axes_gizmo: AxesGizmo,
}

impl Renderer {
    /// Create a new renderer with mesh and SDF pipelines
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, config: RenderConfig) -> Self {
        let surface_format = config.surface_format;
        
        // ====================================================================
        // MESH PIPELINE SETUP
        // ====================================================================
        
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(MESH_SHADER.into()),
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[Uniforms { 
                mvp: Mat4::IDENTITY.to_cols_array_2d(),
                view: Mat4::IDENTITY.to_cols_array_2d(),
                camera_pos: [0.0; 3],
                viewport: [config.width as f32, config.height as f32],
                _pad: [0; 3],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mesh_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mesh Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mesh Bind Group"),
            layout: &mesh_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh Pipeline Layout"),
            bind_group_layouts: &[&mesh_bind_group_layout],
            push_constant_ranges: &[],
        });

        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mesh Pipeline"),
            layout: Some(&mesh_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &mesh_shader,
                entry_point: "vs_main",
                buffers: &[shell_vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &mesh_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        // Default cube mesh
        let default_mesh = create_test_cube(&device);

        // ====================================================================
        // SDF PIPELINE SETUP
        // ====================================================================
        
        let sdf_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Raymarching Shader"),
            source: wgpu::ShaderSource::Wgsl(SDF_SHADER.into()),
        });

        let sdf_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SDF Uniform Buffer"),
            contents: bytemuck::cast_slice(&[SDFUniforms {
                inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                camera_pos: [0.0, 0.0, 3.0],
                time: 0.0,
                resolution: [config.width as f32, config.height as f32],
                instruction_count: 0,
                _pad: 0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sdf_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SDF Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sdf_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SDF Pipeline Layout"),
            bind_group_layouts: &[&sdf_bind_group_layout],
            push_constant_ranges: &[],
        });

        let sdf_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SDF Pipeline"),
            layout: Some(&sdf_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sdf_shader,
                entry_point: "vs_fullscreen",
                buffers: &[],  // Fullscreen triangle - no vertex buffers
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sdf_shader,
                entry_point: "fs_sdf",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            // SDF writes depth via frag_depth; Always compare so the shader controls depth
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        // ====================================================================
        // SPLAT PIPELINE SETUP
        // ====================================================================

        let splat_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Splat Shader"),
            source: wgpu::ShaderSource::Wgsl(SPLAT_SHADER.into()),
        });

        // Reuse mesh bind group layout (binding 0: uniforms)
        let splat_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Splat Pipeline Layout"),
            bind_group_layouts: &[&mesh_bind_group_layout], // Standard Uniforms
            push_constant_ranges: &[],
        });

        let splat_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Splat Pipeline"),
            layout: Some(&splat_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &splat_shader,
                entry_point: "vs_main",
                buffers: &[splat_instance_layout()], // Instanced
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &splat_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    // Alpha blending for gaussian splats
                    // Standard accumulation: src_alpha + dst_alpha * (1 - src_alpha)
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip, // For quad generation
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for billboards
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            // Splats test against SDF depth (LessEqual) but don't write depth
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        // ====================================================================
        // VOLUME PIPELINE SETUP (Dense Grid Raymarching)
        // ====================================================================

        let volume_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Volume Raymarching Shader"),
            source: wgpu::ShaderSource::Wgsl(VOLUME_SHADER.into()),
        });

        let volume_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Volume Uniform Buffer"),
            contents: bytemuck::cast_slice(&[VolumeUniforms {
                inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                camera_pos: [0.0, 0.0, 3.0],
                _pad0: 0.0,
                bounds_min: [-1.0, -1.0, -1.0],
                _pad1: 0.0,
                bounds_max: [1.0, 1.0, 1.0],
                _pad2: 0.0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let volume_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Volume Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let volume_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Volume Pipeline Layout"),
            bind_group_layouts: &[&volume_bind_group_layout],
            push_constant_ranges: &[],
        });

        let volume_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Volume Pipeline"),
            layout: Some(&volume_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &volume_shader,
                entry_point: "vs_fullscreen",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &volume_shader,
                entry_point: "fs_volume",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
            cache: None,
        });

        log::info!("🎨 Renderer initialized with mesh + SDF + Splat + Volume pipelines");

        let depth_texture_view = create_depth_texture(&device, config.width, config.height);
        let view_cube = ViewCube::new(&device, surface_format);
        let axes_gizmo = AxesGizmo::new(&device, surface_format);

        Self {
            device,
            queue,
            width: config.width,
            height: config.height,
            surface_format,
            depth_texture_view,
            mesh_pipeline,
            uniform_buffer,
            uniform_bind_group,
            loaded_meshes: HashMap::new(),
            default_mesh,
            sdf_pipeline,
            sdf_uniform_buffer,
            sdf_bind_group_layout,
            loaded_sdfs: HashMap::new(),
            active_sdf: None,
            splat_pipeline,
            loaded_splats: HashMap::new(),
            active_splat: None,
            volume_pipeline,
            volume_uniform_buffer,
            volume_bind_group_layout,
            loaded_volumes: HashMap::new(),
            active_volume: None,
            camera_pos: [0.0, 0.0, 3.0],
            camera_yaw: -std::f32::consts::FRAC_PI_2, // Look at origin (approx)
            camera_pitch: 0.0,
            view_cube,
            axes_gizmo,
        }
    }

    /// Update viewport dimensions
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.depth_texture_view = create_depth_texture(&self.device, width, height);
    }

    /// Stub for scene/view based rendering (future)
    pub fn render_stub(&mut self) {
        // Will eventually take scene/view pointers
    }

    /// Load geometry from .gve_bin binary data
    /// Returns (success, vertex_count, index_count, error_msg) for WASM logging
    pub fn load_geometry(&mut self, asset_id: u64, data: &[u8]) -> (bool, u32, u32, Option<String>) {
        log::info!("📦 Loading geometry for asset {} ({} bytes)", asset_id, data.len());
        
        let mut stats = (0u32, 0u32); // (primary_count, secondary_count)
        
        match load_geometry_from_binary(
            &self.device,
            &self.queue,
            data,
            &self.sdf_bind_group_layout,
            &self.sdf_uniform_buffer,
            &self.volume_bind_group_layout,
            &self.volume_uniform_buffer,
        ) {
            Ok(result) => {
                // Load primitives into maps
                if let Some(mesh) = result.mesh {
                    stats = (mesh.vertex_count, mesh.index_count);
                    self.loaded_meshes.insert(asset_id, mesh);
                }
                
                if let Some(sdf) = result.sdf {
                    stats.0 = sdf.instruction_count; 
                    self.loaded_sdfs.insert(asset_id, sdf);
                }

                if let Some(splat) = result.splat {
                    if stats.0 == 0 { stats.0 = splat.instance_count; }
                    self.loaded_splats.insert(asset_id, splat);
                }

                if let Some(volume) = result.volume {
                    let vol_size = volume.dims[0] * volume.dims[1] * volume.dims[2];
                    if stats.0 == 0 { stats.0 = vol_size; }
                    self.loaded_volumes.insert(asset_id, volume);
                }

                // Determine default active mode
                // Priority: Volume > SDF (with splats as texture) > Splat-only > Mesh
                if self.loaded_volumes.contains_key(&asset_id) {
                     self.active_volume = Some(asset_id);
                     self.active_sdf = None;
                     self.active_splat = None;
                     log::info!("✅ Asset {}: Defaulting to Volume view", asset_id);
                } else if self.loaded_sdfs.contains_key(&asset_id) {
                     // SDF + splats render together (splats provide color)
                     self.active_sdf = Some(asset_id);
                     self.active_splat = None;
                     self.active_volume = None;
                     let has_splats = self.loaded_splats.contains_key(&asset_id);
                     log::info!("✅ Asset {}: Defaulting to SDF view (splat texture: {})", asset_id, has_splats);
                } else if self.loaded_splats.contains_key(&asset_id) {
                     self.active_splat = Some(asset_id);
                     self.active_sdf = None;
                     self.active_volume = None;
                     log::info!("✅ Asset {}: Defaulting to Splat-only view", asset_id);
                } else {
                     // Mesh is default fallback
                     self.active_sdf = None;
                     self.active_splat = None;
                     self.active_volume = None;
                     log::info!("✅ Asset {}: Defaulting to Mesh view", asset_id);
                }

                (true, stats.0, stats.1, None)
            }
            Err(e) => {
                log::warn!("⚠️ Failed to load geometry for asset {}: {:?}", asset_id, e);
                (false, 0, 0, Some(format!("{:?}", e)))
            }
        }
    }

    /// Load texture from pre-loaded binary data
    pub fn load_texture(&mut self, asset_id: u64, _data: &[u8]) {
        log::info!("🖼️ Loading texture for asset {}", asset_id);
        // Future: Upload to GPU texture
    }

    /// Clear all loaded meshes
    pub fn clear_meshes(&mut self) {
        self.loaded_meshes.clear();
    }

    /// Clear everything (meshes + volumes + SDFs + splats)
    pub fn clear_all(&mut self) {
        self.loaded_meshes.clear();
        self.loaded_sdfs.clear();
        self.loaded_splats.clear();
        self.loaded_volumes.clear();
        self.active_sdf = None;
        self.active_splat = None;
        self.active_volume = None;
    }

    /// Set the active SDF for rendering (None to disable SDF)
    /// Splats for the same asset ID will auto-render as texture via combined pass
    pub fn set_active_sdf(&mut self, asset_id: Option<u64>) {
        self.active_sdf = asset_id;
        if asset_id.is_some() { 
            self.active_splat = None;  // Splats render via SDF combined pass
            self.active_volume = None;
        }
    }

    /// Set the active Splat for standalone rendering (no SDF)
    pub fn set_active_splat(&mut self, asset_id: Option<u64>) {
        self.active_splat = asset_id;
        if asset_id.is_some() { 
            self.active_sdf = None;
            self.active_volume = None;
        }
    }

    /// Set the active Volume for rendering (dense grid raymarching)
    pub fn set_active_volume(&mut self, asset_id: Option<u64>) {
        self.active_volume = asset_id;
        if asset_id.is_some() {
            self.active_sdf = None;
            self.active_splat = None;
        }
    }

    /// Check if asset has mesh data
    pub fn has_mesh(&self, asset_id: u64) -> bool {
        self.loaded_meshes.contains_key(&asset_id)
    }

    /// Check if asset has SDF bytecode data
    pub fn has_sdf(&self, asset_id: u64) -> bool {
        self.loaded_sdfs.contains_key(&asset_id)
    }

    /// Check if asset has Splat data
    pub fn has_splat(&self, asset_id: u64) -> bool {
        self.loaded_splats.contains_key(&asset_id)
    }

    /// Check if asset has volume data (dense grid for raymarching)
    pub fn has_volume(&self, asset_id: u64) -> bool {
        self.loaded_volumes.contains_key(&asset_id)
    }
    
    /// Clear active SDF assignment
    pub fn clear_active_sdf(&mut self) {
        self.active_sdf = None;
    }

    /// Load a test SDF (sphere - box) for debugging
    pub fn load_test_sdf(&mut self, asset_id: u64) {
        let sdf = create_test_sdf(&self.device, &self.sdf_bind_group_layout, &self.sdf_uniform_buffer);
        self.loaded_sdfs.insert(asset_id, sdf);
        self.active_sdf = Some(asset_id);
    }

    /// Unload geometry for an asset
    pub fn unload_geometry(&mut self, asset_id: u64) {
        if self.loaded_meshes.remove(&asset_id).is_some() {
            log::info!("🗑️ Unloaded mesh for asset {}", asset_id);
        }
        if self.loaded_sdfs.remove(&asset_id).is_some() {
            if self.active_sdf == Some(asset_id) {
                self.active_sdf = None;
            }
        }
        if self.loaded_volumes.remove(&asset_id).is_some() {
            if self.active_volume == Some(asset_id) {
                self.active_volume = None;
            }
            log::info!("🗑️ Unloaded volume for asset {}", asset_id);
        }
        if self.loaded_splats.remove(&asset_id).is_some() {
            if self.active_splat == Some(asset_id) {
                self.active_splat = None;
            }
            log::info!("🗑️ Unloaded Splats for asset {}", asset_id);
        }
    }

    /// Read-only scene snapshot for JS: count (u32), then per entry asset_id (u64), type (u8: 0=mesh, 1=sdf), active (u8: 0/1).
    pub fn get_scene_snapshot(&self) -> Vec<u8> {
        const TYPE_MESH: u8 = 0;
        const TYPE_SDF: u8 = 1;
        let mut out = Vec::new();
        let mut count: u32 = 0;
        for (&id, _) in &self.loaded_meshes {
            count = count.saturating_add(1);
        }
        for (&id, _) in &self.loaded_sdfs {
            count = count.saturating_add(1);
        }
        out.extend_from_slice(&count.to_le_bytes());
        for (&id, _) in &self.loaded_meshes {
            out.extend_from_slice(&id.to_le_bytes());
            out.push(TYPE_MESH);
            out.push(0);
        }
        for (&id, _) in &self.loaded_sdfs {
            out.extend_from_slice(&id.to_le_bytes());
            out.push(TYPE_SDF);
            out.push(if self.active_sdf == Some(id) { 1 } else { 0 });
        }
        out
    }

    /// Update camera state
    pub fn update_camera(&mut self, pos: [f32; 3], yaw: f32, pitch: f32) {
        self.camera_pos = pos;
        self.camera_yaw = yaw;
        self.camera_pitch = pitch;
    }

    /// Render to view (Volume > SDF+Splat > Splat > Mesh priority)
    pub fn render_to_view(&mut self, view: &wgpu::TextureView) {
        let aspect = self.width as f32 / self.height.max(1) as f32;
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Priority 1: Render Volume (dense grid raymarching) if active
        if let Some(vol_id) = self.active_volume {
            if let Some(volume) = self.loaded_volumes.get(&vol_id) {
                self.render_volume(&mut encoder, view, volume, aspect);
                self.queue.submit(std::iter::once(encoder.finish()));
                return;
            }
        }

        // Priority 2: SDF + Splat combined rendering
        // SDF writes depth, then splats render with depth test (texture the SDF surface)
        if let Some(sdf_id) = self.active_sdf {
            if let Some(sdf) = self.loaded_sdfs.get(&sdf_id) {
                // Pass 1: SDF raymarching (clears color + depth, writes both)
                self.render_sdf(&mut encoder, view, sdf, aspect);

                // Pass 2: Splats on SDF surface (loads color + depth, depth test LessEqual)
                if let Some(splat) = self.loaded_splats.get(&sdf_id) {
                    self.render_splats_on_sdf(&mut encoder, view, splat, aspect);
                }

                self.queue.submit(std::iter::once(encoder.finish()));
                return;
            }
        }

        // Priority 3: Standalone splat rendering (no SDF)
        if let Some(splat_id) = self.active_splat {
            if let Some(splat) = self.loaded_splats.get(&splat_id) {
                self.render_splats(&mut encoder, view, splat, aspect);
                self.queue.submit(std::iter::once(encoder.finish()));
                return;
            }
        }

        // Priority 4: Fall back to mesh rendering
        self.render_meshes(&mut encoder, view, aspect);
        
        // Render View Cube (always on top)
        let (view_proj, _) = calculate_manual_camera(aspect, self.camera_pos, self.camera_yaw, self.camera_pitch);
        
        // Render Axes Gizmo (using main camera view_proj)
        self.axes_gizmo.render(&self.queue, &mut encoder, view, view_proj);
        // We pass the View Matrix (which is encoded in view_proj, but we can reconstruct or extract it) for rotation syncing.
        // Actually, ViewCube.render takes 'rotation_matrix'.
        // In render_meshes, we calc (view_proj, eye). view_proj = proj * view.
        // Let's just pass view_proj? No, the cube needs to rotate with camera view.
        // Let's reconstruct the view matrix cleanly.
        
        let mut view_cube_view_matrix = Mat4::IDENTITY;
        {
             // Pure rotation view matrix (eye at origin)
             let eye = glam::Vec3::ZERO;
             let direction = glam::Vec3::new(
                self.camera_yaw.cos() * self.camera_pitch.cos(),
                self.camera_pitch.sin(),
                self.camera_yaw.sin() * self.camera_pitch.cos()
            ).normalize();
            let target = eye + direction;
            view_cube_view_matrix = Mat4::look_at_rh(eye, target, glam::Vec3::Y);
        }
        
        self.view_cube.render(&self.queue, &mut encoder, view, view_cube_view_matrix, self.width, self.height);
        
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Render SDF via raymarching (writes color + depth buffer)
    fn render_sdf(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, sdf: &LoadedSDF, aspect: f32) {
        let (view_proj, eye) = calculate_manual_camera(aspect, self.camera_pos, self.camera_yaw, self.camera_pitch);
        let inv_view_proj = view_proj.inverse();

        // Update SDF uniforms (includes view_proj for frag_depth)
        let uniforms = SDFUniforms {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: eye.to_array(),
            time: 0.0,
            resolution: [self.width as f32, self.height as f32],
            instruction_count: sdf.instruction_count,
            _pad: 0,
        };
        self.queue.write_buffer(&self.sdf_uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SDF Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.15, b: 0.2, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.sdf_pipeline);
            // Use cached bind group from LoadedSDF (no per-frame allocation!)
            render_pass.set_bind_group(0, &sdf.bind_group, &[]);
            render_pass.draw(0..3, 0..1);  // Fullscreen triangle
        }
    }

    /// Render volume via dense grid raymarching
    fn render_volume(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, volume: &LoadedVolume, aspect: f32) {
        let (view_proj, eye) = calculate_manual_camera(aspect, self.camera_pos, self.camera_yaw, self.camera_pitch);
        let inv_view_proj = view_proj.inverse();

        // Update volume uniforms with bounds from loaded volume
        let uniforms = VolumeUniforms {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos: eye.to_array(),
            _pad0: 0.0,
            bounds_min: volume.bounds_min,
            _pad1: 0.0,
            bounds_max: volume.bounds_max,
            _pad2: 0.0,
        };
        self.queue.write_buffer(&self.volume_uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Volume Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.15, b: 0.2, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.volume_pipeline);
            render_pass.set_bind_group(0, &volume.bind_group, &[]);
            render_pass.draw(0..3, 0..1);  // Fullscreen triangle
        }
    }

    /// Render meshes via rasterization
    fn render_meshes(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, aspect: f32) {
        let (view_proj, eye) = calculate_manual_camera(aspect, self.camera_pos, self.camera_yaw, self.camera_pitch);
        let mvp = view_proj.to_cols_array_2d();
        
        // Needed for light dir calculation or others
        let view_mat = Mat4::look_at_rh(
            eye,
            eye + glam::Vec3::new(
                self.camera_yaw.cos() * self.camera_pitch.cos(),
                self.camera_pitch.sin(),
                self.camera_yaw.sin() * self.camera_pitch.cos()
            ),
            glam::Vec3::Y
        );

        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[Uniforms { 
            mvp,
            view: view_mat.to_cols_array_2d(),
            camera_pos: eye.to_array(),
            viewport: [self.width as f32, self.height as f32],
            _pad: [0; 3],
        }]));

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mesh Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.15, b: 0.2, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.mesh_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

            // Draw loaded meshes only (no default cube when empty)
            for mesh in self.loaded_meshes.values() {
                self.draw_mesh(&mut render_pass, mesh);
            }
        }
    }

    /// Draw a single mesh
    fn draw_mesh<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, mesh: &'a LoadedMesh) {
        render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        
        if mesh.use_indices {
            if let Some(ref index_buffer) = mesh.index_buffer {
                render_pass.set_index_buffer(index_buffer.slice(..), mesh.index_format);
                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        } else {
            render_pass.draw(0..mesh.vertex_count, 0..1);
        }
    }

    /// Write splat uniforms (shared between standalone and on-SDF modes)
    fn write_splat_uniforms(&self, aspect: f32) {
        let (view_proj, eye) = calculate_manual_camera(aspect, self.camera_pos, self.camera_yaw, self.camera_pitch);
        let view_mat = Mat4::look_at_rh(
            eye,
            eye + glam::Vec3::new(
                self.camera_yaw.cos() * self.camera_pitch.cos(),
                self.camera_pitch.sin(),
                self.camera_yaw.sin() * self.camera_pitch.cos()
            ),
            glam::Vec3::Y
        );
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[Uniforms { 
            mvp: view_proj.to_cols_array_2d(),
            view: view_mat.to_cols_array_2d(),
            camera_pos: eye.to_array(),
            viewport: [self.width as f32, self.height as f32],
            _pad: [0; 3],
        }]));
    }

    /// Render splats on top of SDF depth (Load color + depth, depth test LessEqual)
    fn render_splats_on_sdf(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, splat: &LoadedSplat, aspect: f32) {
        self.write_splat_uniforms(aspect);

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Splat-on-SDF Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,  // Keep SDF color underneath
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,  // Keep SDF depth
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.splat_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, splat.instance_buffer.slice(..));
            render_pass.draw(0..4, 0..splat.instance_count);
        }
    }

    /// Render gaussian splats standalone (clear + depth clear for standalone viewing)
    fn render_splats(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, splat: &LoadedSplat, aspect: f32) {
        self.write_splat_uniforms(aspect);

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Splat Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.15, b: 0.2, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),  // Clear to far plane
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.splat_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, splat.instance_buffer.slice(..));
            render_pass.draw(0..4, 0..splat.instance_count);
        }
    }

    /// Pick View Cube face
    pub fn pick_view_cube(&self, mouse_x: f32, mouse_y: f32) -> Option<CubeFace> {
        // Calculate view matrix same as render (Pure Rotation)
        let eye = glam::Vec3::ZERO;
        let direction = glam::Vec3::new(
            self.camera_yaw.cos() * self.camera_pitch.cos(),
            self.camera_pitch.sin(),
            self.camera_yaw.sin() * self.camera_pitch.cos()
        ).normalize();
        let target = eye + direction;
        let view_matrix = Mat4::look_at_rh(eye, target, glam::Vec3::Y);
        
        self.view_cube.raycast(mouse_x, mouse_y, self.width, self.height, view_matrix)

    }

    /// Toggle axes gizmo visibility
    pub fn toggle_axes(&mut self) {
        self.axes_gizmo.visible = !self.axes_gizmo.visible;
    }
}
