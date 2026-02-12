//! Core renderer implementation
//!
//! Contains the Renderer struct and rendering logic for both
//! mesh rasterization and SDF raymarching.

use std::collections::HashMap;
use wgpu::util::DeviceExt;
use glam::Mat4;

use crate::renderer::shaders::{MESH_SHADER, SPLAT_SHADER, VOLUME_SHADER};
use crate::renderer::types::{
    shell_vertex_layout, splat_instance_layout, DebugState, LoadedMesh, LoadedSplat, LoadedVolume,
    RenderConfig, Uniforms, ViewMode, VolumeUniforms, VOLUME_UNIFORM_BUFFER_SIZE,
};
use crate::renderer::test_geometry::create_test_cube;
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
    pub(crate) surface_format: wgpu::TextureFormat,
    
    // Shared depth buffer (SDF writes, splats test)
    pub(crate) depth_texture_view: wgpu::TextureView,
    
    // Mesh rendering
    pub(crate) mesh_pipeline: wgpu::RenderPipeline,
    pub(crate) uniform_buffer: wgpu::Buffer,
    pub(crate) uniform_bind_group: wgpu::BindGroup,
    pub(crate) loaded_meshes: HashMap<u64, LoadedMesh>,
    pub(crate) default_mesh: LoadedMesh,
    
    // SDF raymarching (Removed in v2.3)


    // Splat rendering
    pub(crate) splat_pipeline: wgpu::RenderPipeline,
    pub(crate) loaded_splats: HashMap<u64, LoadedSplat>,
    pub(crate) active_splat: Option<u64>,

    // Volume raymarching (dense grid)
    pub(crate) volume_pipeline: wgpu::RenderPipeline,
    pub(crate) volume_uniform_buffer: wgpu::Buffer,
    pub(crate) volume_bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) loaded_volumes: HashMap<u64, LoadedVolume>,
    pub(crate) active_volume: Option<u64>,
    
    // ViewMode control
    pub(crate) viewmode: ViewMode,
    
    // Camera & Gizmos
    pub(crate) camera_pos: [f32; 3],
    pub(crate) camera_yaw: f32,
    pub(crate) camera_pitch: f32,
    pub(crate) view_cube: ViewCube,
    pub(crate) axes_gizmo: AxesGizmo,
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
                view_inv: Mat4::IDENTITY.to_cols_array_2d(),
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

        // SDF PIPELINE REMOVED (v2.3)


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
                // Apply depth bias to pull splats slightly towards camera to prevent Z-fighting with SDF
                bias: wgpu::DepthBiasState {
                    constant: -50, 
                    slope_scale: -2.0,
                    clamp: 0.0,
                },
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

        let volume_uniform_buffer = {
            let initial = VolumeUniforms {
                inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                camera_pos: [0.0, 0.0, 3.0],
                _pad0: 0.0,
                bounds_min: [-1.0, -1.0, -1.0],
                _pad1: 0.0,
                bounds_max: [1.0, 1.0, 1.0],
                _pad2: 0.0,
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                use_triplanar: 0,
                _pad_tri: [0.0, 0.0, 0.0],
                triplanar_bounds_min: [-1.0, -1.0, -1.0],
                active_op_count: 0,
                triplanar_bounds_max: [1.0, 1.0, 1.0],
                _pad4: 0.0,
                ops: [crate::renderer::types::RuntimeVolumeOp {
                    op_type: 0, _pad0: [0; 3], pos: [0.0; 3], _pad1: 0,
                    params: [0.0; 8], aabb_min: [0.0; 3], _pad2: 0,
                    aabb_max: [0.0; 3], _pad3: 0,
                }; 16],
            };
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Volume Uniform Buffer"),
                size: VOLUME_UNIFORM_BUFFER_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[initial]));
            buf
        };

        let volume_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Volume Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(VOLUME_UNIFORM_BUFFER_SIZE),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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
            // Volume writes depth via frag_depth; Always compare so shader controls depth
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
            splat_pipeline,
            loaded_splats: HashMap::new(),
            active_splat: None,
            volume_pipeline,
            volume_uniform_buffer,
            volume_bind_group_layout,
            loaded_volumes: HashMap::new(),
            active_volume: None,
            viewmode: ViewMode::default(),
            camera_pos: [0.0, 0.0, 3.0],
            camera_yaw: -std::f32::consts::FRAC_PI_2,
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
    
    /// Set the active viewmode for rendering
    /// Modes: "sdf", "splat", "sdf_overlay"
    pub fn set_viewmode(&mut self, mode: &str) {
        let new_mode = ViewMode::from_str(mode);
        log::info!("🔄 ViewMode changed: {:?} -> {:?}", self.viewmode, new_mode);
        self.viewmode = new_mode;
    }
    
    /// Get the current viewmode
    pub fn get_viewmode(&self) -> ViewMode {
        self.viewmode
    }
    
    /// Get debug state snapshot
    pub fn get_debug_state(&self) -> DebugState {
        DebugState {
            view_mode: format!("{:?}", self.viewmode),
            // active_sdf: None, // Removed

            active_splat: self.active_splat,
            active_volume: self.active_volume,
            camera_pos: self.camera_pos,
            camera_yaw: self.camera_yaw,
            camera_pitch: self.camera_pitch,
        }
    }

    /// Stub for scene/view based rendering (future)
    pub fn render_stub(&mut self) {
        // Will eventually take scene/view pointers
    }

    // =========================================================================
    // Asset management functions moved to asset_management.rs
    // Remaining methods: update_camera, render_to_view, render_* passes
    // =========================================================================


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

        // Determine what passes to run based on ViewMode
        let (render_geo, render_splat) = match self.viewmode {
            ViewMode::Sdf => (true, false), // Maps to Volume now
            ViewMode::Splat => (false, true),
            ViewMode::SdfOverlay => (true, true),
        };

        // Priority 1: Render Volume (dense grid raymarching) if active
        if let Some(vol_id) = self.active_volume {
            if let Some(volume) = self.loaded_volumes.get(&vol_id) {
                let mut screen_cleared = false;

                if render_geo {
                    self.render_volume(&mut encoder, view, volume, aspect);
                    screen_cleared = true;
                }
                
                if render_splat {
                    if let Some(splat) = self.loaded_splats.get(&vol_id) {
                        if !screen_cleared {
                            self.render_splats(&mut encoder, view, splat, aspect);
                            screen_cleared = true;
                        } else {
                            self.render_splats_on_sdf(&mut encoder, view, splat, aspect);
                        }
                    }
                }

                if !screen_cleared {
                    self.clear_pass(&mut encoder, view);
                }

                self.render_gizmos(&mut encoder, view, aspect);
                self.queue.submit(std::iter::once(encoder.finish()));
                return;
            }
        }

        // Priority 2: SDF rendering REMOVED


        // Priority 3: Standalone splat rendering (no SDF)
        if let Some(splat_id) = self.active_splat {
            if let Some(splat) = self.loaded_splats.get(&splat_id) {
                self.render_splats(&mut encoder, view, splat, aspect);
                self.render_gizmos(&mut encoder, view, aspect);
                self.queue.submit(std::iter::once(encoder.finish()));
                return;
            }
        }

        // Priority 4: Fall back to mesh rendering
        self.render_meshes(&mut encoder, view, aspect);
        self.render_gizmos(&mut encoder, view, aspect);
        
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    // =========================================================================
    // Render pass functions moved to render_passes.rs and splat_passes.rs
    // =========================================================================

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
