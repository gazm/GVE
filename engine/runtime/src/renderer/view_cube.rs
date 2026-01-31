use wgpu::util::DeviceExt;
use glam::{Mat4, Vec3, Vec4};
use std::mem;



pub struct ViewCube {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub aspect: f32, // Viewport aspect ratio
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CubeFace {
    Right = 0,
    Left = 1,
    Top = 2,
    Bottom = 3,
    Front = 4,
    Back = 5,
}

impl ViewCube {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("View Cube Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("view_cube.wgsl"))),
        });

        // Uniforms
        let uniform_size = mem::size_of::<[f32; 16]>() as wgpu::BufferAddress;
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("View Cube Uniforms"),
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("View Cube Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT, // Uniforms used in both? Just fragment actually.
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("View Cube Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("View Cube Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("View Cube Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[], // No vertex buffers
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // No culling needed for fullscreen quad
                ..Default::default()
            },
            depth_stencil: None, // No depth test for overlay
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group,
            aspect: 1.0,
        }
    }

    pub fn render(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, rotation_matrix: Mat4, screen_width: u32, screen_height: u32) {
        // View Cube Viewport: Top-Right 100x100
        let cube_size = 120.0;
        let padding = 10.0;
        let x = screen_width as f32 - cube_size - padding;
        let y = padding; 
        
        // Pass rotation matrix directly
        let rotation = rotation_matrix; 
        
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&rotation.to_cols_array()));

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("View Cube Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        rpass.set_viewport(x, y, cube_size, cube_size, 0.0, 1.0);
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..3, 0..1); // Fullscreen triangle (3 verts)
    }

    /// Raycast against the view cube
    pub fn raycast(&self, mouse_x: f32, mouse_y: f32, screen_width: u32, screen_height: u32, view_matrix: Mat4) -> Option<CubeFace> {
        let cube_size = 120.0;
        let padding = 10.0;
        let viewport_x = screen_width as f32 - cube_size - padding;
        let viewport_y = padding;
        
        // Check bounds
        if mouse_x < viewport_x || mouse_x > viewport_x + cube_size ||
           mouse_y < viewport_y || mouse_y > viewport_y + cube_size {
            return None;
        }

        // Normalize to -1..1 in Viewport space
        let ndc_x = ((mouse_x - viewport_x) / cube_size) * 2.0 - 1.0;
        let ndc_y = -(((mouse_y - viewport_y) / cube_size) * 2.0 - 1.0); // Flip Y

        // Ray in View Space (Fixed Camera)
        // Fixed view was LookAt(0,0,3 -> 0,0,0)
        // Proj = Persp(45)
        let proj = Mat4::perspective_rh(45.0f32.to_radians(), 1.0, 0.1, 100.0);
        let view_fixed = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::Y);
        let inv_proj_view = (proj * view_fixed).inverse();
        
        let ray_clip = Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let mut ray_eye = inv_proj_view * ray_clip;
        ray_eye.z = -1.0; 
        ray_eye.w = 0.0;
        // World ray? No, we are in "World" of the cube render pass, where camera is fixed at 0,0,3
        // But the CUBE is rotated.
        // Easiest to Raycast in Object Space (Cube Space).
        
        // FixedView -> World (Inverse FixedView) = Camera Local
        // World -> Object (Inverse Model)
        
        // Matrix to Object Space:
        // InvModel * InvFixedView * InvProj
        // Model = ViewMatrix.Inverse() (CameraToWorld)
        // So InvModel = ViewMatrix (WorldToCamera)
        
        let model = view_matrix.inverse();
        let inv_model = view_matrix; 
        
        let to_object = inv_model * inv_proj_view;
        
        let ray_start_clip = Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let ray_end_clip = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        
        let mut ray_start_obj = to_object * ray_start_clip;
        ray_start_obj /= ray_start_obj.w;
        
        let mut ray_end_obj = to_object * ray_end_clip;
        ray_end_obj /= ray_end_obj.w;
        
        let ray_dir = (ray_end_obj - ray_start_obj).truncate().normalize();
        let ray_origin = ray_start_obj.truncate();

        // Ray-Box Intersection (AABB -0.5 to 0.5)
        let min = Vec3::splat(-0.5);
        let max = Vec3::splat(0.5);
        
        let (t_min, t_max) = slab_intersect(ray_origin, ray_dir, min, max);
        
        if t_min > t_max || t_max < 0.0 {
            return None;
        }

        // Determine hit point and normal
        let hit_point = ray_origin + ray_dir * t_min;
        
        // Find normal (which face)
        let epsilon = 0.001;
        if (hit_point.x - max.x).abs() < epsilon { return Some(CubeFace::Right); }
        if (hit_point.x - min.x).abs() < epsilon { return Some(CubeFace::Left); }
        if (hit_point.y - max.y).abs() < epsilon { return Some(CubeFace::Top); }
        if (hit_point.y - min.y).abs() < epsilon { return Some(CubeFace::Bottom); }
        if (hit_point.z - max.z).abs() < epsilon { return Some(CubeFace::Front); }
        if (hit_point.z - min.z).abs() < epsilon { return Some(CubeFace::Back); }
        
        None
    }
}

fn slab_intersect(ro: Vec3, rd: Vec3, min: Vec3, max: Vec3) -> (f32, f32) {
    let t1 = (min - ro) / rd;
    let t2 = (max - ro) / rd;
    
    let t_min = t1.min(t2).max_element();
    let t_max = t1.max(t2).min_element();
    
    (t_min, t_max)
}
