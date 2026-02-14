//! Node Translate Gizmo - XYZ axes at selected node origin with pick/drag support.
//! Replaces the former AxesGizmo. Renders via SDF raymarching (cylinder + cone per axis).
//! Renders only when a node is selected.

use std::mem;
use glam::{Mat4, Vec3, Vec4};

/// Which part of the gizmo was picked (axis or center)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum GizmoPick {
    None = 0,
    AxisX = 1,
    AxisY = 2,
    AxisZ = 3,
    Center = 4,
}

/// Axis length in world units
const GIZMO_AXIS_LEN: f32 = 0.3;
/// Pick radius in pixels (for center and axis proximity)
const PICK_RADIUS_PX: f32 = 28.0;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GizmoUniforms {
    inv_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad0: f32,
    gizmo_pos: [f32; 3],
    _pad1: f32,
    viewport: [f32; 2],
    _pad2: [f32; 2],
}

pub struct NodeTranslateGizmo {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Gizmo position in world space. When visible, this is the selected node's origin.
    pub position: [f32; 3],
    /// Only render when a node is selected
    pub visible: bool,
}

impl NodeTranslateGizmo {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader_source = concat!(
            include_str!("shaders_sdf.wgsl"),
            include_str!("gizmo_sdf.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gizmo SDF Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_source)),
        });

        let uniform_size = mem::size_of::<GizmoUniforms>() as wgpu::BufferAddress;
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gizmo Uniforms"),
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gizmo Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gizmo Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gizmo Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Node Gizmo SDF Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
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
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group,
            position: [0.0, 0.0, 0.0],
            visible: false,
        }
    }

    pub fn set_position(&mut self, pos: [f32; 3]) {
        self.position = pos;
        self.visible = true;
    }

    pub fn clear(&mut self) {
        self.visible = false;
    }

    pub fn render(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        view_proj: Mat4,
        camera_pos: [f32; 3],
        viewport_width: u32,
        viewport_height: u32,
    ) {
        if !self.visible {
            return;
        }

        let inv_view_proj = view_proj.inverse();
        let uniforms = GizmoUniforms {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos,
            _pad0: 0.0,
            gizmo_pos: self.position,
            _pad1: 0.0,
            viewport: [viewport_width as f32, viewport_height as f32],
            _pad2: [0.0, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Node Gizmo SDF Pass"),
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

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }

    /// Pick which part of the gizmo is under the mouse.
    /// Requires view_proj and screen dimensions.
    pub fn pick(
        &self,
        mouse_x: f32,
        mouse_y: f32,
        width: u32,
        height: u32,
        view_proj: Mat4,
    ) -> GizmoPick {
        if !self.visible {
            return GizmoPick::None;
        }

        let inv_view_proj = view_proj.inverse();
        let ndc_x = (mouse_x / width as f32) * 2.0 - 1.0;
        let ndc_y = -((mouse_y / height as f32) * 2.0 - 1.0);

        let near_clip = inv_view_proj * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far_clip = inv_view_proj * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let _ray_origin = (near_clip / near_clip.w).truncate();
        let _ray_dir = ((far_clip / far_clip.w).truncate() - _ray_origin).normalize();

        let center = Vec3::from(self.position);

        fn project(p: Vec3, view_proj: Mat4, w: f32, h: f32) -> (f32, f32) {
            let clip = view_proj * Vec4::new(p.x, p.y, p.z, 1.0);
            if clip.w <= 0.0 {
                return (f32::NAN, f32::NAN);
            }
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            let scr_x = (ndc_x * 0.5 + 0.5) * w;
            let scr_y = (1.0 - (ndc_y * 0.5 + 0.5)) * h;
            (scr_x, scr_y)
        }

        fn dist_point_to_line(px: f32, py: f32, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
            let dx = x1 - x0;
            let dy = y1 - y0;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-6 {
                return ((px - x0).powi(2) + (py - y0).powi(2)).sqrt();
            }
            let t = ((px - x0) * dx + (py - y0) * dy) / (len * len);
            let t = t.clamp(0.0, 1.0);
            let proj_x = x0 + t * dx;
            let proj_y = y0 + t * dy;
            ((px - proj_x).powi(2) + (py - proj_y).powi(2)).sqrt()
        }

        let (cx, cy) = project(center, view_proj, width as f32, height as f32);
        let d_center = ((mouse_x - cx).powi(2) + (mouse_y - cy).powi(2)).sqrt();
        if d_center <= PICK_RADIUS_PX {
            return GizmoPick::Center;
        }

        let axes = [
            (Vec3::X * GIZMO_AXIS_LEN, GizmoPick::AxisX),
            (Vec3::Y * GIZMO_AXIS_LEN, GizmoPick::AxisY),
            (Vec3::Z * GIZMO_AXIS_LEN, GizmoPick::AxisZ),
        ];

        let mut best: Option<(f32, GizmoPick)> = None;
        for (tip, pick) in axes {
            let tip_world = center + tip;
            let (x0, y0) = project(center, view_proj, width as f32, height as f32);
            let (x1, y1) = project(tip_world, view_proj, width as f32, height as f32);
            let d = dist_point_to_line(mouse_x, mouse_y, x0, y0, x1, y1);
            if d <= PICK_RADIUS_PX {
                let t = if (x1 - x0).abs() > (y1 - y0).abs() {
                    ((mouse_x - x0) / (x1 - x0 + 1e-9)).clamp(0.0, 1.0)
                } else {
                    ((mouse_y - y0) / (y1 - y0 + 1e-9)).clamp(0.0, 1.0)
                };
                if t >= 0.0 && t <= 1.0 {
                    if best.map_or(true, |(bd, _)| d < bd) {
                        best = Some((d, pick));
                    }
                }
            }
        }
        best.map(|(_, p)| p).unwrap_or(GizmoPick::None)
    }

    /// Update position during drag.
    /// axis: 0 = free (view-plane), 1 = X, 2 = Y, 3 = Z.
    /// camera_pos is the eye position for view direction.
    pub fn drag(
        &mut self,
        mouse_x: f32,
        mouse_y: f32,
        prev_mouse_x: f32,
        prev_mouse_y: f32,
        width: u32,
        height: u32,
        view_proj: Mat4,
        camera_pos: [f32; 3],
        axis: u32,
    ) {
        let inv_view_proj = view_proj.inverse();
        let w = width as f32;
        let h = height.max(1) as f32;

        fn unproject_ray(ndc_x: f32, ndc_y: f32, inv: Mat4) -> (Vec3, Vec3) {
            let near = inv * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
            let far = inv * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
            let ro = (near / near.w).truncate();
            let rd = ((far / far.w).truncate() - ro).normalize();
            (ro, rd)
        }

        let ndc_x = |mx: f32| (mx / w) * 2.0 - 1.0;
        let ndc_y = |my: f32| -((my / h) * 2.0 - 1.0);

        let center = Vec3::from(self.position);
        let eye = Vec3::from(camera_pos);
        let view_dir = (center - eye).normalize();

        let (ro_cur, rd_cur) = unproject_ray(ndc_x(mouse_x), ndc_y(mouse_y), inv_view_proj);
        let (ro_prev, rd_prev) = unproject_ray(ndc_x(prev_mouse_x), ndc_y(prev_mouse_y), inv_view_proj);

        let delta = if axis >= 1 && axis <= 3 {
            let axis_vec = match axis {
                1 => Vec3::X,
                2 => Vec3::Y,
                3 => Vec3::Z,
                _ => Vec3::ZERO,
            };
            let plane_normal = axis_vec.cross(view_dir).normalize();
            if plane_normal.length_squared() < 1e-10 {
                return;
            }
            let hit = |ro: Vec3, rd: Vec3| -> f32 {
                let denom = rd.dot(plane_normal);
                if denom.abs() < 1e-6 {
                    return 0.0;
                }
                let t = (center - ro).dot(plane_normal) / denom;
                let p = ro + rd * t;
                (p - center).dot(axis_vec)
            };
            let s_cur = hit(ro_cur, rd_cur);
            let s_prev = hit(ro_prev, rd_prev);
            axis_vec * (s_cur - s_prev)
        } else {
            let plane_normal = view_dir;
            let hit_plane = |ro: Vec3, rd: Vec3| -> Vec3 {
                let denom = rd.dot(plane_normal);
                if denom.abs() < 1e-6 {
                    return center;
                }
                let t = (center - ro).dot(plane_normal) / denom;
                ro + rd * t
            };
            hit_plane(ro_cur, rd_cur) - hit_plane(ro_prev, rd_prev)
        };

        self.position[0] += delta.x;
        self.position[1] += delta.y;
        self.position[2] += delta.z;
    }
}
