//! Bone Gizmo - SDF overlay for skeleton visibility.
//! Draws spheres at joints and capsules between parent and child.

use std::mem;
use glam::Mat4;

use super::types::SkeletonData;

const MAX_BONES: usize = 64;
const NO_PARENT: u16 = 0xFFFF;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BoneGizmoUniforms {
    inv_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    bone_count: u32,
    _pad: u32,
    positions: [[f32; 4]; MAX_BONES],
}

pub struct BoneGizmo {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl BoneGizmo {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader_source = concat!(
            include_str!("shaders_sdf.wgsl"),
            include_str!("bone_gizmo.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bone Gizmo Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_source)),
        });

        let uniform_size = mem::size_of::<BoneGizmoUniforms>() as wgpu::BufferAddress;
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bone Gizmo Uniforms"),
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bone Gizmo Bind Group Layout"),
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
            label: Some("Bone Gizmo Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bone Gizmo Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bone Gizmo Pipeline"),
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

        Self { pipeline, uniform_buffer, bind_group }
    }

    pub fn render(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        view_proj: Mat4,
        camera_pos: [f32; 3],
        skeleton: &SkeletonData,
    ) {
        if skeleton.bones.is_empty() {
            return;
        }

        let n = skeleton.bones.len().min(MAX_BONES);
        let inv_view_proj = view_proj.inverse();

        let mut positions = [[0.0f32; 4]; MAX_BONES];
        let bone_world_positions = compute_bone_world_positions(skeleton);

        for (i, pos) in bone_world_positions.iter().take(MAX_BONES).enumerate() {
            positions[i] = [pos[0], pos[1], pos[2], skeleton.bones[i].parent_idx as f32];
        }

        let uniforms = BoneGizmoUniforms {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos,
            bone_count: n as u32,
            _pad: 0,
            positions,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Bone Gizmo Pass"),
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
}

fn compute_bone_world_positions(skeleton: &SkeletonData) -> Vec<[f32; 3]> {
    use glam::{Mat4, Quat, Vec3};
    let n = skeleton.bones.len();
    let mut world_mats: Vec<Option<Mat4>> = vec![None; n];
    for _ in 0..n {
        for i in 0..n {
            if world_mats[i].is_some() {
                continue;
            }
            let bone = &skeleton.bones[i];
            let pos = Vec3::from_array(bone.rest_pos);
            let rot = Quat::from_xyzw(
                bone.rest_rot[0],
                bone.rest_rot[1],
                bone.rest_rot[2],
                bone.rest_rot[3],
            );
            let local = Mat4::from_rotation_translation(rot, pos);
            let world = if bone.parent_idx == NO_PARENT || bone.parent_idx as usize >= n {
                local
            } else if let Some(pw) = world_mats[bone.parent_idx as usize] {
                pw * local
            } else {
                continue;
            };
            world_mats[i] = Some(world);
        }
    }
    world_mats.iter().map(|m| m.unwrap_or(Mat4::IDENTITY).w_axis.truncate().to_array()).collect()
}
