//! Splat Render Passes for Renderer
//!
//! Gaussian splat rendering including standalone and on-SDF overlay modes.
//! Split from pipeline.rs for maintainability.

use glam::{Mat4, Vec3};
use super::pipeline::Renderer;
use super::types::{LoadedSplat, Uniforms};
use super::math::calculate_manual_camera;

impl Renderer {
    /// Write splat uniforms (shared between standalone and on-SDF modes)
    /// If ortho_bounds is Some((min, max)), uses a fixed orthographic camera for triplanar baking
    pub(crate) fn write_splat_uniforms(&self, aspect: f32, ortho_bounds: Option<([f32; 3], [f32; 3])>) {
        let (view_proj, eye) = if let Some((min_arr, max_arr)) = ortho_bounds {
            let min = Vec3::from(min_arr);
            let max = Vec3::from(max_arr);
            let center = min + (max - min) * 0.5;
            let size = (max - min).max(Vec3::splat(0.01)); // Avoid zero size
            
            // Fixed Top-Down/Front camera aligned with bounds
            // Looking at Center from +Z
            let eye = center + Vec3::new(0.0, 0.0, size.z + 2.0);
            let target = center;
            let up = Vec3::Y;
            
            let view = Mat4::look_at_rh(eye, target, up);
            
            // Ortho matching bounds size
            let w = size.x;
            let h = size.y;
            let projection = Mat4::orthographic_rh(
                -w / 2.0, w / 2.0, 
                -h / 2.0, h / 2.0, 
                0.0, size.z * 4.0 + 10.0 // Ensure we cover the depth
            );
            (projection * view, eye)
        } else {
             calculate_manual_camera(aspect, self.camera_pos, self.camera_yaw, self.camera_pitch)
        };

        let view_mat = Mat4::look_at_rh(
            eye,
            eye + if ortho_bounds.is_some() { -Vec3::Z } else {
                Vec3::new(
                    self.camera_yaw.cos() * self.camera_pitch.cos(),
                    self.camera_pitch.sin(),
                    self.camera_yaw.sin() * self.camera_pitch.cos()
                )
            },
            Vec3::Y
        );
        let view_inv = view_mat.inverse();

        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[Uniforms { 
            mvp: view_proj.to_cols_array_2d(),
            view_inv: view_inv.to_cols_array_2d(),
            camera_pos: eye.to_array(),
            viewport: [self.width as f32, self.height as f32],
            _pad: [0; 3],
        }]));
    }

    /// Render splats on top of SDF depth (Load color + depth, depth test LessEqual)
    pub(crate) fn render_splats_on_sdf(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, splat: &LoadedSplat, aspect: f32) {
        self.write_splat_uniforms(aspect, None);

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
    pub(crate) fn render_splats(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, splat: &LoadedSplat, aspect: f32) {
        self.write_splat_uniforms(aspect, None);

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

}
