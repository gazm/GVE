//! Core Render Passes for Renderer
//!
//! SDF, Volume, Mesh rendering and gizmos.
//! Split from pipeline.rs for maintainability.

use glam::Mat4;
use super::pipeline::Renderer;
use super::types::{LoadedSDF, LoadedMesh, LoadedVolume, SDFUniforms, VolumeUniforms, Uniforms};
use super::math::calculate_manual_camera;

impl Renderer {
    /// Render SDF via raymarching (writes color + depth buffer)
    pub(crate) fn render_sdf(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, sdf: &LoadedSDF, aspect: f32) {
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
    pub(crate) fn render_volume(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, volume: &LoadedVolume, aspect: f32) {
        let (view_proj, eye) = calculate_manual_camera(aspect, self.camera_pos, self.camera_yaw, self.camera_pitch);
        let inv_view_proj = view_proj.inverse();

        // Update volume uniforms with bounds and triplanar flag from loaded volume
        let uniforms = VolumeUniforms {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos: eye.to_array(),
            _pad0: 0.0,
            bounds_min: volume.bounds_min,
            _pad1: 0.0,
            bounds_max: volume.bounds_max,
            _pad2: 0.0,
            view_proj: view_proj.to_cols_array_2d(),
            use_triplanar: if volume.has_triplanar { 1 } else { 0 },
            triplanar_bounds_min: volume.triplanar_bounds_min,
            _pad3: 0.0,
            triplanar_bounds_max: volume.triplanar_bounds_max,
            _pad4: 0.0,
            _pad_end: [0.0, 0.0, 0.0],
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

            render_pass.set_pipeline(&self.volume_pipeline);
            render_pass.set_bind_group(0, &volume.bind_group, &[]);
            render_pass.draw(0..3, 0..1);  // Fullscreen triangle
        }
    }

    /// Render meshes via rasterization
    pub(crate) fn render_meshes(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, aspect: f32) {
        let (view_proj, eye) = calculate_manual_camera(aspect, self.camera_pos, self.camera_yaw, self.camera_pitch);
        let mvp = view_proj.to_cols_array_2d();
        
        // Inverse view for splat billboarding (mesh shader ignores it)
        let view_mat = Mat4::look_at_rh(
            eye,
            eye + glam::Vec3::new(
                self.camera_yaw.cos() * self.camera_pitch.cos(),
                self.camera_pitch.sin(),
                self.camera_yaw.sin() * self.camera_pitch.cos()
            ),
            glam::Vec3::Y
        );
        let view_inv = view_mat.inverse();

        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[Uniforms { 
            mvp,
            view_inv: view_inv.to_cols_array_2d(),
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
    pub(crate) fn draw_mesh<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, mesh: &'a LoadedMesh) {
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

    /// Helper to render gizmos (Axes + ViewCube)
    pub(crate) fn render_gizmos(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, aspect: f32) {
        // Render View Cube (always on top)
        let (view_proj, _) = calculate_manual_camera(aspect, self.camera_pos, self.camera_yaw, self.camera_pitch);
        
        // Render Axes Gizmo (using main camera view_proj)
        self.axes_gizmo.render(&self.queue, encoder, view, view_proj);
        
        // Pure rotation view matrix (eye at origin)
        let eye = glam::Vec3::ZERO;
        let direction = glam::Vec3::new(
            self.camera_yaw.cos() * self.camera_pitch.cos(),
            self.camera_pitch.sin(),
            self.camera_yaw.sin() * self.camera_pitch.cos()
        ).normalize();
        let target = eye + direction;
        let view_cube_view_matrix = Mat4::look_at_rh(eye, target, glam::Vec3::Y);
        
        self.view_cube.render(&self.queue, encoder, view, view_cube_view_matrix, self.width, self.height);
    }

    /// Helper to clear the screen (for Splat mode or empty scenes)
    pub(crate) fn clear_pass(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Clear Pass"),
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
    }
}
