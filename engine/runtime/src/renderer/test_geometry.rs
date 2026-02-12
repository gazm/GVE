//! Test geometry for debugging the rendering pipeline
//!
//! Contains functions to create simple test meshes and SDFs
//! for validating the renderer without loading external assets.

use wgpu::util::DeviceExt;
use shared::ShellVertex;

use crate::renderer::types::LoadedMesh;

/// Create a default test cube mesh for debugging
///
/// Generates a unit cube centered at origin with proper normals
/// for each face. Uses indexed rendering with u16 indices.
pub fn create_test_cube(device: &wgpu::Device) -> LoadedMesh {
    let vertices = vec![
        // Front face (z+)
        ShellVertex { position: [-0.5, -0.5,  0.5], normal: [0.0, 0.0, 1.0] },
        ShellVertex { position: [ 0.5, -0.5,  0.5], normal: [0.0, 0.0, 1.0] },
        ShellVertex { position: [ 0.5,  0.5,  0.5], normal: [0.0, 0.0, 1.0] },
        ShellVertex { position: [-0.5,  0.5,  0.5], normal: [0.0, 0.0, 1.0] },
        // Back face (z-)
        ShellVertex { position: [ 0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0] },
        ShellVertex { position: [-0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0] },
        ShellVertex { position: [-0.5,  0.5, -0.5], normal: [0.0, 0.0, -1.0] },
        ShellVertex { position: [ 0.5,  0.5, -0.5], normal: [0.0, 0.0, -1.0] },
        // Top face (y+)
        ShellVertex { position: [-0.5,  0.5,  0.5], normal: [0.0, 1.0, 0.0] },
        ShellVertex { position: [ 0.5,  0.5,  0.5], normal: [0.0, 1.0, 0.0] },
        ShellVertex { position: [ 0.5,  0.5, -0.5], normal: [0.0, 1.0, 0.0] },
        ShellVertex { position: [-0.5,  0.5, -0.5], normal: [0.0, 1.0, 0.0] },
        // Bottom face (y-)
        ShellVertex { position: [-0.5, -0.5, -0.5], normal: [0.0, -1.0, 0.0] },
        ShellVertex { position: [ 0.5, -0.5, -0.5], normal: [0.0, -1.0, 0.0] },
        ShellVertex { position: [ 0.5, -0.5,  0.5], normal: [0.0, -1.0, 0.0] },
        ShellVertex { position: [-0.5, -0.5,  0.5], normal: [0.0, -1.0, 0.0] },
        // Right face (x+)
        ShellVertex { position: [ 0.5, -0.5,  0.5], normal: [1.0, 0.0, 0.0] },
        ShellVertex { position: [ 0.5, -0.5, -0.5], normal: [1.0, 0.0, 0.0] },
        ShellVertex { position: [ 0.5,  0.5, -0.5], normal: [1.0, 0.0, 0.0] },
        ShellVertex { position: [ 0.5,  0.5,  0.5], normal: [1.0, 0.0, 0.0] },
        // Left face (x-)
        ShellVertex { position: [-0.5, -0.5, -0.5], normal: [-1.0, 0.0, 0.0] },
        ShellVertex { position: [-0.5, -0.5,  0.5], normal: [-1.0, 0.0, 0.0] },
        ShellVertex { position: [-0.5,  0.5,  0.5], normal: [-1.0, 0.0, 0.0] },
        ShellVertex { position: [-0.5,  0.5, -0.5], normal: [-1.0, 0.0, 0.0] },
    ];
    
    let indices: Vec<u16> = vec![
        0, 1, 2, 2, 3, 0,       // front
        4, 5, 6, 6, 7, 4,       // back
        8, 9, 10, 10, 11, 8,    // top
        12, 13, 14, 14, 15, 12, // bottom
        16, 17, 18, 18, 19, 16, // right
        20, 21, 22, 22, 23, 20, // left
    ];
    
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Default Cube Vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Default Cube Indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    
    LoadedMesh {
        vertex_buffer,
        index_buffer: Some(index_buffer),
        vertex_count: vertices.len() as u32,
        index_count: indices.len() as u32,
        use_indices: true,
        index_format: wgpu::IndexFormat::Uint16,  // Test cube uses u16 indices
    }
}



// create_csg_test_sdf REMOVED (v2.3)
