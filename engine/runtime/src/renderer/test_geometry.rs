//! Test geometry for debugging the rendering pipeline
//!
//! Contains functions to create simple test meshes and SDFs
//! for validating the renderer without loading external assets.

use wgpu::util::DeviceExt;
use shared::ShellVertex;

use crate::renderer::types::{GPUSDFInstruction, LoadedMesh, LoadedSDF};
use crate::renderer::loaders::MAX_SDF_INSTRUCTIONS;

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

/// Create a test SDF for debugging
///
/// Default: simple sphere at origin. This is the most basic SDF
/// and validates the entire pipeline works before adding complexity.
pub fn create_test_sdf(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
) -> LoadedSDF {
    log::info!("🔮 Creating test SDF (sphere)");
    
    // Simple sphere at origin - the most basic SDF
    // This validates the entire pipeline works before adding complexity
    let mut instructions = vec![
        // Instruction 0: Sphere at origin, radius 1.0
        // params0.xyz = center (0,0,0), params0.w = radius (1.0)
        GPUSDFInstruction {
            instr_type: 0,  // Primitive
            op: 0x01,       // Sphere
            operand1: 0,
            operand2: 0,
            params0: [0.0, 0.0, 0.0, 1.0],  // center.xyz, radius
            params1: [0.0; 4],               // unused for sphere
        },
    ];

    // Pad to MAX_SDF_INSTRUCTIONS
    while instructions.len() < MAX_SDF_INSTRUCTIONS {
        instructions.push(GPUSDFInstruction::default());
    }

    let instruction_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test SDF Instructions"),
        contents: bytemuck::cast_slice(&instructions),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Test SDF Bind Group"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: instruction_buffer.as_entire_binding(),
            },
        ],
    });

    log::info!("✅ Test SDF created and ready");

    LoadedSDF {
        instruction_buffer,
        bind_group,
        instruction_count: 1,  // Just the sphere
        bounds_min: [-1.5, -1.5, -1.5],
        bounds_max: [1.5, 1.5, 1.5],
    }
}

/// Create a CSG test SDF (sphere with box carved out)
///
/// Use this after verifying the basic sphere works to test
/// binary CSG operations in the SDF VM.
#[allow(dead_code)]
pub fn create_csg_test_sdf(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
) -> LoadedSDF {
    log::info!("🔮 Creating CSG test SDF (sphere - box)");
    
    let mut instructions = vec![
        // Sphere at origin, radius 1.0
        GPUSDFInstruction {
            instr_type: 0,  // Primitive
            op: 0x01,       // Sphere
            operand1: 0,
            operand2: 0,
            params0: [0.0, 0.0, 0.0, 1.0],
            params1: [0.0; 4],
        },
        // Box at origin, size 0.6 x 0.6 x 0.6
        GPUSDFInstruction {
            instr_type: 0,  // Primitive
            op: 0x02,       // Box
            operand1: 0,
            operand2: 0,
            params0: [0.0, 0.0, 0.0, 0.6],  // center.xyz, size.x
            params1: [0.6, 0.6, 0.0, 0.0],   // size.y, size.z
        },
        // Subtract: sphere - box
        GPUSDFInstruction {
            instr_type: 1,  // Binary op
            op: 0x11,       // Subtract
            operand1: 0,
            operand2: 0,
            params0: [0.0; 4],
            params1: [0.0; 4],
        },
    ];

    while instructions.len() < MAX_SDF_INSTRUCTIONS {
        instructions.push(GPUSDFInstruction::default());
    }

    let instruction_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("CSG Test SDF Instructions"),
        contents: bytemuck::cast_slice(&instructions),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("CSG Test SDF Bind Group"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: instruction_buffer.as_entire_binding(),
            },
        ],
    });

    LoadedSDF {
        instruction_buffer,
        bind_group,
        instruction_count: 3,
        bounds_min: [-1.5, -1.5, -1.5],
        bounds_max: [1.5, 1.5, 1.5],
    }
}
