//! Binary asset parsing for the renderer
//!
//! Contains functions for parsing .gve_bin binary data into GPU-ready assets.
//! Handles SDF bytecode and shell mesh extraction from the binary format.

use wgpu::util::DeviceExt;
use shared::{GVEBinaryHeader, ShellVertex, GVE_MAGIC};

use crate::renderer::types::{GPUSDFInstruction, LoadedMesh, LoadedSDF};

// ============================================================================
// Binary Parsing Constants
// ============================================================================

/// SDFBytecodeHeader size: instruction_count(4) + bounds_min(12) + bounds_max(12) + reserved(4)
const SDF_HEADER_SIZE: usize = 32;

/// SDFInstruction size: type(1) + op(1) + operand1(2) + operand2(2) + reserved(2) + params(32)
const SDF_INSTR_SIZE: usize = 40;

/// Maximum instructions per SDF (WebGL2 uniform buffer limit)
pub const MAX_SDF_INSTRUCTIONS: usize = 16;

// ============================================================================
// Geometry Loading
// ============================================================================

/// Load geometry from .gve_bin binary data
///
/// Parses the binary header and extracts either SDF bytecode (priority)
/// or shell mesh data for GPU upload.
///
/// # Arguments
/// * `device` - wgpu device for buffer creation
/// * `data` - Raw .gve_bin binary data
/// * `sdf_bind_group_layout` - Layout for SDF bind groups
/// * `sdf_uniform_buffer` - Shared uniform buffer for SDF rendering
///
/// # Returns
/// `GeometryLoadResult::SDF` if SDF bytecode present, otherwise `GeometryLoadResult::Mesh`
pub fn load_geometry_from_binary(
    device: &wgpu::Device,
    data: &[u8],
    sdf_bind_group_layout: &wgpu::BindGroupLayout,
    sdf_uniform_buffer: &wgpu::Buffer,
) -> Result<GeometryLoadResult, GeometryLoadError> {
    // Validate minimum size for header
    if data.len() < std::mem::size_of::<GVEBinaryHeader>() {
        return Err(GeometryLoadError::DataTooSmall);
    }

    // Parse header
    let header: &GVEBinaryHeader = bytemuck::from_bytes(&data[..std::mem::size_of::<GVEBinaryHeader>()]);
    
    // Validate magic
    if &header.magic != GVE_MAGIC {
        return Err(GeometryLoadError::InvalidMagic);
    }

    // Copy from packed struct to avoid alignment issues
    let version = header.version;
    let vertex_count = header.vertex_count;
    let shell_mesh_offset = header.shell_mesh_offset;
    let volume_data_offset = header.volume_data_offset;
    let sdf_bytecode_offset = header.sdf_bytecode_offset;
    let sdf_bytecode_size = header.sdf_bytecode_size;
    let volume_size = header.volume_size;
    
    log::info!("📄 GVE Header: version={:#x}, vertices={}, shell_offset={}, sdf_offset={}, vol_offset={}, vol_size={}", 
        version, vertex_count, shell_mesh_offset, sdf_bytecode_offset, volume_data_offset, volume_size);

    // Parse SDF bytecode if present (priority over shell mesh)
    if sdf_bytecode_offset > 0 && sdf_bytecode_size > 0 {
        if let Some(sdf) = parse_sdf_bytecode(
            device,
            data,
            sdf_bytecode_offset as usize,
            sdf_bytecode_size as usize,
            sdf_bind_group_layout,
            sdf_uniform_buffer,
        ) {
            log::info!("✅ Loaded SDF bytecode");
            return Ok(GeometryLoadResult::SDF(sdf));
        }
    }

    // Check for Sparse Volume Data (V2.1)
    // We load this IN ADDITION to Shell (Volume provides high-fidelity physics/rendering, Shell provides Early-Z)
    // For now, we return it as a generic GeometryLoadResult variant or attach to SDF?
    // Let's create a new Result variant: Volume
    
    if volume_data_offset > 0 && volume_size > 0 {
         if let Some(vol) = parse_volume_data(device, data, volume_data_offset as usize, volume_size as usize) {
             log::info!("✅ Loaded Volume Data ({} bytes)", volume_size);
             // TODO: Return generic volume result. For now, rely on Shell fallback for rendering.
         }
    }

    // Fall back to shell mesh if no SDF
    if shell_mesh_offset > 0 && vertex_count > 0 {
        if let Some(mesh) = parse_shell_mesh(device, data, header) {
            log::info!("✅ Loaded shell mesh");
            return Ok(GeometryLoadResult::Mesh(mesh));
        }
    }

    Err(GeometryLoadError::NoGeometry {
        header_vertex_count: vertex_count,
        shell_mesh_offset,
        sdf_bytecode_offset,
        sdf_bytecode_size,
    })
}

/// Result of loading geometry - either SDF, Mesh, or Volume
pub enum GeometryLoadResult {
    SDF(LoadedSDF),
    Mesh(LoadedMesh),
    // Volume(LoadedVolume),
}

/// Errors that can occur during geometry loading
#[derive(Debug)]
pub enum GeometryLoadError {
    DataTooSmall,
    InvalidMagic,
    NoGeometry {
        header_vertex_count: u32,
        shell_mesh_offset: u64,
        sdf_bytecode_offset: u64,
        sdf_bytecode_size: u32,
    },
}

// ============================================================================
// Volume Parsing (Stub)
// ============================================================================

fn parse_volume_data(
    device: &wgpu::Device,
    data: &[u8],
    offset: usize,
    size: usize,
) -> Option<wgpu::Buffer> {
    if offset + size > data.len() {
        log::warn!("⚠️ Volume data offset out of bounds");
        return None;
    }
    
    let vol_data = &data[offset..offset + size];
    
    // Create generic storage buffer for VDB data
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("VDB Volume Data"),
        contents: vol_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    
    Some(buffer)
}

// ============================================================================
// SDF Parsing
// ============================================================================

/// Parse SDF bytecode section from binary data
///
/// Uses manual byte reading to avoid alignment issues with unaligned input.
fn parse_sdf_bytecode(
    device: &wgpu::Device,
    data: &[u8],
    offset: usize,
    size: usize,
    bind_group_layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
) -> Option<LoadedSDF> {
    if offset + size > data.len() {
        log::warn!("⚠️ SDF bytecode offset out of bounds");
        return None;
    }

    let sdf_data = &data[offset..offset + size];
    
    if sdf_data.len() < SDF_HEADER_SIZE {
        log::warn!("⚠️ SDF data too small for header");
        return None;
    }

    // Read header manually (avoids alignment issues)
    let instruction_count = u32::from_le_bytes([sdf_data[0], sdf_data[1], sdf_data[2], sdf_data[3]]);
    let bounds_min = [
        f32::from_le_bytes([sdf_data[4], sdf_data[5], sdf_data[6], sdf_data[7]]),
        f32::from_le_bytes([sdf_data[8], sdf_data[9], sdf_data[10], sdf_data[11]]),
        f32::from_le_bytes([sdf_data[12], sdf_data[13], sdf_data[14], sdf_data[15]]),
    ];
    let bounds_max = [
        f32::from_le_bytes([sdf_data[16], sdf_data[17], sdf_data[18], sdf_data[19]]),
        f32::from_le_bytes([sdf_data[20], sdf_data[21], sdf_data[22], sdf_data[23]]),
        f32::from_le_bytes([sdf_data[24], sdf_data[25], sdf_data[26], sdf_data[27]]),
    ];
    
    log::info!("🔮 SDF: {} instructions, bounds: {:?} to {:?}", 
        instruction_count, bounds_min, bounds_max);

    let expected_size = SDF_HEADER_SIZE + (instruction_count as usize * SDF_INSTR_SIZE);
    
    if sdf_data.len() < expected_size {
        log::warn!("⚠️ SDF data too small for {} instructions", instruction_count);
        return None;
    }

    // Convert to GPU format
    let gpu_instructions = parse_sdf_instructions(sdf_data, instruction_count);

    // Create GPU uniform buffer
    let instruction_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SDF Instruction Buffer"),
        contents: bytemuck::cast_slice(&gpu_instructions),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create cached bind group (avoids per-frame allocation)
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SDF Bind Group"),
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

    Some(LoadedSDF {
        instruction_buffer,
        bind_group,
        instruction_count,
        bounds_min,
        bounds_max,
    })
}

/// Parse SDF instructions from raw bytes into GPU format
fn parse_sdf_instructions(sdf_data: &[u8], instruction_count: u32) -> Vec<GPUSDFInstruction> {
    let mut gpu_instructions = Vec::with_capacity(MAX_SDF_INSTRUCTIONS);
    
    for i in 0..instruction_count.min(MAX_SDF_INSTRUCTIONS as u32) as usize {
        let base = SDF_HEADER_SIZE + i * SDF_INSTR_SIZE;
        
        let instr_type = sdf_data[base] as u32;
        let op = sdf_data[base + 1] as u32;
        
        // Read 8 float params starting at byte 8
        let mut params = [0.0f32; 8];
        for j in 0..8 {
            let p = base + 8 + j * 4;
            params[j] = f32::from_le_bytes([sdf_data[p], sdf_data[p+1], sdf_data[p+2], sdf_data[p+3]]);
        }
        
        gpu_instructions.push(GPUSDFInstruction {
            instr_type,
            op,
            operand1: 0,
            operand2: 0,
            params0: [params[0], params[1], params[2], params[3]],
            params1: [params[4], params[5], params[6], params[7]],
        });
    }

    // Pad to MAX_SDF_INSTRUCTIONS (WebGL2 uniform buffer size requirement)
    while gpu_instructions.len() < MAX_SDF_INSTRUCTIONS {
        gpu_instructions.push(GPUSDFInstruction::default());
    }

    gpu_instructions
}

// ============================================================================
// Shell Mesh Parsing
// ============================================================================

/// Parse shell mesh section from binary data
fn parse_shell_mesh(device: &wgpu::Device, data: &[u8], header: &GVEBinaryHeader) -> Option<LoadedMesh> {
    let offset = header.shell_mesh_offset as usize;
    
    if offset + 8 > data.len() {
        log::warn!("⚠️ Shell mesh offset out of bounds");
        return None;
    }

    // Read vertex_count and index_count
    let vertex_count = u32::from_le_bytes(data[offset..offset+4].try_into().ok()?);
    let index_count = u32::from_le_bytes(data[offset+4..offset+8].try_into().ok()?);
    
    log::info!("📐 Shell mesh: {} vertices, {} indices", vertex_count, index_count);

    // Calculate sizes
    let vertex_size = std::mem::size_of::<ShellVertex>();
    let vertices_start = offset + 8;
    let vertices_end = vertices_start + (vertex_count as usize * vertex_size);
    
    if vertices_end > data.len() {
        log::warn!("⚠️ Vertex data out of bounds");
        return None;
    }

    // Create vertex buffer
    let vertex_data = &data[vertices_start..vertices_end];
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Loaded Mesh Vertices"),
        contents: vertex_data,
        usage: wgpu::BufferUsages::VERTEX,
    });

    // Parse indices if present
    // Note: Python shell_gen.py always writes u32 indices ('I' format)
    let (index_buffer, use_indices, index_format) = if index_count > 0 {
        let indices_start = vertices_end;
        // Always use 4-byte indices to match Python's struct.pack('I')
        let index_size = 4;
        let index_format = wgpu::IndexFormat::Uint32;
        let indices_end = indices_start + (index_count as usize * index_size);
        
        log::info!("📐 Parsing indices: start={}, end={}, count={}, size={}", 
            indices_start, indices_end, index_count, index_size);
        
        if indices_end <= data.len() {
            let index_data = &data[indices_start..indices_end];
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Loaded Mesh Indices"),
                contents: index_data,
                usage: wgpu::BufferUsages::INDEX,
            });
            (Some(buffer), true, index_format)
        } else {
            log::warn!("⚠️ Index data out of bounds (need {} bytes, have {}), using vertices only", 
                indices_end, data.len());
            (None, false, wgpu::IndexFormat::Uint32)
        }
    } else {
        (None, false, wgpu::IndexFormat::Uint32)
    };

    Some(LoadedMesh {
        vertex_buffer,
        index_buffer,
        vertex_count,
        index_count,
        use_indices,
        index_format,
    })
}
