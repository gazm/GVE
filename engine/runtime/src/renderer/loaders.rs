//! Binary asset parsing for the renderer
//!
//! Contains functions for parsing .gve_bin binary data into GPU-ready assets.
//! Handles dense volumes, SDF bytecode, and shell mesh extraction.

use wgpu::util::DeviceExt;
use shared::{GVEBinaryHeader, ShellVertex, GVE_MAGIC};

use crate::renderer::types::{GPUSDFInstruction, LoadedMesh, LoadedSDF, LoadedSplat, LoadedVolume};

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
/// Parses the binary header and extracts all available representations:
/// - Dense volume (for GPU raymarching)
/// - SDF bytecode (legacy CSG raymarching)
/// - Shell mesh (rasterized view)
/// - Splats (gaussian splatting)
pub fn load_geometry_from_binary(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[u8],
    sdf_bind_group_layout: &wgpu::BindGroupLayout,
    sdf_uniform_buffer: &wgpu::Buffer,
    volume_bind_group_layout: &wgpu::BindGroupLayout,
    volume_uniform_buffer: &wgpu::Buffer,
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
    let splat_data_offset = header.splat_data_offset;
    let splat_count = header.splat_count;
    
    log::info!("📄 GVE Header: version={:#x}, vertices={}, shell_offset={}, volume_offset={}, volume_size={}", 
        version, vertex_count, shell_mesh_offset, volume_data_offset, volume_size);

    let mut result = GeometryLoadResult {
        sdf: None,
        mesh: None,
        splat: None,
        volume: None,
    };
    
    // Parse dense volume (for GPU raymarching) - priority for SDF view
    log::info!("📦 Checking volume: offset={}, size={}", volume_data_offset, volume_size);
    if volume_data_offset > 0 && volume_size > 0 {
        if let Some(vol) = parse_dense_volume(
            device,
            queue,
            data,
            volume_data_offset as usize,
            volume_size as usize,
            volume_bind_group_layout,
            volume_uniform_buffer,
        ) {
            log::info!("✅ Loaded dense volume: {}x{}x{}", vol.dims[0], vol.dims[1], vol.dims[2]);
            result.volume = Some(vol);
        }
    }
    
    // Parse SDF bytecode (legacy CSG raymarching)
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
            result.sdf = Some(sdf);
        }
    }

    // Check for Splat Data
    if splat_data_offset > 0 && splat_count > 0 {
        if let Some(splat) = parse_splat_data(device, data, splat_data_offset as usize, splat_count as usize) {
            log::info!("✅ Loaded Splats: {}", splat_count);
            result.splat = Some(splat);
        }
    }

    // Check for shell mesh
    if shell_mesh_offset > 0 && vertex_count > 0 {
        if let Some(mesh) = parse_shell_mesh(device, data, header) {
            log::info!("✅ Loaded shell mesh");
            result.mesh = Some(mesh);
        }
    }

    // Check if we loaded anything
    if result.sdf.is_none() && result.mesh.is_none() && result.splat.is_none() && result.volume.is_none() {
        return Err(GeometryLoadError::NoGeometry);
    }

    Ok(result)
}

/// Result of loading geometry - can contain multiple representations
pub struct GeometryLoadResult {
    pub sdf: Option<LoadedSDF>,
    pub mesh: Option<LoadedMesh>,
    pub splat: Option<LoadedSplat>,
    pub volume: Option<LoadedVolume>,
}

/// Errors that can occur during geometry loading
#[derive(Debug)]
pub enum GeometryLoadError {
    DataTooSmall,
    InvalidMagic,
    NoGeometry, // Simplified
}

// ============================================================================
// Dense Volume Parsing (3D Texture for Raymarching)
// ============================================================================

/// Dense grid header size: dims(12) + bounds_min(12) + bounds_max(12) + uncompressed_size(4) = 40 bytes
const DENSE_VOLUME_HEADER_SIZE: usize = 40;

/// Parse LZ4-compressed dense volume section and create 3D texture for GPU raymarching
fn parse_dense_volume(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[u8],
    offset: usize,
    size: usize,
    bind_group_layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
) -> Option<LoadedVolume> {
    if offset + size > data.len() {
        log::warn!("⚠️ Volume data offset out of bounds");
        return None;
    }
    
    if size < DENSE_VOLUME_HEADER_SIZE {
        log::warn!("⚠️ Volume data too small for header");
        return None;
    }
    
    let vol_data = &data[offset..offset + size];
    
    // Parse header: dims (3x u32) + bounds_min (3x f32) + bounds_max (3x f32) + uncompressed_size (u32)
    let dims = [
        u32::from_le_bytes([vol_data[0], vol_data[1], vol_data[2], vol_data[3]]),
        u32::from_le_bytes([vol_data[4], vol_data[5], vol_data[6], vol_data[7]]),
        u32::from_le_bytes([vol_data[8], vol_data[9], vol_data[10], vol_data[11]]),
    ];
    
    // Validate dimensions are reasonable (max 512 per axis for WebGPU 3D textures)
    // This also catches old VDB format data which would parse as garbage dimensions
    const MAX_DIM: u32 = 512;
    if dims[0] == 0 || dims[1] == 0 || dims[2] == 0 
        || dims[0] > MAX_DIM || dims[1] > MAX_DIM || dims[2] > MAX_DIM {
        log::warn!("⚠️ Volume dims invalid or too large: {}x{}x{} (max {}). Likely old format - skipping.", 
            dims[0], dims[1], dims[2], MAX_DIM);
        return None;
    }
    
    let bounds_min = [
        f32::from_le_bytes([vol_data[12], vol_data[13], vol_data[14], vol_data[15]]),
        f32::from_le_bytes([vol_data[16], vol_data[17], vol_data[18], vol_data[19]]),
        f32::from_le_bytes([vol_data[20], vol_data[21], vol_data[22], vol_data[23]]),
    ];
    let bounds_max = [
        f32::from_le_bytes([vol_data[24], vol_data[25], vol_data[26], vol_data[27]]),
        f32::from_le_bytes([vol_data[28], vol_data[29], vol_data[30], vol_data[31]]),
        f32::from_le_bytes([vol_data[32], vol_data[33], vol_data[34], vol_data[35]]),
    ];
    
    let uncompressed_size = u32::from_le_bytes([vol_data[36], vol_data[37], vol_data[38], vol_data[39]]) as usize;
    
    // Validate bounds are finite floats (not NaN/Inf from garbage data)
    let bounds_valid = bounds_min.iter().chain(bounds_max.iter())
        .all(|v| v.is_finite() && v.abs() < 1000.0);
    if !bounds_valid {
        log::warn!("⚠️ Volume bounds invalid (NaN/Inf or too large). Likely old format - skipping.");
        return None;
    }
    
    // Validate uncompressed size matches expected voxel count
    let expected_voxels = (dims[0] as u64 * dims[1] as u64 * dims[2] as u64) as usize;
    let expected_uncompressed = expected_voxels * 4; // f32 per voxel
    if uncompressed_size != expected_uncompressed {
        log::warn!("⚠️ Volume uncompressed size mismatch: header says {} but dims imply {}", 
            uncompressed_size, expected_uncompressed);
        return None;
    }
    
    // Get compressed data (after 40-byte header)
    let compressed_data = &vol_data[DENSE_VOLUME_HEADER_SIZE..];
    let compressed_size = size - DENSE_VOLUME_HEADER_SIZE;
    
    // Decompress LZ4
    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&format!(
        "🔧 LZ4 decompressing: {} compressed bytes -> {} expected",
        compressed_size, uncompressed_size
    ).into());
    
    let voxel_data = match lz4_flex::decompress(compressed_data, uncompressed_size) {
        Ok(decompressed) => {
            let ratio = uncompressed_size as f32 / compressed_size as f32;
            log::info!("📦 Dense volume: {}x{}x{}, decompressed {} -> {} bytes ({:.1}x)", 
                dims[0], dims[1], dims[2],
                compressed_size, uncompressed_size, ratio);
            #[cfg(target_arch = "wasm32")]
            web_sys::console::log_1(&format!(
                "✅ LZ4 decompressed: {}x{}x{}, {} -> {} bytes ({:.1}x)",
                dims[0], dims[1], dims[2], compressed_size, uncompressed_size, ratio
            ).into());
            decompressed
        }
        Err(e) => {
            log::warn!("⚠️ LZ4 decompression failed: {:?}. Likely old format - skipping.", e);
            #[cfg(target_arch = "wasm32")]
            web_sys::console::error_1(&format!(
                "❌ LZ4 decompression FAILED: {:?}",
                e
            ).into());
            return None;
        }
    };
    
    log::info!("📦 Bounds: [{:.2},{:.2},{:.2}] to [{:.2},{:.2},{:.2}]",
        bounds_min[0], bounds_min[1], bounds_min[2],
        bounds_max[0], bounds_max[1], bounds_max[2]);
    
    // Create 3D texture
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Volume 3D Texture"),
        size: wgpu::Extent3d {
            width: dims[0],
            height: dims[1],
            depth_or_array_layers: dims[2],
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // Upload decompressed voxel data to GPU texture
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &voxel_data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(dims[0] * 4), // f32 = 4 bytes
            rows_per_image: Some(dims[1]),
        },
        wgpu::Extent3d {
            width: dims[0],
            height: dims[1],
            depth_or_array_layers: dims[2],
        },
    );
    
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Volume Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    
    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Volume Bind Group"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    
    Some(LoadedVolume {
        texture,
        texture_view,
        sampler,
        bind_group,
        dims,
        bounds_min,
        bounds_max,
    })
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

// ============================================================================
// Splat Parsing
// ============================================================================

fn parse_splat_data(
    device: &wgpu::Device,
    data: &[u8],
    offset: usize,
    count: usize,
) -> Option<LoadedSplat> {
    if offset > data.len() {
        log::warn!("⚠️ Splat offset out of bounds");
        return None;
    }
    
    // Splat size is 48 bytes
    let splat_size = std::mem::size_of::<shared::Splat>();
    let total_size = count * splat_size;
    
    if offset + total_size > data.len() {
        log::warn!("⚠️ Splat data out of bounds");
        return None;
    }
    
    let splat_data = &data[offset..offset + total_size];
    
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Splat Instance Buffer"),
        contents: splat_data,
        usage: wgpu::BufferUsages::VERTEX,
    });
    
    Some(LoadedSplat {
        instance_buffer,
        instance_count: count as u32,
    })
}
