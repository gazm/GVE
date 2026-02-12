//! Binary asset parsing for the renderer
//!
//! Contains functions for parsing .gve_bin binary data into GPU-ready assets.
//! GVE 3.0 chunk-based format: each section is a self-describing chunk with
//! a FourCC identifier (VOLM, MESH, SPLT, TRIP, ROPS) and explicit size.

use wgpu::util::DeviceExt;
use shared::ShellVertex;
use shared::binary_format::{GVE3Header, ChunkHeader, GVE3_MAGIC, chunk_id, align_to_16};

use crate::renderer::types::{LoadedMesh, LoadedSplat, LoadedVolume};

// ============================================================================
// Binary Parsing Constants
// ============================================================================

const GVE3_HEADER_SIZE: usize = std::mem::size_of::<GVE3Header>();
const CHUNK_HEADER_SIZE: usize = std::mem::size_of::<ChunkHeader>();

// ============================================================================
// Geometry Loading (GVE 3.0 Chunk-Based)
// ============================================================================

/// Load geometry from .gve_bin binary data (GVE 3.0 chunk format)
///
/// Iterates over self-describing chunks and extracts:
/// - Dense volume (VOLM) for GPU raymarching
/// - Shell mesh (MESH) for rasterized view
/// - Splats (SPLT) for gaussian splatting
/// - Triplanar textures (TRIP) for surface coloring
pub fn load_geometry_from_binary(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[u8],
    volume_bind_group_layout: &wgpu::BindGroupLayout,
    volume_uniform_buffer: &wgpu::Buffer,
) -> Result<GeometryLoadResult, GeometryLoadError> {
    // Validate minimum size for GVE3 header
    if data.len() < GVE3_HEADER_SIZE {
        return Err(GeometryLoadError::DataTooSmall);
    }

    // Parse GVE3 header
    let header: &GVE3Header = bytemuck::from_bytes(&data[..GVE3_HEADER_SIZE]);

    // Validate magic
    if header.magic != GVE3_MAGIC {
        return Err(GeometryLoadError::InvalidMagic);
    }

    let chunk_count = header.chunk_count;
    log::info!("\u{1f4e6} GVE3 file: {} chunks", chunk_count);

    let mut result = GeometryLoadResult {
        mesh: None,
        splat: None,
        volume: None,
    };

    // Collect chunk locations: (fourcc, data_offset, data_size)
    let mut chunks: Vec<([u8; 4], usize, usize)> = Vec::new();
    let mut cursor = GVE3_HEADER_SIZE;

    for i in 0..chunk_count as usize {
        if cursor + CHUNK_HEADER_SIZE > data.len() {
            log::warn!("\u{26a0}\u{fe0f} Chunk {} header extends past EOF", i);
            break;
        }
        let chunk_hdr: &ChunkHeader = bytemuck::from_bytes(&data[cursor..cursor + CHUNK_HEADER_SIZE]);
        let fourcc = chunk_hdr.fourcc;
        let size = chunk_hdr.size as usize;
        let data_start = cursor + CHUNK_HEADER_SIZE;

        if data_start + size > data.len() {
            log::warn!("\u{26a0}\u{fe0f} Chunk {} ({}) data extends past EOF", i,
                String::from_utf8_lossy(&fourcc));
            break;
        }

        log::info!("  \u{1f4e6} Chunk {}: {} ({} bytes)", i, String::from_utf8_lossy(&fourcc), size);
        chunks.push((fourcc, data_start, size));

        // Advance past data + padding to 16-byte boundary
        cursor = data_start + align_to_16(size as u64) as usize;
    }

    // Find the TRIP chunk offset for volume parsing (needed to bind triplanar textures)
    let trip_chunk = chunks.iter().find(|(cc, _, _)| *cc == chunk_id::TRIP);
    let triplanar_global_offset = trip_chunk.map(|(_, off, _)| *off as u64).unwrap_or(0);

    // Parse ROPS chunk if present
    let rops_chunk = chunks.iter().find(|(cc, _, _)| *cc == chunk_id::ROPS);
    let patches = if let Some((_, offset, size)) = rops_chunk {
        parse_runtime_ops(data, *offset, *size).unwrap_or_else(|| {
            log::warn!("⚠️ Failed to parse ROPS chunk");
            Vec::new()
        })
    } else {
        Vec::new()
    };

    // Process chunks
    for (fourcc, offset, size) in &chunks {
        if *fourcc == chunk_id::VOLM {
            if let Some(mut vol) = parse_dense_volume(
                device, queue, data,
                *offset, *size,
                triplanar_global_offset,
                volume_bind_group_layout,
                volume_uniform_buffer,
            ) {
                // Attach parsed runtime patches
                vol.patches = patches.clone();
                
                log::info!("\u{2705} Loaded dense volume: {}x{}x{} (triplanar: {}, patches: {})",
                    vol.dims[0], vol.dims[1], vol.dims[2], vol.has_triplanar, vol.patches.len());
                result.volume = Some(vol);
            }
        } else if *fourcc == chunk_id::MESH {
            if let Some(mesh) = parse_shell_mesh(device, data, *offset, *size) {
                log::info!("\u{2705} Loaded shell mesh");
                result.mesh = Some(mesh);
            }
        } else if *fourcc == chunk_id::SPLT {
            // Splat count = chunk size / per-splat size
            let splat_size = std::mem::size_of::<shared::Splat>();
            let count = *size / splat_size;
            if count > 0 {
                if let Some(splat) = parse_splat_data(device, data, *offset, count) {
                    log::info!("\u{2705} Loaded Splats: {}", count);
                    result.splat = Some(splat);
                }
            }
        } else if *fourcc == chunk_id::ROPS {
            // Handled above
            log::info!("\u{2705} Loaded Runtime Ops: {}", patches.len());
        } else {
            // TRIP consumed via triplanar_global_offset, ROPS/META handled
            if *fourcc != chunk_id::TRIP {
                log::info!("  \u{23ed}\u{fe0f} Skipping chunk: {}", String::from_utf8_lossy(fourcc));
            }
        }
    }

    // Check if we loaded anything
    if result.mesh.is_none() && result.splat.is_none() && result.volume.is_none() {
        return Err(GeometryLoadError::NoGeometry);
    }

    Ok(result)
}

/// Result of loading geometry - can contain multiple representations
pub struct GeometryLoadResult {
    // sdf: Option<LoadedSDF>, // Removed
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
// Runtime Ops Parsing
// ============================================================================

fn parse_runtime_ops(data: &[u8], offset: usize, size: usize) -> Option<Vec<crate::renderer::types::RuntimeVolumeOp>> {
    let op_size = std::mem::size_of::<crate::renderer::types::RuntimeVolumeOp>();
    if size % op_size != 0 {
        log::warn!("⚠️ ROPS chunk size {} not divisible by struct size {}", size, op_size);
        return None;
    }
    
    let count = size / op_size;
    let mut ops = Vec::with_capacity(count);
    let mut cursor = offset;
    
    for _ in 0..count {
        if cursor + op_size > data.len() {
            break;
        }
        let op: crate::renderer::types::RuntimeVolumeOp = bytemuck::cast_slice(&data[cursor..cursor+op_size])[0];
        ops.push(op);
        cursor += op_size;
    }
    
    Some(ops)
}

// ============================================================================
// Dense Volume Parsing (3D Texture for Raymarching)
// ============================================================================

/// Dense grid header size: dims(12) + bounds_min(12) + bounds_max(12) + uncompressed_size(4) = 40 bytes
const DENSE_VOLUME_HEADER_SIZE: usize = 40;

/// TRI1 header: magic(4) + resolution(4) + bounds_min(12) + bounds_max(12) = 32 bytes
const TRI1_HEADER_SIZE: usize = 32;
const TRI1_MAGIC: &[u8; 4] = b"TRI1";

/// Parse optional triplanar block (TRI1). Returns (xy_view, xz_view, yz_view, bounds_min, bounds_max) or None.
fn parse_triplanar(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[u8],
    offset: usize,
) -> Option<(
    wgpu::TextureView,
    wgpu::TextureView,
    wgpu::TextureView,
    [f32; 3],
    [f32; 3],
)> {
    if offset + TRI1_HEADER_SIZE > data.len() {
        return None;
    }
    let tri = &data[offset..];
    if &tri[0..4] != TRI1_MAGIC {
        log::warn!("⚠️ Triplanar magic mismatch");
        return None;
    }
    let resolution = u32::from_le_bytes([tri[4], tri[5], tri[6], tri[7]]) as usize;
    if resolution == 0 || resolution > 2048 {
        log::warn!("⚠️ Triplanar resolution invalid: {}", resolution);
        return None;
    }
    let bounds_min = [
        f32::from_le_bytes([tri[8], tri[9], tri[10], tri[11]]),
        f32::from_le_bytes([tri[12], tri[13], tri[14], tri[15]]),
        f32::from_le_bytes([tri[16], tri[17], tri[18], tri[19]]),
    ];
    let bounds_max = [
        f32::from_le_bytes([tri[20], tri[21], tri[22], tri[23]]),
        f32::from_le_bytes([tri[24], tri[25], tri[26], tri[27]]),
        f32::from_le_bytes([tri[28], tri[29], tri[30], tri[31]]),
    ];
    let slice_bytes = resolution * resolution * 4;
    let total_data = TRI1_HEADER_SIZE + slice_bytes * 3;
    if offset + total_data > data.len() {
        log::warn!("⚠️ Triplanar data truncated");
        return None;
    }
    let create_tex = |label: &str, slice: &[u8]| {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: resolution as u32, height: resolution as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let res_u = resolution as u32;
        queue.write_texture(
            wgpu::ImageCopyTexture { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            slice,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(res_u * 4), rows_per_image: Some(res_u) },
            wgpu::Extent3d { width: res_u, height: res_u, depth_or_array_layers: 1 },
        );
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    };
    let xy = create_tex("Triplanar XY", &tri[TRI1_HEADER_SIZE..TRI1_HEADER_SIZE + slice_bytes]);
    let xz = create_tex("Triplanar XZ", &tri[TRI1_HEADER_SIZE + slice_bytes..TRI1_HEADER_SIZE + slice_bytes * 2]);
    let yz = create_tex("Triplanar YZ", &tri[TRI1_HEADER_SIZE + slice_bytes * 2..total_data]);
    log::info!("✅ Triplanar loaded: {}x{}", resolution, resolution);
    Some((xy, xz, yz, bounds_min, bounds_max))
}

/// Create a 1x1 RGBA dummy texture (grey) for unused triplanar bindings
fn create_dummy_triplanar_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Dummy Triplanar"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let grey: [u8; 4] = [128, 128, 128, 255];
    queue.write_texture(
        wgpu::ImageCopyTexture { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &grey,
        wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Parse LZ4-compressed dense volume section and create 3D texture for GPU raymarching
fn parse_dense_volume(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[u8],
    offset: usize,
    size: usize,
    triplanar_offset: u64,
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

    let triplanar_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Triplanar Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let (tri_xy, tri_xz, tri_yz, tri_bounds_min, tri_bounds_max, has_triplanar) = if triplanar_offset > 0 {
        if let Some((xy, xz, yz, bmin, bmax)) = parse_triplanar(device, queue, data, triplanar_offset as usize) {
            let vol_eq = bmin == bounds_min && bmax == bounds_max;
            log::info!("📐 Triplanar vs volume bounds: tri=([{:.4},{:.4},{:.4}]..[{:.4},{:.4},{:.4}]) vol=([{:.4},{:.4},{:.4}]..[{:.4},{:.4},{:.4}]) match={}",
                bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2],
                bounds_min[0], bounds_min[1], bounds_min[2], bounds_max[0], bounds_max[1], bounds_max[2],
                vol_eq);
            #[cfg(target_arch = "wasm32")]
            web_sys::console::log_1(&format!(
                "📐 Triplanar bounds: [{:.4},{:.4},{:.4}]..[{:.4},{:.4},{:.4}] | Volume: [{:.4},{:.4},{:.4}]..[{:.4},{:.4},{:.4}] | match={}",
                bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2],
                bounds_min[0], bounds_min[1], bounds_min[2], bounds_max[0], bounds_max[1], bounds_max[2],
                vol_eq
            ).into());
            (xy, xz, yz, bmin, bmax, true)
        } else {
            let d1 = create_dummy_triplanar_texture(device, queue);
            let d2 = create_dummy_triplanar_texture(device, queue);
            let d3 = create_dummy_triplanar_texture(device, queue);
            (d1, d2, d3, bounds_min, bounds_max, false)
        }
    } else {
        let d1 = create_dummy_triplanar_texture(device, queue);
        let d2 = create_dummy_triplanar_texture(device, queue);
        let d3 = create_dummy_triplanar_texture(device, queue);
        (d1, d2, d3, bounds_min, bounds_max, false)
    };

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Volume Bind Group"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&texture_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&tri_xy) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&tri_xz) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&tri_yz) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&triplanar_sampler) },
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
        has_triplanar,
        triplanar_bounds_min: tri_bounds_min,
        triplanar_bounds_max: tri_bounds_max,
        patches: Vec::new(),
    })
}

// SDF Parsing Functions REMOVED (v2.3)

// ============================================================================
// Shell Mesh Parsing
// ============================================================================

/// Parse shell mesh from MESH chunk data
///
/// MESH chunk layout: vertex_count(u32) + index_count(u32) + vertices + indices
fn parse_shell_mesh(device: &wgpu::Device, data: &[u8], offset: usize, size: usize) -> Option<LoadedMesh> {
    if size < 8 {
        log::warn!("\u{26a0}\u{fe0f} MESH chunk too small");
        return None;
    }

    let chunk_end = offset + size;

    // Read vertex_count and index_count from start of chunk data
    let vertex_count = u32::from_le_bytes(data[offset..offset+4].try_into().ok()?);
    let index_count = u32::from_le_bytes(data[offset+4..offset+8].try_into().ok()?);
    
    log::info!("\u{1f4d0} Shell mesh: {} vertices, {} indices", vertex_count, index_count);

    // Calculate sizes
    let vertex_size = std::mem::size_of::<ShellVertex>();
    let vertices_start = offset + 8;
    let vertices_end = vertices_start + (vertex_count as usize * vertex_size);
    
    if vertices_end > chunk_end {
        log::warn!("\u{26a0}\u{fe0f} Vertex data extends past MESH chunk");
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
    let (index_buffer, use_indices, index_format) = if index_count > 0 {
        let indices_start = vertices_end;
        let index_size = 4; // u32 indices
        let index_format = wgpu::IndexFormat::Uint32;
        let indices_end = indices_start + (index_count as usize * index_size);
        
        if indices_end <= chunk_end {
            let index_data = &data[indices_start..indices_end];
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Loaded Mesh Indices"),
                contents: index_data,
                usage: wgpu::BufferUsages::INDEX,
            });
            (Some(buffer), true, index_format)
        } else {
            log::warn!("\u{26a0}\u{fe0f} Index data extends past MESH chunk, using vertices only");
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
