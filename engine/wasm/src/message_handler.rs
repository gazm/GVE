use shared::{MessageType, MessageHeader};
use shared::binary_format::{GVE3Header, ChunkHeader, GVE3_MAGIC, chunk_id, align_to_16};
use crate::wasm_engine::WasmEngine;

/// Debug: inspect GVE3 binary header + chunks
#[cfg(debug_assertions)]
fn debug_gve_header(payload: &[u8]) {
    let hdr_size = std::mem::size_of::<GVE3Header>();
    let chunk_hdr_size = std::mem::size_of::<ChunkHeader>();

    if payload.len() < hdr_size {
        web_sys::console::warn_1(&"Payload too small for GVE3 header".into());
        return;
    }

    let header: &GVE3Header = bytemuck::from_bytes(&payload[..hdr_size]);

    if header.magic != GVE3_MAGIC {
        web_sys::console::error_1(&format!("\u{274c} Invalid magic: {:?} (expected GVE3)", header.magic).into());
        return;
    }

    let chunk_count = header.chunk_count;
    web_sys::console::log_1(&format!(
        "\u{1f50d} GVE3 Header: version={:#x}, chunks={}",
        header.version, chunk_count
    ).into());

    let mut cursor = hdr_size;
    for i in 0..chunk_count as usize {
        if cursor + chunk_hdr_size > payload.len() { break; }
        let chunk_hdr: &ChunkHeader = bytemuck::from_bytes(&payload[cursor..cursor + chunk_hdr_size]);
        let fourcc = String::from_utf8_lossy(&chunk_hdr.fourcc);
        let size = chunk_hdr.size;
        web_sys::console::log_1(&format!(
            "  \u{1f4e6} Chunk {}: {} ({} bytes)", i, fourcc, size
        ).into());

        // Extra volume debug
        if chunk_hdr.fourcc == chunk_id::VOLM {
            let data_start = cursor + chunk_hdr_size;
            if data_start + 40 <= payload.len() && size >= 40 {
                let vol = &payload[data_start..];
                let dims = [
                    u32::from_le_bytes([vol[0], vol[1], vol[2], vol[3]]),
                    u32::from_le_bytes([vol[4], vol[5], vol[6], vol[7]]),
                    u32::from_le_bytes([vol[8], vol[9], vol[10], vol[11]]),
                ];
                let uncompressed = u32::from_le_bytes([vol[36], vol[37], vol[38], vol[39]]);
                web_sys::console::log_1(&format!(
                    "    \u{1f50d} Volume: dims={}x{}x{}, uncompressed={}",
                    dims[0], dims[1], dims[2], uncompressed
                ).into());
            }
        }

        cursor = cursor + chunk_hdr_size + align_to_16(size) as usize;
    }
}

pub fn handle_binary_message(_engine: &mut WasmEngine, data: &[u8]) {
    if data.len() < std::mem::size_of::<MessageHeader>() {
        #[cfg(debug_assertions)]
        web_sys::console::warn_1(&"Binary message too short for header".into());
        return;
    }

    let header: &MessageHeader = bytemuck::from_bytes(&data[..std::mem::size_of::<MessageHeader>()]);
    
    let msg_type = header.msg_type;
    let asset_id = header.asset_id;
    let version = header.version;
    let payload_size = header.payload_size;

    #[cfg(debug_assertions)]
    web_sys::console::log_1(&format!(
        "WASM received message: Type={:?}, AssetID={}, Version={}, PayloadSize={}", 
        msg_type, asset_id, version, payload_size
    ).into());

    match msg_type {
        MessageType::AssetReady => {
            #[cfg(debug_assertions)]
            web_sys::console::log_1(&format!("Asset Ready message received for asset {}", asset_id).into());
            
            let payload = &data[std::mem::size_of::<MessageHeader>()..];
            
            #[cfg(debug_assertions)]
            {
                web_sys::console::log_1(&format!("📦 Calling load_geometry with {} bytes", payload.len()).into());
                debug_gve_header(payload);
            }
            
            let (success, count1, count2, err) = _engine.renderer.load_geometry(asset_id, payload);
            
            #[cfg(debug_assertions)]
            if success {
                web_sys::console::log_1(&format!("✅ Geometry loaded: {} verts/instrs, {} indices", count1, count2).into());
                // Debug: check what was actually loaded
                let has_mesh = _engine.renderer.has_mesh(asset_id);
                let has_volume = _engine.renderer.has_volume(asset_id);
                let has_splat = _engine.renderer.has_splat(asset_id);
                let has_triplanar = _engine.renderer.has_triplanar(asset_id);
                web_sys::console::log_1(&format!("📊 Asset {} modes: mesh={}, volume={}, splat={}, triplanar={}", 
                    asset_id, has_mesh, has_volume, has_splat, has_triplanar).into());
            } else {
                web_sys::console::error_1(&format!("❌ Failed to load geometry: {:?}", err).into());
            }
        }
        MessageType::VersionBump => {
            #[cfg(debug_assertions)]
            web_sys::console::log_1(&format!("Version Bump message received for asset {}", asset_id).into());
        }
        MessageType::UpdateCamera => {
            let payload = &data[std::mem::size_of::<MessageHeader>()..];
            if payload.len() >= 20 {
                // Fixed-size array — no heap allocation on this hot path
                let mut vals = [0f32; 5];
                for (i, chunk) in payload.chunks_exact(4).take(5).enumerate() {
                    vals[i] = f32::from_le_bytes(chunk.try_into().unwrap());
                }
                _engine.renderer.update_camera(
                    [vals[0], vals[1], vals[2]], vals[3], vals[4],
                );
            }
        }
        MessageType::LoadChunk => {
            let payload = &data[std::mem::size_of::<MessageHeader>()..];
            if payload.len() >= 16 {
                let chunk_id = u64::from_le_bytes(payload[..8].try_into().unwrap());
                let x = i32::from_le_bytes(payload[8..12].try_into().unwrap());
                let z = i32::from_le_bytes(payload[12..16].try_into().unwrap());
                _engine.handle_load_chunk(chunk_id, x, z);
            }
        }
        MessageType::TranslateNode => {
            let payload = &data[std::mem::size_of::<MessageHeader>()..];
            if payload.len() >= 20 {
                let node_id = u64::from_le_bytes(payload[..8].try_into().unwrap());
                let dx = f32::from_le_bytes(payload[8..12].try_into().unwrap());
                let dy = f32::from_le_bytes(payload[12..16].try_into().unwrap());
                let dz = f32::from_le_bytes(payload[16..20].try_into().unwrap());
                _engine.translate_node(node_id, dx, dy, dz);
            }
        }
        MessageType::UpdateJoint => {
            let payload = &data[std::mem::size_of::<MessageHeader>()..];
            if payload.len() >= 24 {
                let joint_id = u64::from_le_bytes(payload[..8].try_into().unwrap());
                let qx = f32::from_le_bytes(payload[8..12].try_into().unwrap());
                let qy = f32::from_le_bytes(payload[12..16].try_into().unwrap());
                let qz = f32::from_le_bytes(payload[16..20].try_into().unwrap());
                let qw = f32::from_le_bytes(payload[20..24].try_into().unwrap());
                _engine.update_joint(joint_id, [qx, qy, qz, qw]);
            }
        }
        _ => {
            #[cfg(debug_assertions)]
            web_sys::console::log_1(&format!("Unhandled message type: {:?}", msg_type).into());
        }
    }
}
