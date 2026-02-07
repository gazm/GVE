use shared::{MessageType, MessageHeader, GVEBinaryHeader, GVE_MAGIC};
use crate::wasm_engine::WasmEngine;

/// Debug: inspect GVE binary header to understand why volume isn't loading
#[cfg(debug_assertions)]
fn debug_gve_header(payload: &[u8]) {
    if payload.len() < std::mem::size_of::<GVEBinaryHeader>() {
        web_sys::console::warn_1(&"Payload too small for GVE header".into());
        return;
    }
    
    let header: &GVEBinaryHeader = bytemuck::from_bytes(&payload[..std::mem::size_of::<GVEBinaryHeader>()]);
    
    // Check magic
    if &header.magic != GVE_MAGIC {
        web_sys::console::error_1(&format!("❌ Invalid magic: {:?}", header.magic).into());
        return;
    }
    
    // Copy fields from packed struct to avoid unaligned access
    let version = header.version;
    let volume_offset = header.volume_data_offset;
    let volume_size = header.volume_size;
    let shell_offset = header.shell_mesh_offset;
    let vertex_count = header.vertex_count;
    
    web_sys::console::log_1(&format!(
        "🔍 GVE Header Debug:\n  version={:#x}\n  volume_offset={}\n  volume_size={}\n  shell_offset={}\n  vertex_count={}",
        version, volume_offset, volume_size, shell_offset, vertex_count
    ).into());
    
    // Check if volume data exists
    if volume_offset > 0 && volume_size > 0 {
        web_sys::console::log_1(&format!(
            "🔍 Volume data: offset={} size={} (expected at bytes {}..{})",
            volume_offset, volume_size, volume_offset, volume_offset + volume_size as u64
        ).into());
        
        // Peek at first 40 bytes of volume data (header)
        let vol_start = volume_offset as usize;
        let vol_end = vol_start + 40.min(volume_size as usize);
        if vol_end <= payload.len() {
            let vol_header = &payload[vol_start..vol_end];
            let dims = [
                u32::from_le_bytes([vol_header[0], vol_header[1], vol_header[2], vol_header[3]]),
                u32::from_le_bytes([vol_header[4], vol_header[5], vol_header[6], vol_header[7]]),
                u32::from_le_bytes([vol_header[8], vol_header[9], vol_header[10], vol_header[11]]),
            ];
            let uncompressed = u32::from_le_bytes([vol_header[36], vol_header[37], vol_header[38], vol_header[39]]);
            web_sys::console::log_1(&format!(
                "🔍 Volume header: dims={}x{}x{}, uncompressed_size={}",
                dims[0], dims[1], dims[2], uncompressed
            ).into());
        }
    } else {
        web_sys::console::warn_1(&format!(
            "⚠️ No volume data in header: offset={}, size={}",
            volume_offset, volume_size
        ).into());
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
                web_sys::console::log_1(&format!("📊 Asset {} modes: mesh={}, volume={}, splat={}", 
                    asset_id, has_mesh, has_volume, has_splat).into());
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
