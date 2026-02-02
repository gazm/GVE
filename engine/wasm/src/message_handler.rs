use shared::{MessageType, MessageHeader};
use crate::wasm_engine::WasmEngine;

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
            web_sys::console::log_1(&format!("📦 Calling load_geometry with {} bytes", payload.len()).into());
            
            let (success, count1, count2, err) = _engine.renderer.load_geometry(asset_id, payload);
            
            #[cfg(debug_assertions)]
            if success {
                web_sys::console::log_1(&format!("✅ Geometry loaded: {} verts/instrs, {} indices", count1, count2).into());
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
                // Manually read f32s to avoid alignment issues
                let vals: Vec<f32> = payload.chunks_exact(4)
                    .take(5)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                
                let pos = [vals[0], vals[1], vals[2]];
                let yaw = vals[3];
                let pitch = vals[4];
                
                _engine.renderer.update_camera(pos, yaw, pitch);
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
