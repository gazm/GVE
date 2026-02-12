//! Asset Management for Renderer
//!
//! Handles loading, unloading, and tracking of GPU assets.
//! Split from pipeline.rs for maintainability.

use super::pipeline::Renderer;
use super::loaders::load_geometry_from_binary;
// create_test_sdf removed (v2.3)

impl Renderer {
    /// Load geometry from .gve_bin binary data
    /// Returns (success, vertex_count, index_count, error_msg) for WASM logging
    pub fn load_geometry(&mut self, asset_id: u64, data: &[u8]) -> (bool, u32, u32, Option<String>) {
        log::info!("📦 Loading geometry for asset {} ({} bytes)", asset_id, data.len());
        
        let mut stats = (0u32, 0u32); // (primary_count, secondary_count)
        
        match load_geometry_from_binary(
            &self.device,
            &self.queue,
            data,
            &self.volume_bind_group_layout,
            &self.volume_uniform_buffer,
        ) {
            Ok(result) => {
                // Load primitives into maps
                if let Some(mesh) = result.mesh {
                    stats = (mesh.vertex_count, mesh.index_count);
                    self.loaded_meshes.insert(asset_id, mesh);
                }
                
                // SDF loading removed (v2.3)

                if let Some(splat) = result.splat {
                    if stats.0 == 0 { stats.0 = splat.instance_count; }
                    self.loaded_splats.insert(asset_id, splat);
                }

                if let Some(volume) = result.volume {
                    let vol_size = volume.dims[0] * volume.dims[1] * volume.dims[2];
                    if stats.0 == 0 { stats.0 = vol_size; }
                    self.loaded_volumes.insert(asset_id, volume);
                }

                // Determine default active mode
                // Priority: Volume > Splat-only > Mesh
                if self.loaded_volumes.contains_key(&asset_id) {
                     self.active_volume = Some(asset_id);
                     self.active_splat = None;
                     log::info!("✅ Asset {}: Defaulting to Volume view", asset_id);
                } else if self.loaded_splats.contains_key(&asset_id) {
                     self.active_splat = Some(asset_id);
                     self.active_volume = None;
                     log::info!("✅ Asset {}: Defaulting to Splat-only view", asset_id);
                } else {
                     // Mesh is default fallback
                     self.active_splat = None;
                     self.active_volume = None;
                     log::info!("✅ Asset {}: Defaulting to Mesh view", asset_id);
                }

                (true, stats.0, stats.1, None)
            }
            Err(e) => {
                log::warn!("⚠️ Failed to load geometry for asset {}: {:?}", asset_id, e);
                (false, 0, 0, Some(format!("{:?}", e)))
            }
        }
    }

    /// Load texture from pre-loaded binary data
    pub fn load_texture(&mut self, asset_id: u64, _data: &[u8]) {
        log::info!("🖼️ Loading texture for asset {}", asset_id);
        // Future: Upload to GPU texture
    }

    /// Clear all loaded meshes
    pub fn clear_meshes(&mut self) {
        self.loaded_meshes.clear();
    }

    /// Clear everything (meshes + volumes + splats)
    pub fn clear_all(&mut self) {
        self.loaded_meshes.clear();
        self.loaded_splats.clear();
        self.loaded_volumes.clear();
        self.active_splat = None;
        self.active_volume = None;
    }

    // set_active_sdf REMOVED (v2.3)

    /// Set the active Splat for rendering
    pub fn set_active_splat(&mut self, asset_id: Option<u64>) {
        self.active_splat = asset_id;
        // Note: Don't clear other active assets here - caller handles exclusivity
        // SdfTextured mode requires both volume AND splat active
    }

    /// Set the active Volume for rendering (dense grid raymarching)
    pub fn set_active_volume(&mut self, asset_id: Option<u64>) {
        self.active_volume = asset_id;
        // Note: Don't clear other active assets here - caller handles exclusivity
        // SdfTextured mode requires both volume AND splat active
    }

    /// Check if asset has mesh data
    pub fn has_mesh(&self, asset_id: u64) -> bool {
        self.loaded_meshes.contains_key(&asset_id)
    }

    // has_sdf REMOVED (v2.3)

    /// Check if asset has Splat data
    pub fn has_splat(&self, asset_id: u64) -> bool {
        self.loaded_splats.contains_key(&asset_id)
    }

    /// Check if asset has volume data (dense grid for raymarching)
    pub fn has_volume(&self, asset_id: u64) -> bool {
        self.loaded_volumes.contains_key(&asset_id)
    }

    /// Check if asset's volume has triplanar textures (for SDF surface color)
    pub fn has_triplanar(&self, asset_id: u64) -> bool {
        self.loaded_volumes
            .get(&asset_id)
            .map(|v| v.has_triplanar)
            .unwrap_or(false)
    }
    
    // clear_active_sdf and load_test_sdf REMOVED (v2.3)

    /// Unload geometry for an asset
    pub fn unload_geometry(&mut self, asset_id: u64) {
        if self.loaded_meshes.remove(&asset_id).is_some() {
            log::info!("🗑️ Unloaded mesh for asset {}", asset_id);
        }
        // Unload SDF logic REMOVED (v2.3)
        if self.loaded_volumes.remove(&asset_id).is_some() {
            if self.active_volume == Some(asset_id) {
                self.active_volume = None;
            }
            log::info!("🗑️ Unloaded volume for asset {}", asset_id);
        }
        if self.loaded_splats.remove(&asset_id).is_some() {
            if self.active_splat == Some(asset_id) {
                self.active_splat = None;
            }
            log::info!("🗑️ Unloaded Splats for asset {}", asset_id);
        }
    }

    /// Read-only scene snapshot for JS: count (u32), then per entry asset_id (u64), type (u8: 0=mesh, 1=sdf), active (u8: 0/1).
    pub fn get_scene_snapshot(&self) -> Vec<u8> {
        const TYPE_MESH: u8 = 0;
        let mut out = Vec::new();
        let mut count: u32 = 0;
        for (&_id, _) in &self.loaded_meshes {
            count = count.saturating_add(1);
        }
        // SDF count logic REMOVED (v2.3)
        out.extend_from_slice(&count.to_le_bytes());
        for (&id, _) in &self.loaded_meshes {
            out.extend_from_slice(&id.to_le_bytes());
            out.push(TYPE_MESH);
            out.push(0);
        }
        // SDF snapshot logic REMOVED (v2.3)
        out
    }
    /// Add a runtime patch (e.g., bullet hole) to the active volume
    pub fn add_runtime_patch(&mut self, asset_id: u64, op_type: u32, pos: [f32; 3], params: [f32; 8]) {
        if let Some(volume) = self.loaded_volumes.get_mut(&asset_id) {
            
            // Calculate AABB on CPU once
            let (aabb_min, aabb_max) = match op_type {
                1 => { // Sphere: params.x = radius
                    let r = params[0];
                    (
                        [pos[0] - r, pos[1] - r, pos[2] - r],
                        [pos[0] + r, pos[1] + r, pos[2] + r]
                    )
                },
                2 => { // Box: params = [sx, sy, sz]
                    (
                        [pos[0] - params[0], pos[1] - params[1], pos[2] - params[2]],
                        [pos[0] + params[0], pos[1] + params[1], pos[2] + params[2]]
                    )
                },
                _ => ([0.0; 3], [0.0; 3]),
            };

            let op = crate::renderer::types::RuntimeVolumeOp {
                op_type,
                _pad0: [0; 3],
                pos,
                _pad1: 0,
                params,
                aabb_min,
                _pad2: 0,
                aabb_max,
                _pad3: 0,
            };
            
            volume.patches.push(op);
            log::info!("💥 Added patch to asset {}: type={} pos={:?}", asset_id, op_type, pos);
            
            // Limit to 16 for now (shader limit)
            if volume.patches.len() > 16 {
                volume.patches.remove(0);
            }
        }
    }

    /// Clear all runtime patches for an asset
    pub fn clear_runtime_patches(&mut self, asset_id: u64) {
        if let Some(volume) = self.loaded_volumes.get_mut(&asset_id) {
            volume.patches.clear();
        }
    }
}
