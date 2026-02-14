mod wasm_engine;
mod message_handler;

use wasm_bindgen::prelude::*;
use std::cell::RefCell;
use once_cell::unsync::Lazy;
use crate::wasm_engine::WasmEngine;
use crate::message_handler::handle_binary_message;

// SAFETY: WASM is single-threaded — RefCell is safe and avoids Mutex atomic overhead.
thread_local! {
    static ENGINE: Lazy<RefCell<Option<WasmEngine>>> = Lazy::new(|| RefCell::new(None));
}

/// Borrow the engine mutably and run a closure. Returns None if engine is not initialized or if already borrowed (re-entrant call).
fn with_engine_mut<R>(f: impl FnOnce(&mut WasmEngine) -> R) -> Option<R> {
    ENGINE.with(|cell| {
        let mut borrow = match cell.try_borrow_mut() {
            Ok(b) => b,
            Err(_) => return None, // Re-entrant: skip this call (e.g. render_frame during load_geometry)
        };
        borrow.as_mut().map(f)
    })
}

/// Borrow the engine immutably and run a closure. Returns None if engine is not initialized or if already borrowed.
fn with_engine<R>(f: impl FnOnce(&WasmEngine) -> R) -> Option<R> {
    ENGINE.with(|cell| {
        let borrow = match cell.try_borrow() {
            Ok(b) => b,
            Err(_) => return None,
        };
        borrow.as_ref().map(f)
    })
}

#[wasm_bindgen]
pub async fn init_engine(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Warn);
    let window = web_sys::window().expect("no global `window` exists");
    let document = window.document().expect("should have a document on window");
    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or_else(|| JsValue::from_str("Canvas not found"))?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;

    let engine = WasmEngine::new(canvas).await?;

    ENGINE.with(|cell| {
        *cell.borrow_mut() = Some(engine);
    });

    #[cfg(debug_assertions)]
    web_sys::console::log_1(&"🚀 GVE WASM Engine Initialized".into());
    Ok(())
}

#[wasm_bindgen]
pub fn render_frame(_dt: f32) {
    with_engine_mut(|engine| {
        if let Err(e) = engine.render() {
            web_sys::console::error_1(&format!("Render error: {:?}", e).into());
        }
    });
}

#[wasm_bindgen]
pub fn resize_viewport(width: u32, height: u32) {
    with_engine_mut(|engine| engine.resize(width, height));
}

#[wasm_bindgen]
pub fn handle_message(data: &[u8]) {
    with_engine_mut(|engine| handle_binary_message(engine, data));
}

// load_test_sdf REMOVED (v2.3)

/// Clear all loaded SDFs and return to default mesh
#[wasm_bindgen]
pub fn clear_viewport() {
    with_engine_mut(|engine| engine.renderer.clear_all());
}

// clear_sdf REMOVED (v2.3)

#[wasm_bindgen]
pub fn set_view_mode(mode: u32, asset_id: u64) {
    with_engine_mut(|engine| {
        match mode {
            0 => { // Mesh (Shell)
                 engine.renderer.set_active_splat(None);
                 engine.renderer.set_active_volume(None);
                 engine.renderer.set_viewmode("mesh"); 
                 #[cfg(debug_assertions)] {
                     if engine.renderer.has_mesh(asset_id) {
                         web_sys::console::log_1(&format!("👁️ Mesh mode for asset {}", asset_id).into());
                     } else {
                         web_sys::console::warn_1(&format!("⚠️ Asset {} has no mesh data!", asset_id).into());
                     }
                 }
            },
            1 => { // Volume (formerly SDF)
                 let has_volume = engine.renderer.has_volume(asset_id);
                 if has_volume {
                     engine.renderer.set_active_volume(Some(asset_id));
                 }
                 engine.renderer.set_active_splat(None);
                 engine.renderer.set_viewmode("sdf"); // Keep "sdf" string for compatibility or change to "volume"?
                 #[cfg(debug_assertions)] {
                     if has_volume {
                         web_sys::console::log_1(&format!("👁️ Volume mode for asset {}", asset_id).into());
                     } else {
                         web_sys::console::warn_1(&format!("⚠️ Asset {} has no Volume data!", asset_id).into());
                     }
                 }
            },
            2 => { // Splat
                 if engine.renderer.has_splat(asset_id) {
                     engine.renderer.set_active_splat(Some(asset_id));
                     engine.renderer.set_active_volume(None);
                     engine.renderer.set_viewmode("splat");
                 }
                 #[cfg(debug_assertions)] {
                     if engine.renderer.has_splat(asset_id) {
                         web_sys::console::log_1(&format!("👁️ Splat mode for asset {}", asset_id).into());
                     } else {
                         web_sys::console::warn_1(&format!("⚠️ Asset {} has no Splat data!", asset_id).into());
                     }
                 }
            },
            3 => { // Overlay (Volume + Splats)
                 let has_volume = engine.renderer.has_volume(asset_id);
                 
                 if has_volume {
                     engine.renderer.set_active_volume(Some(asset_id));
                 }
                 if engine.renderer.has_splat(asset_id) {
                     engine.renderer.set_active_splat(Some(asset_id));
                 }
                 engine.renderer.set_viewmode("sdf_overlay");
                 #[cfg(debug_assertions)]
                 web_sys::console::log_1(&"👁️ Overlay mode: Volume + Splats".into());
            },
            _ => {}
        }
    });
}

#[wasm_bindgen]
pub fn set_show_skeleton(visible: bool) {
    with_engine_mut(|engine| engine.renderer.set_show_skeleton(visible));
}

/// Get available view modes for an asset as a packed bitmask byte.
/// Bit layout: bit0=mesh, bit1=sdf (volume||bytecode), bit2=splat, bit3=volume
/// Returns a single u8 (0 if engine not ready).
#[wasm_bindgen]
pub fn get_asset_modes(asset_id: u64) -> u8 {
    with_engine(|engine| {
        let has_mesh = engine.renderer.has_mesh(asset_id);
        let has_splat = engine.renderer.has_splat(asset_id);
        let has_volume = engine.renderer.has_volume(asset_id);
        (has_mesh as u8)
            | ((has_volume as u8) << 1) // Reusing bit 1 for volume
            | ((has_splat as u8) << 2)
            | ((has_volume as u8) << 3) // Duplicate for compatibility? Or just use bit 1?
    }).unwrap_or(0)
}

/// Check if the View Cube was clicked
#[wasm_bindgen]
pub fn pick_view_cube(mouse_x: f32, mouse_y: f32) -> JsValue {
    with_engine_mut(|engine| {
        engine.renderer.pick_view_cube(mouse_x, mouse_y)
            .map(|face| JsValue::from(face as u32))
            .unwrap_or(JsValue::NULL)
    }).unwrap_or(JsValue::NULL)
}

/// Move the camera instantly to an arbitrary position + orientation
#[wasm_bindgen]
pub fn snap_camera_to(pos_x: f32, pos_y: f32, pos_z: f32, yaw: f32, pitch: f32) {
    with_engine_mut(|engine| engine.snap_camera_to([pos_x, pos_y, pos_z], yaw, pitch));
}

/// Set selected node position (shows gizmo at that point)
#[wasm_bindgen]
pub fn set_selected_node_pos(x: f32, y: f32, z: f32) {
    with_engine_mut(|engine| engine.set_selected_node_pos(x, y, z));
}

/// Clear node selection (hides gizmo)
#[wasm_bindgen]
pub fn clear_node_selection() {
    with_engine_mut(|engine| engine.clear_node_selection());
}

/// Pick gizmo part under mouse. Returns 0=none, 1=X, 2=Y, 3=Z, 4=center.
#[wasm_bindgen]
pub fn pick_gizmo(mouse_x: f32, mouse_y: f32) -> u32 {
    with_engine(|engine| engine.pick_gizmo(mouse_x, mouse_y)).unwrap_or(0)
}

/// Drag gizmo. axis: 0=free, 1=X, 2=Y, 3=Z.
#[wasm_bindgen]
pub fn drag_gizmo(
    mouse_x: f32,
    mouse_y: f32,
    prev_mouse_x: f32,
    prev_mouse_y: f32,
    axis: u32,
) {
    with_engine_mut(|engine| {
        engine.drag_gizmo(mouse_x, mouse_y, prev_mouse_x, prev_mouse_y, axis);
    });
}

/// Get current selected node position [x, y, z]
#[wasm_bindgen]
pub fn get_selected_node_pos() -> Vec<f32> {
    with_engine(|engine| engine.get_selected_node_pos()).unwrap_or_default()
}

/// Return current scene as binary: u32 count, then per entry asset_id (u64), type (u8: 0=mesh, 1=sdf), active (u8: 0/1).
#[wasm_bindgen]
pub fn get_scene_snapshot() -> Vec<u8> {
    with_engine(|engine| engine.renderer.get_scene_snapshot()).unwrap_or_default()
}

/// Get debug info as JSON string
#[wasm_bindgen]
pub fn get_debug_info() -> String {
    with_engine(|engine| {
        let state = engine.renderer.get_debug_state();
        format!(
            r#"{{"view_mode": "{}", "show_skeleton": {}, "active_assets": {{ "splat": {}, "volume": {} }}, "camera": {{ "pos": [{:.2}, {:.2}, {:.2}], "yaw": {:.2}, "pitch": {:.2} }} }}"#,
            state.view_mode,
            state.show_skeleton,
            state.active_splat.map(|id| id.to_string()).unwrap_or("null".to_string()),
            state.active_volume.map(|id| id.to_string()).unwrap_or("null".to_string()),
            state.camera_pos[0], state.camera_pos[1], state.camera_pos[2],
            state.camera_yaw, state.camera_pitch
        )
    }).unwrap_or("{}".to_string())
}
