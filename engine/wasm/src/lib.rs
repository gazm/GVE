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

/// Borrow the engine mutably and run a closure. Returns None if engine is not initialized.
fn with_engine_mut<R>(f: impl FnOnce(&mut WasmEngine) -> R) -> Option<R> {
    ENGINE.with(|cell| {
        let mut borrow = cell.borrow_mut();
        borrow.as_mut().map(f)
    })
}

/// Borrow the engine immutably and run a closure. Returns None if engine is not initialized.
fn with_engine<R>(f: impl FnOnce(&WasmEngine) -> R) -> Option<R> {
    ENGINE.with(|cell| {
        let borrow = cell.borrow();
        borrow.as_ref().map(f)
    })
}

#[wasm_bindgen]
pub async fn init_engine(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
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

/// Load a test SDF (sphere with box carved out) for debugging
#[wasm_bindgen]
pub fn load_test_sdf() {
    with_engine_mut(|engine| {
        engine.renderer.load_test_sdf(999);
        #[cfg(debug_assertions)]
        web_sys::console::log_1(&"🔮 Test SDF loaded!".into());
    });
}

/// Clear all loaded SDFs and return to default mesh
#[wasm_bindgen]
pub fn clear_viewport() {
    with_engine_mut(|engine| engine.renderer.clear_all());
}

#[wasm_bindgen]
pub fn clear_sdf() {
    with_engine_mut(|engine| {
        engine.renderer.set_active_sdf(None);
        #[cfg(debug_assertions)]
        web_sys::console::log_1(&"🧹 SDF cleared, showing mesh".into());
    });
}

/// Set view mode: 0=Mesh, 1=SDF (uses Volume raymarching), 2=Splat
#[wasm_bindgen]
pub fn set_view_mode(mode: u32, asset_id: u64) {
    with_engine_mut(|engine| {
        match mode {
            0 => { // Mesh (Shell)
                 engine.renderer.set_active_sdf(None);
                 engine.renderer.set_active_splat(None);
                 engine.renderer.set_active_volume(None);
                 #[cfg(debug_assertions)] {
                     if engine.renderer.has_mesh(asset_id) {
                         web_sys::console::log_1(&format!("👁️ Mesh mode for asset {}", asset_id).into());
                     } else {
                         web_sys::console::warn_1(&format!("⚠️ Asset {} has no mesh data!", asset_id).into());
                     }
                 }
            },
            1 => { // SDF - prefer Volume raymarching over SDF bytecode
                 let has_volume = engine.renderer.has_volume(asset_id);
                 if has_volume {
                     engine.renderer.set_active_volume(Some(asset_id));
                 } else if engine.renderer.has_sdf(asset_id) {
                     engine.renderer.set_active_sdf(Some(asset_id));
                 }
                 #[cfg(debug_assertions)] {
                     if has_volume {
                         web_sys::console::log_1(&format!("👁️ SDF mode (volume) for asset {}", asset_id).into());
                     } else if engine.renderer.has_sdf(asset_id) {
                         web_sys::console::log_1(&format!("👁️ SDF mode (bytecode) for asset {}", asset_id).into());
                     } else {
                         web_sys::console::warn_1(&format!("⚠️ Asset {} has no SDF/Volume data!", asset_id).into());
                     }
                 }
            },
            2 => { // Splat
                 if engine.renderer.has_splat(asset_id) {
                     engine.renderer.set_active_splat(Some(asset_id));
                 }
                 #[cfg(debug_assertions)] {
                     if engine.renderer.has_splat(asset_id) {
                         web_sys::console::log_1(&format!("👁️ Splat mode for asset {}", asset_id).into());
                     } else {
                         web_sys::console::warn_1(&format!("⚠️ Asset {} has no Splat data!", asset_id).into());
                     }
                 }
            },
            _ => {}
        }
    });
}

/// Get available view modes for an asset as a packed bitmask byte.
/// Bit layout: bit0=mesh, bit1=sdf (volume||bytecode), bit2=splat, bit3=volume
/// Returns a single u8 (0 if engine not ready).
#[wasm_bindgen]
pub fn get_asset_modes(asset_id: u64) -> u8 {
    with_engine(|engine| {
        let has_mesh = engine.renderer.has_mesh(asset_id);
        let has_sdf = engine.renderer.has_sdf(asset_id);
        let has_splat = engine.renderer.has_splat(asset_id);
        let has_volume = engine.renderer.has_volume(asset_id);
        let has_sdf_mode = has_volume || has_sdf;
        (has_mesh as u8)
            | ((has_sdf_mode as u8) << 1)
            | ((has_splat as u8) << 2)
            | ((has_volume as u8) << 3)
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

/// Toggle Axes visualization
#[wasm_bindgen]
pub fn toggle_axes() {
    with_engine_mut(|engine| engine.renderer.toggle_axes());
}

/// Return current scene as binary: u32 count, then per entry asset_id (u64), type (u8: 0=mesh, 1=sdf), active (u8: 0/1).
#[wasm_bindgen]
pub fn get_scene_snapshot() -> Vec<u8> {
    with_engine(|engine| engine.renderer.get_scene_snapshot()).unwrap_or_default()
}
