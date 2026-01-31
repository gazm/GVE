mod wasm_engine;
mod message_handler;

use wasm_bindgen::prelude::*;
use std::sync::Mutex;
use once_cell::sync::Lazy;
use crate::wasm_engine::WasmEngine;
use crate::message_handler::handle_binary_message;

static ENGINE: Lazy<Mutex<Option<WasmEngine>>> = Lazy::new(|| Mutex::new(None));

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
    
    let mut global_engine = ENGINE.lock().map_err(|_| JsValue::from_str("Failed to lock engine"))?;
    *global_engine = Some(engine);

    web_sys::console::log_1(&"GVE WASM Engine Initialized".into());
    Ok(())
}

#[wasm_bindgen]
pub fn render_frame(_dt: f32) {
    if let Ok(mut engine_opt) = ENGINE.lock() {
        if let Some(engine) = engine_opt.as_mut() {
            if let Err(e) = engine.render() {
                web_sys::console::error_1(&format!("Render error: {:?}", e).into());
            }
        }
    }
}

#[wasm_bindgen]
pub fn resize_viewport(width: u32, height: u32) {
    if let Ok(mut engine_opt) = ENGINE.lock() {
        if let Some(engine) = engine_opt.as_mut() {
            engine.resize(width, height);
        }
    }
}

#[wasm_bindgen]
pub fn handle_message(data: &[u8]) {
    if let Ok(mut engine_opt) = ENGINE.lock() {
        if let Some(engine) = engine_opt.as_mut() {
            handle_binary_message(engine, data);
        }
    }
}

/// Load a test SDF (sphere with box carved out) for debugging
#[wasm_bindgen]
pub fn load_test_sdf() {
    if let Ok(mut engine_opt) = ENGINE.lock() {
        if let Some(engine) = engine_opt.as_mut() {
            engine.renderer.load_test_sdf(999);  // Use asset ID 999 for test
            web_sys::console::log_1(&"🔮 Test SDF loaded!".into());
        }
    }
}

/// Clear all loaded SDFs and return to default mesh
#[wasm_bindgen]
pub fn clear_viewport() {
    if let Ok(mut engine_opt) = ENGINE.lock() {
        if let Some(engine) = engine_opt.as_mut() {
            engine.renderer.clear_all();
        }
    }
}

#[wasm_bindgen]
pub fn clear_sdf() {
    if let Ok(mut engine_opt) = ENGINE.lock() {
        if let Some(engine) = engine_opt.as_mut() {
            engine.renderer.set_active_sdf(None);
            web_sys::console::log_1(&"🧹 SDF cleared, showing mesh".into());
        }
    }
}

/// Check if the View Cube was clicked
#[wasm_bindgen]
pub fn pick_view_cube(mouse_x: f32, mouse_y: f32) -> JsValue {
    if let Ok(mut engine_opt) = ENGINE.lock() {
        if let Some(engine) = engine_opt.as_mut() {
            if let Some(face) = engine.renderer.pick_view_cube(mouse_x, mouse_y) {
                 // Return the face index/name
                 // 0=Right, 1=Left, 2=Top, 3=Bottom, 4=Front, 5=Back
                 return JsValue::from(face as u32);
            }
        }
    }
    JsValue::NULL
}

/// Toggle Axes visualization
#[wasm_bindgen]
pub fn toggle_axes() {
    if let Ok(mut engine_opt) = ENGINE.lock() {
        if let Some(engine) = engine_opt.as_mut() {
            engine.renderer.toggle_axes();
        }
    }
}

/// Return current scene as binary: u32 count, then per entry asset_id (u64), type (u8: 0=mesh, 1=sdf), active (u8: 0/1).
#[wasm_bindgen]
pub fn get_scene_snapshot() -> Vec<u8> {
    if let Ok(engine_opt) = ENGINE.lock() {
        if let Some(engine) = engine_opt.as_ref() {
            return engine.renderer.get_scene_snapshot();
        }
    }
    Vec::new()
}
