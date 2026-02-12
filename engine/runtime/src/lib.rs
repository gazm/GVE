pub mod renderer;

pub use renderer::{Renderer, RenderConfig, Scene, View, create_renderer, render_frame};


#[no_mangle]
#[no_mangle]
pub extern "C" fn add_runtime_patch(renderer: &mut Renderer, asset_id: u64, op_type: u32, pos_x: f32, pos_y: f32, pos_z: f32, p0: f32, p1: f32, p2: f32, p3: f32, p4: f32, p5: f32, p6: f32, p7: f32) {
    renderer.add_runtime_patch(asset_id, op_type, [pos_x, pos_y, pos_z], [p0, p1, p2, p3, p4, p5, p6, p7]);
}

#[no_mangle]
pub extern "C" fn clear_runtime_patches(renderer: &mut Renderer, asset_id: u64) {
    renderer.clear_runtime_patches(asset_id);
}
