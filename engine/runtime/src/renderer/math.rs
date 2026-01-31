//! Matrix utilities for camera and projection calculations
//!
//! Contains helper functions for MVP matrix generation and
//! camera setup for both mesh and SDF rendering.

use glam::{Mat4, Vec3};

/// Calculate Model-View-Projection matrix for orbiting camera
///
/// Creates a perspective projection with an orbiting camera that
/// rotates around the origin at a fixed distance.
///
/// # Arguments
/// * `aspect` - Viewport aspect ratio (width / height)
/// * `rotation` - Current rotation angle in radians
/// * `distance` - Camera distance from origin
///
/// # Returns
/// 4x4 MVP matrix as column-major array
pub fn calculate_mvp(aspect: f32, rotation: f32, distance: f32) -> [[f32; 4]; 4] {
    let eye = Vec3::new(
        distance * rotation.sin(),
        distance * 0.5,  // Slight elevation
        distance * rotation.cos(),
    );
    let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
    
    (proj * view).to_cols_array_2d()
}

/// Calculate camera matrices for SDF raymarching
///
/// Generates the inverse view-projection matrix needed for
/// reconstructing world-space rays from screen coordinates.
///
/// # Arguments
/// * `aspect` - Viewport aspect ratio (width / height)
/// * `rotation` - Current rotation angle in radians
/// * `distance` - Camera distance from origin
///
/// # Returns
/// Tuple of (inverse_view_proj_matrix, camera_position)
pub fn calculate_sdf_camera(aspect: f32, rotation: f32, distance: f32) -> (Mat4, Vec3) {
    let eye = Vec3::new(
        distance * rotation.sin(),
        distance * 0.5,
        distance * rotation.cos(),
    );
    let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
    let view_proj = proj * view;
    let inv_view_proj = view_proj.inverse();
    
    (inv_view_proj, eye)
}

/// Calculate camera matrices from explicit position and rotation
pub fn calculate_manual_camera(aspect: f32, pos: [f32; 3], yaw: f32, pitch: f32) -> (Mat4, Vec3) {
    let eye = Vec3::from(pos);
    
    // Calculate direction from yaw/pitch
    let direction = Vec3::new(
        yaw.cos() * pitch.cos(),
        pitch.sin(),
        yaw.sin() * pitch.cos()
    ).normalize();
    
    let target = eye + direction;
    let view = Mat4::look_at_rh(eye, target, Vec3::Y);
    let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
    
    let view_proj = proj * view;
    
    (view_proj, eye)
}
