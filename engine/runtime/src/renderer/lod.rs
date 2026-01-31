//! LOD (Level of Detail) management for the renderer
//!
//! Handles distance-based LOD transitions for efficient rendering
//! of objects at varying distances from the camera.

/// Manages LOD transitions based on camera distance
#[derive(Debug, Clone)]
pub struct LodManager {
    /// Distance threshold for high-detail rendering
    pub near_distance: f32,
    /// Distance threshold for low-detail rendering
    pub far_distance: f32,
    /// Size of the transition zone between LOD levels
    pub transition_zone: f32,
}

impl LodManager {
    /// Create a new LOD manager with default settings
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Calculate LOD level (0.0 = highest detail, 1.0 = lowest)
    #[allow(dead_code)]
    pub fn calculate_lod(&self, distance: f32) -> f32 {
        if distance <= self.near_distance {
            0.0
        } else if distance >= self.far_distance {
            1.0
        } else {
            (distance - self.near_distance) / (self.far_distance - self.near_distance)
        }
    }
    
    /// Check if an object is within the transition zone
    #[allow(dead_code)]
    pub fn in_transition(&self, distance: f32) -> bool {
        let mid = (self.near_distance + self.far_distance) / 2.0;
        let half_zone = self.transition_zone / 2.0;
        distance >= mid - half_zone && distance <= mid + half_zone
    }
}

impl Default for LodManager {
    fn default() -> Self {
        Self {
            near_distance: 10.0,
            far_distance: 100.0,
            transition_zone: 20.0,
        }
    }
}
