use glam::{DMat4, DVec3};

/// A 3D camera with smooth movement and rotation controls.
///
/// The camera uses a position-target system with yaw and pitch angles for rotation.
/// It implements smooth movement with acceleration and deceleration for better control.
///
/// # Properties
/// * Position and orientation are stored using 64-bit floating point values for precision
/// * Supports field of view (FOV) adjustment
/// * Implements velocity-based movement with acceleration and max speed limits
pub struct Camera {
  /// Current position in 3D space
  pub position: DVec3,
  /// Horizontal rotation angle in degrees
  pub yaw: f64,
  /// Vertical rotation angle in degrees
  pub pitch: f64,
  /// Direction the camera is facing
  pub front: DVec3,
  /// Current movement velocity
  pub velocity: DVec3,
  /// Rate of acceleration when movement input is received
  pub movement_acceleration: f64,
  /// Rate of deceleration when no movement input is present
  pub movement_deceleration: f64,
  /// Maximum movement speed
  pub max_speed: f64,
  /// Current movement input vector
  pub movement_input: DVec3,
  /// Field of view in degrees
  pub fov: f32,
}

impl Camera {
  /// Creates a new camera with default settings.
  ///
  /// Initial position is set to (-1.1, 0.1, 1.0) with the camera facing diagonally (-45° yaw).
  pub fn new() -> Self {
    Self {
      position: DVec3::new(-1.1, 0.1, 1.0),
      yaw: -std::f64::consts::FRAC_PI_4, // -45 degrees
      pitch: 0.0,
      front: DVec3::new(0.0, 0.0, -1.0),
      velocity: DVec3::ZERO,
      movement_acceleration: 20.0,
      movement_deceleration: 10.0,
      max_speed: 2.0,
      movement_input: DVec3::ZERO,
      fov: 45.0,
    }
  }

  /// Updates the camera's movement based on current velocity and input.
  ///
  /// This method handles:
  /// * Acceleration based on movement input
  /// * Deceleration when no input is present
  /// * Velocity clamping to max speed
  ///
  /// # Parameters
  /// * `delta_time` - Time elapsed since last update in seconds
  pub fn update_movement(&mut self, delta_time: f64) {
    // Apply acceleration based on input
    let acceleration = self.movement_input * self.movement_acceleration;
    self.velocity += acceleration * delta_time;

    // Apply deceleration when no input
    if self.movement_input.length_squared() < 0.1 {
      let deceleration = -self.velocity.normalize_or_zero() * self.movement_deceleration;
      self.velocity += deceleration * delta_time;

      // Stop completely if velocity is very small
      if self.velocity.length_squared() < 0.01 {
        self.velocity = DVec3::ZERO;
      }
    }

    // Clamp velocity to max speed
    if self.velocity.length_squared() > self.max_speed * self.max_speed {
      self.velocity = self.velocity.normalize() * self.max_speed;
    }

    // Update position
    self.position += self.velocity * delta_time;
  }

  /// Computes the view matrix for the camera's current position and orientation.
  ///
  /// Returns a 4x4 matrix that transforms world space to camera space.
  pub fn get_view_matrix(&self) -> DMat4 {
    DMat4::look_at_rh(self.position, self.position + self.front, DVec3::Y)
  }
}

impl Default for Camera {
  fn default() -> Self {
    Self::new()
  }
}
