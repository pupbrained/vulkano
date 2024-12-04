//! Camera system with physics-based movement.
//!
//! This module implements a 3D camera system with smooth movement and rotation controls.
//! Features include:
//! * Position and orientation tracking with 64-bit precision
//! * Physics-based movement with acceleration and deceleration
//! * Configurable field of view (FOV)
//! * View matrix generation for rendering
//! * Smooth mouse-look controls
//!
//! # Example
//! ```
//! use vulkano_app::render::camera::Camera;
//! use glam::DVec3;
//!
//! let mut camera = Camera::new();
//!
//! // Set initial position and orientation
//! camera.position = DVec3::new(0.0, 1.0, -5.0);
//! camera.yaw = std::f64::consts::FRAC_PI_4;  // 45 degrees right
//!
//! // Move camera forward
//! camera.movement_input = DVec3::new(0.0, 0.0, -1.0);
//! camera.update_movement(0.016);  // Update with 16ms frame time
//! ```

use glam::{DMat4, DVec3};

/// A 3D camera with smooth movement and rotation controls.
///
/// The camera uses a position-target system with yaw and pitch angles for rotation.
/// It implements smooth movement with acceleration and deceleration for better control.
/// The camera's orientation is determined by its yaw (horizontal) and pitch (vertical) angles,
/// which are used to calculate the front vector.
///
/// # Properties
/// * Position and orientation are stored using 64-bit floating point values for precision
/// * Supports field of view (FOV) adjustment
/// * Implements velocity-based movement with acceleration and max speed limits
///
/// # Example
/// ```
/// use vulkano_app::Camera;
/// use glam::DVec3;
///
/// let mut camera = Camera::new();
///
/// // Update camera position
/// camera.position = DVec3::new(0.0, 1.0, -5.0);
///
/// // Rotate camera 45 degrees right
/// camera.yaw = std::f64::consts::FRAC_PI_4;
///
/// // Move camera forward
/// camera.movement_input = DVec3::new(0.0, 0.0, -1.0);
/// camera.update_movement(0.016); // Update with 16ms frame time
/// ```
#[derive(Debug)]
pub struct Camera {
  /// Current position in 3D space
  pub position:              DVec3,
  /// Horizontal rotation angle in degrees
  pub yaw:                   f64,
  /// Vertical rotation angle in degrees
  pub pitch:                 f64,
  /// Direction the camera is facing
  pub front:                 DVec3,
  /// Current movement velocity
  pub velocity:              DVec3,
  /// Rate of acceleration when movement input is received
  pub movement_acceleration: f64,
  /// Rate of deceleration when no movement input is present
  pub movement_deceleration: f64,
  /// Maximum movement speed
  pub max_speed:             f64,
  /// Current movement input vector
  pub movement_input:        DVec3,
  /// Field of view in degrees
  pub fov:                   f32,
  /// Mouse/Gamepad look sensitivity
  pub mouse_sensitivity:     f64,
}

impl Camera {
  /// Creates a new camera with default settings.
  ///
  /// Initial position is set to (-1.1, 0.1, 1.0) with the camera facing diagonally (-45° yaw).
  pub fn new() -> Self {
    Self {
      position:              DVec3::new(-1.1, 0.1, 1.0),
      yaw:                   -std::f64::consts::FRAC_PI_4, // -45 degrees
      pitch:                 0.0,
      front:                 DVec3::new(0.0, 0.0, -1.0),
      velocity:              DVec3::ZERO,
      movement_acceleration: 20.0,
      movement_deceleration: 10.0,
      max_speed:             2.0,
      movement_input:        DVec3::ZERO,
      fov:                   45.0,
      mouse_sensitivity:     0.005,
    }
  }

  /// Updates camera position based on velocity and movement input
  ///
  /// # Parameters
  /// * `movement_input` - Movement input vector in world space
  /// * `delta_time` - Time elapsed since last update in seconds
  pub fn update_movement(&mut self, movement_input: DVec3, delta_time: f64) {
    // Update velocity based on input
    let target_velocity = if movement_input.length_squared() > 0.0 {
      movement_input * self.max_speed
    } else {
      DVec3::ZERO
    };

    // Apply acceleration/deceleration
    let accel = if movement_input.length_squared() > 0.0 {
      self.movement_acceleration
    } else {
      self.movement_deceleration
    };

    // Update velocity with acceleration
    self.velocity = self.velocity.lerp(target_velocity, accel * delta_time);

    // Apply movement if velocity is non-zero
    if self.velocity.length_squared() > 0.0 {
      let movement = self.velocity * delta_time;
      self.position += movement;
    }
  }

  /// Rotates the camera by the given yaw and pitch deltas.
  ///
  /// This method handles all camera rotation, including:
  /// * Yaw (horizontal) rotation with proper wrapping around ±π
  /// * Pitch (vertical) rotation with clamping to prevent flipping
  /// * Automatic front vector updates
  ///
  /// # Parameters
  /// * `yaw_delta` - Change in horizontal rotation (negative = left, positive = right)
  /// * `pitch_delta` - Change in vertical rotation (negative = up, positive = down)
  pub fn rotate(&mut self, yaw_delta: f64, pitch_delta: f64) {
    // Update yaw and normalize to [-π, π]
    self.yaw += yaw_delta;

    // Normalize yaw to [-π, π] range
    let two_pi = 2.0 * std::f64::consts::PI;
    self.yaw = self.yaw - (two_pi * (self.yaw / two_pi).floor());
    if self.yaw > std::f64::consts::PI {
      self.yaw -= two_pi;
    }

    // Update pitch with clamping
    self.pitch += pitch_delta;
    self.pitch = self
      .pitch
      .clamp(-89.0f64.to_radians(), 89.0f64.to_radians());

    // Update front vector
    let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
    let (pitch_sin, pitch_cos) = self.pitch.sin_cos();

    self.front = DVec3::new(yaw_cos * pitch_cos, pitch_sin, yaw_sin * pitch_cos).normalize();
  }

  /// Updates camera rotation based on gamepad right stick input
  ///
  /// Applies a non-linear response curve and deadzone for smooth control
  ///
  /// # Parameters
  /// * `right_stick_x` - X-axis input from gamepad's right stick (-1.0 to 1.0)
  /// * `right_stick_y` - Y-axis input from gamepad's right stick (-1.0 to 1.0)
  pub fn update_gamepad_rotation(&mut self, right_stick_x: f64, right_stick_y: f64) {
    // Apply right stick for camera rotation with improved sensitivity
    let rotation_sensitivity = self.mouse_sensitivity * 0.5;

    // Right stick values are already processed with deadzone and curve
    if right_stick_x != 0.0 || right_stick_y != 0.0 {
      self.rotate(
        -right_stick_x * rotation_sensitivity,
        -right_stick_y * rotation_sensitivity,
      );
    }
  }

  /// Computes the view matrix for the camera's current position and orientation.
  ///
  /// The view matrix transforms world space coordinates into camera space coordinates.
  /// It is constructed by combining:
  /// * The camera's position (translation)
  /// * The camera's orientation (rotation from yaw and pitch)
  ///
  /// # Returns
  /// A 4x4 matrix that transforms world space to camera space, suitable for use
  /// in vertex shaders and other rendering calculations.
  pub fn get_view_matrix(&self) -> DMat4 {
    DMat4::look_at_rh(self.position, self.position + self.front, DVec3::Y)
  }
}

impl Default for Camera {
  fn default() -> Self {
    Self::new()
  }
}
