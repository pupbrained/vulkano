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
  /// Mouse/Gamepad look sensitivity
  pub mouse_sensitivity: f64,
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
      mouse_sensitivity: 0.005,
    }
  }

  /// Updates the camera's movement based on current velocity and input.
  ///
  /// This method implements a physics-based movement system with the following features:
  /// * Acceleration when movement input is present
  /// * Deceleration when no input is present (friction)
  /// * Maximum speed limiting
  /// * Frame-rate independent movement using delta time
  ///
  /// The movement input is expected to be a normalized direction vector where:
  /// * X: Positive = Right, Negative = Left
  /// * Y: Positive = Up, Negative = Down
  /// * Z: Positive = Backward, Negative = Forward
  ///
  /// # Parameters
  /// * `delta_time` - Time elapsed since last update in seconds
  ///
  /// # Example
  /// ```
  /// let mut camera = Camera::new();
  ///
  /// // Move forward
  /// camera.movement_input = DVec3::new(0.0, 0.0, -1.0);
  /// camera.update_movement(0.016);
  ///
  /// // Stop moving (will decelerate)
  /// camera.movement_input = DVec3::ZERO;
  /// camera.update_movement(0.016);
  /// ```
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
