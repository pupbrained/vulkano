use glam::{DMat4, DVec3};

pub struct Camera {
  pub position: DVec3,
  pub yaw: f64,
  pub pitch: f64,
  pub front: DVec3,
  pub velocity: DVec3,
  pub movement_acceleration: f64,
  pub movement_deceleration: f64,
  pub max_speed: f64,
  pub movement_input: DVec3,
  pub fov: f32,
}

impl Camera {
  pub fn new() -> Self {
    Self {
      position: DVec3::new(0.0, 0.0, -2.0),
      yaw: -90.0,
      pitch: 0.0,
      front: DVec3::new(0.0, 0.0, -1.0),
      velocity: DVec3::ZERO,
      movement_acceleration: 20.0,
      movement_deceleration: 10.0,
      max_speed: 5.0,
      movement_input: DVec3::ZERO,
      fov: 45.0,
    }
  }

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

  pub fn get_view_matrix(&self) -> DMat4 {
    DMat4::look_at_rh(self.position, self.position + self.front, DVec3::Y)
  }
}

impl Default for Camera {
  fn default() -> Self {
    Self::new()
  }
}
