use std::time::Instant;

use egui_winit_vulkano::Gui;
use glam::DVec3;

#[derive(Clone, Copy)]
pub struct GuiState {
  pub fps: f32,
  pub last_frame_time: Instant,
  pub needs_pipeline_update: bool,
  pub wireframe_mode: bool,
  pub line_width: f32,
  pub max_line_width: f32,
  pub supports_wide_lines: bool,
  pub fov: f32,
}

impl Default for GuiState {
  fn default() -> Self {
    Self {
      fps: 0.0,
      last_frame_time: Instant::now(),
      needs_pipeline_update: false,
      wireframe_mode: false,
      line_width: 1.0,
      max_line_width: 1.0,
      supports_wide_lines: false,
      fov: 90.0,
    }
  }
}

/// Represents changes made by the GUI that need to be synced back to the App
pub struct GuiStateChanges {
  pub wireframe_mode: Option<bool>,
  pub line_width: Option<f32>,
  pub fov: Option<f32>,
  pub camera_reset: bool,
  pub movement_reset: bool,
  pub pass_events_to_game: bool,
  pub max_speed: Option<f64>,
  pub movement_acceleration: Option<f64>,
  pub movement_deceleration: Option<f64>,
}

impl Default for GuiStateChanges {
  fn default() -> Self {
    Self {
      wireframe_mode: None,
      line_width: None,
      fov: None,
      camera_reset: false,
      movement_reset: false,
      pass_events_to_game: true,
      max_speed: None,
      movement_acceleration: None,
      movement_deceleration: None,
    }
  }
}

pub fn draw_gui(
  gui: &mut Gui,
  state: &mut GuiState,
  camera_pos: &DVec3,
  camera_yaw: &f64,
  camera_pitch: &f64,
  max_speed: &f64,
  movement_acceleration: &f64,
  movement_deceleration: &f64,
  current_movement_speed: &f64,
) -> GuiStateChanges {
  let mut changes = GuiStateChanges::default();

  // Calculate FPS
  let now = Instant::now();
  let frame_time = now.duration_since(state.last_frame_time).as_secs_f32();
  state.fps = 1.0 / frame_time;
  state.last_frame_time = now;

  gui.immediate_ui(|gui| {
    let ctx = gui.context();
    changes.pass_events_to_game = !ctx.wants_pointer_input() && !ctx.wants_keyboard_input();

    egui::Window::new("Stats & Controls")
      .default_pos([10.0, 10.0])
      .show(&gui.context(), |ui| {
        // Performance stats
        ui.heading("Performance");
        ui.label(format!("FPS: {:.1}", state.fps));
        ui.label(format!("Frame Time: {:.2}ms", frame_time * 1000.0));

        ui.separator();

        // Camera position info
        ui.heading("Camera Position");
        ui.label(format!("X: {:.2}", camera_pos.x));
        ui.label(format!("Y: {:.2}", camera_pos.y));
        ui.label(format!("Z: {:.2}", camera_pos.z));
        ui.label(format!("Yaw: {:.1}°", camera_yaw.to_degrees()));
        ui.label(format!("Pitch: {:.1}°", camera_pitch.to_degrees()));

        ui.separator();

        // Movement settings
        ui.heading("Movement Settings");
        
        // Speed control
        let mut current_speed = *max_speed;
        ui.horizontal(|ui| {
          ui.label("Speed:");
          if ui
            .add(egui::Slider::new(&mut current_speed, 0.1..=10.0).step_by(0.1))
            .changed()
          {
            changes.max_speed = Some(current_speed);
          }
        });

        // Acceleration control
        let mut current_accel = *movement_acceleration;
        ui.horizontal(|ui| {
          ui.label("Acceleration:");
          if ui
            .add(egui::Slider::new(&mut current_accel, 1.0..=50.0).step_by(1.0))
            .changed()
          {
            changes.movement_acceleration = Some(current_accel);
          }
        });

        // Deceleration control
        let mut current_decel = *movement_deceleration;
        ui.horizontal(|ui| {
          ui.label("Deceleration:");
          if ui
            .add(egui::Slider::new(&mut current_decel, 1.0..=50.0).step_by(1.0))
            .changed()
          {
            changes.movement_deceleration = Some(current_decel);
          }
        });

        // Current velocity display
        ui.label(format!("Current Speed: {:.2} units/s", current_movement_speed));

        ui.separator();

        // Rendering settings
        ui.heading("Rendering");
        if ui
          .checkbox(&mut state.wireframe_mode, "Wireframe Mode")
          .changed()
        {
          changes.wireframe_mode = Some(state.wireframe_mode);
          state.needs_pipeline_update = true;
        }

        if state.wireframe_mode {
          if state.supports_wide_lines {
            if ui
              .add(
                egui::Slider::new(&mut state.line_width, 1.0..=state.max_line_width)
                  .text("Line Width"),
              )
              .changed()
            {
              changes.line_width = Some(state.line_width);
              state.needs_pipeline_update = true;
            }
          } else {
            ui.label("Wide lines not supported on this device");
          }
        }

        // Add FOV slider
        ui.horizontal(|ui| {
          ui.label("Field of View:");
          if ui
            .add(egui::Slider::new(&mut state.fov, 30.0..=120.0).step_by(1.0))
            .changed()
          {
            changes.fov = Some(state.fov);
            state.needs_pipeline_update = true;
          }
        });

        ui.separator();

        // Controls help
        ui.heading("Controls");
        ui.label("WASD - Move horizontally");
        ui.label("Space/Shift - Move up/down");

        ui.separator();

        // Reset buttons
        if ui.button("Reset Camera Position").clicked() {
          changes.camera_reset = true;
        }
        if ui.button("Reset Movement Settings").clicked() {
          changes.movement_reset = true;
        }
      });
  });

  changes
}
