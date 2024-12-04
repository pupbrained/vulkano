//! GUI Implementation using egui
//!
//! This module implements the application's graphical user interface using egui.
//! It provides real-time statistics, rendering options, and camera controls.

use std::time::Instant;

use egui_winit_vulkano::Gui;

use crate::Camera;

/// Stores the current state of the GUI including performance metrics and rendering options
#[derive(Clone, Copy)]
pub struct GuiState {
  /// Current frames per second
  pub fps:                    f32,
  /// Average frames per second over time
  pub avg_fps:                f32,
  /// Total number of frames rendered
  pub frame_count:            u32,
  /// Accumulated frame times for averaging
  pub frame_time_accumulator: f32,
  /// Timestamp of the last frame
  pub last_frame_time:        Instant,
  /// Timestamp of last FPS average calculation
  pub last_avg_update:        Instant,
  /// Flag indicating if the rendering pipeline needs to be rebuilt
  pub needs_pipeline_update:  bool,
  /// Toggle for wireframe rendering mode
  pub wireframe_mode:         bool,
  /// Current line width for wireframe rendering
  pub line_width:             f32,
  /// Maximum supported line width by the GPU
  pub max_line_width:         f32,
  /// Whether the GPU supports wide lines
  pub supports_wide_lines:    bool,
  /// Field of view in degrees
  pub fov:                    f32,
}

impl Default for GuiState {
  /// Creates a default `GuiState` instance with initial values
  fn default() -> Self {
    Self {
      fps:                    0.0,
      avg_fps:                0.0,
      frame_count:            0,
      frame_time_accumulator: 0.0,
      last_frame_time:        Instant::now(),
      last_avg_update:        Instant::now(),
      needs_pipeline_update:  false,
      wireframe_mode:         false,
      line_width:             1.0,
      max_line_width:         1.0,
      supports_wide_lines:    false,
      fov:                    90.0,
    }
  }
}

/// Represents changes made by the GUI that need to be synced back to the App
#[derive(Default)]
pub struct GuiStateChanges {
  /// Whether the rendering pipeline needs to be rebuilt
  pub rebuild_pipeline:      bool,
  /// New wireframe rendering state
  pub wireframe_mode:        Option<bool>,
  /// New line width for wireframe rendering
  pub line_width:            Option<f32>,
  /// New field of view setting
  pub fov:                   Option<f32>,
  /// Flag to reset camera position
  pub camera_reset:          bool,
  /// Flag to reset movement settings
  pub movement_reset:        bool,
  /// Flag to pass events to the game
  pub pass_events_to_game:   bool,
  /// New maximum speed setting
  pub max_speed:             Option<f64>,
  /// New movement acceleration setting
  pub movement_acceleration: Option<f64>,
  /// New movement deceleration setting
  pub movement_deceleration: Option<f64>,
}

/// Draws the GUI frame and handles user interactions
///
/// # Arguments
/// * `gui` - The egui context
/// * `state` - Current GUI state
/// * `camera` - Camera controller for view manipulation
///
/// # Returns
/// A `GuiStateChanges` struct containing any modifications made through the GUI
pub fn draw_gui(gui: &mut Gui, state: &mut GuiState, camera: &mut Camera) -> GuiStateChanges {
  let mut changes = GuiStateChanges::default();

  // Calculate FPS
  let now = Instant::now();
  let frame_time = now.duration_since(state.last_frame_time).as_secs_f32();
  state.fps = 1.0 / frame_time;
  state.frame_time_accumulator += frame_time;
  state.frame_count += 1;

  // Update average FPS once per second
  let time_since_last_update = now.duration_since(state.last_avg_update).as_secs_f32();
  if time_since_last_update >= 1.0 {
    state.avg_fps = state.frame_count as f32 / state.frame_time_accumulator;
    state.frame_count = 0;
    state.frame_time_accumulator = 0.0;
    state.last_avg_update = now;
  }

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
        ui.label(format!("Avg FPS: {:.1}", state.avg_fps));
        ui.label(format!("Frame Time: {:.2}ms", frame_time * 1000.0));

        ui.separator();

        // Camera position info
        ui.heading("Camera Position");
        ui.label(format!("X: {:.2}", camera.position.x));
        ui.label(format!("Y: {:.2}", camera.position.y));
        ui.label(format!("Z: {:.2}", camera.position.z));
        ui.label(format!("Yaw: {:.1}°", camera.yaw.to_degrees()));
        ui.label(format!("Pitch: {:.1}°", camera.pitch.to_degrees()));

        ui.separator();

        // Movement settings
        ui.heading("Movement Settings");

        // Speed control
        ui.horizontal(|ui| {
          ui.label("Speed:");
          if ui
            .add(egui::Slider::new(&mut camera.max_speed, 0.1..=10.0).step_by(0.1))
            .changed()
          {
            changes.max_speed = Some(camera.max_speed);
          }
        });

        // Acceleration control
        ui.horizontal(|ui| {
          ui.label("Acceleration:");
          if ui
            .add(egui::Slider::new(&mut camera.movement_acceleration, 1.0..=50.0).step_by(1.0))
            .changed()
          {
            changes.movement_acceleration = Some(camera.movement_acceleration);
          }
        });

        // Deceleration control
        ui.horizontal(|ui| {
          ui.label("Deceleration:");
          if ui
            .add(egui::Slider::new(&mut camera.movement_deceleration, 1.0..=50.0).step_by(1.0))
            .changed()
          {
            changes.movement_deceleration = Some(camera.movement_deceleration);
          }
        });

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
