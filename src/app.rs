//! Main application logic and Vulkan initialization.
//!
//! This module handles:
//! * Vulkan device selection and initialization
//! * Window and event management
//! * Resource creation and management
//! * Render loop coordination
//! * Camera and input handling

use std::{
  sync::Arc,
  time::Instant,
};

use egui_winit_vulkano::{Gui, GuiConfig};
use glam::{DMat3, DMat4, DVec3, Mat4};
use vulkano::{
  buffer::allocator::SubbufferAllocator,
  command_buffer::{
    allocator::StandardCommandBufferAllocator,
    AutoCommandBufferBuilder,
    CommandBufferUsage,
  },
  descriptor_set::{allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet},
  device::{Device, Queue},
  format::Format,
  image::{sampler::Sampler, view::ImageView, ImageUsage},
  instance::Instance,
  memory::allocator::StandardMemoryAllocator,
  render_pass::Subpass,
  swapchain::{acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
  sync::{self, GpuFuture},
  Validated,
  VulkanError,
};
use vulkano::pipeline::Pipeline;
use winit::{
  application::ApplicationHandler,
  dpi::{LogicalPosition, LogicalSize},
  event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
  event_loop::{ActiveEventLoop, EventLoop},
  window::{Window, WindowId},
};

use crate::{
  command_buffer_builder_ext::AutoCommandBufferBuilderExt,
  init::initialize_vulkan,
  model::VikingRoomModelBuffers,
  render::{window_size_dependent_setup, RenderContext, WindowSizeSetupConfig},
  shaders::{fs, vs},
};

/// Main application state containing all Vulkan and window resources.
///
/// This struct manages the complete state of the Vulkan application, including:
/// * Vulkan instance, device, and resource management
/// * Camera and movement controls
/// * Rendering pipeline configuration
/// * Performance monitoring
pub struct App {
    // Vulkan core resources
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,

    // Memory and resource allocators
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    uniform_buffer_allocator: SubbufferAllocator,

    // 3D model and texture resources
    model_buffers: VikingRoomModelBuffers,
    texture: Arc<ImageView>,
    sampler: Arc<Sampler>,

    // Rendering context and UI
    rcx: Option<RenderContext>,
    gui: Option<Gui>,

    // Performance monitoring
    last_frame_time: Instant,
    fps: f32,

    // Camera position and orientation
    camera_pos: DVec3,
    camera_yaw: f64,
    camera_pitch: f64,
    camera_front: DVec3,

    // Camera movement parameters
    camera_velocity: DVec3,
    movement_acceleration: f64,
    movement_deceleration: f64,
    max_speed: f64,
    movement_input: DVec3,

    // Rendering settings
    wireframe_mode: bool,
    line_width: f32,
    max_line_width: f32,
    needs_pipeline_update: bool,
    supports_wide_lines: bool,
    cursor_captured: bool,
    fov: f32, // Field of view in degrees
}

impl App {
  /// Creates a new application instance and initializes all Vulkan resources.
  ///
  /// This function:
  /// * Creates the Vulkan instance and selects a suitable physical device
  /// * Sets up the logical device and command queues
  /// * Creates the swapchain and render passes
  /// * Loads the 3D model and textures
  /// * Initializes the GUI system
  ///
  /// # Parameters
  /// * `event_loop` - The winit event loop to create the window for
  pub fn new(event_loop: &EventLoop<()>) -> Self {
    let initialized = initialize_vulkan(event_loop);
    
    App {
      instance: initialized.instance,
      device: initialized.device,
      queue: initialized.queue,
      memory_allocator: initialized.memory_allocator,
      descriptor_set_allocator: initialized.descriptor_set_allocator,
      command_buffer_allocator: initialized.command_buffer_allocator,
      model_buffers: initialized.model_buffers,
      uniform_buffer_allocator: initialized.uniform_buffer_allocator,
      texture: initialized.texture,
      sampler: initialized.sampler,
      rcx: None,
      gui: None,
      last_frame_time: Instant::now(),
      fps: 0.0,
      // Camera state
      camera_pos: DVec3::new(-1.1, 0.1, 1.0),
      camera_yaw: -std::f64::consts::FRAC_PI_4,
      camera_pitch: 0.0,
      camera_front: DVec3::new(
        (-std::f64::consts::FRAC_PI_4).cos() * 0.0f64.cos(),
        0.0f64.sin(),
        (-std::f64::consts::FRAC_PI_4).sin() * 0.0f64.cos(),
      )
      .normalize(),
      camera_velocity: DVec3::ZERO,
      movement_acceleration: 20.0,
      movement_deceleration: 10.0,
      max_speed: 2.0,
      movement_input: DVec3::ZERO,
      // Rendering settings
      wireframe_mode: false,
      line_width: 1.0,
      max_line_width: initialized.max_line_width,
      needs_pipeline_update: false,
      supports_wide_lines: initialized.supports_wide_lines,
      cursor_captured: false,
      fov: 90.0, // Default 90 degree FOV
    }
  }

  /// Updates camera position and orientation based on current movement state.
  ///
  /// Applies velocity and acceleration to smoothly move the camera
  /// according to user input.
  ///
  /// # Parameters
  /// * `delta_time` - Time elapsed since last update in seconds
  pub fn update_camera_movement(&mut self, delta_time: f64) {
    // Calculate movement direction based on input
    let forward = DVec3::new(self.camera_yaw.cos(), 0.0, self.camera_yaw.sin()).normalize();

    let right = forward.cross(DVec3::new(0.0, -1.0, 0.0)).normalize();

    // Calculate target velocity based on input
    let mut target_velocity = DVec3::ZERO;
    if self.movement_input.length() > 0.0 {
      // Combine horizontal movement
      target_velocity += forward * self.movement_input.z;
      target_velocity += right * self.movement_input.x;
      // Add vertical movement
      target_velocity.y = self.movement_input.y;

      // Normalize and scale to max speed if moving diagonally
      if target_velocity.length() > 1.0 {
        target_velocity = target_velocity.normalize();
      }
      target_velocity *= self.max_speed;
    }

    // Accelerate or decelerate towards target velocity
    let acceleration = if target_velocity.length() > 0.0 {
      self.movement_acceleration
    } else {
      self.movement_deceleration
    };

    // Update velocity with acceleration
    let velocity_delta = (target_velocity - self.camera_velocity) * acceleration * delta_time;
    self.camera_velocity += velocity_delta;

    // Update position
    self.camera_pos += self.camera_velocity * delta_time;
  }
}

impl ApplicationHandler for App {
  /// Handles application resume events.
  ///
  /// Called when the application window gains focus or is restored.
  /// Recreates any resources that may have been lost during suspension.
  fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    let window = Arc::new(
      event_loop
        .create_window(
          Window::default_attributes()
            .with_decorations(true)
            .with_title("Vulkano App")
            .with_inner_size(LogicalSize::new(1280, 720))
            .with_position(if !cfg!(target_os = "linux") {
              LogicalPosition::new(
                (event_loop.primary_monitor().unwrap().size().width as i32 - 1280) / 2,
                (event_loop.primary_monitor().unwrap().size().height as i32 - 720) / 2,
              )
            } else {
              LogicalPosition::new(0, 0) // Default position on Linux
            }),
        )
        .unwrap(),
    );

    let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
    let window_size = window.inner_size();

    let (swapchain, images) = {
      let surface_capabilities = self
        .device
        .physical_device()
        .surface_capabilities(&surface, Default::default())
        .unwrap();

      println!(
        "Supported composite alpha modes: {:?}",
        surface_capabilities.supported_composite_alpha
      );

      let (image_format, _) = self
        .device
        .physical_device()
        .surface_formats(&surface, Default::default())
        .unwrap()
        .into_iter()
        .find(|(format, _)| {
          matches!(
            format,
            Format::B8G8R8A8_UNORM | Format::R8G8B8A8_UNORM | Format::A8B8G8R8_UNORM_PACK32
          )
        })
        .unwrap_or_else(|| {
          self
            .device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
        });

      println!("Selected format: {:?}", image_format);

      Swapchain::new(
        self.device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
          min_image_count: surface_capabilities.min_image_count.max(2),
          image_format,
          image_extent: window_size.into(),
          image_usage: ImageUsage::COLOR_ATTACHMENT
            | ImageUsage::TRANSFER_DST
            | ImageUsage::TRANSFER_SRC,
          composite_alpha: vulkano::swapchain::CompositeAlpha::Opaque,
          pre_transform: surface_capabilities.current_transform,
          clipped: true,
          ..Default::default()
        },
      )
      .unwrap()
    };

    let render_pass = vulkano::ordered_passes_renderpass!(
        self.device.clone(),
        attachments: {
            msaa_color: {
                format: swapchain.image_format(),
                samples: 4,
                load_op: Clear,
                store_op: DontCare,
            },
            final_color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: DontCare,
                store_op: Store,
            },
            depth: {
                format: Format::D32_SFLOAT,
                samples: 4,
                load_op: Clear,
                store_op: DontCare,
            }
        },
        passes: [
            {
                color: [msaa_color],
                color_resolve: [final_color],
                depth_stencil: {depth},
                input: []
            },
            {
                color: [final_color],
                depth_stencil: {},
                input: []
            }
        ]
    )
    .unwrap();

    let vs = vs::load(self.device.clone())
      .unwrap()
      .entry_point("main")
      .unwrap();
    let fs = fs::load(self.device.clone())
      .unwrap()
      .entry_point("main")
      .unwrap();

    let swapchain_image_views: Vec<_> = images
      .iter()
      .map(|image| ImageView::new_default(image.clone()).unwrap())
      .collect();

    let (framebuffers, pipeline) = window_size_dependent_setup(WindowSizeSetupConfig {
      window_size,
      images: &images,
      render_pass: &render_pass,
      memory_allocator: &self.memory_allocator,
      vertex_shader: &vs,
      fragment_shader: &fs,
      wireframe_mode: self.wireframe_mode,
      line_width: self.line_width,
    });

    let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

    self.gui = Some(Gui::new_with_subpass(
      event_loop,
      surface.clone(),
      self.queue.clone(),
      Subpass::from(render_pass.clone(), 1).unwrap(),
      swapchain.image_format(),
      GuiConfig::default(),
    ));

    self.rcx = Some(RenderContext {
      window,
      swapchain,
      render_pass,
      framebuffers,
      vs,
      fs,
      pipeline,
      recreate_swapchain: false,
      previous_frame_end,
      swapchain_image_views,
    });
  }

  /// Processes window events such as resizing, keyboard input, and mouse movement.
  ///
  /// Handles:
  /// * Window resize events by recreating the swapchain
  /// * Keyboard input for camera movement
  /// * Mouse input for camera rotation
  /// * Window close events
  fn window_event(
    &mut self,
    event_loop: &ActiveEventLoop,
    _window_id: WindowId,
    event: WindowEvent,
  ) {
    let mut pass_events_to_game = true;
    if let Some(gui) = &mut self.gui {
      pass_events_to_game = !gui.update(&event);
    }

    let rcx = self.rcx.as_mut().unwrap();

    match event {
      WindowEvent::CloseRequested => {
        event_loop.exit();
      }
      WindowEvent::Resized(_) => {
        rcx.recreate_swapchain = true;
      }
      WindowEvent::KeyboardInput {
        event:
          winit::event::KeyEvent {
            physical_key: key,
            state,
            ..
          },
        ..
      } => {
        use winit::{event::ElementState, keyboard::PhysicalKey};

        if !self.cursor_captured {
          // Only handle Escape key when cursor is not captured
          if let PhysicalKey::Code(winit::keyboard::KeyCode::Escape) = key {
            if state == ElementState::Pressed {
              self.cursor_captured = false;
              rcx
                .window
                .set_cursor_grab(winit::window::CursorGrabMode::None)
                .unwrap();
              rcx.window.set_cursor_visible(true);
            }
          }
          return;
        }

        let value = match state {
          ElementState::Pressed => 1.0,
          ElementState::Released => 0.0,
        };

        match key {
          PhysicalKey::Code(winit::keyboard::KeyCode::KeyW) => {
            self.movement_input.z = value;
          }
          PhysicalKey::Code(winit::keyboard::KeyCode::KeyS) => {
            self.movement_input.z = -value;
          }
          PhysicalKey::Code(winit::keyboard::KeyCode::KeyA) => {
            self.movement_input.x = -value;
          }
          PhysicalKey::Code(winit::keyboard::KeyCode::KeyD) => {
            self.movement_input.x = value;
          }
          PhysicalKey::Code(winit::keyboard::KeyCode::Space) => {
            self.movement_input.y = value;
          }
          PhysicalKey::Code(winit::keyboard::KeyCode::ShiftLeft) => {
            self.movement_input.y = -value;
          }
          PhysicalKey::Code(winit::keyboard::KeyCode::Escape) => {
            if state == ElementState::Pressed {
              self.cursor_captured = false;
              rcx
                .window
                .set_cursor_grab(winit::window::CursorGrabMode::None)
                .unwrap();
              rcx.window.set_cursor_visible(true);
            }
          }
          _ => {}
        }

        // Wait for any pending operations to complete before updating the pipeline
        if let Some(rcx) = &mut self.rcx {
          rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();
        }
      }
      WindowEvent::MouseInput {
        state,
        button,
        ..
      } => {
        if button == MouseButton::Left && pass_events_to_game && state == ElementState::Pressed {
          self.cursor_captured = true;
          // Try Locked mode first, fall back to Confined if not supported
          if rcx
            .window
            .set_cursor_grab(winit::window::CursorGrabMode::Locked)
            .is_err()
          {
            rcx
              .window
              .set_cursor_grab(winit::window::CursorGrabMode::Confined)
              .unwrap();
          }
          rcx.window.set_cursor_visible(false);
        }
      }
      WindowEvent::CursorMoved { .. } => {}
      WindowEvent::MouseWheel { delta, .. } => {
        if self.cursor_captured {
          match delta {
            winit::event::MouseScrollDelta::LineDelta(_, y) => {
              // Adjust FOV by 5 degrees per scroll line, inverted direction
              self.fov = (self.fov - y * 5.0).clamp(30.0, 120.0);
              self.needs_pipeline_update = true;
            }
            winit::event::MouseScrollDelta::PixelDelta(pos) => {
              // Adjust FOV by 5 degrees per 50 pixels, inverted direction
              self.fov = (self.fov - (pos.y as f32 / 50.0) * 5.0).clamp(30.0, 120.0);
              self.needs_pipeline_update = true;
            }
          }
        }
      }
      WindowEvent::RedrawRequested => {
        let now = Instant::now();
        let frame_time = now.duration_since(self.last_frame_time).as_secs_f64();
        self.update_camera_movement(frame_time);

        let rcx = self.rcx.as_mut().unwrap();
        let window_size = rcx.window.inner_size();

        if window_size.width == 0 || window_size.height == 0 {
          return;
        }

        rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if rcx.recreate_swapchain || self.needs_pipeline_update {
          let (new_swapchain, new_images) = rcx
            .swapchain
            .recreate(SwapchainCreateInfo {
              image_extent: window_size.into(),
              ..rcx.swapchain.create_info()
            })
            .expect("failed to recreate swapchain");

          rcx.swapchain = new_swapchain;
          let swapchain_image_views: Vec<_> = new_images
            .iter()
            .map(|image| ImageView::new_default(image.clone()).unwrap())
            .collect();
          rcx.swapchain_image_views = swapchain_image_views;
          (rcx.framebuffers, rcx.pipeline) = window_size_dependent_setup(WindowSizeSetupConfig {
            window_size,
            images: &new_images,
            render_pass: &rcx.render_pass,
            memory_allocator: &self.memory_allocator,
            vertex_shader: &rcx.vs,
            fragment_shader: &rcx.fs,
            wireframe_mode: self.wireframe_mode,
            line_width: self.line_width,
          });
          rcx.recreate_swapchain = false;
          self.needs_pipeline_update = false;
        }

        let uniform_buffer = {
          // Apply fixed rotations to orient the model correctly
          let vertical_rotation = DMat3::from_rotation_x(-std::f64::consts::FRAC_PI_2);
          let horizontal_rotation = DMat3::from_rotation_y(std::f64::consts::PI); // 180 degree rotation
          let initial_rotation = horizontal_rotation * vertical_rotation;

          let aspect_ratio =
            rcx.swapchain.image_extent()[0] as f32 / rcx.swapchain.image_extent()[1] as f32;

          let proj = Mat4::perspective_rh(self.fov.to_radians(), aspect_ratio, 0.01, 100.0);

          // Update view matrix based on camera position
          let view = DMat4::look_at_rh(
            self.camera_pos,
            self.camera_pos + self.camera_front,
            DVec3::new(0.0, -1.0, 0.0),
          );

          let scale = DMat4::from_scale(DVec3::splat(1.0));

          let uniform_data = vs::Data {
            world: DMat4::from_mat3(initial_rotation)
              .to_cols_array_2d()
              .map(|row| row.map(|val| val as f32)),
            view: (view * scale)
              .to_cols_array_2d()
              .map(|row| row.map(|val| val as f32)),
            proj: proj.to_cols_array_2d(),
          };

          let buffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
          *buffer.write().unwrap() = uniform_data;

          buffer
        };

        let layout = &rcx.pipeline.layout().set_layouts()[0];
        let descriptor_set = DescriptorSet::new(
          self.descriptor_set_allocator.clone(),
          layout.clone(),
          [
            WriteDescriptorSet::buffer(0, uniform_buffer.clone()),
            WriteDescriptorSet::image_view_sampler(1, self.texture.clone(), self.sampler.clone()),
          ],
          [],
        )
        .unwrap();

        let (image_index, suboptimal, acquire_future) =
          match acquire_next_image(rcx.swapchain.clone(), None).map_err(Validated::unwrap) {
            Ok(r) => r,
            Err(VulkanError::OutOfDate) => {
              rcx.recreate_swapchain = true;
              return;
            }
            Err(e) => panic!("failed to acquire next image: {e}"),
          };

        if suboptimal {
          rcx.recreate_swapchain = true;
        }

        // Update egui UI before rendering
        if let Some(gui) = &mut self.gui {
          gui.immediate_ui(|gui| {
            egui::Window::new("Stats & Controls")
              .default_pos([10.0, 10.0])
              .show(&gui.context(), |ui| {
                // Performance stats
                ui.heading("Performance");
                let now = Instant::now();
                let frame_time = now.duration_since(self.last_frame_time).as_secs_f32();
                self.fps = 1.0 / frame_time;
                self.last_frame_time = now;
                ui.label(format!("FPS: {:.1}", self.fps));
                ui.label(format!("Frame Time: {:.2}ms", frame_time * 1000.0));

                ui.separator();

                // Camera position info
                ui.heading("Camera Position");
                ui.label(format!("X: {:.2}", self.camera_pos.x));
                ui.label(format!("Y: {:.2}", self.camera_pos.y));
                ui.label(format!("Z: {:.2}", self.camera_pos.z));
                ui.label(format!("Yaw: {:.1}°", self.camera_yaw.to_degrees()));
                ui.label(format!("Pitch: {:.1}°", self.camera_pitch.to_degrees()));

                ui.separator();

                // Movement settings
                ui.heading("Movement Settings");
                ui.horizontal(|ui| {
                  ui.label("Speed:");
                  if ui.small_button("-").clicked() && self.max_speed > 0.5 {
                    self.max_speed -= 0.5;
                  }
                  ui.label(format!("{:.1}", self.max_speed));
                  if ui.small_button("+").clicked() {
                    self.max_speed += 0.5;
                  }
                });

                ui.horizontal(|ui| {
                  ui.label("Acceleration:");
                  if ui.small_button("-").clicked() && self.movement_acceleration > 1.0 {
                    self.movement_acceleration -= 1.0;
                  }
                  ui.label(format!("{:.1}", self.movement_acceleration));
                  if ui.small_button("+").clicked() {
                    self.movement_acceleration += 1.0;
                  }
                });

                ui.horizontal(|ui| {
                  ui.label("Deceleration:");
                  if ui.small_button("-").clicked() && self.movement_deceleration > 1.0 {
                    self.movement_deceleration -= 1.0;
                  }
                  ui.label(format!("{:.1}", self.movement_deceleration));
                  if ui.small_button("+").clicked() {
                    self.movement_deceleration += 1.0;
                  }
                });

                // Current velocity display
                ui.label(format!(
                  "Current Speed: {:.2}",
                  self.camera_velocity.length()
                ));

                ui.separator();

                // Rendering settings
                ui.heading("Rendering");
                let mut wireframe = self.wireframe_mode;
                if ui.checkbox(&mut wireframe, "Wireframe Mode").changed() {
                  self.wireframe_mode = wireframe;
                  self.needs_pipeline_update = true;
                }

                if self.wireframe_mode {
                  if self.supports_wide_lines {
                    let mut line_width = self.line_width;
                    if ui.add(egui::Slider::new(&mut line_width, 1.0..=self.max_line_width).text("Line Width")).changed() {
                      self.line_width = line_width;
                      self.needs_pipeline_update = true;
                    }
                  } else {
                    ui.label("Wide lines not supported on this device");
                  }
                }

                // Add FOV slider
                ui.horizontal(|ui| {
                  ui.label("Field of View:");
                  if ui
                    .add(egui::Slider::new(&mut self.fov, 30.0..=120.0).step_by(1.0))
                    .changed()
                  {
                    self.needs_pipeline_update = true;
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
                  self.camera_pos = DVec3::new(-1.1, 0.1, 1.0);
                  self.camera_yaw = -std::f64::consts::FRAC_PI_4;
                  self.camera_pitch = 0.0;
                  self.camera_velocity = DVec3::ZERO;
                }
                if ui.button("Reset Movement Settings").clicked() {
                  self.max_speed = 2.0;
                  self.movement_acceleration = 20.0;
                  self.movement_deceleration = 10.0;
                }
              });
          });
        }

        let mut builder = AutoCommandBufferBuilder::primary(
          self.command_buffer_allocator.clone(),
          self.queue.queue_family_index(),
          CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder.build_app_render_pass(
          rcx,
          &descriptor_set,
          image_index,
          &self.model_buffers,
          &mut self.gui
        );

        // Build and execute the command buffer
        let command_buffer = builder.build().unwrap();
        let final_future = rcx
          .previous_frame_end
          .take()
          .unwrap()
          .join(acquire_future)
          .then_execute(self.queue.clone(), command_buffer)
          .unwrap()
          .then_swapchain_present(
            self.queue.clone(),
            SwapchainPresentInfo::swapchain_image_index(rcx.swapchain.clone(), image_index),
          )
          .then_signal_fence_and_flush();

        match final_future.map_err(Validated::unwrap) {
          Ok(future) => {
            // Wait for the GPU to finish the previous frame before starting the next one
            future.wait(None).unwrap();
            rcx.previous_frame_end = Some(future.boxed());
          }
          Err(VulkanError::OutOfDate) => {
            rcx.recreate_swapchain = true;
            rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
          }
          Err(e) => {
            println!("Failed to flush future: {e}");
            rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
          }
        }
      }
      _ => {}
    }
  }

  /// Processes raw device events such as mouse movement.
  ///
  /// Used primarily for camera control, converting raw mouse movement
  /// into camera rotation.
  fn device_event(
    &mut self,
    _event_loop: &ActiveEventLoop,
    _device_id: DeviceId,
    event: DeviceEvent,
  ) {
    if let DeviceEvent::MouseMotion { delta } = event {
      if self.cursor_captured {
        let sensitivity = 0.005;
        let (delta_x, delta_y) = delta;

        self.camera_yaw -= delta_x * sensitivity; // Inverted horizontal movement
                                                  // Clamp yaw to keep it within -2π to 2π range
        self.camera_yaw %= 2.0 * std::f64::consts::PI;

        self.camera_pitch -= delta_y * sensitivity;
        // Clamp the pitch to prevent flipping
        self.camera_pitch = self
          .camera_pitch
          .clamp(-89.0f64.to_radians(), 89.0f64.to_radians());

        // Update the camera's direction
        let direction = DVec3::new(
          self.camera_yaw.cos() * self.camera_pitch.cos(),
          self.camera_pitch.sin(),
          self.camera_yaw.sin() * self.camera_pitch.cos(),
        );
        self.camera_front = direction.normalize();
      }
    }
  }

  /// Called when the event loop is about to wait for new events.
  ///
  /// Used to perform any necessary cleanup or state updates between frames.
  fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
    let rcx = self.rcx.as_mut().unwrap();
    rcx.window.request_redraw();
  }
}
