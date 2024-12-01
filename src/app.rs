//! Main application logic and Vulkan initialization.
//!
//! This module implements the core application functionality:
//! * Vulkan instance and device initialization with feature selection
//! * Window creation and event handling (resize, input, focus)
//! * Resource management (buffers, textures, shaders)
//! * Render loop coordination and frame timing
//! * Camera control system with smooth movement
//!
//! # Architecture
//! The application uses a modular architecture:
//! * `App`: Main application state and event handling
//! * `RenderContext`: Vulkan rendering pipeline and resources
//! * `Camera`: 3D camera system with physics-based movement
//! * `Gui`: Egui-based user interface integration
//!
//! # Frame Loop
//! Each frame follows this sequence:
//! 1. Process window and input events
//! 2. Update camera position and orientation
//! 3. Acquire next swapchain image
//! 4. Record command buffer with render commands
//! 5. Submit commands and present frame

use std::{sync::Arc, time::Instant};

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
  pipeline::Pipeline,
  render_pass::Subpass,
  swapchain::{acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
  sync::{self, GpuFuture},
  Validated,
  VulkanError,
};
#[cfg(not(target_os = "linux"))]
use winit::dpi::LogicalPosition;
use winit::{
  application::ApplicationHandler,
  dpi::LogicalSize,
  event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
  event_loop::{ActiveEventLoop, EventLoop},
  window::{Window, WindowId},
};

use crate::{
  core::{command_buffer_builder_ext::AutoCommandBufferBuilderExt, init::initialize_vulkan},
  render::{
    model::VikingRoomModelBuffers,
    pipeline::{window_size_dependent_setup, RenderContext, WindowSizeSetupConfig},
  },
  shaders::{fs, vs},
  Camera,
};

/// Main application state containing all Vulkan and window resources.
///
/// This struct manages the complete state of the Vulkan application, including:
/// * Vulkan instance, physical device, and logical device
/// * Command pools and memory allocators
/// * Descriptor sets and pipeline resources
/// * Camera system and movement state
/// * Performance monitoring (frame timing)
/// * GUI integration
///
/// The application handles:
/// * Window management and event processing
/// * Resource creation and cleanup
/// * Frame timing and vsync
/// * Input processing for camera control
///
/// # Resource Management
/// Critical resources are stored in `Arc` to allow safe sharing:
/// * Vulkan device and queues
/// * Memory allocators
/// * Command pools
/// * Descriptor sets
///
/// # Example Usage
/// ```
/// let event_loop = EventLoop::new();
/// let app = App::new(&event_loop);
/// event_loop.run(app); // Starts the main application loop
/// ```
pub struct App {
  // Vulkan resources
  instance: Arc<Instance>,
  device: Arc<Device>,
  queue: Arc<Queue>,
  memory_allocator: Arc<StandardMemoryAllocator>,
  descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
  command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
  model_buffers: VikingRoomModelBuffers,
  uniform_buffer_allocator: SubbufferAllocator,
  texture: Arc<ImageView>,
  sampler: Arc<Sampler>,

  // Rendering context and UI
  rcx: Option<RenderContext>,
  gui: Option<Gui>,
  gui_state: GuiState,

  // Performance monitoring
  last_frame_time: Instant,

  // Camera state
  camera: Camera,
  current_movement_speed: f64,

  // Input state
  forward_pressed: bool,
  back_pressed: bool,
  left_pressed: bool,
  right_pressed: bool,
  up_pressed: bool,
  down_pressed: bool,

  // Rendering settings
  wireframe_mode: bool,
  line_width: f32,
  needs_pipeline_update: bool,
  cursor_captured: bool,
  fov: f32, // Field of View in degrees
}

impl App {
  /// Creates a new application instance and initializes all Vulkan resources.
  ///
  /// This function performs the complete Vulkan initialization sequence:
  /// 1. Creates Vulkan instance with debug validation layers
  /// 2. Selects physical device with required features:
  ///    * Geometry shader support
  ///    * Anisotropic filtering
  ///    * Wide lines for debug rendering
  /// 3. Creates logical device and command queues
  /// 4. Initializes memory allocators and descriptor pools
  /// 5. Creates the window and swapchain
  /// 6. Loads the Viking Room model and textures
  /// 7. Sets up the GUI system with Egui
  ///
  /// # Parameters
  /// * `event_loop` - The winit event loop to create the window for
  ///
  /// # Returns
  /// A fully initialized `App` instance ready to start rendering
  ///
  /// # Panics
  /// Will panic if:
  /// * No suitable Vulkan device is found
  /// * Required device features are not available
  /// * Window creation fails
  /// * Resource allocation fails
  pub fn new(event_loop: &EventLoop<()>) -> Self {
    let initialized = initialize_vulkan(event_loop);

    App {
      instance: initialized.instance,
      device: initialized.device,
      queue: initialized.queue,
      memory_allocator: initialized.memory_allocator.clone(),
      descriptor_set_allocator: initialized.descriptor_set_allocator.clone(),
      command_buffer_allocator: initialized.command_buffer_allocator.clone(),
      model_buffers: initialized.model_buffers,
      uniform_buffer_allocator: initialized.uniform_buffer_allocator,
      texture: initialized.texture.clone(),
      sampler: initialized.sampler.clone(),
      rcx: None,
      gui: None,
      gui_state: GuiState {
        max_line_width: initialized.max_line_width,
        supports_wide_lines: initialized.supports_wide_lines,
        ..GuiState::default()
      },
      last_frame_time: Instant::now(),
      // Camera state
      camera: Camera::new(),
      current_movement_speed: 0.0,
      // Input state
      forward_pressed: false,
      back_pressed: false,
      left_pressed: false,
      right_pressed: false,
      up_pressed: false,
      down_pressed: false,
      // Rendering settings
      wireframe_mode: false,
      line_width: 1.0,
      needs_pipeline_update: false,
      cursor_captured: false,
      fov: 90.0, // Default 90 degree FOV
    }
  }

  /// Updates camera position and orientation based on current movement state.
  ///
  /// Applies velocity and acceleration to smoothly move the camera
  /// according to user input. The movement system features:
  /// * Smooth acceleration and deceleration
  /// * Maximum speed limiting
  /// * Frame-rate independent movement
  /// * Combined movement in multiple directions
  ///
  /// The camera's movement is controlled by:
  /// * WASD keys for forward/backward/strafe
  /// * Space/Shift for up/down
  /// * Mouse for looking around
  ///
  /// # Parameters
  /// * `delta_time` - Time elapsed since last update in seconds
  pub fn update_camera_movement(&mut self, delta_time: f64) {
    // Calculate movement direction based on input
    let forward = DVec3::new(self.camera.yaw.cos(), 0.0, self.camera.yaw.sin()).normalize();
    if !forward.is_finite() {
      return;
    }

    let right = DVec3::Y.cross(forward).normalize();
    if !right.is_finite() {
      return;
    }

    // Calculate target velocity based on input
    let mut target_velocity = DVec3::ZERO;
    if self.forward_pressed {
      target_velocity += forward;
    }
    if self.back_pressed {
      target_velocity -= forward;
    }
    if self.left_pressed {
      target_velocity -= right;
    }
    if self.right_pressed {
      target_velocity += right;
    }
    if self.up_pressed {
      target_velocity.y += 1.0;
    }
    if self.down_pressed {
      target_velocity.y -= 1.0;
    }

    // Normalize and scale target velocity
    if target_velocity.length() > 0.0 {
      target_velocity = target_velocity.normalize() * self.camera.max_speed;
    }

    // Update velocity based on input state
    self.camera.velocity = if target_velocity.length() == 0.0 {
      let current_speed = self.camera.velocity.length();
      if current_speed < self.camera.movement_deceleration * delta_time {
        DVec3::ZERO
      } else {
        let decel_factor =
          1.0 - (self.camera.movement_deceleration * delta_time / current_speed).clamp(0.0, 1.0);
        self.camera.velocity * decel_factor
      }
    } else {
      let accel_factor = (-self.camera.movement_acceleration * delta_time).exp();
      target_velocity + (self.camera.velocity - target_velocity) * accel_factor
    };

    self.current_movement_speed = self.camera.velocity.length();

    // Apply movement
    let movement = self.camera.velocity * delta_time;
    if movement.is_finite() {
      self.camera.position += movement;
    }
  }
}

impl ApplicationHandler for App {
  /// Handles application resume events.
  ///
  /// Called when the application window gains focus or is restored.
  /// Recreates any resources that may have been lost during suspension.
  fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    let window_attrs = {
      let base_attrs = Window::default_attributes()
        .with_decorations(true)
        .with_title("Vulkano App")
        .with_inner_size(LogicalSize::new(1280, 720));

      #[cfg(not(target_os = "linux"))]
      {
        base_attrs.with_position(LogicalPosition::new(
          (event_loop.primary_monitor().unwrap().size().width as i32 - 1280) / 2,
          (event_loop.primary_monitor().unwrap().size().height as i32 - 720) / 2,
        ))
      }

      #[cfg(target_os = "linux")]
      base_attrs
    };

    let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

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

      // Query supported present modes
      let present_modes = self
        .device
        .physical_device()
        .surface_present_modes(&surface, Default::default())
        .unwrap();

      // Try to use IMMEDIATE if supported, fall back to MAILBOX (triple buffering) if not, then FIFO (vsync)
      let present_mode = if present_modes.contains(&vulkano::swapchain::PresentMode::Immediate) {
        println!("Using IMMEDIATE present mode (vsync off)");
        vulkano::swapchain::PresentMode::Immediate
      } else if present_modes.contains(&vulkano::swapchain::PresentMode::Mailbox) {
        println!("Using MAILBOX present mode (triple buffering)");
        vulkano::swapchain::PresentMode::Mailbox
      } else {
        println!("Using FIFO present mode (vsync on)");
        vulkano::swapchain::PresentMode::Fifo
      };

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
          present_mode,
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
  /// Handles the following event types:
  /// * Window resize: Triggers swapchain recreation
  /// * Window close: Initiates cleanup and exit
  /// * Keyboard events: Updates camera movement state
  /// * Mouse events: Updates camera orientation
  /// * Focus events: Handles cursor capture
  ///
  /// Input processing includes:
  /// * WASD keys for camera movement
  /// * Space/Shift for vertical movement
  /// * Escape to toggle cursor capture
  /// * Mouse movement for camera rotation
  /// * Mouse buttons for GUI interaction
  ///
  /// # Parameters
  /// * `event_loop` - Active event loop reference
  /// * `window_id` - ID of the window generating the event
  /// * `event` - The window event to process
  fn window_event(
    &mut self,
    event_loop: &ActiveEventLoop,
    _window_id: WindowId,
    event: WindowEvent,
  ) {
    // Handle egui events and check if we should pass events to game
    let mut pass_events_to_game = true;
    if let Some(gui) = &mut self.gui {
      if gui.update(&event) {
        pass_events_to_game = false;
      }
    }

    let rcx = self.rcx.as_mut().unwrap();

    match event {
      WindowEvent::CloseRequested => {
        event_loop.exit();
      }
      WindowEvent::Resized(_) => {
        rcx.recreate_swapchain = true;
      }
      WindowEvent::MouseInput {
        state: ElementState::Pressed,
        button: MouseButton::Left,
        ..
      } => {
        if pass_events_to_game {
          rcx
            .window
            .set_cursor_grab(
              #[cfg(not(target_os = "linux"))]
              winit::window::CursorGrabMode::Confined,
              #[cfg(target_os = "linux")]
              winit::window::CursorGrabMode::Locked,
            )
            .unwrap();
          rcx.window.set_cursor_visible(false);
          self.cursor_captured = true;
        }
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

        match key {
          PhysicalKey::Code(winit::keyboard::KeyCode::KeyW) => {
            self.forward_pressed = state == ElementState::Pressed;
          }
          PhysicalKey::Code(winit::keyboard::KeyCode::KeyS) => {
            self.back_pressed = state == ElementState::Pressed;
          }
          PhysicalKey::Code(winit::keyboard::KeyCode::KeyA) => {
            self.left_pressed = state == ElementState::Pressed;
          }
          PhysicalKey::Code(winit::keyboard::KeyCode::KeyD) => {
            self.right_pressed = state == ElementState::Pressed;
          }
          PhysicalKey::Code(winit::keyboard::KeyCode::Space) => {
            self.up_pressed = state == ElementState::Pressed;
          }
          PhysicalKey::Code(winit::keyboard::KeyCode::ShiftLeft) => {
            self.down_pressed = state == ElementState::Pressed;
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
      }
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
        self.last_frame_time = now;

        // Clamp frame time to avoid huge jumps
        let clamped_frame_time = frame_time.min(0.1); // Max 100ms per frame
        self.update_camera_movement(clamped_frame_time);

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

        // Draw GUI and handle changes
        if let Some(gui) = &mut self.gui {
          let changes = gui::draw_gui(
            gui,
            &mut self.gui_state,
            &mut self.camera,
            &self.current_movement_speed,
          );

          // Apply any changes from GUI
          if let Some(wireframe) = changes.wireframe_mode {
            if wireframe != self.wireframe_mode {
              self.wireframe_mode = wireframe;
              self.needs_pipeline_update = true;
            }
          }
          if let Some(line_width) = changes.line_width {
            if line_width != self.line_width {
              self.line_width = line_width;
              self.needs_pipeline_update = true;
            }
          }
          if let Some(fov) = changes.fov {
            self.fov = fov;
            self.needs_pipeline_update = true;
          }
          if changes.camera_reset {
            self.camera.position = DVec3::new(-1.1, 0.1, 1.0);
            self.camera.yaw = -std::f64::consts::FRAC_PI_4;
            self.camera.pitch = 0.0;
          }
          if changes.movement_reset {
            // Reset velocity and all movement settings to default values
            self.camera.velocity = DVec3::ZERO;
            self.camera.movement_acceleration = 20.0;
            self.camera.movement_deceleration = 10.0;
            self.camera.max_speed = 2.0;
          }
          if let Some(speed) = changes.max_speed {
            self.camera.max_speed = speed;
          }
          if let Some(accel) = changes.movement_acceleration {
            self.camera.movement_acceleration = accel;
          }
          if let Some(decel) = changes.movement_deceleration {
            self.camera.movement_deceleration = decel;
          }
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
            self.camera.position,
            self.camera.position
              + DVec3::new(
                self.camera.yaw.cos() * self.camera.pitch.cos(),
                self.camera.pitch.sin(),
                self.camera.yaw.sin() * self.camera.pitch.cos(),
              ),
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

        // Make sure the previous frame is completely finished before starting a new one
        if let Some(previous_frame_end) = rcx.previous_frame_end.as_mut() {
          previous_frame_end.cleanup_finished();
          match previous_frame_end.flush() {
            Ok(_) => (),
            Err(e) => {
              println!("Failed to wait for previous frame: {e}");
              rcx.recreate_swapchain = true;
              return;
            }
          }
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
          &mut self.gui,
        );

        // Build and execute the command buffer
        let command_buffer = builder.build().unwrap();

        let acquire_semaphore = sync::now(self.device.clone());

        let final_future = acquire_semaphore
          .join(acquire_future)
          .then_execute(self.queue.clone(), command_buffer)
          .unwrap()
          .then_signal_semaphore()
          .then_swapchain_present(
            self.queue.clone(),
            SwapchainPresentInfo::swapchain_image_index(rcx.swapchain.clone(), image_index),
          )
          .then_signal_fence_and_flush();

        match final_future.map_err(Validated::unwrap) {
          Ok(future) => {
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
  /// into camera rotation. Features include:
  /// * Configurable mouse sensitivity
  /// * Vertical rotation limits (no over-rotation)
  /// * Smooth motion interpolation
  ///
  /// Only processes mouse motion events when the cursor is captured
  /// to prevent unwanted camera movement during GUI interaction.
  ///
  /// # Parameters
  /// * `event_loop` - Active event loop reference
  /// * `device_id` - ID of the input device
  /// * `event` - The device event to process
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

        self.camera.yaw -= delta_x * sensitivity; // Inverted horizontal movement
                                                  // Clamp yaw to keep it within -2π to 2π range
        self.camera.yaw %= 2.0 * std::f64::consts::PI;

        self.camera.pitch -= delta_y * sensitivity;
        // Clamp the pitch to prevent flipping
        self.camera.pitch = self
          .camera
          .pitch
          .clamp(-89.0f64.to_radians(), 89.0f64.to_radians());
      }
    }
  }

  /// Called when the event loop is about to wait for new events.
  ///
  /// Used to perform any necessary cleanup or state updates between frames:
  /// * Updates GUI state
  /// * Processes deferred events
  /// * Updates performance metrics
  /// * Triggers resource cleanup if needed
  ///
  /// # Parameters
  /// * `event_loop` - Active event loop reference
  fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
    let rcx = self.rcx.as_mut().unwrap();
    rcx.window.request_redraw();
  }
}

use crate::gui::{self, GuiState};
