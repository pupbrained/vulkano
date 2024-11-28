use std::{error::Error, sync::Arc, time::Instant, time::Duration};

use self::model::{Normal, Position, INDICES, NORMALS, POSITIONS};

use egui_winit_vulkano::{Gui, GuiConfig};

use glam::{
  f32::{Mat3, Vec3},
  Mat4,
};

use vulkano::{
  buffer::{
    allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
    Buffer, BufferCreateInfo, BufferUsage, Subbuffer,
  },
  command_buffer::{
    allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
  },
  descriptor_set::{allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet},
  device::{
    physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
    DeviceOwned, Queue, QueueCreateInfo, QueueFlags,
  },
  format::Format,
  image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
  instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
  memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
  pipeline::{
    graphics::{
      color_blend::{ColorBlendAttachmentState, ColorBlendState},
      depth_stencil::{DepthState, DepthStencilState},
      input_assembly::InputAssemblyState,
      multisample::MultisampleState,
      rasterization::{PolygonMode, RasterizationState},
      vertex_input::{Vertex, VertexDefinition},
      viewport::{Viewport, ViewportState},
      GraphicsPipelineCreateInfo,
    },
    layout::PipelineDescriptorSetLayoutCreateInfo,
    GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
  },
  render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
  shader::EntryPoint,
  swapchain::{acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
  sync::{self, GpuFuture},
  Validated, VulkanError, VulkanLibrary,
};

use winit::{
  application::ApplicationHandler,
  dpi::{LogicalSize, PhysicalSize},
  event::WindowEvent,
  event_loop::{ActiveEventLoop, EventLoop},
  window::{Window, WindowId},
};

#[cfg(target_os = "windows")]
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

#[cfg(target_os = "windows")]
use windows::{
  Win32::Foundation::HWND,
  Win32::Graphics::Dwm::{DwmSetWindowAttribute, DWMSBT_MAINWINDOW, DWMWA_SYSTEMBACKDROP_TYPE},
};

mod model;

fn main() -> Result<(), impl Error> {
  // The start of this example is exactly the same as `triangle`. You should read the `triangle`
  // example if you haven't done so yet.

  let event_loop = EventLoop::new().unwrap();
  let mut app = App::new(&event_loop);

  event_loop.run_app(&mut app)
}

struct App {
  instance: Arc<Instance>,
  device: Arc<Device>,
  queue: Arc<Queue>,
  memory_allocator: Arc<StandardMemoryAllocator>,
  descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
  command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
  vertex_buffer: Subbuffer<[Position]>,
  normals_buffer: Subbuffer<[Normal]>,
  index_buffer: Subbuffer<[u16]>,
  uniform_buffer_allocator: SubbufferAllocator,
  rcx: Option<RenderContext>,
  gui: Option<Gui>,
  last_frame_time: Instant,
  fps: f32,
  // Camera state
  camera_pos: Vec3,
  camera_yaw: f32,
  camera_pitch: f32,
  // Smooth movement
  camera_velocity: Vec3,
  movement_acceleration: f32,
  movement_deceleration: f32,
  max_speed: f32,
  movement_input: Vec3,
  // Rendering settings
  wireframe_mode: bool,
  line_width: f32,
  max_line_width: f32,
  needs_pipeline_update: bool,
  last_line_width_update: Instant,
  line_width_update_interval: Duration,
}

struct RenderContext {
  window: Arc<Window>,
  swapchain: Arc<Swapchain>,
  render_pass: Arc<RenderPass>,
  framebuffers: Vec<Arc<Framebuffer>>,
  vs: EntryPoint,
  fs: EntryPoint,
  pipeline: Arc<GraphicsPipeline>,
  recreate_swapchain: bool,
  previous_frame_end: Option<Box<dyn GpuFuture>>,
  swapchain_image_views: Vec<Arc<ImageView>>,
  rotation_start: Instant,
}

impl App {
  fn new(event_loop: &EventLoop<()>) -> Self {
    let library = VulkanLibrary::new().unwrap();
    let required_extensions = Surface::required_extensions(event_loop).unwrap();
    let instance = Instance::new(
      library,
      InstanceCreateInfo {
        flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
        enabled_extensions: required_extensions,
        ..Default::default()
      },
    )
    .unwrap();

    let device_extensions = DeviceExtensions {
      khr_swapchain: true,
      ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
      .enumerate_physical_devices()
      .unwrap()
      .filter(|p| p.supported_extensions().contains(&device_extensions))
      .filter_map(|p| {
        p.queue_family_properties()
          .iter()
          .enumerate()
          .position(|(i, q)| {
            q.queue_flags.intersects(QueueFlags::GRAPHICS)
              && p.presentation_support(i as u32, event_loop).unwrap()
          })
          .map(|i| (p, i as u32))
      })
      .min_by_key(|(p, _)| match p.properties().device_type {
        PhysicalDeviceType::DiscreteGpu => 0,
        PhysicalDeviceType::IntegratedGpu => 1,
        PhysicalDeviceType::VirtualGpu => 2,
        PhysicalDeviceType::Cpu => 3,
        PhysicalDeviceType::Other => 4,
        _ => 5,
      })
      .unwrap();

    println!(
      "Using device: {} (type: {:?})",
      physical_device.properties().device_name,
      physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
      physical_device.clone(),
      DeviceCreateInfo {
        enabled_extensions: device_extensions,
        enabled_features: DeviceFeatures {
          fill_mode_non_solid: true, // Enable wireframe mode
          wide_lines: true,          // Enable adjustable line width
          image_view_format_swizzle: true, // Enable image view format swizzling
          ..DeviceFeatures::empty()
        },
        queue_create_infos: vec![QueueCreateInfo {
          queue_family_index,
          ..Default::default()
        }],
        ..Default::default()
      },
    )
    .unwrap();

    let supports_wide_lines = physical_device.supported_features().wide_lines;

    // Query maximum line width
    let max_line_width = if supports_wide_lines {
      let properties = physical_device.properties();
      properties.line_width_range[1]
    } else {
      1.0
    };

    println!("Wide lines support: {}", supports_wide_lines);
    println!("Maximum line width: {:.1}", max_line_width);

    let queue = queues.next().unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
      device.clone(),
      Default::default(),
    ));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
      device.clone(),
      Default::default(),
    ));

    let vertex_buffer = Buffer::from_iter(
      memory_allocator.clone(),
      BufferCreateInfo {
        usage: BufferUsage::VERTEX_BUFFER,
        ..Default::default()
      },
      AllocationCreateInfo {
        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
          | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        ..Default::default()
      },
      POSITIONS,
    )
    .unwrap();
    let normals_buffer = Buffer::from_iter(
      memory_allocator.clone(),
      BufferCreateInfo {
        usage: BufferUsage::VERTEX_BUFFER,
        ..Default::default()
      },
      AllocationCreateInfo {
        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
          | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        ..Default::default()
      },
      NORMALS,
    )
    .unwrap();
    let index_buffer = Buffer::from_iter(
      memory_allocator.clone(),
      BufferCreateInfo {
        usage: BufferUsage::INDEX_BUFFER,
        ..Default::default()
      },
      AllocationCreateInfo {
        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
          | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        ..Default::default()
      },
      INDICES,
    )
    .unwrap();

    let uniform_buffer_allocator = SubbufferAllocator::new(
      memory_allocator.clone(),
      SubbufferAllocatorCreateInfo {
        buffer_usage: BufferUsage::UNIFORM_BUFFER,
        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
          | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        ..Default::default()
      },
    );

    App {
      instance,
      device,
      queue,
      memory_allocator,
      descriptor_set_allocator,
      command_buffer_allocator,
      vertex_buffer,
      normals_buffer,
      index_buffer,
      uniform_buffer_allocator,
      rcx: None,
      gui: None,
      last_frame_time: Instant::now(),
      fps: 0.0,
      // Camera settings
      camera_pos: Vec3::new(-1.1, 0.1, 1.0),
      camera_yaw: -std::f32::consts::FRAC_PI_4,
      camera_pitch: 0.0,
      // Smooth movement settings
      camera_velocity: Vec3::ZERO,
      movement_acceleration: 20.0,
      movement_deceleration: 10.0,
      max_speed: 2.0,
      movement_input: Vec3::ZERO,
      // Rendering settings
      wireframe_mode: false,
      line_width: 1.0,
      max_line_width,
      needs_pipeline_update: false,
      last_line_width_update: Instant::now(),
      line_width_update_interval: Duration::from_millis(100),
    }
  }

  fn update_camera_movement(&mut self, delta_time: f32) {
    // Calculate movement direction based on input
    let forward = Vec3::new(self.camera_yaw.cos(), 0.0, self.camera_yaw.sin()).normalize();

    let right = forward.cross(Vec3::new(0.0, -1.0, 0.0)).normalize();

    // Calculate target velocity based on input
    let mut target_velocity = Vec3::ZERO;
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
  fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    let window = Arc::new(
      event_loop
        .create_window(
          Window::default_attributes()
            .with_decorations(true)
            .with_transparent(true)
            .with_title("Vulkano App")
            .with_inner_size(LogicalSize::new(800, 600)),
        )
        .unwrap(),
    );

    #[cfg(target_os = "windows")]
    {
      // Apply Mica effect (Windows 11 only)
      if let Ok(handle) = window.window_handle() {
        if let RawWindowHandle::Win32(handle) = handle.as_raw() {
          unsafe {
            DwmSetWindowAttribute(
              HWND(handle.hwnd.get() as *mut _),
              DWMWA_SYSTEMBACKDROP_TYPE,
              &DWMSBT_MAINWINDOW as *const _ as *const _,
              std::mem::size_of::<i32>() as u32,
            )
            .ok();
          }
        }
      }
    }

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
          image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
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
            final_color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
            depth: {
                format: Format::D16_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
            }
        },
        passes: [
            {
                color: [final_color],
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

    let (framebuffers, pipeline) = window_size_dependent_setup(
      window_size,
      &images,
      &render_pass,
      &self.memory_allocator,
      &vs,
      &fs,
      self.wireframe_mode,
      self.line_width,
    );

    let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

    let rotation_start = Instant::now();

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
      rotation_start,
    });
  }

  fn window_event(
    &mut self,
    event_loop: &ActiveEventLoop,
    _window_id: WindowId,
    event: WindowEvent,
  ) {
    if let Some(gui) = &mut self.gui {
      let _pass_events_to_game = !gui.update(&event);
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
        use winit::event::ElementState;
        use winit::keyboard::PhysicalKey;

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
          _ => {}
        }

        // Wait for any pending operations to complete before updating the pipeline
        if let Some(rcx) = &mut self.rcx {
          rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();
        }
      }
      WindowEvent::RedrawRequested => {
        let now = Instant::now();
        let frame_time = now.duration_since(self.last_frame_time).as_secs_f32();
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
          (rcx.framebuffers, rcx.pipeline) = window_size_dependent_setup(
            window_size,
            &new_images,
            &rcx.render_pass,
            &self.memory_allocator,
            &rcx.vs,
            &rcx.fs,
            self.wireframe_mode,
            self.line_width,
          );
          rcx.recreate_swapchain = false;
          self.needs_pipeline_update = false;
        }

        let uniform_buffer = {
          let elapsed = rcx.rotation_start.elapsed();
          let rotation = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
          let rotation = Mat3::from_rotation_y(rotation as f32);

          let aspect_ratio =
            rcx.swapchain.image_extent()[0] as f32 / rcx.swapchain.image_extent()[1] as f32;

          let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, aspect_ratio, 0.01, 100.0);

          // Update view matrix based on camera position
          let view = Mat4::look_at_rh(
            self.camera_pos,
            self.camera_pos + Vec3::new(self.camera_yaw.cos(), 0.0, self.camera_yaw.sin()),
            Vec3::new(0.0, -1.0, 0.0), // Keep Y-axis inverted for Vulkan
          );
          let scale = Mat4::from_scale(Vec3::splat(0.01));

          let uniform_data = vs::Data {
            world: Mat4::from_mat3(rotation).to_cols_array_2d(),
            view: (view * scale).to_cols_array_2d(),
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
          [WriteDescriptorSet::buffer(0, uniform_buffer)],
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
                  ui.horizontal(|ui| {
                    ui.label("Line Width:");
                    let device = self.device.physical_device();
                    let supports_wide_lines = device.supported_features().wide_lines;

                    if supports_wide_lines {
                      let mut width = self.line_width;
                      if ui
                        .add(egui::Slider::new(&mut width, 1.0..=self.max_line_width).step_by(0.1))
                        .changed()
                      {
                        if now.duration_since(self.last_line_width_update) > self.line_width_update_interval {
                          self.line_width = width;
                          self.needs_pipeline_update = true;
                          self.last_line_width_update = now;
                        }
                      }
                    } else {
                      ui.label("1.0 (Wide lines not supported)");
                      self.line_width = 1.0;
                    }
                  });
                }

                ui.separator();

                // Controls help
                ui.heading("Controls");
                ui.label("WASD - Move horizontally");
                ui.label("Space/Shift - Move up/down");

                ui.separator();

                // Reset buttons
                if ui.button("Reset Camera Position").clicked() {
                  self.camera_pos = Vec3::new(-1.1, 0.1, 1.0);
                  self.camera_yaw = -std::f32::consts::FRAC_PI_4;
                  self.camera_pitch = 0.0;
                  self.camera_velocity = Vec3::ZERO;
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

        builder
          .begin_render_pass(
            RenderPassBeginInfo {
              clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into()), Some(1.0.into())],
              ..RenderPassBeginInfo::framebuffer(rcx.framebuffers[image_index as usize].clone())
            },
            SubpassBeginInfo {
              contents: SubpassContents::Inline,
              ..Default::default()
            },
          )
          .unwrap();

        builder
          .bind_pipeline_graphics(rcx.pipeline.clone())
          .unwrap()
          .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            rcx.pipeline.layout().clone(),
            0,
            descriptor_set,
          )
          .unwrap()
          .bind_vertex_buffers(0, (self.vertex_buffer.clone(), self.normals_buffer.clone()))
          .unwrap()
          .bind_index_buffer(self.index_buffer.clone())
          .unwrap();

        unsafe { builder.draw_indexed(self.index_buffer.len() as u32, 1, 0, 0, 0) }.unwrap();

        // Move to the egui subpass
        builder
          .next_subpass(
            SubpassEndInfo::default(),
            SubpassBeginInfo {
              contents: SubpassContents::SecondaryCommandBuffers,
              ..Default::default()
            },
          )
          .unwrap();

        // Draw egui in the second subpass
        if let Some(gui) = &mut self.gui {
          let cb = gui.draw_on_subpass_image([
            rcx.swapchain.image_extent()[0],
            rcx.swapchain.image_extent()[1],
          ]);
          builder.execute_commands(cb).unwrap();
        }

        // End the render pass
        builder.end_render_pass(SubpassEndInfo::default()).unwrap();

        // Build and execute the command buffer
        let command_buffer = builder.build().unwrap();
        let final_future = rcx
          .previous_frame_end
          .take()
          .unwrap()
          .join(acquire_future)
          .then_execute(self.queue.clone(), command_buffer)
          .unwrap();

        // Present the final image
        let future = final_future
          .then_swapchain_present(
            self.queue.clone(),
            SwapchainPresentInfo::swapchain_image_index(rcx.swapchain.clone(), image_index),
          )
          .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
          Ok(future) => {
            rcx.previous_frame_end = Some(future.boxed());
          }
          Err(VulkanError::OutOfDate) => {
            rcx.recreate_swapchain = true;
            rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
          }
          Err(e) => {
            println!("failed to flush future: {e}");
            rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
          }
        }
      }
      _ => {}
    }
  }

  fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
    let rcx = self.rcx.as_mut().unwrap();
    rcx.window.request_redraw();
  }
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
  window_size: PhysicalSize<u32>,
  images: &[Arc<Image>],
  render_pass: &Arc<RenderPass>,
  memory_allocator: &Arc<StandardMemoryAllocator>,
  vs: &EntryPoint,
  fs: &EntryPoint,
  wireframe_mode: bool,
  line_width: f32,
) -> (Vec<Arc<Framebuffer>>, Arc<GraphicsPipeline>) {
  let device = memory_allocator.device();

  // Always use line width 1.0 if wide lines are not supported
  let actual_line_width = if device.physical_device().supported_features().wide_lines {
    line_width
  } else {
    1.0
  };

  let depth_buffer = ImageView::new_default(
    Image::new(
      memory_allocator.clone(),
      ImageCreateInfo {
        image_type: ImageType::Dim2d,
        format: Format::D16_UNORM,
        extent: images[0].extent(),
        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
        ..Default::default()
      },
      AllocationCreateInfo::default(),
    )
    .unwrap(),
  )
  .unwrap();

  let framebuffers = images
    .iter()
    .map(|image| {
      let view = ImageView::new_default(image.clone()).unwrap();

      Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
          attachments: vec![view, depth_buffer.clone()],
          ..Default::default()
        },
      )
      .unwrap()
    })
    .collect::<Vec<_>>();

  // In the triangle example we use a dynamic viewport, as its a simple example. However in the
  // teapot example, we recreate the pipelines with a hardcoded viewport instead. This allows the
  // driver to optimize things, at the cost of slower window resizes.
  // https://computergraphics.stackexchange.com/questions/5742/vulkan-best-way-of-updating-pipeline-viewport
  let pipeline = {
    let vertex_input_state = [Position::per_vertex(), Normal::per_vertex()]
      .definition(vs)
      .unwrap();

    let stages = [
      PipelineShaderStageCreateInfo::new(vs.clone()),
      PipelineShaderStageCreateInfo::new(fs.clone()),
    ];

    let layout = PipelineLayout::new(
      device.clone(),
      PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
        .into_pipeline_layout_create_info(device.clone())
        .unwrap(),
    )
    .unwrap();

    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    GraphicsPipeline::new(
      device.clone(),
      None,
      GraphicsPipelineCreateInfo {
        stages: stages.into_iter().collect(),
        vertex_input_state: Some(vertex_input_state),
        input_assembly_state: Some(InputAssemblyState::default()),
        viewport_state: Some(ViewportState {
          viewports: [Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
          }]
          .into_iter()
          .collect(),
          ..Default::default()
        }),
        rasterization_state: Some(RasterizationState {
          cull_mode: vulkano::pipeline::graphics::rasterization::CullMode::None,
          polygon_mode: if wireframe_mode {
            PolygonMode::Line
          } else {
            PolygonMode::Fill
          },
          line_width: actual_line_width,
          ..Default::default()
        }),
        depth_stencil_state: Some(DepthStencilState {
          depth: Some(DepthState::simple()),
          ..Default::default()
        }),
        multisample_state: Some(MultisampleState::default()),
        color_blend_state: Some(ColorBlendState::with_attachment_states(
          subpass.num_color_attachments(),
          ColorBlendAttachmentState {
            blend: Some(vulkano::pipeline::graphics::color_blend::AttachmentBlend {
              src_color_blend_factor:
                vulkano::pipeline::graphics::color_blend::BlendFactor::SrcAlpha,
              dst_color_blend_factor:
                vulkano::pipeline::graphics::color_blend::BlendFactor::OneMinusSrcAlpha,
              color_blend_op: vulkano::pipeline::graphics::color_blend::BlendOp::Add,
              src_alpha_blend_factor: vulkano::pipeline::graphics::color_blend::BlendFactor::One,
              dst_alpha_blend_factor: vulkano::pipeline::graphics::color_blend::BlendFactor::Zero,
              alpha_blend_op: vulkano::pipeline::graphics::color_blend::BlendOp::Add,
            }),
            color_write_mask: vulkano::pipeline::graphics::color_blend::ColorComponents::all(),
            ..Default::default()
          },
        )),
        subpass: Some(subpass.into()),
        ..GraphicsPipelineCreateInfo::layout(layout)
      },
    )
    .unwrap()
  };

  (framebuffers, pipeline)
}

mod vs {
  vulkano_shaders::shader! {
    ty: "vertex",
    path: "src/vert.glsl",
  }
}

mod fs {
  vulkano_shaders::shader! {
    ty: "fragment",
    path: "src/frag.glsl",
  }
}
