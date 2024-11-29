use std::{error::Error, sync::Arc, time::Duration, time::Instant};

use egui_winit_vulkano::{Gui, GuiConfig};

use glam::{DMat3, DMat4, DVec3, Mat4};

use vulkano::{
  buffer::{
    allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
    Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
  },
  command_buffer::{
    allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    CopyBufferToImageInfo, PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassBeginInfo,
    SubpassContents, SubpassEndInfo,
  },
  descriptor_set::{allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet},
  device::{
    physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
    DeviceOwned, Queue, QueueCreateInfo, QueueFlags,
  },
  format::Format,
  image::{
    sampler::{Sampler, SamplerCreateInfo},
    view::ImageView,
    Image, ImageCreateInfo, ImageType, ImageUsage,
  },
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
  dpi::{LogicalPosition, LogicalSize, PhysicalSize},
  event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
  event_loop::{ActiveEventLoop, EventLoop},
  window::{Window, WindowId},
};

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct Position {
  #[format(R32G32B32_SFLOAT)]
  position: [f32; 3],
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct Normal {
  #[format(R32G32B32_SFLOAT)]
  normal: [f32; 3],
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct TexCoord {
  #[format(R32G32_SFLOAT)]
  tex_coord: [f32; 2],
}

struct VikingRoomModelBuffers {
  positions: Subbuffer<[Position]>,
  normals: Subbuffer<[Normal]>,
  tex_coords: Subbuffer<[TexCoord]>,
  indices: Subbuffer<[u32]>,
}

fn load_viking_room_model(
  memory_allocator: &Arc<StandardMemoryAllocator>,
) -> VikingRoomModelBuffers {
  let (models, _materials) =
    tobj::load_obj("models/viking_room.obj", &tobj::LoadOptions::default()).unwrap();
  let mesh = &models[0].mesh;

  let positions: Vec<Position> = mesh
    .positions
    .chunks(3)
    .map(|chunk| Position {
      position: [chunk[0], chunk[1], chunk[2]],
    })
    .collect();

  let normals: Vec<Normal> = mesh
    .normals
    .chunks(3)
    .map(|chunk| Normal {
      normal: [chunk[0], chunk[1], chunk[2]],
    })
    .collect();

  let tex_coords: Vec<TexCoord> = mesh
    .texcoords
    .chunks(2)
    .map(|chunk| TexCoord {
      tex_coord: [chunk[0], 1.0 - chunk[1]], // Flip Y coordinate
    })
    .collect();

  let indices: Vec<u32> = mesh.indices.clone();

  let vertex_buffer = Buffer::from_iter(
    memory_allocator.clone(),
    BufferCreateInfo {
      usage: BufferUsage::VERTEX_BUFFER,
      ..Default::default()
    },
    AllocationCreateInfo {
      memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
      ..Default::default()
    },
    positions,
  )
  .unwrap();

  let normal_buffer = Buffer::from_iter(
    memory_allocator.clone(),
    BufferCreateInfo {
      usage: BufferUsage::VERTEX_BUFFER,
      ..Default::default()
    },
    AllocationCreateInfo {
      memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
      ..Default::default()
    },
    normals,
  )
  .unwrap();

  let tex_coord_buffer = Buffer::from_iter(
    memory_allocator.clone(),
    BufferCreateInfo {
      usage: BufferUsage::VERTEX_BUFFER,
      ..Default::default()
    },
    AllocationCreateInfo {
      memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
      ..Default::default()
    },
    tex_coords,
  )
  .unwrap();

  let index_buffer = Buffer::from_iter(
    memory_allocator.clone(),
    BufferCreateInfo {
      usage: BufferUsage::INDEX_BUFFER,
      ..Default::default()
    },
    AllocationCreateInfo {
      memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
      ..Default::default()
    },
    indices,
  )
  .unwrap();

  VikingRoomModelBuffers {
    positions: vertex_buffer,
    normals: normal_buffer,
    tex_coords: tex_coord_buffer,
    indices: index_buffer,
  }
}

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
  model_buffers: VikingRoomModelBuffers,
  uniform_buffer_allocator: SubbufferAllocator,
  texture: Arc<ImageView>,
  sampler: Arc<Sampler>,
  rcx: Option<RenderContext>,
  gui: Option<Gui>,
  last_frame_time: Instant,
  fps: f32,
  // Camera state
  camera_pos: DVec3,
  camera_yaw: f64,
  camera_pitch: f64,
  camera_front: DVec3,
  // Smooth movement
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
  last_line_width_update: Instant,
  line_width_update_interval: Duration,
  cursor_captured: bool,
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
          #[cfg(target_os = "macos")]
          image_view_format_swizzle: true, // Enable image view format swizzling
          #[cfg(not(target_os = "macos"))]
          image_view_format_swizzle: false,
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

    let model_buffers = load_viking_room_model(&memory_allocator);

    let uniform_buffer_allocator = SubbufferAllocator::new(
      memory_allocator.clone(),
      SubbufferAllocatorCreateInfo {
        buffer_usage: BufferUsage::UNIFORM_BUFFER,
        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
          | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        ..Default::default()
      },
    );

    // Load the texture
    let texture_path = std::path::Path::new("textures/viking_room.png");
    let texture_data = image::open(texture_path).unwrap().to_rgba8();
    let dimensions = texture_data.dimensions();

    let image = Image::new(
      memory_allocator.clone(),
      ImageCreateInfo {
        image_type: ImageType::Dim2d,
        format: Format::R8G8B8A8_SRGB,
        extent: [dimensions.0, dimensions.1, 1],
        usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
        ..Default::default()
      },
      AllocationCreateInfo {
        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
        ..Default::default()
      },
    )
    .unwrap();

    let texture = ImageView::new_default(image.clone()).unwrap();

    // Create staging buffer
    let staging_buffer = Buffer::from_iter(
      memory_allocator.clone(),
      BufferCreateInfo {
        usage: BufferUsage::TRANSFER_SRC,
        ..Default::default()
      },
      AllocationCreateInfo {
        memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        ..Default::default()
      },
      texture_data.into_raw(),
    )
    .unwrap();

    // Create command buffer for texture upload
    let mut texture_upload = AutoCommandBufferBuilder::primary(
      command_buffer_allocator.clone(),
      queue.queue_family_index(),
      CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    texture_upload
      .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging_buffer, image))
      .unwrap();

    let texture_upload = texture_upload.build().unwrap();
    texture_upload
      .execute(queue.clone())
      .unwrap()
      .then_signal_fence_and_flush()
      .unwrap()
      .wait(None)
      .unwrap();

    let sampler = Sampler::new(
      device.clone(),
      SamplerCreateInfo {
        mag_filter: vulkano::image::sampler::Filter::Linear,
        min_filter: vulkano::image::sampler::Filter::Linear,
        address_mode: [vulkano::image::sampler::SamplerAddressMode::Repeat; 3],
        ..Default::default()
      },
    )
    .unwrap();

    App {
      instance,
      device,
      queue,
      memory_allocator,
      descriptor_set_allocator,
      command_buffer_allocator,
      model_buffers,
      uniform_buffer_allocator,
      texture,
      sampler,
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
      max_line_width,
      needs_pipeline_update: false,
      last_line_width_update: Instant::now(),
      line_width_update_interval: Duration::from_millis(100),
      cursor_captured: false,
    }
  }

  fn update_camera_movement(&mut self, delta_time: f64) {
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
  fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    let window = Arc::new(
      event_loop
        .create_window(
          Window::default_attributes()
            .with_decorations(true)
            .with_title("Vulkano App")
            .with_inner_size(LogicalSize::new(1280, 720))
            .with_position(LogicalPosition::new(
              (event_loop.primary_monitor().unwrap().size().width as i32 - 1280) / 2,
              (event_loop.primary_monitor().unwrap().size().height as i32 - 720) / 2,
            )),
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
                format: Format::D16_UNORM,
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
          rcx
            .window
            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
            .unwrap();
          rcx.window.set_cursor_visible(false);
        }
      }
      WindowEvent::CursorMoved { .. } => {}
      WindowEvent::MouseWheel { .. } => {}
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

          let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, aspect_ratio, 0.01, 100.0);

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
                  ui.horizontal(|ui| {
                    ui.label("Line Width:");
                    let device = self.device.physical_device();
                    let supports_wide_lines = device.supported_features().wide_lines;

                    if supports_wide_lines {
                      let mut width = self.line_width;
                      if ui
                        .add(egui::Slider::new(&mut width, 1.0..=self.max_line_width).step_by(0.1))
                        .changed()
                        && now.duration_since(self.last_line_width_update)
                          > self.line_width_update_interval
                      {
                        self.line_width = width;
                        self.needs_pipeline_update = true;
                        self.last_line_width_update = now;
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

        builder
          .begin_render_pass(
            RenderPassBeginInfo {
              clear_values: vec![
                Some([0.0, 0.0, 0.0, 1.0].into()), // msaa_color clear value
                None,                              // final_color (DontCare)
                Some(1.0.into()),                  // depth clear value
              ],
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
          .bind_vertex_buffers(
            0,
            (
              self.model_buffers.positions.clone(),
              self.model_buffers.normals.clone(),
              self.model_buffers.tex_coords.clone(),
            ),
          )
          .unwrap()
          .bind_index_buffer(self.model_buffers.indices.clone())
          .unwrap();

        unsafe { builder.draw_indexed(self.model_buffers.indices.len() as u32, 1, 0, 0, 0) }
          .unwrap();

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

  fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
    let rcx = self.rcx.as_mut().unwrap();
    rcx.window.request_redraw();
  }
}

/// Configuration for window size dependent setup
#[derive(Clone)]
struct WindowSizeSetupConfig<'a> {
  window_size: PhysicalSize<u32>,
  images: &'a [Arc<Image>],
  render_pass: &'a Arc<RenderPass>,
  memory_allocator: &'a Arc<StandardMemoryAllocator>,
  vertex_shader: &'a EntryPoint,
  fragment_shader: &'a EntryPoint,
  wireframe_mode: bool,
  line_width: f32,
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
  config: WindowSizeSetupConfig,
) -> (Vec<Arc<Framebuffer>>, Arc<GraphicsPipeline>) {
  let device = config.memory_allocator.device();

  // Create multisampled depth buffer
  let depth_buffer = ImageView::new_default(
    Image::new(
      config.memory_allocator.clone(),
      ImageCreateInfo {
        image_type: ImageType::Dim2d,
        format: Format::D16_UNORM,
        extent: config.images[0].extent(),
        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
        samples: vulkano::image::SampleCount::Sample4,
        ..Default::default()
      },
      AllocationCreateInfo::default(),
    )
    .unwrap(),
  )
  .unwrap();

  let framebuffers = config
    .images
    .iter()
    .map(|image| {
      let view = ImageView::new_default(image.clone()).unwrap();
      let msaa_color = ImageView::new_default(
        Image::new(
          config.memory_allocator.clone(),
          ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: image.format(),
            extent: image.extent(),
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
            samples: vulkano::image::SampleCount::Sample4,
            ..Default::default()
          },
          AllocationCreateInfo::default(),
        )
        .unwrap(),
      )
      .unwrap();

      Framebuffer::new(
        config.render_pass.clone(),
        FramebufferCreateInfo {
          attachments: vec![msaa_color, view, depth_buffer.clone()],
          ..Default::default()
        },
      )
      .unwrap()
    })
    .collect::<Vec<_>>();

  // Always use line width 1.0 if wide lines are not supported
  let actual_line_width = if device.physical_device().supported_features().wide_lines {
    config.line_width
  } else {
    1.0
  };

  let pipeline = {
    let vertex_input_state = [
      Position::per_vertex(),
      Normal::per_vertex(),
      TexCoord::per_vertex(),
    ]
    .definition(config.vertex_shader)
    .unwrap();

    let stages = [
      PipelineShaderStageCreateInfo::new(config.vertex_shader.clone()),
      PipelineShaderStageCreateInfo::new(config.fragment_shader.clone()),
    ];

    let layout = PipelineLayout::new(
      device.clone(),
      PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
        .into_pipeline_layout_create_info(device.clone())
        .unwrap(),
    )
    .unwrap();

    let subpass = Subpass::from(config.render_pass.clone(), 0).unwrap();

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
            extent: config.window_size.into(),
            depth_range: 0.0..=1.0,
          }]
          .into_iter()
          .collect(),
          ..Default::default()
        }),
        rasterization_state: Some(RasterizationState {
          cull_mode: vulkano::pipeline::graphics::rasterization::CullMode::None,
          polygon_mode: if config.wireframe_mode {
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
        multisample_state: Some(MultisampleState {
          rasterization_samples: vulkano::image::SampleCount::Sample4,
          ..Default::default()
        }),
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
