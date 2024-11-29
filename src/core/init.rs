use std::sync::Arc;

use vulkano::{
  buffer::{
    allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
    Buffer,
    BufferCreateInfo,
    BufferUsage,
  },
  command_buffer::{
    allocator::StandardCommandBufferAllocator,
    AutoCommandBufferBuilder,
    CommandBufferUsage,
    CopyBufferToImageInfo,
    PrimaryCommandBufferAbstract,
  },
  descriptor_set::allocator::StandardDescriptorSetAllocator,
  device::{
    physical::PhysicalDeviceType,
    Device,
    DeviceCreateInfo,
    DeviceExtensions,
    DeviceFeatures,
    Queue,
    QueueCreateInfo,
    QueueFlags,
  },
  format::Format,
  image::{
    sampler::{Sampler, SamplerCreateInfo},
    view::ImageView,
    Image,
    ImageCreateInfo,
    ImageType,
    ImageUsage,
  },
  instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
  memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
  swapchain::Surface,
  sync::GpuFuture,
  VulkanLibrary,
};
use winit::event_loop::EventLoop;

use crate::render::model::{load_viking_room_model, VikingRoomModelBuffers};

pub struct InitializedVulkan {
  pub instance: Arc<Instance>,
  pub device: Arc<Device>,
  pub queue: Arc<Queue>,
  pub memory_allocator: Arc<StandardMemoryAllocator>,
  pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
  pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
  pub model_buffers: VikingRoomModelBuffers,
  pub uniform_buffer_allocator: SubbufferAllocator,
  pub texture: Arc<ImageView>,
  pub sampler: Arc<Sampler>,
  pub max_line_width: f32,
  pub supports_wide_lines: bool,
}

pub fn initialize_vulkan(event_loop: &EventLoop<()>) -> InitializedVulkan {
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
        image_view_format_swizzle: true, // Enable image view
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

  let model_buffers = load_viking_room_model(memory_allocator.clone());

  let uniform_buffer_allocator = SubbufferAllocator::new(
    memory_allocator.clone(),
    SubbufferAllocatorCreateInfo {
      buffer_usage: BufferUsage::UNIFORM_BUFFER,
      memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
      ..Default::default()
    },
  );

  // Load the texture
  let texture_path = std::path::Path::new("src/textures/viking_room.png");
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

  InitializedVulkan {
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
    max_line_width,
    supports_wide_lines,
  }
}
