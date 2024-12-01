//! Vulkan rendering pipeline and context management.
//!
//! This module implements the core rendering infrastructure including:
//! * Swapchain: Manages the presentation of rendered frames to the window
//! * Pipeline: Configures the graphics pipeline stages (vertex/fragment shaders, blending, etc.)
//! * Framebuffers: Manages render targets for each swapchain image
//! * Render Pass: Defines the sequence of rendering operations
//!
//! # Pipeline Configuration
//! The graphics pipeline is configured with:
//! * Vertex input: Position, normal, and texture coordinate attributes
//! * Shaders: Vertex and fragment shaders for 3D rendering
//! * Depth testing: Enabled with less-equal comparison
//! * Alpha blending: Disabled for opaque rendering
//! * Multisampling: No multisampling (1 sample per pixel)
//!
//! # Usage
//! The pipeline is automatically recreated when:
//! * The window is resized
//! * The swapchain becomes invalid
//! * Render pass configuration changes

use std::sync::Arc;

use vulkano::{
  device::DeviceOwned,
  format::Format,
  image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
  memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator},
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
    GraphicsPipeline,
    PipelineLayout,
    PipelineShaderStageCreateInfo,
  },
  render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
  shader::EntryPoint,
  swapchain::Swapchain,
  sync::GpuFuture,
};
use winit::{dpi::PhysicalSize, window::Window};

use crate::render::vertex::{Normal, Position, TexCoord};

/// Core rendering context containing all Vulkan resources needed for rendering.
///
/// This struct maintains ownership of critical Vulkan objects and manages their lifecycle:
/// * `window`: The window surface for presentation
/// * `swapchain`: Manages frame presentation and image acquisition
/// * `render_pass`: Defines the rendering operations sequence
/// * `framebuffers`: Render targets for each swapchain image
/// * `viewport`: Current rendering viewport dimensions
///
/// The context automatically handles:
/// * Window resize events by recreating the swapchain
/// * Pipeline recreation when needed
/// * Synchronization of rendering and presentation
///
/// # Example
/// ```
/// let context = RenderContext {
///     window: window.clone(),
///     swapchain: swapchain.clone(),
///     render_pass: render_pass.clone(),
///     framebuffers: framebuffers,
///     viewport: Viewport {
///         offset: [0.0, 0.0],
///         extent: [width as f32, height as f32],
///         depth_range: 0.0..1.0,
///     },
/// };
/// ```
pub struct RenderContext {
  /// The window being rendered to
  pub window: Arc<Window>,
  /// Vulkan swapchain for presenting rendered images
  pub swapchain: Arc<Swapchain>,
  /// Render pass defining the rendering operations
  pub render_pass: Arc<RenderPass>,
  /// Framebuffers for each swapchain image
  pub framebuffers: Vec<Arc<Framebuffer>>,
  /// Compiled vertex shader
  pub vs: EntryPoint,
  /// Compiled fragment shader
  pub fs: EntryPoint,
  /// Graphics pipeline containing all render state
  pub pipeline: Arc<GraphicsPipeline>,
  /// Flag indicating if swapchain needs recreation
  pub recreate_swapchain: bool,
  /// Synchronization primitive for frame rendering
  pub previous_frame_end: Option<Box<dyn GpuFuture>>,
  /// Views into the swapchain images
  pub swapchain_image_views: Vec<Arc<ImageView>>,
}

/// Configuration for window size dependent setup.
///
/// Contains all resources needed to recreate the pipeline and framebuffers
/// when the window is resized. This includes:
/// * Memory allocators for buffers and descriptor sets
/// * Shader entry points for vertex and fragment shaders
/// * Vertex buffer layouts and attributes
/// * Pipeline layout configuration
///
/// This configuration is used to ensure proper recreation of resources
/// when the window size changes, maintaining render quality and correctness.
#[derive(Clone)]
pub struct WindowSizeSetupConfig<'a> {
  /// Current window dimensions
  pub window_size: PhysicalSize<u32>,
  /// Swapchain images
  pub images: &'a [Arc<Image>],
  /// Render pass to create framebuffers for
  pub render_pass: &'a Arc<RenderPass>,
  /// Memory allocator for creating new resources
  pub memory_allocator: &'a Arc<StandardMemoryAllocator>,
  /// Vertex shader to use in pipeline
  pub vertex_shader: &'a EntryPoint,
  /// Fragment shader to use in pipeline
  pub fragment_shader: &'a EntryPoint,
  /// Flag to enable wireframe rendering
  pub wireframe_mode: bool,
  /// Line width for wireframe rendering
  pub line_width: f32,
}

/// Creates or recreates window size dependent resources.
///
/// This function is called during initialization and whenever the window is resized.
/// It performs the following steps:
/// 1. Creates new framebuffers matching the current window size
/// 2. Configures the graphics pipeline with:
///    * Viewport and scissor matching window dimensions
///    * Vertex input layout (position, normal, texcoord)
///    * Shader stages and specialization constants
///    * Fixed-function state (blending, depth testing, etc.)
///    * Dynamic state for viewport updates
/// 3. Sets up render states:
///    * Depth testing enabled (LESS_OR_EQUAL)
///    * No color blending (opaque rendering)
///    * Clockwise face culling
///    * Fill polygon mode
///
/// # Parameters
/// * `config` - Configuration containing all required resources
///
/// # Returns
/// A tuple containing:
/// * Vector of framebuffers, one for each swapchain image
/// * Configured graphics pipeline ready for rendering
///
/// # Panics
/// Will panic if:
/// * Pipeline creation fails
/// * Framebuffer creation fails
/// * Window size is invalid (zero width/height)
pub fn window_size_dependent_setup(
  config: WindowSizeSetupConfig,
) -> (Vec<Arc<Framebuffer>>, Arc<GraphicsPipeline>) {
  let device = config.memory_allocator.device();

  // Create multisampled depth buffer for MSAA (4x anti-aliasing)
  // This buffer stores depth values and is used for depth testing during rendering
  // The TRANSIENT_ATTACHMENT flag optimizes for GPU-local memory since we don't need to read it back
  let depth_buffer = ImageView::new_default(
    Image::new(
      config.memory_allocator.clone(),
      ImageCreateInfo {
        image_type: ImageType::Dim2d,
        format: Format::D32_SFLOAT, // 32-bit float depth buffer for high precision
        extent: config.images[0].extent(),
        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
        samples: vulkano::image::SampleCount::Sample4, // 4x MSAA
        ..Default::default()
      },
      AllocationCreateInfo::default(),
    )
    .unwrap(),
  )
  .unwrap();

  // Create framebuffers for each swapchain image
  let framebuffers = config
    .images
    .iter()
    .map(|image| {
      // Create a view into the swapchain image
      let view = ImageView::new_default(image.clone()).unwrap();

      // Create a multisampled color buffer for MSAA
      // This is where we'll render to before resolving to the final swapchain image
      let msaa_color = ImageView::new_default(
        Image::new(
          config.memory_allocator.clone(),
          ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: image.format(), // Match swapchain format
            extent: image.extent(),
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
            samples: vulkano::image::SampleCount::Sample4, // 4x MSAA
            ..Default::default()
          },
          AllocationCreateInfo::default(),
        )
        .unwrap(),
      )
      .unwrap();

      // Create framebuffer with MSAA color, final color, and depth attachments
      // The render pass will automatically resolve the MSAA color to the final image
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

  // Check if wide lines are supported (useful for debug rendering)
  // Some devices (particularly mobile) don't support lines wider than 1.0
  let actual_line_width = if device.physical_device().supported_features().wide_lines {
    config.line_width
  } else {
    1.0
  };

  let pipeline = {
    // Define vertex attributes and their memory layout
    // This must match the vertex shader input layout
    let vertex_input_state = [
      Position::per_vertex(), // vec3 position
      Normal::per_vertex(),   // vec3 normal
      TexCoord::per_vertex(), // vec2 texcoord
    ]
    .definition(config.vertex_shader)
    .unwrap();

    // Configure shader stages (vertex and fragment)
    let stages = [
      PipelineShaderStageCreateInfo::new(config.vertex_shader.clone()),
      PipelineShaderStageCreateInfo::new(config.fragment_shader.clone()),
    ];

    // Create pipeline layout which defines shader resource bindings
    // This includes uniform buffers, textures, and push constants
    let layout = PipelineLayout::new(
      device.clone(),
      PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
        .into_pipeline_layout_create_info(device.clone())
        .unwrap(),
    )
    .unwrap();

    let subpass = Subpass::from(config.render_pass.clone(), 0).unwrap();

    // Configure the graphics pipeline with all our settings
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
