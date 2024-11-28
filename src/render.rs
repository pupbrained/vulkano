//! Vulkan rendering pipeline and context management.
//!
//! This module handles the core rendering functionality including:
//! * Swapchain management
//! * Pipeline creation and configuration
//! * Framebuffer setup
//! * Render pass configuration

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

use crate::vertex::{Normal, Position, TexCoord};

/// Core rendering context containing all Vulkan resources needed for rendering.
///
/// This struct maintains ownership of critical Vulkan objects and manages their lifecycle.
/// It includes the window, swapchain, render pass, framebuffers, and graphics pipeline.
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
/// when the window is resized.
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
/// It handles:
/// * Creating new framebuffers for the current window size
/// * Setting up the graphics pipeline with appropriate viewport
/// * Configuring render states (blending, depth testing, etc.)
///
/// # Parameters
/// * `config` - Configuration containing all required resources
///
/// # Returns
/// A tuple containing the new framebuffers and graphics pipeline
pub fn window_size_dependent_setup(
  config: WindowSizeSetupConfig,
) -> (Vec<Arc<Framebuffer>>, Arc<GraphicsPipeline>) {
  let device = config.memory_allocator.device();

  // Create multisampled depth buffer
  let depth_buffer = ImageView::new_default(
    Image::new(
      config.memory_allocator.clone(),
      ImageCreateInfo {
        image_type: ImageType::Dim2d,
        format: Format::D32_SFLOAT,
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
