//! A Vulkan-based 3D renderer for displaying textured models.
//!
//! This crate provides a simple yet efficient implementation of a 3D renderer using the Vulkan graphics API
//! through the vulkano Rust bindings. It supports:
//! * 3D model loading and rendering
//! * Texture mapping
//! * Camera controls
//! * Vertex attribute handling

// Core Vulkan functionality
pub mod core {
  pub mod command_buffer_builder_ext;
  pub mod init;
}

// Rendering components
pub mod render {
  pub mod camera;
  pub mod model;
  pub mod render;
  pub mod vertex;
}

/// Core application functionality
pub mod app;
/// GLSL shader management
pub mod shaders;
/// GUI management
pub mod gui;

// Re-export commonly used items
pub use core::init::{initialize_vulkan, InitializedVulkan};

pub use app::App;
pub use gui::GuiState;
pub use render::{
  camera::Camera,
  model::{load_viking_room_model, VikingRoomModelBuffers},
  render::{window_size_dependent_setup, RenderContext, WindowSizeSetupConfig},
  vertex::{Normal, Position, TexCoord},
};
pub use shaders::{fs, vs};
