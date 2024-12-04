//! A Vulkan-based 3D renderer for displaying textured models.
//!
//! This crate provides a simple yet efficient implementation of a 3D renderer using the Vulkan graphics API
//! through the vulkano Rust bindings. It supports:
//! * 3D model loading and rendering with efficient vertex buffer management
//! * Texture mapping with mipmaps and anisotropic filtering
//! * Interactive camera controls with smooth movement
//! * Configurable vertex attribute handling
//! * Real-time performance monitoring
//! * Wireframe rendering mode for debugging
//!
//! # Architecture
//! The crate is organized into several modules:
//! * `core`: Low-level Vulkan initialization and command buffer management
//! * `render`: High-level rendering components (camera, models, pipelines)
//! * `app`: Main application state and event handling
//! * `gui`: User interface implementation using egui
//! * `shaders`: GLSL shader compilation and management
//!
//! # Example
//! ```no_run
//! use winit::event_loop::EventLoop;
//! use vulkano_app::App;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let event_loop = EventLoop::new()?;
//!     let mut app = App::new(&event_loop);
//!     Ok(event_loop.run_app(&mut app)?)
//! }
//! ```

// Core Vulkan functionality
pub mod core {
  /// Extensions for command buffer creation and management
  pub mod command_buffer_builder_ext;
  /// Vulkan instance and device initialization
  pub mod init;
}

// Rendering components
pub mod render {
  /// Camera system with physics-based movement
  pub mod camera;
  /// 3D model loading and management
  pub mod model;
  /// Vulkan graphics pipeline configuration
  pub mod pipeline;
  /// Vertex attribute definitions and handling
  pub mod vertex;
}

/// Core application functionality and state management
pub mod app;
/// GUI implementation using egui
pub mod gui;
/// GLSL shader compilation and resource management
pub mod shaders;

// Re-export commonly used items
pub use core::init::{initialize_vulkan, InitializedVulkan};

pub use app::App;
pub use gui::GuiState;
pub use render::{
  camera::Camera,
  model::{load_viking_room_model, VikingRoomModelBuffers},
  pipeline::{window_size_dependent_setup, RenderContext, WindowSizeSetupConfig},
};
pub use shaders::{fs, vs};
