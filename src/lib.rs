//! A Vulkan-based 3D renderer for displaying textured models.
//!
//! This crate provides a simple yet efficient implementation of a 3D renderer using the Vulkan graphics API
//! through the vulkano Rust bindings. It supports:
//! * 3D model loading and rendering
//! * Texture mapping
//! * Camera controls
//! * Vertex attribute handling

/// Core application functionality and initialization
pub mod app;
/// Camera management and transformation matrices
pub mod camera;
/// Vulkan initialization and device setup
pub mod init;
/// 3D model loading and GPU buffer management
pub mod model;
/// Vulkan rendering pipeline and setup
pub mod render;
/// GLSL shader management
pub mod shaders;
/// Vertex attribute definitions and format specifications
pub mod vertex;

// command buffer builder functions
pub mod command_buffer_builder_ext;

// Re-export commonly used items
pub use app::App;
pub use camera::Camera;
pub use init::{initialize_vulkan, InitializedVulkan};
pub use model::{load_viking_room_model, VikingRoomModelBuffers};
pub use render::{window_size_dependent_setup, RenderContext, WindowSizeSetupConfig};
pub use shaders::{fs, vs};
pub use vertex::{Normal, Position, TexCoord};
