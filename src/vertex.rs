use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

/// Represents a vertex position in 3D space.
///
/// Uses 32-bit floating point values for each coordinate (x, y, z).
/// The position data is stored in a format compatible with Vulkan's R32G32B32_SFLOAT format.
#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Position {
  #[format(R32G32B32_SFLOAT)]
  pub position: [f32; 3],
}

/// Represents a vertex normal vector used for lighting calculations.
///
/// Uses 32-bit floating point values for each component (x, y, z).
/// The normal data is stored in a format compatible with Vulkan's R32G32B32_SFLOAT format.
#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Normal {
  #[format(R32G32B32_SFLOAT)]
  pub normal: [f32; 3],
}

/// Represents texture coordinates (UV coordinates) for texture mapping.
///
/// Uses 32-bit floating point values for each component (u, v).
/// The texture coordinate data is stored in a format compatible with Vulkan's R32G32_SFLOAT format.
#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct TexCoord {
  #[format(R32G32_SFLOAT)]
  pub tex_coord: [f32; 2],
}
