use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Position {
  #[format(R32G32B32_SFLOAT)]
  pub position: [f32; 3],
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Normal {
  #[format(R32G32B32_SFLOAT)]
  pub normal: [f32; 3],
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct TexCoord {
  #[format(R32G32_SFLOAT)]
  pub tex_coord: [f32; 2],
}
