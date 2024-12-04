use std::sync::Arc;

use vulkano::{
  buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
  memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

use crate::render::vertex::{Normal, Position, TexCoord};

/// Represents the GPU buffers containing the 3D model data for the Viking Room.
///
/// This struct holds four Vulkan buffers that store different vertex attributes:
/// * `positions`: (x, y, z) coordinates of each vertex in 3D space
/// * `normals`: Surface normal vectors used for lighting calculations
/// * `tex_coords`: (u, v) texture coordinates for mapping textures
/// * `indices`: Vertex indices that define triangles (3 indices per triangle)
///
/// The buffers are created on the GPU for optimal rendering performance.
/// Each buffer is stored as a Vulkan Subbuffer, allowing for efficient memory
/// management and data transfer between CPU and GPU.
///
/// # Memory Layout
/// * positions: `[x, y, z, x, y, z, ...]` - 3 floats per vertex
/// * normals: `[nx, ny, nz, nx, ny, nz, ...]` - 3 floats per vertex
/// * tex_coords: `[u, v, u, v, ...]` - 2 floats per vertex
/// * indices: `[i1, i2, i3, i1, i2, i3, ...]` - 3 u32s per triangle
pub struct VikingRoomModelBuffers {
  pub positions:  Subbuffer<[Position]>,
  pub normals:    Subbuffer<[Normal]>,
  pub tex_coords: Subbuffer<[TexCoord]>,
  pub indices:    Subbuffer<[u32]>,
}

/// Loads the Viking Room 3D model from an OBJ file and creates GPU buffers for rendering.
///
/// This function performs the following steps:
/// 1. Loads and triangulates the OBJ model from "models/viking_room.obj"
/// 2. Extracts vertex attributes (positions, normals, texture coordinates)
/// 3. Creates optimized Vulkan buffers with appropriate memory flags:
///    * `BufferUsage::VERTEX_BUFFER` for vertex data
///    * `BufferUsage::INDEX_BUFFER` for indices
///    * `MemoryTypeFilter::PREFER_DEVICE` for GPU-local storage
///
/// # Parameters
/// * `memory_allocator` - The Vulkan memory allocator used to create the GPU buffers
///
/// # Returns
/// Returns a `VikingRoomModelBuffers` containing all the necessary buffers for rendering
///
/// # Panics
/// Will panic if:
/// * The OBJ file cannot be found or is invalid
/// * There is insufficient GPU memory
/// * Buffer creation fails due to Vulkan errors
///
/// # Example
/// ```
/// let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
/// let model_buffers = load_viking_room_model(memory_allocator);
///
/// // Use the buffers in a vertex buffer binding
/// command_buffer.bind_vertex_buffers(0, (
///     model_buffers.positions.clone(),
///     model_buffers.normals.clone(),
///     model_buffers.tex_coords.clone()
/// ));
/// command_buffer.bind_index_buffer(model_buffers.indices.clone());
/// ```
pub fn load_viking_room_model(
  memory_allocator: Arc<StandardMemoryAllocator>,
) -> VikingRoomModelBuffers {
  let (positions, normals, tex_coords, indices) = {
    let model = tobj::load_obj(
      "src/models/viking_room.obj",
      &tobj::LoadOptions {
        triangulate: true,
        ..Default::default()
      },
    )
    .unwrap();

    let mesh = &model.0[0].mesh;

    let positions = mesh
      .positions
      .chunks(3)
      .map(|xyz| Position {
        position: [xyz[0], xyz[1], xyz[2]],
      })
      .collect::<Vec<_>>();

    let normals = mesh
      .normals
      .chunks(3)
      .map(|xyz| Normal {
        normal: [xyz[0], xyz[1], xyz[2]],
      })
      .collect::<Vec<_>>();

    let tex_coords = mesh
      .texcoords
      .chunks(2)
      .map(|uv| TexCoord {
        tex_coord: [uv[0], 1.0 - uv[1]],
      })
      .collect::<Vec<_>>();

    let indices = mesh.indices.clone();

    (positions, normals, tex_coords, indices)
  };

  let positions = Buffer::from_iter(
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

  let normals = Buffer::from_iter(
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

  let tex_coords = Buffer::from_iter(
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

  let indices = Buffer::from_iter(
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
    positions,
    normals,
    tex_coords,
    indices,
  }
}
