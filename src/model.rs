use std::sync::Arc;
use vulkano::{
  buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
  memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

use crate::{Normal, Position, TexCoord};

pub struct VikingRoomModelBuffers {
  pub positions: Subbuffer<[Position]>,
  pub normals: Subbuffer<[Normal]>,
  pub tex_coords: Subbuffer<[TexCoord]>,
  pub indices: Subbuffer<[u32]>,
}

pub fn load_viking_room_model(
  memory_allocator: Arc<StandardMemoryAllocator>,
) -> VikingRoomModelBuffers {
  let (positions, normals, tex_coords, indices) = {
    let model = tobj::load_obj(
      "models/viking_room.obj",
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
