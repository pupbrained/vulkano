//! GLSL shader compilation and loading.
//!
//! This module provides compile-time shader loading using the vulkano_shaders macro.
//! The shaders are compiled from GLSL source files during build time.

/// Vertex shader module.
///
/// Loads and compiles the vertex shader from "src/vert.glsl".
/// The shader handles vertex transformations and attribute passing.
pub mod vs {
  vulkano_shaders::shader! {
    ty: "vertex",
    path: "src/shaders/vert.glsl",
  }
}

/// Fragment shader module.
///
/// Loads and compiles the fragment shader from "src/frag.glsl".
/// The shader handles texture sampling and final color output.
pub mod fs {
  vulkano_shaders::shader! {
    ty: "fragment",
    path: "src/shaders/frag.glsl",
  }
}
