pub mod app;
pub mod camera;
pub mod model;
pub mod render;
pub mod shaders;
pub mod vertex;

// Re-export commonly used items
pub use app::App;
pub use camera::Camera;
pub use model::{load_viking_room_model, VikingRoomModelBuffers};
pub use render::{window_size_dependent_setup, RenderContext, WindowSizeSetupConfig};
pub use shaders::{fs, vs};
pub use vertex::{Normal, Position, TexCoord};
