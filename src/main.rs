use std::error::Error;
use winit::event_loop::EventLoop;

fn main() -> Result<(), Box<dyn Error>> {
  let event_loop = EventLoop::new()?;
  let mut app = vulkano_app::App::new(&event_loop);
  Ok(event_loop.run_app(&mut app)?)
}
