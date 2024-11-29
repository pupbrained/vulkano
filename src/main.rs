use std::error::Error;

use winit::event_loop::EventLoop;

use vulkano_app::App;

fn main() -> Result<(), Box<dyn Error>> {
  let event_loop = EventLoop::new()?;
  let mut app = App::new(&event_loop);
  Ok(event_loop.run_app(&mut app)?)
}
