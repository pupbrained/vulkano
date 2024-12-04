//! Vulkano Application Entry Point
//!
//! This is the main entry point for the Vulkano-based application.
//! It initializes the event loop and application state, then runs
//! the main application loop.

use vulkano_app::App;
use winit::{error::EventLoopError, event_loop::EventLoop};

/// Application entry point that sets up and runs the main event loop.
///
/// # Returns
/// * `Ok(())` if the application exits normally
/// * `Err` if initialization or execution fails
///
/// # Errors
/// This function will return an error if:
/// * Event loop creation fails
/// * Application initialization fails
/// * Runtime errors occur during execution
fn main() -> Result<(), EventLoopError> {
  let event_loop = EventLoop::new()?;
  let mut app = App::new(&event_loop);
  event_loop.run_app(&mut app)
}
