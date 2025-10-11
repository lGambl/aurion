pub mod audio;

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use self::audio::AudioCapture;
use anyhow::{Context, Result};
use crossbeam::queue::ArrayQueue;
use wgpu::SurfaceError;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

const METER_SAMPLES_PER_FRAME: usize = 4_096;
const AUDIO_ATTACK: f32 = 0.2;
const AUDIO_DECAY: f32 = 0.96;
const MIN_LEVEL_DB: f32 = -90.0;
const MIN_AUDIO_LEVEL: f32 = 1.0e-6;

pub struct AurionApp {
    window: Option<Arc<Window>>,
    window_id: Option<WindowId>,
    state: Option<RendererState>,
}

impl Default for AurionApp {
    fn default() -> Self {
        Self {
            window: None,
            window_id: None,
            state: None,
        }
    }
}

impl ApplicationHandler for AurionApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs: WindowAttributes = Window::default_attributes()
            .with_title("Aurion (starting)")
            .with_inner_size(PhysicalSize::new(1280, 720));

        let raw_window = event_loop
            .create_window(attrs)
            .expect("failed to create window");
        let window_id = raw_window.id();
        let window = Arc::new(raw_window);

        let state = pollster::block_on(RendererState::new(Arc::clone(&window)))
            .expect("failed to initialize GPU/audio state");

        self.state = Some(state);
        self.window_id = Some(window_id);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if Some(window_id) != self.window_id {
            return;
        }

        let (Some(window), Some(state)) = (self.window.as_ref(), self.state.as_mut()) else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::ScaleFactorChanged {
                mut inner_size_writer,
                ..
            } => {
                let new_size = PhysicalSize::new(state.size.width.max(1), state.size.height.max(1));
                let _ = inner_size_writer.request_inner_size(new_size);
                state.resize(new_size);
            }
            WindowEvent::RedrawRequested => {
                state.update_audio_meter();
                state.update_window_title(window);
                match state.render_frame() {
                    Ok(()) => {}
                    Err(SurfaceError::Lost) => state.resize(state.size),
                    Err(SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(_) => {}
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl AurionApp {
    pub fn run() -> Result<()> {
        let event_loop = EventLoop::new()?;
        let mut app = Self::default();
        event_loop.run_app(&mut app)?;
        Ok(())
    }
}

struct RendererState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    _instance: wgpu::Instance,
    _window_handle: Arc<Window>,
    frame_counter: u32,
    fps_timer: Instant,
    _audio: AudioCapture,
    audio_queue: Arc<ArrayQueue<f32>>,
    audio_level: f32,
}

impl RendererState {
    async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(Arc::clone(&window))
            .context("failed to create surface")?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("failed to find a suitable GPU adapter")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Aurion Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await
            .context("failed to create device")?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|format| format.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let present_mode = if surface_caps
            .present_modes
            .contains(&wgpu::PresentMode::Fifo)
        {
            wgpu::PresentMode::Fifo
        } else {
            surface_caps.present_modes[0]
        };

        let alpha_mode = surface_caps
            .alpha_modes
            .first()
            .copied()
            .unwrap_or(wgpu::CompositeAlphaMode::Opaque);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        let audio = AudioCapture::new().context("failed to start audio capture")?;
        let audio_queue = audio.queue();

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            _instance: instance,
            _window_handle: window,
            frame_counter: 0,
            fps_timer: Instant::now(),
            _audio: audio,
            audio_queue,
            audio_level: 0.0,
        })
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
    }

    fn render_frame(&mut self) -> Result<(), SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Aurion Render Encoder"),
            });

        {
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Aurion Clear Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.06,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        Ok(())
    }

    fn update_audio_meter(&mut self) {
        let mut sum = 0.0;
        let mut count = 0;

        while count < METER_SAMPLES_PER_FRAME {
            match self.audio_queue.pop() {
                Some(sample) => {
                    sum += sample * sample;
                    count += 1;
                }
                None => break,
            }
        }

        if count > 0 {
            let rms = (sum / count as f32).sqrt();
            self.audio_level = self.audio_level * (1.0 - AUDIO_ATTACK) + rms * AUDIO_ATTACK;
        } else {
            self.audio_level *= AUDIO_DECAY;
        }
    }

    fn update_window_title(&mut self, window: &Window) {
        self.frame_counter += 1;
        let now = Instant::now();
        let elapsed = now - self.fps_timer;

        if elapsed >= Duration::from_secs(1) {
            let fps = self.frame_counter as f64 / elapsed.as_secs_f64();
            let level_db = if self.audio_level <= MIN_AUDIO_LEVEL {
                MIN_LEVEL_DB
            } else {
                20.0 * self.audio_level.log10()
            };
            window.set_title(&format!("Aurion ({fps:.0} FPS | {level_db:.1} dBFS)"));
            self.frame_counter = 0;
            self.fps_timer = now;
        }
    }
}
