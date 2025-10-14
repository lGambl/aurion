pub mod audio;

use std::{
    collections::VecDeque,
    f32::consts::PI,
    mem::size_of,
    sync::Arc,
    time::{Duration, Instant},
};

use self::audio::AudioCapture;
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use crossbeam::queue::ArrayQueue;
use rustfft::{num_complex::Complex32, FftPlanner};
use wgpu::SurfaceError;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes, WindowId, WindowLevel},
};

const METER_SAMPLES_PER_FRAME: usize = 4_096;
const AUDIO_ATTACK: f32 = 0.2;
const AUDIO_DECAY: f32 = 0.96;
const MIN_LEVEL_DB: f32 = -90.0;
const MIN_AUDIO_LEVEL: f32 = 1.0e-6;
const FFT_SIZE: usize = 2_048;
const MAX_AUDIO_SAMPLES: usize = FFT_SIZE * 4;
const INITIAL_SPECTRUM_BINS: usize = 256;
const SPECTRUM_ATTACK: f32 = 0.25;
const SPECTRUM_DECAY: f32 = 0.85;
// Visual tuning defaults (now adjustable at runtime)
const DEFAULT_GAP_FRACTION: f32 = 0.01; // 1% of each bar width is gap
const DEFAULT_GAIN: f32 = 1.6; // vertical gain before clamping
const DEFAULT_SIDE_FRACTION: f32 = 0.25; // each side uses up to 25% width
const DEFAULT_WIDTH_SCALE: f32 = 1.2; // expand bar width within its cell

pub struct AurionApp {
    window: Option<Arc<Window>>,
    window_id: Option<WindowId>,
    state: Option<RendererState>,
    click_through: bool,
    always_on_top: bool,
}

impl Default for AurionApp {
    fn default() -> Self {
        Self {
            window: None,
            window_id: None,
            state: None,
            click_through: false,
            always_on_top: true,
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
            .with_inner_size(PhysicalSize::new(1280, 720))
            .with_transparent(true)
            .with_decorations(true)
            .with_window_level(WindowLevel::AlwaysOnTop);

        let raw_window = event_loop
            .create_window(attrs)
            .expect("failed to create window");
        let window_id = raw_window.id();
        let window = Arc::new(raw_window);

        // Enable click-through by default (toggle with F8)
        if self.click_through {
            window.set_cursor_hittest(false);
        }

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
            WindowEvent::KeyboardInput { event, .. } => {
                // Toggle click-through with F8, close with Escape
                if event.state == winit::event::ElementState::Pressed {
                    match &event.logical_key {
                        Key::Named(NamedKey::F8) => {
                            self.click_through = !self.click_through;
                            window.set_cursor_hittest(!self.click_through);
                        }
                        Key::Named(NamedKey::F7) => {
                            // Toggle always-on-top
                            self.always_on_top = !self.always_on_top;
                            use winit::window::WindowLevel as WL;
                            let next = if self.always_on_top { WL::AlwaysOnTop } else { WL::Normal };
                            window.set_window_level(next);
                        }
                        Key::Named(NamedKey::F9) => {
                            // Increase number of bars
                            if let Some(state) = self.state.as_mut() {
                                let half_len = FFT_SIZE / 2;
                                let max_bins = half_len.max(1);
                                let step = 16usize;
                                let new_bins = (state.num_bins + step).min(max_bins);
                                if new_bins != state.num_bins {
                                    state.num_bins = new_bins;
                                    state.spectrum_bins.resize(new_bins, 0.0);
                                    state.spectrum_renderer = SpectrumRenderer::new(
                                        &state.device,
                                        state.config.format,
                                        new_bins,
                                        state.blend,
                                    );
                                }
                            }
                        }
                        Key::Named(NamedKey::F10) => {
                            // Decrease number of bars
                            if let Some(state) = self.state.as_mut() {
                                let min_bins = 16usize;
                                let step = 16usize;
                                let new_bins = state.num_bins.saturating_sub(step).max(min_bins);
                                if new_bins != state.num_bins {
                                    state.num_bins = new_bins;
                                    state.spectrum_bins.resize(new_bins, 0.0);
                                    state.spectrum_renderer = SpectrumRenderer::new(
                                        &state.device,
                                        state.config.format,
                                        new_bins,
                                        state.blend,
                                    );
                                }
                            }
                        }
                        Key::Named(NamedKey::ArrowUp) => {
                            if let Some(state) = self.state.as_mut() {
                                state.gain = (state.gain + 0.1).clamp(0.1, 5.0);
                            }
                        }
                        Key::Named(NamedKey::ArrowDown) => {
                            if let Some(state) = self.state.as_mut() {
                                state.gain = (state.gain - 0.1).clamp(0.1, 5.0);
                            }
                        }
                        Key::Named(NamedKey::ArrowRight) => {
                            if let Some(state) = self.state.as_mut() {
                                state.gap_fraction = (state.gap_fraction + 0.01).clamp(0.0, 0.4);
                            }
                        }
                        Key::Named(NamedKey::ArrowLeft) => {
                            if let Some(state) = self.state.as_mut() {
                                state.gap_fraction = (state.gap_fraction - 0.01).clamp(0.0, 0.4);
                            }
                        }
                        Key::Character(ch) if ch == "r" || ch == "R" => {
                            if let Some(state) = self.state.as_mut() {
                                state.gap_fraction = DEFAULT_GAP_FRACTION;
                                state.gain = DEFAULT_GAIN;
                                state.side_fraction = DEFAULT_SIDE_FRACTION;
                                state.width_scale = DEFAULT_WIDTH_SCALE;
                            }
                        }
                        Key::Named(NamedKey::F11) => {
                            // Increase side coverage up to 50%
                            if let Some(state) = self.state.as_mut() {
                                state.side_fraction = (state.side_fraction + 0.05).clamp(0.05, 0.5);
                            }
                        }
                        Key::Named(NamedKey::F12) => {
                            // Decrease side coverage down to 5%
                            if let Some(state) = self.state.as_mut() {
                                state.side_fraction = (state.side_fraction - 0.05).clamp(0.05, 0.5);
                            }
                        }
                        Key::Character(ch) if ch == "]" => {
                            if let Some(state) = self.state.as_mut() {
                                state.width_scale = (state.width_scale + 0.05).clamp(0.1, 2.0);
                            }
                        }
                        Key::Character(ch) if ch == "[" => {
                            if let Some(state) = self.state.as_mut() {
                                state.width_scale = (state.width_scale - 0.05).clamp(0.1, 2.0);
                            }
                        }
                        Key::Named(NamedKey::Escape) => {
                            event_loop.exit();
                        }
                        _ => {}
                    }
                }
            }
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
                state.update_audio_state();
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
    audio_history: VecDeque<f32>,
    fft_buffer: Vec<Complex32>,
    fft: Arc<dyn rustfft::Fft<f32> + Send + Sync>,
    fft_window: Vec<f32>,
    spectrum_bins: Vec<f32>,
    spectrum_renderer: SpectrumRenderer,
    num_bins: usize,
    gap_fraction: f32,
    gain: f32,
    side_fraction: f32,
    width_scale: f32,
    blend: wgpu::BlendState,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

struct SpectrumRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    cpu_vertices: Vec<Vertex>,
    vertex_capacity: usize,
    vertex_count: u32,
}

impl SpectrumRenderer {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        bins: usize,
        blend: wgpu::BlendState,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Aurion Spectrum Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/spectrum.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Aurion Spectrum Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Aurion Spectrum Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Vertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let vertex_capacity = bins.max(1) * 6;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Aurion Spectrum Vertex Buffer"),
            size: (vertex_capacity * size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            vertex_buffer,
            cpu_vertices: Vec::with_capacity(vertex_capacity),
            vertex_capacity,
            vertex_count: 0,
        }
    }

    fn update(&mut self, queue: &wgpu::Queue, levels: &[f32], gap_fraction: f32, gain: f32, side_fraction: f32, width_scale: f32) {
        self.cpu_vertices.clear();

        if levels.is_empty() {
            self.vertex_count = 0;
            return;
        }

        let base_line = -1.0f32; // bottom of NDC
        let max_height = 2.0f32; // full vertical range
        let side = side_fraction.clamp(0.0, 0.5);
        let total = levels.len();
        let left_count = total / 2;
        let right_count = total - left_count;

        // Left band occupies left side_fraction of width
        if left_count > 0 && side > 0.0 {
            let left_x0 = -1.0f32;
            let left_x1 = -1.0f32 + 2.0f32 * side;
            let band_width = left_x1 - left_x0;
            let cell_width = band_width / left_count as f32;
            let gap = cell_width * gap_fraction.clamp(0.0, 0.9);
            for i in 0..left_count {
                let level = levels[i];
                let clamped = (level * gain).clamp(0.0, 1.0);
                let center = left_x0 + (i as f32 + 0.5) * cell_width;
                let mut bar_w = (cell_width - gap) * width_scale.max(0.1);
                if bar_w > cell_width { bar_w = cell_width; }
                let x0 = center - bar_w * 0.5;
                let x1 = center + bar_w * 0.5;
                let y0 = base_line;
                let y1 = (y0 + clamped * max_height).clamp(-1.0, 1.0);

                let bottom_color = color_for_level(clamped * 0.4);
                let top_color = color_for_level(clamped);

                self.cpu_vertices.extend_from_slice(&[
                    Vertex { position: [x0, y0], color: bottom_color },
                    Vertex { position: [x1, y0], color: bottom_color },
                    Vertex { position: [x1, y1], color: top_color },
                    Vertex { position: [x0, y0], color: bottom_color },
                    Vertex { position: [x1, y1], color: top_color },
                    Vertex { position: [x0, y1], color: top_color },
                ]);
            }
        }

        // Right band occupies right side_fraction of width
        if right_count > 0 && side > 0.0 {
            let right_x0 = 1.0f32 - 2.0f32 * side;
            let right_x1 = 1.0f32;
            let band_width = right_x1 - right_x0;
            let cell_width = band_width / right_count as f32;
            let gap = cell_width * gap_fraction.clamp(0.0, 0.9);
            for j in 0..right_count {
                let level = levels[left_count + j];
                let clamped = (level * gain).clamp(0.0, 1.0);
                let center = right_x0 + (j as f32 + 0.5) * cell_width;
                let mut bar_w = (cell_width - gap) * width_scale.max(0.1);
                if bar_w > cell_width { bar_w = cell_width; }
                let x0 = center - bar_w * 0.5;
                let x1 = center + bar_w * 0.5;
                let y0 = base_line;
                let y1 = (y0 + clamped * max_height).clamp(-1.0, 1.0);

                let bottom_color = color_for_level(clamped * 0.4);
                let top_color = color_for_level(clamped);

                self.cpu_vertices.extend_from_slice(&[
                    Vertex { position: [x0, y0], color: bottom_color },
                    Vertex { position: [x1, y0], color: bottom_color },
                    Vertex { position: [x1, y1], color: top_color },
                    Vertex { position: [x0, y0], color: bottom_color },
                    Vertex { position: [x1, y1], color: top_color },
                    Vertex { position: [x0, y1], color: top_color },
                ]);
            }
        }

        debug_assert!(self.cpu_vertices.len() <= self.vertex_capacity);

        if self.cpu_vertices.is_empty() {
            self.vertex_count = 0;
            return;
        }

        queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(&self.cpu_vertices),
        );
        self.vertex_count = self.cpu_vertices.len() as u32;
    }

    fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.vertex_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..self.vertex_count, 0..1);
    }
}

fn color_for_level(level: f32) -> [f32; 3] {
    let l = level.clamp(0.0, 1.0);
    if l <= 0.5 {
        let t = l / 0.5;
        lerp_color([0.08, 0.1, 0.4], [0.12, 0.75, 0.95], t)
    } else {
        let t = (l - 0.5) / 0.5;
        lerp_color([0.12, 0.75, 0.95], [1.0, 0.78, 0.2], t)
    }
}

fn lerp_color(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

impl RendererState {
    async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
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

        // Optional debug of surface capabilities
        eprintln!(
            "formats: {:?}\npresent_modes: {:?}\nalpha_modes: {:?}",
            surface_caps.formats,
            surface_caps.present_modes,
            surface_caps.alpha_modes
        );

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

        // Prefer transparent compositing when supported
        let alpha_mode = surface_caps.alpha_modes
            .iter()
            .copied()
            .find(|m| *m == wgpu::CompositeAlphaMode::PreMultiplied)
            .or_else(|| surface_caps.alpha_modes
                .iter()
                .copied()
                .find(|m| *m == wgpu::CompositeAlphaMode::PostMultiplied))
            .unwrap_or(wgpu::CompositeAlphaMode::Opaque);

        // Helpful heads-up if the platform can't do true transparency
        if alpha_mode == wgpu::CompositeAlphaMode::Opaque {
            eprintln!(
                "WARNING: Surface alpha mode is Opaque; real desktop-through transparency \
                 won't be available on this platform/driver."
            );
        }

        // Choose appropriate blending for the selected alpha mode
        let premul = alpha_mode == wgpu::CompositeAlphaMode::PreMultiplied;
        let blend = if premul {
            wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
        } else {
            wgpu::BlendState::ALPHA_BLENDING
        };

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

        let mut fft_planner = FftPlanner::new();
        let fft = fft_planner.plan_fft_forward(FFT_SIZE);
        let fft_buffer = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
        let fft_window = (0..FFT_SIZE)
            .map(|i| {
                let position = i as f32 / (FFT_SIZE - 1) as f32;
                0.5 - 0.5 * (2.0 * PI * position).cos()
            })
            .collect();
        let audio_history = VecDeque::with_capacity(MAX_AUDIO_SAMPLES);
        let spectrum_bins = vec![0.0; INITIAL_SPECTRUM_BINS];
        let spectrum_renderer = SpectrumRenderer::new(&device, config.format, INITIAL_SPECTRUM_BINS, blend);

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
            audio_history,
            fft_buffer,
            fft,
            fft_window,
            spectrum_bins,
            spectrum_renderer,
            num_bins: INITIAL_SPECTRUM_BINS,
            gap_fraction: DEFAULT_GAP_FRACTION,
            gain: DEFAULT_GAIN,
            side_fraction: DEFAULT_SIDE_FRACTION,
            width_scale: DEFAULT_WIDTH_SCALE,
            blend,
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
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Aurion Spectrum Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.spectrum_renderer.draw(&mut pass);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        Ok(())
    }

    fn update_audio_state(&mut self) {
        let mut meter_sum = 0.0;
        let mut meter_count = 0;
        let mut popped = 0usize;

        while let Some(sample) = self.audio_queue.pop() {
            self.audio_history.push_back(sample);
            if self.audio_history.len() > MAX_AUDIO_SAMPLES {
                self.audio_history.pop_front();
            }

            if popped < METER_SAMPLES_PER_FRAME {
                meter_sum += sample * sample;
                meter_count += 1;
            }
            popped += 1;
        }

        if meter_count > 0 {
            let rms = (meter_sum / meter_count as f32).sqrt();
            self.audio_level = self.audio_level * (1.0 - AUDIO_ATTACK) + rms * AUDIO_ATTACK;
        } else {
            self.audio_level *= AUDIO_DECAY;
        }

        if self.audio_history.len() >= FFT_SIZE {
            let start_index = self.audio_history.len() - FFT_SIZE;
            for (i, sample) in self
                .audio_history
                .iter()
                .skip(start_index)
                .enumerate()
            {
                let windowed = *sample * self.fft_window[i];
                self.fft_buffer[i] = Complex32::new(windowed, 0.0);
            }

            self.fft.process(&mut self.fft_buffer);

            let half_len = FFT_SIZE / 2;
            let bins = self.num_bins.min(half_len.max(1));
            for bin in 0..bins {
                let start = (bin * half_len) / bins;
                let end = ((bin + 1) * half_len) / bins;
                let mut max_power = 0.0;
                for freq in start..end.max(start + 1) {
                    let power = self.fft_buffer[freq].norm_sqr();
                    if power > max_power {
                        max_power = power;
                    }
                }

                let amplitude = (max_power / FFT_SIZE as f32).sqrt();
                let current = self.spectrum_bins[bin];
                let updated = if amplitude > current {
                    current + (amplitude - current) * SPECTRUM_ATTACK
                } else {
                    current * SPECTRUM_DECAY + amplitude * (1.0 - SPECTRUM_DECAY)
                };
                self.spectrum_bins[bin] = updated.clamp(0.0, 1.0);
            }

            for level in self.spectrum_bins.iter_mut().skip(bins) {
                *level *= SPECTRUM_DECAY;
            }
        } else {
            for level in &mut self.spectrum_bins {
                *level *= SPECTRUM_DECAY;
            }
        }

        self.spectrum_renderer
            .update(&self.queue, &self.spectrum_bins, self.gap_fraction, self.gain, self.side_fraction, self.width_scale);
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
