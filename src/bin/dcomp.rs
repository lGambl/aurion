#![cfg(windows)]

use std::mem;

use anyhow::{Context, Result};
use windows::core::{Interface, PCSTR};
use windows::Win32::Foundation::{BOOL, HWND, RECT};
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE};
use windows::Win32::Graphics::Direct3D::Fxc::{D3DCompile, D3DCOMPILE_ENABLE_STRICTNESS};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D11::{
    D3D11CreateDevice,
    ID3D11Device,
    ID3D11DeviceContext,
    ID3D11RenderTargetView,
    ID3D11VertexShader,
    ID3D11PixelShader,
    ID3D11InputLayout,
    ID3D11Buffer,
    ID3D11Texture2D,
    D3D11_CREATE_DEVICE_BGRA_SUPPORT,
    D3D11_SDK_VERSION,
    D3D11_BUFFER_DESC,
    D3D11_INPUT_ELEMENT_DESC,
    D3D11_INPUT_PER_VERTEX_DATA,
    D3D11_USAGE_DYNAMIC,
    D3D11_CPU_ACCESS_WRITE,
    D3D11_VIEWPORT,
    D3D11_MAPPED_SUBRESOURCE,
    D3D11_MAP_WRITE_DISCARD,
};
use windows::Win32::Graphics::Dxgi::{Common::*, IDXGIDevice, IDXGIFactory2, IDXGISwapChain1, DXGI_SWAP_CHAIN_DESC1, DXGI_SCALING, DXGI_SWAP_EFFECT};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_ALPHA_MODE, DXGI_ALPHA_MODE_PREMULTIPLIED};
use windows::Win32::Graphics::DirectComposition::{DCompositionCreateDevice, IDCompositionDevice, IDCompositionTarget, IDCompositionVisual};
use windows::Win32::UI::WindowsAndMessaging::{GetClientRect, GetWindowLongPtrW, SetWindowLongPtrW, GWL_EXSTYLE, WS_EX_TRANSPARENT};

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId, WindowLevel};

#[path = "../app/audio.rs"]
mod audio;

use crossbeam::queue::ArrayQueue;
use rustfft::{FftPlanner, num_complex::Complex32};
use std::collections::VecDeque;
use std::sync::Arc;

const FFT_SIZE: usize = 2048;
const INITIAL_BINS: usize = 192;
const SPECTRUM_ATTACK: f32 = 0.25;
const SPECTRUM_DECAY: f32 = 0.85;

struct Spectrum {
    audio: audio::AudioCapture,
    queue: Arc<ArrayQueue<f32>>,
    channels: u16,
    ring: VecDeque<f32>,
    fft_buf: Vec<Complex32>,
    window: Vec<f32>,
    fft: Arc<dyn rustfft::Fft<f32> + Send + Sync>,
    bins: Vec<f32>,
}

impl Spectrum {
    fn new() -> Result<Self> {
        let audio = audio::AudioCapture::new()?;
        let queue = audio.queue();
        let channels = audio.channels();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / FFT_SIZE as f32).cos())
            .collect();
        Ok(Self {
            audio,
            queue,
            channels,
            ring: VecDeque::with_capacity(FFT_SIZE * 4),
            fft_buf: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
            window,
            fft,
            bins: vec![0.0; INITIAL_BINS],
        })
    }

    fn update(&mut self) {
        while let Some(s) = self.queue.pop() {
            self.ring.push_back(s);
            if self.ring.len() > FFT_SIZE * 4 { self.ring.pop_front(); }
        }
        if self.ring.len() < FFT_SIZE * (self.channels as usize) { return; }
        // Take last FFT_SIZE frames, downmix
        let mut mono = vec![0.0f32; FFT_SIZE];
        let len = self.ring.len();
        let start = len - FFT_SIZE * (self.channels as usize);
        // Copy from ring into a slice
        let buf: Vec<f32> = self.ring.iter().cloned().collect();
        for i in 0..FFT_SIZE {
            let mut v = 0.0f32;
            for c in 0..self.channels as usize {
                v += buf[start + i * self.channels as usize + c];
            }
            mono[i] = v / self.channels as f32;
        }
        // Window + FFT
        for i in 0..FFT_SIZE {
            self.fft_buf[i].re = mono[i] * self.window[i];
            self.fft_buf[i].im = 0.0;
        }
        self.fft.process(&mut self.fft_buf);

        // Magnitudes and smoothing
        let half = FFT_SIZE / 2;
        let mut mags = vec![0.0f32; half];
        for i in 0..half {
            let c = self.fft_buf[i];
            mags[i] = (c.re * c.re + c.im * c.im).sqrt();
        }

        let bins = self.bins.len();
        for b in 0..bins {
            let a = (b * half) / bins;
            let z = ((b + 1) * half) / bins;
            let mut sum = 0.0f32;
            let mut count = 0;
            for i in a..z.max(a + 1) {
                sum += mags[i];
                count += 1;
            }
            let v = if count > 0 { sum / count as f32 } else { 0.0 };
            let prev = self.bins[b];
            let target = v;
            let next = if target > prev {
                prev + (target - prev) * SPECTRUM_ATTACK
            } else {
                prev * SPECTRUM_DECAY
            };
            self.bins[b] = next.min(1.0);
        }
    }

    fn vertices_ndc(&self, width: u32, height: u32) -> Vec<f32> {
        // Build CPU vertices in NDC: float2 pos, float4 color per vertex (6 floats)
        let n = self.bins.len();
        let gap = 0.08f32; // fraction of each cell
        let mut out = Vec::with_capacity(n * 6 * 6);
        let cell = 2.0f32 / n as f32; // NDC width per bar
        let bw = cell * (1.0 - gap);
        for i in 0..n {
            let x0 = -1.0 + i as f32 * cell + (cell - bw) * 0.5;
            let x1 = x0 + bw;
            let h = (self.bins[i] * 1.8).min(1.0);
            let y0 = -1.0;
            let y1 = -1.0 + h * 2.0; // up towards 1
            let color = color_for(h);
            let a = 1.0f32;
            let premul = [color[0] * a, color[1] * a, color[2] * a, a];
            // two triangles
            push_vtx(&mut out, x0, y0, premul);
            push_vtx(&mut out, x1, y0, premul);
            push_vtx(&mut out, x1, y1, premul);
            push_vtx(&mut out, x0, y0, premul);
            push_vtx(&mut out, x1, y1, premul);
            push_vtx(&mut out, x0, y1, premul);
        }
        out
    }
}

fn push_vtx(out: &mut Vec<f32>, x: f32, y: f32, rgba: [f32; 4]) {
    out.push(x); out.push(y);
    out.extend_from_slice(&rgba);
}

fn color_for(level: f32) -> [f32; 3] {
    // Match existing palette idea: blue→cyan→yellow
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
fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }
struct DCompPresenter {
    d3d: ID3D11Device,
    ctx: ID3D11DeviceContext,
    dxgi_factory: IDXGIFactory2,
    dcomp: IDCompositionDevice,
    target: IDCompositionTarget,
    visual: IDCompositionVisual,
    swapchain: IDXGISwapChain1,
    format: DXGI_FORMAT,
    // simple pipeline
    vs: ID3D11VertexShader,
    ps: ID3D11PixelShader,
    layout: ID3D11InputLayout,
    vbuf: ID3D11Buffer,
    vbuf_capacity: usize,
}

impl DCompPresenter {
    fn new(hwnd: HWND, size: PhysicalSize<u32>) -> Result<Self> {
        unsafe {
            // constants not all exposed; build enum values via transmute
            fn scaling_stretch() -> DXGI_SCALING { unsafe { core::mem::transmute(0i32) } }
            fn swap_effect_flip_sequential() -> DXGI_SWAP_EFFECT { unsafe { core::mem::transmute(3i32) } }
            fn alpha_premultiplied() -> DXGI_ALPHA_MODE { unsafe { core::mem::transmute(1i32) } }
            // D3D11 device (BGRA for interop)
            let mut d3d: Option<ID3D11Device> = None;
            let mut ctx: Option<ID3D11DeviceContext> = None;
            let flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT.0 as u32;
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                None,
                windows::Win32::Graphics::Direct3D11::D3D11_CREATE_DEVICE_FLAG(flags),
                None,
                D3D11_SDK_VERSION,
                Some(&mut d3d),
                None,
                Some(&mut ctx),
            )
            .ok()
            .context("D3D11CreateDevice failed")?;

            let d3d = d3d.unwrap();
            let ctx = ctx.unwrap();

            // DXGI factory
            let dxgi_device: IDXGIDevice = d3d.cast()?;
            let adapter = dxgi_device.GetAdapter()?;
            let factory: IDXGIFactory2 = adapter.GetParent()?;

            // DirectComposition device
            let dcomp: IDCompositionDevice = DCompositionCreateDevice(&dxgi_device)?;

            // Swapchain for composition
            let format = DXGI_FORMAT_B8G8R8A8_UNORM;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: size.width,
                Height: size.height,
                Format: format,
                Stereo: BOOL::from(false),
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                BufferUsage: windows::Win32::Graphics::Dxgi::DXGI_USAGE(32), // DXGI_USAGE_RENDER_TARGET_OUTPUT
                BufferCount: 2,
                Scaling: scaling_stretch(),
                SwapEffect: swap_effect_flip_sequential(),
                AlphaMode: alpha_premultiplied(),
                Flags: 0,
            };

            let swapchain = factory.CreateSwapChainForComposition(&d3d, &desc, None)?;

            // Composition target + visual
            let target = dcomp.CreateTargetForHwnd(hwnd, true)?;
            let visual = dcomp.CreateVisual()?;
            visual.SetContent(&swapchain)?;
            target.SetRoot(&visual)?;
            dcomp.Commit()?;

            // Compile tiny shaders
            let hlsl_vs = b"
struct VSIn { float2 pos : POSITION; float4 color : COLOR0; };
struct VSOut { float4 pos : SV_Position; float4 color : COLOR0; };
VSOut main(VSIn i) { VSOut o; o.pos = float4(i.pos, 0, 1); o.color = i.color; return o; }
";
            let hlsl_ps = b"
struct PSIn { float4 pos : SV_Position; float4 color : COLOR0; };
float4 main(PSIn i) : SV_Target { return i.color; }
";
            let mut vs_blob: Option<ID3DBlob> = None;
            let mut ps_blob: Option<ID3DBlob> = None;
            let mut err_blob: Option<ID3DBlob> = None;
            D3DCompile(
                hlsl_vs.as_ptr() as _,
                hlsl_vs.len(),
                None,
                None,
                None,
                PCSTR(b"main\0".as_ptr()),
                PCSTR(b"vs_5_0\0".as_ptr()),
                D3DCOMPILE_ENABLE_STRICTNESS,
                0,
                &mut vs_blob,
                Some(&mut err_blob),
            )
            .ok()
            .context("D3DCompile VS failed")?;
            err_blob = None;
            D3DCompile(
                hlsl_ps.as_ptr() as _,
                hlsl_ps.len(),
                None,
                None,
                None,
                PCSTR(b"main\0".as_ptr()),
                PCSTR(b"ps_5_0\0".as_ptr()),
                D3DCOMPILE_ENABLE_STRICTNESS,
                0,
                &mut ps_blob,
                Some(&mut err_blob),
            )
            .ok()
            .context("D3DCompile PS failed")?;
            let vsb = vs_blob.as_ref().unwrap();
            let psb = ps_blob.as_ref().unwrap();

            let vs_code = std::slice::from_raw_parts(vsb.GetBufferPointer() as *const u8, vsb.GetBufferSize());
            let ps_code = std::slice::from_raw_parts(psb.GetBufferPointer() as *const u8, psb.GetBufferSize());
            let mut vs_opt: Option<ID3D11VertexShader> = None;
            d3d.CreateVertexShader(vs_code, None, Some(&mut vs_opt))?;
            let vs = vs_opt.unwrap();
            let mut ps_opt: Option<ID3D11PixelShader> = None;
            d3d.CreatePixelShader(ps_code, None, Some(&mut ps_opt))?;
            let ps = ps_opt.unwrap();

            let il_desc = [
                D3D11_INPUT_ELEMENT_DESC {
                    SemanticName: PCSTR(b"POSITION\0".as_ptr()),
                    SemanticIndex: 0,
                    Format: DXGI_FORMAT_R32G32_FLOAT,
                    InputSlot: 0,
                    AlignedByteOffset: 0,
                    InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
                    InstanceDataStepRate: 0,
                },
                D3D11_INPUT_ELEMENT_DESC {
                    SemanticName: PCSTR(b"COLOR\0".as_ptr()),
                    SemanticIndex: 0,
                    Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                    InputSlot: 0,
                    AlignedByteOffset: 8,
                    InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
                    InstanceDataStepRate: 0,
                },
            ];
            let vs_code2 = std::slice::from_raw_parts(vsb.GetBufferPointer() as *const u8, vsb.GetBufferSize());
            let mut layout_opt: Option<ID3D11InputLayout> = None;
            d3d.CreateInputLayout(&il_desc, vs_code2, Some(&mut layout_opt))?;
            let layout = layout_opt.unwrap();

            // dynamic vertex buffer (start small, grow as needed)
            let vbuf_capacity = 4096usize; // floats count for vertices array later
            let bd = D3D11_BUFFER_DESC {
                ByteWidth: (vbuf_capacity * mem::size_of::<f32>()) as u32,
                Usage: D3D11_USAGE_DYNAMIC,
                BindFlags: windows::Win32::Graphics::Direct3D11::D3D11_BIND_VERTEX_BUFFER.0 as u32,
                CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
                MiscFlags: 0,
                StructureByteStride: 0,
            };
            let mut vbuf: Option<ID3D11Buffer> = None;
            d3d.CreateBuffer(&bd, None, Some(&mut vbuf))?;
            let vbuf = vbuf.unwrap();

            Ok(Self {
                d3d,
                ctx,
                dxgi_factory: factory,
                dcomp,
                target,
                visual,
                swapchain,
                format,
                vs,
                ps,
                layout,
                vbuf,
                vbuf_capacity,
            })
        }
    }

    fn resize(&self, size: PhysicalSize<u32>) -> Result<()> {
        if size.width == 0 || size.height == 0 { return Ok(()); }
        unsafe {
            self.swapchain
                .ResizeBuffers(0, size.width, size.height, self.format, 0)
                .ok()
                .context("ResizeBuffers failed")?;
            self.dcomp.Commit()?;
        }
        Ok(())
    }

    fn render_clear(&self, color: [f32; 4]) -> Result<()> {
        unsafe {
            // Acquire back buffer and clear it. For premultiplied alpha, RGB must be pre-multiplied by A.
            let tex2d: ID3D11Texture2D = self.swapchain.GetBuffer(0)?;
            let mut rtv_opt: Option<ID3D11RenderTargetView> = None;
            self.d3d.CreateRenderTargetView(&tex2d, None, Some(&mut rtv_opt))?;
            let rtv = rtv_opt.expect("RTV");

            // Clear with premultiplied color
            let premul = [color[0] * color[3], color[1] * color[3], color[2] * color[3], color[3]];
            self.ctx.OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);
            self.ctx.ClearRenderTargetView(&rtv, &premul);

            self.swapchain.Present(0, 0).ok()?;
            self.dcomp.Commit()?;
        }
        Ok(())
    }

    fn render_bars(&mut self, verts: &[f32], width: u32, height: u32) -> Result<()> {
        unsafe {
            let tex2d: ID3D11Texture2D = self.swapchain.GetBuffer(0)?;
            let mut rtv_opt: Option<ID3D11RenderTargetView> = None;
            self.d3d.CreateRenderTargetView(&tex2d, None, Some(&mut rtv_opt))?;
            let rtv = rtv_opt.unwrap();

            // Clear transparent
            self.ctx.OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);
            self.ctx.ClearRenderTargetView(&rtv, &[0.0, 0.0, 0.0, 0.0]);

            // Resize vbuf if needed
            if verts.len() > self.vbuf_capacity {
                let new_cap = verts.len().next_power_of_two();
                let bd = D3D11_BUFFER_DESC {
                    ByteWidth: (new_cap * mem::size_of::<f32>()) as u32,
                    Usage: D3D11_USAGE_DYNAMIC,
                    BindFlags: windows::Win32::Graphics::Direct3D11::D3D11_BIND_VERTEX_BUFFER.0 as u32,
                    CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
                    MiscFlags: 0,
                    StructureByteStride: 0,
                };
                let mut new_buf: Option<ID3D11Buffer> = None;
                self.d3d.CreateBuffer(&bd, None, Some(&mut new_buf))?;
                self.vbuf = new_buf.unwrap();
                self.vbuf_capacity = new_cap;
            }

            // Map and upload
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            self.ctx.Map(&self.vbuf, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut mapped))?;
            std::ptr::copy_nonoverlapping(
                verts.as_ptr() as *const u8,
                mapped.pData as *mut u8,
                verts.len() * mem::size_of::<f32>(),
            );
            self.ctx.Unmap(&self.vbuf, 0);

            // Pipeline state
            self.ctx.IASetInputLayout(&self.layout);
            let stride = (6 * mem::size_of::<f32>()) as u32; // float2 pos + float4 color
            let offset = 0u32;
            let bufs = [Some(self.vbuf.clone())];
            let strides = [stride];
            let offsets = [offset];
            self.ctx.IASetVertexBuffers(0, 1, Some(bufs.as_ptr()), Some(strides.as_ptr()), Some(offsets.as_ptr()));
            self.ctx.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            self.ctx.VSSetShader(&self.vs, None);
            self.ctx.PSSetShader(&self.ps, None);

            // Viewport
            let vp = D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            self.ctx.RSSetViewports(Some(&[vp]));

            let vertex_count = (verts.len() / 6) as u32;
            self.ctx.Draw(vertex_count, 0);

            self.swapchain.Present(0, 0).ok()?;
            self.dcomp.Commit()?;
        }
        Ok(())
    }
}

struct App {
    window: Option<Window>,
    window_id: Option<WindowId>,
    presenter: Option<DCompPresenter>,
    spectrum: Option<Spectrum>,
}

impl Default for App { fn default() -> Self { Self { window: None, window_id: None, presenter: None, spectrum: None } } }

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        let attrs: WindowAttributes = Window::default_attributes()
            .with_title("Aurion DComp Overlay")
            .with_inner_size(PhysicalSize::new(1280, 720))
            .with_decorations(false)
            .with_window_level(WindowLevel::AlwaysOnTop);

        let window = event_loop.create_window(attrs).expect("create window");
        use raw_window_handle::HasWindowHandle;
        let raw = window.window_handle().expect("handle").as_raw();
        let hwnd = match raw {
            raw_window_handle::RawWindowHandle::Win32(h) => HWND(h.hwnd.get() as isize),
            _ => panic!("not a Win32 window"),
        };

        // Optional click-through: WS_EX_TRANSPARENT + disable hit-test
        unsafe {
            let ex = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
            SetWindowLongPtrW(hwnd, GWL_EXSTYLE, ex | (WS_EX_TRANSPARENT.0 as isize));
        }
        let _ = window.set_cursor_hittest(false);

        // Initial size from client rect
        let size = {
            unsafe {
                let mut rc = RECT::default();
                GetClientRect(hwnd, &mut rc).ok().unwrap();
                PhysicalSize::new((rc.right - rc.left) as u32, (rc.bottom - rc.top) as u32)
            }
        };

        let presenter = DCompPresenter::new(hwnd, size).expect("init dcomp");
        let spectrum = Spectrum::new().expect("audio+fft init");

        self.window_id = Some(window.id());
        self.presenter = Some(presenter);
        self.spectrum = Some(spectrum);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        if Some(window_id) != self.window_id { return; }
        let (Some(window), Some(presenter)) = (self.window.as_ref(), self.presenter.as_ref()) else { return; };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                let _ = presenter.resize(size);
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                if let Some(spec) = self.spectrum.as_mut() {
                    spec.update();
                    let size = window.inner_size();
                    let verts = spec.vertices_ndc(size.width, size.height);
                    let _ = self.presenter.as_mut().unwrap().render_bars(&verts, size.width, size.height);
                }
                window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}
