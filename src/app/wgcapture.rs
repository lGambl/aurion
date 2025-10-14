#![cfg(all(windows, feature = "wgcapture"))]

use std::mem::MaybeUninit;
use windows::core::{HSTRING, Interface};
use windows::Graphics::Capture::{Direct3D11CaptureFramePool, GraphicsCaptureItem, GraphicsCaptureSession};
use windows::Graphics::DirectX::Direct3D11::{IDirect3DDevice, IDirect3DSurface};
use windows::Graphics::DirectX::DirectXPixelFormat;
use windows::Win32::Foundation::{BOOL, HWND, LPARAM, RECT};
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_11_0};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::{Common::*, IDXGIDevice};
use windows::Win32::Graphics::Gdi::{EnumDisplayMonitors, HDC, HMONITOR};
use windows::Win32::System::WinRT::Direct3D11::{CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess};
use windows::Win32::System::WinRT::Graphics::Capture::IGraphicsCaptureItemInterop;
use windows::Win32::System::WinRT::RoGetActivationFactory;
use windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow;

#[derive(Clone, Copy, Debug)]
pub enum Source {
    None,
    ForegroundWindow,
    MonitorIndex(usize),
}

pub struct WgcCapture {
    pub source: Source,
    d3d_device: Option<ID3D11Device>,
    d3d_ctx: Option<ID3D11DeviceContext>,
    wgc_device: Option<IDirect3DDevice>,
    frame_pool: Option<Direct3D11CaptureFramePool>,
    session: Option<GraphicsCaptureSession>,
    item: Option<GraphicsCaptureItem>,
    size: (u32, u32),
    buffer: Vec<u8>,
    monitors: Vec<HMONITOR>,
    monitor_index: usize,
}

impl WgcCapture {
    pub fn new() -> Self {
        Self {
            source: Source::None,
            d3d_device: None,
            d3d_ctx: None,
            wgc_device: None,
            frame_pool: None,
            session: None,
            item: None,
            size: (0, 0),
            buffer: Vec::new(),
            monitors: enumerate_monitors(),
            monitor_index: 0,
        }
    }

    pub fn set_source(&mut self, source: Source) {
        self.source = source;
        self.restart_capture();
    }

    pub fn next_monitor(&mut self) {
        if self.monitors.is_empty() {
            self.monitors = enumerate_monitors();
        }
        if !self.monitors.is_empty() {
            self.monitor_index = (self.monitor_index + 1) % self.monitors.len();
            self.source = Source::MonitorIndex(self.monitor_index);
            self.restart_capture();
        }
    }

    pub fn prev_monitor(&mut self) {
        if self.monitors.is_empty() {
            self.monitors = enumerate_monitors();
        }
        if !self.monitors.is_empty() {
            if self.monitor_index == 0 { self.monitor_index = self.monitors.len() - 1; } else { self.monitor_index -= 1; }
            self.source = Source::MonitorIndex(self.monitor_index);
            self.restart_capture();
        }
    }

    // Returns (width, height, BGRA bytes). If no new frame, returns the last one.
    pub fn frame_bgra<'a>(&'a mut self) -> Option<(u32, u32, &'a [u8])> {
        let frame_pool = self.frame_pool.as_ref()?;
        // Try to get a new frame; if none, fall back to last buffer
        let frame = match frame_pool.TryGetNextFrame() {
            Ok(f) => Some(f),
            Err(_) => None,
        };

        if frame.is_none() {
            if self.size != (0, 0) && !self.buffer.is_empty() {
                return Some((self.size.0, self.size.1, &self.buffer));
            } else {
                return None;
            }
        }
        let frame = frame.unwrap();

        // Resize if content size changed
        if let Ok(content_size) = frame.ContentSize() {
            let w = content_size.Width as u32;
            let h = content_size.Height as u32;
            if w != 0 && h != 0 && (w, h) != self.size {
                self.resize_pool(w, h);
            }
        }

        let surface: IDirect3DSurface = match frame.Surface() { Ok(s) => s, Err(_) => return None };
        let tex: ID3D11Texture2D = unsafe {
            let access: IDirect3DDxgiInterfaceAccess = surface.cast().ok()?;
            access.GetInterface().ok()?
        };

        let device = self.d3d_device.as_ref()?;
        let ctx = self.d3d_ctx.as_ref()?;

        unsafe {
            let mut src_desc = MaybeUninit::<D3D11_TEXTURE2D_DESC>::zeroed().assume_init();
            tex.GetDesc(&mut src_desc);

            let w = src_desc.Width;
            let h = src_desc.Height;
            self.size = (w, h);

            // Create staging texture if needed
            let staging_desc = D3D11_TEXTURE2D_DESC {
                Width: w,
                Height: h,
                MipLevels: 1,
                ArraySize: 1,
                Format: src_desc.Format,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Usage: D3D11_USAGE_STAGING,
                BindFlags: 0,
                CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                MiscFlags: 0,
            };
            let mut staging: Option<ID3D11Texture2D> = None;
            if device.CreateTexture2D(&staging_desc, None, Some(&mut staging)).is_err() {
                return None;
            }
            let staging = staging.unwrap();

            // Copy GPU texture to CPU-readable staging
            ctx.CopyResource(&staging, &tex);

            // Map and copy into our buffer
            let mut mapped = MaybeUninit::<D3D11_MAPPED_SUBRESOURCE>::zeroed().assume_init();
            if ctx.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped)).is_err() {
                return None;
            }

            let src_row_pitch = mapped.RowPitch as usize;
            let dst_row_pitch = (w as usize) * 4;
            let total = dst_row_pitch * (h as usize);
            self.buffer.resize(total, 0);
            let src_ptr = mapped.pData as *const u8;
            for row in 0..(h as usize) {
                let src = std::slice::from_raw_parts(src_ptr.add(row * src_row_pitch), dst_row_pitch);
                let dst = &mut self.buffer[row * dst_row_pitch .. (row+1) * dst_row_pitch];
                dst.copy_from_slice(src);
            }
            ctx.Unmap(&staging, 0);
        }

        Some((self.size.0, self.size.1, &self.buffer))
    }

    fn restart_capture(&mut self) {
        self.stop_capture();
        if let Err(_) = self.start_capture_for_source() {
            self.stop_capture();
        }
    }

    fn stop_capture(&mut self) {
        if let Some(session) = self.session.take() { let _ = session.Close(); }
        if let Some(pool) = self.frame_pool.take() { let _ = pool.Close(); }
        self.item = None;
    }

    fn start_capture_for_source(&mut self) -> windows::core::Result<()> {
        let (device, ctx, wgc_dev) = create_d3d_devices()?;
        self.d3d_device = Some(device.clone());
        self.d3d_ctx = Some(ctx.clone());
        self.wgc_device = Some(wgc_dev.clone());

        let item = match self.source {
            Source::ForegroundWindow => {
                unsafe {
                    let hwnd: HWND = GetForegroundWindow();
                    if hwnd.0 == 0 { return Ok(()); }
                    let class_name = HSTRING::from("Windows.Graphics.Capture.GraphicsCaptureItem");
                    let interop: IGraphicsCaptureItemInterop = RoGetActivationFactory(&class_name)?;
                    let item: GraphicsCaptureItem = interop.CreateForWindow(hwnd)?;
                    item
                }
            }
            Source::MonitorIndex(i) => {
                if self.monitors.is_empty() {
                    self.monitors = enumerate_monitors();
                }
                let idx = i.min(self.monitors.len().saturating_sub(1));
                let hmon = self.monitors.get(idx).copied().unwrap_or(HMONITOR(0));
                if hmon.0 == 0 { return Ok(()); }
                unsafe {
                    let class_name = HSTRING::from("Windows.Graphics.Capture.GraphicsCaptureItem");
                    let interop: IGraphicsCaptureItemInterop = RoGetActivationFactory(&class_name)?;
                    let item: GraphicsCaptureItem = interop.CreateForMonitor(hmon)?;
                    item
                }
            }
            Source::None => return Ok(()),
        };

        let size = item.Size()?;
        let w = size.Width as u32;
        let h = size.Height as u32;
        self.size = (w, h);
        let frame_pool = Direct3D11CaptureFramePool::Create(
            &wgc_dev,
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            3,
            size,
        )?;
        let session = frame_pool.CreateCaptureSession(&item)?;
        let _ = session.SetIsCursorCaptureEnabled(true);
        session.StartCapture()?;

        self.item = Some(item);
        self.frame_pool = Some(frame_pool);
        self.session = Some(session);
        Ok(())
    }

    fn resize_pool(&mut self, w: u32, h: u32) {
        if let (Some(dev), Some(_item), Some(pool)) = (&self.wgc_device, &self.item, &self.frame_pool) {
            let size = windows::Graphics::SizeInt32 { Width: w as i32, Height: h as i32 };
            let _ = pool.Recreate(dev, DirectXPixelFormat::B8G8R8A8UIntNormalized, 3, size);
            self.size = (w, h);
        }
    }
}

fn enumerate_monitors() -> Vec<HMONITOR> {
    unsafe extern "system" fn enum_proc(hmon: HMONITOR, _hdc: HDC, _rc: *mut RECT, data: LPARAM) -> BOOL {
        let vec = &mut *(data.0 as *mut Vec<HMONITOR>);
        vec.push(hmon);
        BOOL(1)
    }
    let mut monitors: Vec<HMONITOR> = Vec::new();
    unsafe {
        let data = LPARAM(&mut monitors as *mut _ as isize);
        let _ = EnumDisplayMonitors(HDC(0), None, Some(enum_proc), data);
    }
    monitors
}

fn create_d3d_devices() -> windows::core::Result<(ID3D11Device, ID3D11DeviceContext, IDirect3DDevice)> {
    unsafe {
        let mut device: Option<ID3D11Device> = None;
        let mut ctx: Option<ID3D11DeviceContext> = None;
        let mut fl: D3D_FEATURE_LEVEL = D3D_FEATURE_LEVEL_11_0;
        let flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            None,
            flags,
            Some(&[D3D_FEATURE_LEVEL_11_0]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            Some(&mut fl),
            Some(&mut ctx),
        )?;
        let device = device.unwrap();
        let ctx = ctx.unwrap();

        let dxgi: IDXGIDevice = device.cast()?;
        let inspectable = CreateDirect3D11DeviceFromDXGIDevice(&dxgi)?;
        let wgc_dev: IDirect3DDevice = inspectable.cast()?;
        Ok((device, ctx, wgc_dev))
    }
}
