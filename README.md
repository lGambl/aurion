# Aurion

Aurion is a GPU‑accelerated, always‑on‑top audio spectrum visualizer. It captures your system audio in real time, computes an FFT, and renders smooth, color‑graded bars along the left and right edges of a transparent window. On Windows, it can optionally draw a live desktop/window capture behind the spectrum using Windows Graphics Capture.

## Features
- Real‑time audio capture via `cpal` (defaults to system output when available)
- Smooth RMS meter + FFT spectrum using `rustfft`
- WGPU renderer with transparent window and alpha blending
- Click‑through overlay and always‑on‑top toggle
- Runtime‑tunable visuals: bar count, spacing, gain, band width, side coverage
- Optional Windows Graphics Capture background (desktop/foreground window)

## Build
Prerequisites:
- Rust (edition 2024 compatible toolchain)
- GPU drivers/runtime that support Vulkan (WGPU will use the Vulkan backend here)
- Windows 10 1903+ for Windows Graphics Capture (only when using the `wgcapture` feature)

Commands:
- Cross‑platform (no background capture): `cargo run`
- Windows with desktop/window background: `cargo run --features wgcapture`

> Note: If Vulkan isn’t available on your system, install the latest GPU drivers. On Windows, most vendor drivers include Vulkan support.

## Usage & Controls
General:
- `Esc` — Exit
- `F7` — Toggle always‑on‑top
- `F8` — Toggle click‑through (let mouse clicks pass through the window)

Spectrum tuning:
- `F9` / `F10` — Increase / decrease number of bars
- `Arrow Up` / `Arrow Down` — Increase / decrease vertical gain
- `Arrow Right` / `Arrow Left` — Increase / decrease gap between bars
- `]` / `[` — Increase / decrease individual bar width within its cell
- `R` — Reset visuals to defaults
- `F11` / `F12` — Increase / decrease side coverage (how much of each edge is used)

Windows Graphics Capture (only with `--features wgcapture`):
- `F1` — Capture the foreground window
- `F2` — Cycle to the next monitor
- `F3` — Cycle to the previous monitor
- `F4` — Disable capture (no background)

## How it works
- Audio is captured from the default output device when possible (falls back to default input). Samples are windowed and transformed via FFT.
- The spectrum is rendered with WGPU using a simple WGSL shader (`src/app/shaders/spectrum.wgsl`). Bars are color‑interpolated based on level.
- On Windows (with `wgcapture`), a background pass uploads the latest BGRA frame from Windows Graphics Capture and draws it beneath the spectrum.

## Notes
- Changing the audio device is not exposed via CLI/UI; the app follows your system default. Adjust in code if needed.
- Transparency and click‑through behavior depend on platform support provided by `winit`.
- If you experience a black/blank window on startup, ensure Vulkan is available and a discrete GPU is selected.

## Project layout
- `src/main.rs` — App entry point
- `src/app/mod.rs` — App, rendering, FFT, input handling
- `src/app/audio.rs` — Audio capture (cpal)
- `src/app/wgcapture.rs` — Windows Graphics Capture (gated by the `wgcapture` feature)
- `src/app/shaders/` — WGSL shaders

