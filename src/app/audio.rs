use std::sync::Arc;

use anyhow::{Context, Result};
use cpal::{
    SampleFormat, SampleRate, Stream, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use crossbeam::queue::ArrayQueue;

pub const TARGET_SAMPLE_RATE: u32 = 48_000;
pub const AUDIO_RING_CAPACITY: usize = 192_000; // ~2 seconds of stereo sample data at 48 kHz

pub struct AudioCapture {
    queue: Arc<ArrayQueue<f32>>,
    channels: u16,
    sample_rate: u32,
    _stream: Stream,
}

impl AudioCapture {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device =
            pick_loopback_device(&host).context("no suitable audio capture device found")?;
        let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());

        let (stream_config, sample_format) = select_stream_config(&device)?;
        let sample_rate = stream_config.sample_rate.0;
        let channels = stream_config.channels;

        let estimated_capacity = (sample_rate as usize * channels as usize).max(1024);
        let capacity = AUDIO_RING_CAPACITY.max(estimated_capacity);
        let queue = Arc::new(ArrayQueue::new(capacity));

        let err_fn = |err: cpal::StreamError| {
            eprintln!("audio stream error: {err}");
        };

        let stream = build_input_stream(
            &device,
            &stream_config,
            sample_format,
            Arc::clone(&queue),
            err_fn,
        )?;
        stream.play().context("failed to start audio stream")?;

        println!(
            "Audio capture started on '{device_name}' ({channels} ch @ {} Hz, {:?})",
            sample_rate, sample_format
        );

        Ok(Self {
            queue,
            channels,
            sample_rate,
            _stream: stream,
        })
    }

    pub fn queue(&self) -> Arc<ArrayQueue<f32>> {
        Arc::clone(&self.queue)
    }

    #[allow(dead_code)]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[allow(dead_code)]
    pub fn channels(&self) -> u16 {
        self.channels
    }
}

fn pick_loopback_device(host: &cpal::Host) -> Option<cpal::Device> {
    host.default_output_device()
        .or_else(|| host.default_input_device())
}

fn select_stream_config(device: &cpal::Device) -> Result<(StreamConfig, SampleFormat)> {
    let supported_configs = device
        .supported_input_configs()
        .map(|configs| configs.collect::<Vec<_>>())
        .unwrap_or_default();

    if !supported_configs.is_empty() {
        for range in &supported_configs {
            if range.sample_format() == SampleFormat::F32
                && range.min_sample_rate().0 <= TARGET_SAMPLE_RATE
                && range.max_sample_rate().0 >= TARGET_SAMPLE_RATE
            {
                let cfg = range
                    .clone()
                    .with_sample_rate(SampleRate(TARGET_SAMPLE_RATE));
                return Ok((cfg.config(), SampleFormat::F32));
            }
        }

        if let Some(range) = supported_configs
            .iter()
            .find(|range| range.sample_format() == SampleFormat::F32)
        {
            let cfg = range.clone().with_max_sample_rate();
            return Ok((cfg.config(), SampleFormat::F32));
        }
    }

    let default_config = device
        .default_input_config()
        .or_else(|_| device.default_output_config())
        .context("device does not advertise a default input format")?;

    let sample_format = default_config.sample_format();
    Ok((default_config.config(), sample_format))
}

fn build_input_stream<E>(
    device: &cpal::Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    queue: Arc<ArrayQueue<f32>>,
    err_fn: E,
) -> Result<Stream>
where
    E: FnMut(cpal::StreamError) + Send + 'static,
{
    match sample_format {
        SampleFormat::F32 => {
            let queue = queue;
            let stream = device.build_input_stream(
                config,
                move |data: &[f32], _| push_samples(&queue, data.iter().copied()),
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        SampleFormat::I16 => {
            let queue = queue;
            let stream = device.build_input_stream(
                config,
                move |data: &[i16], _| {
                    let scale = i16::MAX as f32;
                    push_samples(
                        &queue,
                        data.iter().map(move |&sample| sample as f32 / scale),
                    );
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        SampleFormat::U16 => {
            let queue = queue;
            let stream = device.build_input_stream(
                config,
                move |data: &[u16], _| {
                    let scale = u16::MAX as f32;
                    push_samples(
                        &queue,
                        data.iter()
                            .map(move |&sample| (sample as f32 / scale) * 2.0 - 1.0),
                    );
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        other => Err(anyhow::anyhow!(
            "unsupported input sample format: {other:?}"
        )),
    }
}

fn push_samples<I>(queue: &ArrayQueue<f32>, samples: I)
where
    I: IntoIterator<Item = f32>,
{
    for sample in samples {
        if queue.push(sample).is_err() {
            let _ = queue.pop();
            let _ = queue.push(sample);
        }
    }
}
