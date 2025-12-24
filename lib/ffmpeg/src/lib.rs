use anyhow::{Context, Result, anyhow, bail};
use ffmpeg_sidecar::{
    command::FfmpegCommand,
    event::{
        AudioStream, FfmpegDuration, FfmpegEvent, FfmpegProgress, OutputVideoFrame, Stream,
        StreamTypeSpecificData::{Audio, Video},
        VideoStream,
    },
    ffprobe,
};
use image::RgbImage;
use log::{info, warn};
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use spin_sleep::SpinSleeper;
use std::fmt;
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Debug, Clone)]
pub struct SubtitleConfig {
    pub path: PathBuf,
    pub font_name: String,
    pub font_size: u32,
    pub is_white_font_color: bool,
    pub enable_background: bool,
    pub is_embedded: bool,
    pub margin_v: Option<u32>,
}

impl SubtitleConfig {
    pub fn new(path: impl AsRef<Path>) -> SubtitleConfig {
        SubtitleConfig {
            path: PathBuf::from(path.as_ref()),
            font_name: "Source Han Sans SC Medium".to_string(),
            font_size: 20,
            is_white_font_color: true,
            enable_background: false,
            is_embedded: true,
            margin_v: None,
        }
    }

    pub fn with_font_name(mut self, font_name: &str) -> Self {
        self.font_name = font_name.to_string();
        self
    }

    pub fn with_font_size(mut self, font_size: u32) -> Self {
        self.font_size = font_size;
        self
    }

    pub fn with_is_embedded(mut self, is_embedded: bool) -> Self {
        self.is_embedded = is_embedded;
        self
    }

    pub fn with_margin_v(mut self, margin: u32) -> Self {
        self.margin_v = Some(margin);
        self
    }

    pub fn with_is_white_font_color(mut self, is_white_font_color: bool) -> Self {
        self.is_white_font_color = is_white_font_color;
        self
    }

    pub fn with_enable_background(mut self, enable: bool) -> Self {
        self.enable_background = enable;
        self
    }
}

#[derive(Debug, Default, Clone)]
pub struct AudioMetadata {
    pub format: String,
    pub sample_rate: u32,
    pub channels: String,
    pub duration: f64,
}

#[derive(Debug, Default, Clone)]
pub struct VideoMetadata {
    pub format: String,
    pub pix_fmt: String,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub duration: f64, // second
    pub auido_metadata: AudioMetadata,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum MediaType {
    Video,
    Audio,
    Unknown,
}

impl Default for MediaType {
    fn default() -> Self {
        MediaType::Video
    }
}

impl Serialize for MediaType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            MediaType::Video => serializer.serialize_str("Video"),
            MediaType::Audio => serializer.serialize_str("Audio"),
            MediaType::Unknown => serializer.serialize_str("Unknown"),
        }
    }
}

impl<'de> Deserialize<'de> for MediaType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct MediaTypeVisitor;

        impl<'de> Visitor<'de> for MediaTypeVisitor {
            type Value = MediaType;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter
                    .write_str("a string representing MediaType ('Video', 'Audio' or 'Unknown')")
            }

            fn visit_str<E>(self, value: &str) -> Result<MediaType, E>
            where
                E: de::Error,
            {
                match value {
                    "Video" => Ok(MediaType::Video),
                    "Audio" => Ok(MediaType::Audio),
                    "Unknown" => Ok(MediaType::Unknown),
                    _ => Err(E::custom(format!("unknown MediaType variant: {}", value))),
                }
            }
        }

        deserializer.deserialize_str(MediaTypeVisitor)
    }
}

#[derive(Debug, Clone)]
pub enum VideoResolution {
    Origin,
    P480,
    P720,
    P1080,
    P2K,
    P4K,
    P8K,
}

impl Default for VideoResolution {
    fn default() -> Self {
        Self::Origin
    }
}

#[derive(Clone, Debug)]
pub enum VideoExitStatus {
    Finished,
    Stop,
}

#[derive(Default, Debug, Clone)]
pub struct VideoFramesIterConfig {
    pub offset_ms: Option<u64>,
    pub duration_ms: Option<u64>,
    pub fps: Option<f32>,
    pub resolution: VideoResolution,
}

impl VideoFramesIterConfig {
    pub fn with_offset_ms(mut self, ms: u64) -> Self {
        self.offset_ms = Some(ms);
        self
    }

    pub fn with_duration_ms(mut self, ms: u64) -> Self {
        self.duration_ms = Some(ms);
        self
    }

    pub fn with_fps(mut self, fps: f32) -> Self {
        self.fps = Some(fps);
        self
    }

    pub fn with_resolution(mut self, resolution: VideoResolution) -> Self {
        self.resolution = resolution;
        self
    }
}

pub fn is_installed() -> bool {
    ffmpeg_sidecar::command::ffmpeg_is_installed()
}

pub fn auto_download() -> Result<()> {
    ffmpeg_sidecar::download::auto_download().context("Dowonlad ffmpeg failed")?;

    Ok(())
}

pub fn media_type(path: impl AsRef<Path>) -> Result<MediaType> {
    #[derive(Serialize, Deserialize)]
    struct FfprobeStreamsOutput {
        codec_type: String,
    }

    #[derive(Serialize, Deserialize)]
    struct FfprobeOutput {
        streams: Vec<FfprobeStreamsOutput>,
    }

    if !ffprobe::ffprobe_is_installed() {
        bail!("ffprobe is not install");
    }

    let mut ty = MediaType::Unknown;
    let path = path.as_ref().to_string_lossy();

    let output = duct::cmd!(
        ffprobe::ffprobe_path().to_string_lossy().to_string(),
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        path.to_string(),
    )
    .read()?
    .to_string();

    let output = serde_json::from_str::<FfprobeOutput>(&output)
        .with_context(|| format!("parse {output} failed"))?;

    for stream in output.streams.into_iter() {
        match stream.codec_type.as_str() {
            "video" => ty = MediaType::Video,
            "audio" if ty == MediaType::Unknown => ty = MediaType::Audio,
            _ => (),
        }
    }

    Ok(ty)
}

pub fn audio_metadata(path: impl AsRef<str>) -> Result<AudioMetadata> {
    let mut ffmpeg_runner = FfmpegCommand::new()
        .input(path.as_ref())
        .overwrite()
        .print_command()
        .spawn()
        .with_context(|| format!("Can't get {} metadata", path.as_ref()))?;

    let mut metadata = AudioMetadata::default();

    ffmpeg_runner.iter()?.for_each(|e| match e {
        FfmpegEvent::ParsedDuration(FfmpegDuration { duration, .. }) => {
            metadata.duration = duration;
        }
        FfmpegEvent::ParsedInputStream(Stream {
            format,
            type_specific_data:
                Audio(AudioStream {
                    sample_rate,
                    channels,
                }),
            ..
        }) => {
            metadata.format = format.clone();
            metadata.sample_rate = sample_rate;
            metadata.channels = channels.clone();
        }
        _ => {}
    });

    _ = ffmpeg_runner.kill();
    _ = ffmpeg_runner.wait();
    Ok(metadata)
}

pub fn video_metadata(path: impl AsRef<str>) -> Result<VideoMetadata> {
    let mut ffmpeg_runner = FfmpegCommand::new()
        .input(path.as_ref())
        .overwrite()
        .print_command()
        .spawn()
        .with_context(|| format!("Can't get {} metadata", path.as_ref()))?;

    let mut metadata = VideoMetadata::default();

    ffmpeg_runner.iter()?.for_each(|e| match e {
        FfmpegEvent::ParsedDuration(FfmpegDuration { duration, .. }) => {
            metadata.duration = duration;
        }
        FfmpegEvent::ParsedInputStream(Stream {
            format,
            type_specific_data:
                Audio(AudioStream {
                    sample_rate,
                    channels,
                }),
            ..
        }) => {
            metadata.auido_metadata.format = format.clone();
            metadata.auido_metadata.sample_rate = sample_rate;
            metadata.auido_metadata.channels = channels.clone();
        }

        FfmpegEvent::ParsedInputStream(Stream {
            format,
            type_specific_data:
                Video(VideoStream {
                    pix_fmt,
                    width,
                    height,
                    fps,
                }),
            ..
        }) => {
            metadata.format = format.clone();
            metadata.pix_fmt = pix_fmt.clone();
            metadata.width = width;
            metadata.height = height;
            metadata.fps = fps;
        }

        _ => {}
    });

    _ = ffmpeg_runner.kill();
    _ = ffmpeg_runner.wait();
    Ok(metadata)
}

fn timestamp_to_ms(timestamp: &str) -> Result<u64> {
    let parts: Vec<&str> = timestamp.split(':').collect();
    if parts.len() != 3 {
        bail!("Invalid timestamp format: {timestamp}");
    }

    let seconds_parts: Vec<&str> = parts[2].split('.').collect();
    if seconds_parts.len() != 2 {
        bail!("Invalid seconds format: {:?}", seconds_parts);
    }

    let hours: u64 = parts[0].parse().map_err(|e| anyhow!("Invalid hours {e}"))?;
    let minutes: u64 = parts[1]
        .parse()
        .map_err(|e| anyhow!("Invalid minutes {e}"))?;
    let seconds: u64 = seconds_parts[0]
        .parse()
        .map_err(|e| anyhow!("Invalid seconds {e}"))?;
    let milliseconds: u64 = seconds_parts[1]
        .parse()
        .map_err(|e| anyhow!("Invalid milliseconds {e}"))?;

    let total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds;

    Ok(total_ms)
}

pub fn convert_to_whisper_compatible_audio(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    cancel: Arc<AtomicBool>,
    progress_cb: impl FnMut(i32) + 'static,
) -> Result<()> {
    convert_to_audio(input, output, true, cancel, progress_cb)
}

pub fn convert_to_audio(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    is_mono: bool,
    cancel: Arc<AtomicBool>,
    mut progress_cb: impl FnMut(i32) + 'static,
) -> Result<()> {
    let mut audio_duration = None;
    let input = input.as_ref().display().to_string();

    let arg_string = if is_mono {
        "-filter:a aformat=sample_fmts=s16:channel_layouts=mono:sample_rates=16000"
    } else {
        "-filter:a aformat=sample_fmts=s16:sample_rates=16000"
    };

    let mut process = FfmpegCommand::new()
        .input(&input)
        .args(arg_string.split(' '))
        .overwrite()
        .output(output.as_ref().display().to_string())
        .print_command()
        .spawn()
        .with_context(|| format!("ffmpeg spawn child process for converting {input} failed"))?;

    let iter = process
        .iter()
        .with_context(|| format!("ffmpeg iter for converting {input} failed"))?;

    for event in iter.into_iter() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }

        match event {
            FfmpegEvent::ParsedDuration(FfmpegDuration { duration, .. }) => {
                audio_duration = Some((duration * 1000.0) as u64);
            }
            FfmpegEvent::Progress(FfmpegProgress { time, .. }) => match timestamp_to_ms(&time) {
                Ok(ms) if ms > 0 => {
                    if let Some(duration) = audio_duration {
                        progress_cb((100 * ms / duration) as i32);
                    }
                }
                Err(e) => warn!("{e}"),
                _ => (),
            },
            _ => (),
        }
    }

    _ = process.kill();
    _ = process.wait();

    Ok(())
}

pub fn frame_to_rgb_ppm(frame: &OutputVideoFrame) -> String {
    let mut ppm = format!("P3\n{} {}\n255\n", frame.width, frame.height);

    for y in 0..frame.height {
        for x in 0..frame.width {
            let idx = (y * frame.width + x) as usize * 3;
            let r = frame.data[idx] as u32;
            let g = frame.data[idx + 1] as u32;
            let b = frame.data[idx + 2] as u32;
            ppm.push_str(&format!("{r} {g} {b}\n"));
        }
    }

    ppm
}

pub fn frame_to_image(frame: &OutputVideoFrame) -> Result<RgbImage> {
    let img =
        RgbImage::from_raw(frame.width, frame.height, frame.data.clone()).with_context(|| {
            format!(
                "Invalid image dimensions ({}x{}) or data length: {}",
                frame.width,
                frame.height,
                frame.data.len()
            )
        })?;

    Ok(img)
}

pub fn video_frames_iter(
    path: impl AsRef<Path>,
    config: VideoFramesIterConfig,
    cancel: Arc<AtomicBool>,
    mut cb: impl FnMut(RgbImage, f32, usize),
) -> Result<VideoExitStatus> {
    let VideoFramesIterConfig {
        offset_ms,
        duration_ms,
        fps,
        resolution,
    } = config;

    let path = path.as_ref().to_string_lossy();
    let interval_ms = fps.map(|v| 1000.0 / v as f64);

    let mut cmd = FfmpegCommand::new();
    if let Some(ms) = duration_ms {
        cmd.duration(format!("{}ms", ms));
    }

    if let Some(ms) = offset_ms {
        cmd.seek(format!("{}ms", ms));
    }

    let cmd = cmd.input(&path);

    if let Some(fps) = fps {
        cmd.args(&["-r", &fps.to_string()]);
    }

    match resolution {
        VideoResolution::P480 => cmd.args(&["-vf", "scale=-2:480"]),
        VideoResolution::P720 => cmd.args(&["-vf", "scale=-2:720"]),
        VideoResolution::P1080 => cmd.args(&["-vf", "scale=-2:1080"]),
        VideoResolution::P2K => cmd.args(&["-vf", "scale=-2:1440"]),
        VideoResolution::P4K => cmd.args(&["-vf", "scale=-2:2160"]),
        VideoResolution::P8K => cmd.args(&["-vf", "scale=-2:4320"]),
        _ => cmd,
    };

    let mut process = cmd
        .rawvideo()
        .overwrite()
        .print_command()
        .spawn()
        .with_context(|| format!("ffmpeg spawn child process for video frames {path} failed"))?;

    let sleeper = SpinSleeper::default();
    let start_time = std::time::Instant::now();

    let iter = process
        .iter()
        .with_context(|| format!("ffmpeg iter for video frames {path} failed"))?
        .filter_frames();

    for (index, frame) in iter.into_iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }

        match frame_to_image(&frame) {
            Ok(img) => cb(img, frame.timestamp, index),
            Err(e) => {
                warn!("{e}");
                continue;
            }
        }

        if let Some(interval) = interval_ms {
            let target_time = start_time
                + std::time::Duration::from_millis((interval * (index as u64 + 1) as f64) as u64);
            sleeper.sleep_until(target_time);
        }
    }

    _ = process.kill();
    _ = process.wait();

    if cancel.load(Ordering::Relaxed) {
        info!("Exit ffmpeg child process after cancelling");
        return Ok(VideoExitStatus::Stop);
    }

    info!("Exit ffmpeg child process after finishing frame iteration");
    Ok(VideoExitStatus::Finished)
}

pub fn total_video_frames(path: impl AsRef<Path>) -> Result<u32> {
    if !ffprobe::ffprobe_is_installed() {
        bail!("ffprobe is not install");
    }

    let path = path.as_ref().to_string_lossy();

    let output = duct::cmd!(
        ffprobe::ffprobe_path().to_string_lossy().to_string(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        path.to_string(),
    )
    .read()?
    .to_string();

    Ok(output
        .trim()
        .parse::<u32>()
        .with_context(|| format!("parse `{}` to u32 error", output))?)
}

pub fn video_screenshots(path: impl AsRef<Path>, count: u32) -> Result<Vec<RgbImage>> {
    let mut screenshots = vec![];
    let path = path.as_ref().to_string_lossy();
    let duration = video_metadata(&path)?.duration;

    if duration == 0.0 || count == 0 {
        return Ok(screenshots);
    }

    let interval = duration / count as f64;

    for i in 0..count {
        let timestamp = i as f64 * interval;

        let mut process = FfmpegCommand::new()
            .input(&path)
            .args(&["-ss", &timestamp.to_string(), "-vframes", "1"])
            .rawvideo()
            .overwrite()
            .print_command()
            .spawn()
            .with_context(|| format!("ffmpeg spawn for screenshot at {}s failed", timestamp))?;

        let frame = process
            .iter()
            .with_context(|| format!("ffmpeg iter for screenshot at {}s failed", timestamp))?
            .filter_frames()
            .next()
            .ok_or_else(|| anyhow!("No frame found at timestamp {}", timestamp))?;

        let img = frame_to_image(&frame)?;
        screenshots.push(img);

        _ = process.kill();
        _ = process.wait();
    }

    Ok(screenshots)
}

pub fn adjust_normalized_voice(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    multiple: f32,
    cancel: Arc<AtomicBool>,
    mut progress_cb: impl FnMut(i32) + 'static,
) -> Result<()> {
    let mut audio_duration = None;
    let input_path = input_path.as_ref().to_string_lossy();

    // I=-16：目标响度（-16 LUFS是广播常用标准） LRA=11：动态范围控制 TP=-1.5：最大真实峰值（防止削波） volume=1.3 声音调成原来的1.3倍
    let mut process = FfmpegCommand::new()
        .input(&input_path)
        .args(&[
            "-af",
            &format!("loudnorm=I=-16:LRA=11:TP=-1.5,volume={multiple}"),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
        ])
        .overwrite()
        .output(output_path.as_ref().to_string_lossy())
        .print_command()
        .spawn()
        .with_context(|| format!("ffmpeg spawn for increase voice {multiple} failed"))?;

    let iter = process
        .iter()
        .with_context(|| format!("ffmpeg iter for increase voice {multiple} failed"))?;

    for event in iter.into_iter() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }

        match event {
            FfmpegEvent::ParsedDuration(FfmpegDuration { duration, .. }) => {
                audio_duration = Some((duration * 1000.0) as u64);
            }
            FfmpegEvent::Progress(FfmpegProgress { time, .. }) => match timestamp_to_ms(&time) {
                Ok(ms) if ms > 0 => {
                    if let Some(duration) = audio_duration {
                        progress_cb((100 * ms / duration) as i32);
                    }
                }
                Err(e) => warn!("{e}"),
                _ => (),
            },
            _ => (),
        }
    }

    _ = process.kill();
    _ = process.wait();

    Ok(())
}

pub fn add_subtitle<P>(
    input_path: P,
    output_path: P,
    subtitle_config: SubtitleConfig,
    cancel: Arc<AtomicBool>,
    mut progress_cb: impl FnMut(i32) + 'static,
) -> Result<()>
where
    P: AsRef<Path>,
{
    let mut audio_duration = None;
    let input_path = input_path.as_ref().to_string_lossy();
    let subtitle_path = subtitle_config.path.as_path().to_string_lossy();

    let mut command = FfmpegCommand::new();
    command.input(&input_path);

    let background = {
        let backcolour = if subtitle_config.enable_background {
            if subtitle_config.is_white_font_color {
                ",BackColour=&H00000000,BorderStyle=3"
            } else {
                ",BackColour=&H00FFFFFF,BorderStyle=3"
            }
        } else {
            ",BorderStyle=1"
        };

        if subtitle_config.is_white_font_color {
            format!(",PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000{backcolour}")
        } else {
            format!(",PrimaryColour=&H00000000,OutlineColour=&H00FFFFFF{backcolour}")
        }
    };

    if subtitle_config.is_embedded {
        #[cfg(target_os = "windows")]
        let subtitle_path = subtitle_path.replace("\\", "/").replacen(":", "\\:", 1);

        let filter = format!(
            "subtitles='{}':force_style='FontName={},FontSize={}{}{}'",
            subtitle_path,
            subtitle_config.font_name,
            subtitle_config.font_size,
            match subtitle_config.margin_v {
                Some(margin) => format!(",MarginV={margin}"),
                _ => "".to_string(),
            },
            background,
        );

        command.args(&["-vf", &filter]).args(&["-c:a", "copy"]);
    } else {
        command
            .input(&subtitle_path)
            .args(&["-c", "copy"])
            .args(&["-c:s", "mov_text"]) // 对于MP4使用mov_text编码
            .args(&["-disposition:s:0", "default"]);
    }

    let mut process = command
        .overwrite()
        .output(output_path.as_ref().to_string_lossy())
        .print_command()
        .spawn()
        .with_context(|| format!("ffmpeg spawn for add subtitle {subtitle_path} failed"))?;

    let iter = process
        .iter()
        .with_context(|| format!("ffmpeg iter for add subtitle {subtitle_path} failed"))?;

    for event in iter.into_iter() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }

        match event {
            FfmpegEvent::ParsedDuration(FfmpegDuration { duration, .. }) => {
                audio_duration = Some((duration * 1000.0) as u64);
            }
            FfmpegEvent::Progress(FfmpegProgress { time, .. }) => match timestamp_to_ms(&time) {
                Ok(ms) if ms > 0 => {
                    if let Some(duration) = audio_duration {
                        progress_cb((100 * ms / duration) as i32);
                    }
                }
                Err(e) => warn!("{e}"),
                _ => (),
            },
            _ => (),
        }
    }

    _ = process.kill();
    _ = process.wait();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // cargo test test_metadata -- --no-capture
    #[test]
    fn test_metadata() -> Result<()> {
        let audio_metadata = audio_metadata("./data/test.mp3")?;
        println!("{audio_metadata:?}");

        let video_metadata = video_metadata("./data/test.mp4")?;
        println!("{video_metadata:?}");

        Ok(())
    }

    // cargo test test_convert_to_whisper_audio -- --no-capture
    #[test]
    fn test_convert_to_whisper_audio() -> Result<()> {
        convert_to_whisper_compatible_audio(
            "./data/test.mp4",
            "./tmp/output.wav",
            Arc::new(AtomicBool::new(false)),
            |progress| println!("convert video progress: {}%", progress),
        )?;

        convert_to_whisper_compatible_audio(
            "./data/test.mp3",
            "./tmp/output.wav",
            Arc::new(AtomicBool::new(false)),
            |progress| println!("convert audio progress: {}%", progress),
        )?;

        Ok(())
    }

    // cargo test test_convert_to_audio -- --no-capture
    #[test]
    fn test_convert_to_audio() -> Result<()> {
        convert_to_audio(
            "./data/test.mp4",
            "./tmp/output.wav",
            false,
            Arc::new(AtomicBool::new(false)),
            |progress| println!("convert video progress: {}%", progress),
        )?;

        convert_to_audio(
            "./data/test.mp3",
            "./tmp/output.wav",
            false,
            Arc::new(AtomicBool::new(false)),
            |progress| println!("convert audio progress: {}%", progress),
        )?;

        Ok(())
    }

    // cargo test test_video_frames_iter -- --no-capture
    #[test]
    fn test_video_frames_iter() -> Result<()> {
        let path = "./data/test.mp4";
        let metadata = video_metadata(path)?;

        println!("{metadata:?}");

        let config = VideoFramesIterConfig::default()
            .with_offset_ms(3000)
            .with_duration_ms(1000)
            .with_resolution(VideoResolution::P720)
            .with_fps(metadata.fps);

        video_frames_iter(
            path,
            config,
            Arc::new(AtomicBool::new(false)),
            |img, timestamp, index| {
                let file = std::path::PathBuf::from(format!("./tmp/test-{index}-{timestamp}.png"));
                _ = img.save(file);
            },
        )?;
        Ok(())
    }

    // cargo test test_total_video_frames -- --no-capture
    #[test]
    fn test_total_video_frames() -> Result<()> {
        let frames_count = total_video_frames("./data/test.mp4")?;
        assert!(frames_count > 0);
        println!("frames count: {frames_count}");
        Ok(())
    }

    // cargo test test_video_screenshots -- --no-capture
    #[test]
    fn test_video_screenshots() -> Result<()> {
        let screenshots = video_screenshots("./data/test.mp4", 10)?;
        assert!(screenshots.len() > 0);
        println!("screenshots count: {}", screenshots.len());

        for (index, img) in screenshots.into_iter().enumerate() {
            let file = std::path::PathBuf::from(format!("./tmp/screenshots-{index}.png"));
            _ = img.save(file);
        }
        Ok(())
    }

    // cargo test test_media_type -- --no-capture
    #[test]
    fn test_media_type() -> Result<()> {
        let ty = media_type("./data/test.mp4")?;
        assert!(ty == MediaType::Video);

        let ty = media_type("./data/test.mp3")?;
        assert!(ty == MediaType::Audio);

        let ty = media_type("./Cargo.toml")?;
        assert!(ty == MediaType::Unknown);

        Ok(())
    }

    // cargo test test_adjust_normalized_voice -- --no-capture
    #[test]
    fn test_adjust_normalized_voice() -> Result<()> {
        adjust_normalized_voice(
            "./data/test.mp3",
            "./tmp/test_voice.mp3",
            1.,
            Arc::new(AtomicBool::new(false)),
            |progress| println!("adjust normalized voice progress: {}%", progress),
        )?;

        adjust_normalized_voice(
            "./data/test.mp4",
            "./tmp/test_voice.mp4",
            1.,
            Arc::new(AtomicBool::new(false)),
            |progress| println!("adjust normalized voice progress: {}%", progress),
        )?;

        Ok(())
    }

    // cargo test test_add_subtitle -- --no-capture
    #[test]
    fn test_add_subtitle() -> Result<()> {
        let config = SubtitleConfig::new("./data/test.srt")
            .with_font_name("Source Han Sans SC Medium")
            .with_font_size(20)
            .with_is_white_font_color(true)
            // .with_enable_background(true)
            .with_is_embedded(true);

        add_subtitle(
            "./data/test.mp4",
            "./tmp/test_embedded_subtitle.mp4",
            config,
            Arc::new(AtomicBool::new(false)),
            |progress| println!("adjust add embedded subtitle progress: {}%", progress),
        )?;

        let config = SubtitleConfig::new("./data/test.srt").with_is_embedded(false);

        add_subtitle(
            "./data/test.mp4",
            "./tmp/test_attach_subtitle.mp4",
            config,
            Arc::new(AtomicBool::new(false)),
            |progress| println!("adjust add attach subtitle progress: {}%", progress),
        )?;

        Ok(())
    }
}
