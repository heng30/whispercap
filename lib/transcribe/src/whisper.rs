use super::vad::EnergyVAD;
use super::wav::{self, AudioData};
use anyhow::{Context, Result, anyhow, bail};
use log::debug;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
};
use whisper_rs::{
    FullParams, SamplingStrategy, SegmentCallbackData, WhisperContext, WhisperContextParameters,
    WhisperState, WhisperVadParams,
};

const GGML_SILERO_VAD_MODEL: &'static [u8] = include_bytes!("../data/ggml-silero-v5.1.2.bin");

#[derive(Clone, Debug)]
pub struct WhisperConfig {
    pub model_path: PathBuf,
    pub vad_model_path: Option<PathBuf>,
    pub language: Option<String>, // "zh", "en"，None is auto detect
    pub translate: bool,
    pub n_threads: i32,
    pub temperature: f32,
    pub max_segment_length: Option<u32>,
    pub initial_prompt: Option<String>,
    pub debug_mode: bool,

    // Chunking configuration for long audio files to avoid timestamp drift
    pub chunk_length_ms: Option<u64>, // Length of each chunk in milliseconds, default 60000 (60s)
    pub chunk_overlap_ms: Option<u64>, // Overlap between chunks in milliseconds, default 1000 (1s)
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/ggml-base.bin"),
            vad_model_path: None,
            language: None,
            translate: false,
            n_threads: num_cpus::get().min(8) as i32,
            temperature: 0.0,
            max_segment_length: None,
            initial_prompt: None,
            debug_mode: false,
            chunk_length_ms: None,
            chunk_overlap_ms: None,
        }
    }
}

impl WhisperConfig {
    pub fn new<P: Into<PathBuf>>(model_path: P) -> Self {
        Self {
            model_path: model_path.into(),
            ..Default::default()
        }
    }

    pub fn with_vad_model_path<S: Into<PathBuf>>(mut self, path: S) -> Self {
        self.vad_model_path = Some(path.into());
        self
    }

    pub fn with_language<S: Into<String>>(mut self, language: S) -> Self {
        self.language = Some(language.into());
        self
    }

    pub fn with_translate(mut self, translate: bool) -> Self {
        self.translate = translate;
        self
    }

    pub fn with_threads(mut self, n_threads: i32) -> Self {
        self.n_threads = n_threads;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.clamp(0.0, 1.0);
        self
    }

    pub fn with_initial_prompt<S: Into<String>>(mut self, prompt: S) -> Self {
        self.initial_prompt = Some(prompt.into());
        self
    }

    pub fn with_debug_mode(mut self, debug_mode: bool) -> Self {
        self.debug_mode = debug_mode;
        self
    }

    pub fn with_chunk_length_ms(mut self, length_ms: u64) -> Self {
        self.chunk_length_ms = Some(length_ms);
        self
    }

    pub fn with_chunk_overlap_ms(mut self, overlap_ms: u64) -> Self {
        self.chunk_overlap_ms = Some(overlap_ms);
        self
    }

    pub fn should_use_chunking(&self) -> bool {
        self.chunk_length_ms.is_some() && self.chunk_length_ms.unwrap() > 0
    }

    pub fn validate(&self) -> Result<()> {
        if !self.model_path.exists() {
            bail!("model path not exist: {}", self.model_path.display());
        }

        if self.n_threads <= 0 {
            bail!("n_threads is 0");
        }

        if !(0.0..=1.0).contains(&self.temperature) {
            bail!("temperature should between 0.0 and 1.0");
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub index: i32,
    pub start_time: u64, // ms
    pub end_time: u64,   // ms
    pub text: String,
    pub confidence: f32, // (0.0-1.0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub language: Option<String>,
    pub segments: Vec<TranscriptionSegment>,
    pub processing_time: u64, // ms
    pub audio_duration: u64,  // ms
}

#[derive(Debug, Clone)]
struct AudioChunk {
    samples: Vec<f32>,
    start_offset_ms: u64,
}

impl TranscriptionResult {
    pub fn real_time_factor(&self) -> f64 {
        if self.audio_duration == 0 {
            return 0.0;
        }
        self.processing_time as f64 / self.audio_duration as f64
    }

    pub fn average_confidence(&self) -> f32 {
        if self.segments.is_empty() {
            return 0.0;
        }

        let total: f32 = self.segments.iter().map(|s| s.confidence).sum();
        total / self.segments.len() as f32
    }

    pub fn filter_by_confidence(&self, min_confidence: f32) -> TranscriptionResult {
        let filtered_segments: Vec<_> = self
            .segments
            .iter()
            .filter(|s| s.confidence >= min_confidence)
            .cloned()
            .collect();

        let filtered_text = filtered_segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        TranscriptionResult {
            text: filtered_text,
            language: self.language.clone(),
            segments: filtered_segments,
            processing_time: self.processing_time,
            audio_duration: self.audio_duration,
        }
    }
}

pub struct WhisperTranscriber {
    context: Arc<WhisperContext>,
    config: WhisperConfig,
}

impl WhisperTranscriber {
    pub fn new(config: WhisperConfig) -> Result<Self> {
        config.validate()?;

        debug!("Load Whisper model: {}", config.model_path.display());

        let ctx_params = WhisperContextParameters::default();
        let context = WhisperContext::new_with_params(
            config.model_path.to_string_lossy().as_ref(),
            ctx_params,
        )
        .map_err(|e| anyhow!("Load Whisper model error: {e}"))?;

        Ok(Self {
            context: Arc::new(context),
            config,
        })
    }

    pub async fn transcribe_file<P: AsRef<Path>>(
        &self,
        audio_path: P,
        progress_cb: impl FnMut(i32) + 'static,
        segmemnt_cb: impl FnMut(SegmentCallbackData) + 'static,
        abort_cb: impl FnMut() -> bool + 'static,
    ) -> Result<TranscriptionResult> {
        is_valid_aduio_file(&audio_path)?;
        debug!("Start transcribe: {}", audio_path.as_ref().display());

        let audio_data = wav::read_file(&audio_path)?;

        if self.config.should_use_chunking() {
            self.transcribe_audio_data_chunked(&audio_data, progress_cb, segmemnt_cb, abort_cb)
                .await
        } else {
            self.transcribe_audio_data(&audio_data, progress_cb, segmemnt_cb, abort_cb)
                .await
        }
    }

    pub async fn transcribe_audio_data(
        &self,
        audio_data: &AudioData,
        progress_cb: impl FnMut(i32) + 'static,
        segmemnt_cb: impl FnMut(SegmentCallbackData) + 'static,
        abort_cb: impl FnMut() -> bool + 'static,
    ) -> Result<TranscriptionResult> {
        let start_time = std::time::Instant::now();

        let audio_samples = if !audio_data.is_whisper_compatible() {
            self.prepare_audio_samples(audio_data)?
        } else {
            audio_data.samples.clone()
        };

        debug!(
            "Start whisper infer，audio duration: {:.2}s",
            audio_data.duration()
        );

        let mut state = self
            .context
            .create_state()
            .map_err(|e| anyhow!("Create whisper state failed: {e}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.config.n_threads);
        params.set_translate(self.config.translate);
        params.set_debug_mode(self.config.debug_mode);
        params.set_temperature(self.config.temperature);
        params.set_language(self.config.language.as_ref().map(|x| x.as_str()));
        params.set_token_timestamps(true);

        params.set_progress_callback_safe(progress_cb);
        params.set_segment_callback_safe(segmemnt_cb);
        params.set_abort_callback_safe(abort_cb);

        if let Some(path) = &self.config.vad_model_path {
            if !path.exists() {
                bail!("No found vad model path: {}", path.display());
            }

            params.set_vad_model_path(Some(&path.to_string_lossy().to_string()));
            params.set_vad_params(WhisperVadParams::default());
            params.enable_vad(true);
        }

        if let Some(prompt) = &self.config.initial_prompt {
            params.set_initial_prompt(prompt.as_str());
        }

        state
            .full(params, &audio_samples)
            .map_err(|e| anyhow!("Whisper transcribe failed: {e}"))?;

        let result =
            self.extract_transcription_result(&state, audio_data.duration(), start_time)?;

        debug!(
            "Transcript finished，real time factor: {:.2}x",
            result.real_time_factor()
        );

        Ok(result)
    }

    async fn transcribe_audio_data_chunked(
        &self,
        audio_data: &AudioData,
        mut progress_cb: impl FnMut(i32) + 'static,
        mut segmemnt_cb: impl FnMut(SegmentCallbackData) + 'static,
        mut abort_cb: impl FnMut() -> bool + 'static,
    ) -> Result<TranscriptionResult> {
        let start_time = std::time::Instant::now();

        let audio_samples = if !audio_data.is_whisper_compatible() {
            self.prepare_audio_samples(audio_data)?
        } else {
            audio_data.samples.clone()
        };

        let audio_duration = audio_data.duration();

        debug!(
            "Start chunked whisper infer，audio duration: {:.2}s",
            audio_duration
        );

        // Create a temporary audio data for chunking
        let temp_audio_data = AudioData {
            samples: audio_samples,
            config: audio_data.config.clone(),
        };

        let chunks = self.split_audio_into_chunks(&temp_audio_data);
        let total_chunks = chunks.len();

        debug!("Transcribing in {} chunks", total_chunks);

        let mut all_segments = Vec::new();
        let mut full_text = String::new();
        let mut global_segment_index = 0i32;

        for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
            if abort_cb() {
                bail!("Transcription aborted");
            }

            debug!(
                "Processing chunk {}/{} (offset: {}ms, samples: {})",
                chunk_idx + 1,
                total_chunks,
                chunk.start_offset_ms,
                chunk.samples.len()
            );

            let chunk_result = self
                .transcribe_chunk_internal(&chunk.samples, global_segment_index)
                .await?;

            global_segment_index += chunk_result.segments.len() as i32;

            // Adjust segment timestamps with chunk offset and call callback
            for segment in chunk_result.segments {
                let adjusted_segment = TranscriptionSegment {
                    start_time: segment.start_time + chunk.start_offset_ms,
                    end_time: segment.end_time + chunk.start_offset_ms,
                    ..segment
                };

                // Convert to SegmentCallbackData for callback
                let callback_data = SegmentCallbackData {
                    text: adjusted_segment.text.clone(),
                    start_timestamp: (adjusted_segment.start_time / 10) as i64,
                    end_timestamp: (adjusted_segment.end_time / 10) as i64,
                    segment: adjusted_segment.index - 1,
                };

                segmemnt_cb(callback_data);
                all_segments.push(adjusted_segment);
            }

            if !full_text.is_empty() {
                full_text.push(' ');
            }
            full_text.push_str(&chunk_result.text);

            progress_cb(((chunk_idx + 1) * 100 / total_chunks) as i32);
        }

        let processing_time = start_time.elapsed().as_millis() as u64;
        let audio_duration_ms = (audio_duration * 1000.0) as u64;

        progress_cb(100);

        let result = TranscriptionResult {
            text: full_text,
            language: self.config.language.clone(),
            segments: all_segments,
            processing_time,
            audio_duration: audio_duration_ms,
        };

        debug!(
            "Chunked transcript finished，real time factor: {:.2}x",
            result.real_time_factor()
        );

        Ok(result)
    }

    async fn transcribe_chunk_internal(
        &self,
        samples: &[f32],
        start_segment_index: i32,
    ) -> Result<TranscriptionResult> {
        let chunk_duration = samples.len() as f64 / 16000.0;
        let start_time = std::time::Instant::now();

        let mut state = self
            .context
            .create_state()
            .map_err(|e| anyhow!("Create whisper state for chunk failed: {e}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.config.n_threads);
        params.set_translate(self.config.translate);
        params.set_debug_mode(self.config.debug_mode);
        params.set_temperature(self.config.temperature);
        params.set_language(self.config.language.as_ref().map(|x| x.as_str()));
        params.set_token_timestamps(true);

        // No callbacks for internal chunk processing
        params.set_progress_callback_safe(|_| {});
        params.set_segment_callback_safe(|_| {});

        if let Some(path) = &self.config.vad_model_path {
            if path.exists() {
                params.set_vad_model_path(Some(&path.to_string_lossy().to_string()));
                params.set_vad_params(WhisperVadParams::default());
                params.enable_vad(true);
            }
        }

        if let Some(prompt) = &self.config.initial_prompt {
            params.set_initial_prompt(prompt.as_str());
        }

        state
            .full(params, samples)
            .map_err(|e| anyhow!("Whisper transcribe chunk failed: {e}"))?;

        self.extract_transcription_result_with_offset(
            &state,
            chunk_duration,
            start_time,
            start_segment_index,
        )
    }

    fn extract_transcription_result_with_offset(
        &self,
        state: &WhisperState,
        audio_duration: f64,
        start_time: std::time::Instant,
        start_segment_index: i32,
    ) -> Result<TranscriptionResult> {
        let audio_duration_ms = (audio_duration * 1000.0) as u64;

        let num_segments = state.full_n_segments();

        let mut segments = Vec::new();
        let mut full_text = String::new();

        for i in 0..num_segments {
            let Some(segment) = state.get_segment(i) else {
                continue;
            };

            let segment_text = segment.to_str().unwrap_or("").trim().to_string();

            if segment_text.is_empty() {
                continue;
            }

            let start_time_ms = (segment.start_timestamp() as u64) * 10;
            let end_time_ms = (segment.end_timestamp() as u64) * 10;
            let confidence = self.calculate_segment_confidence(state, i)?;

            segments.push(TranscriptionSegment {
                index: start_segment_index + i as i32 + 1,
                start_time: start_time_ms,
                end_time: end_time_ms,
                text: segment_text.clone(),
                confidence,
            });

            if !full_text.is_empty() {
                full_text.push(' ');
            }
            full_text.push_str(&segment_text);
        }

        let processing_time = start_time.elapsed().as_millis() as u64;
        Ok(TranscriptionResult {
            text: full_text,
            language: self.config.language.clone(),
            segments,
            processing_time,
            audio_duration: audio_duration_ms,
        })
    }

    fn prepare_audio_samples(&self, audio_data: &AudioData) -> Result<Vec<f32>> {
        let mut samples = audio_data.samples.clone();

        if audio_data.config.channels > 1 {
            let mono_data = audio_data.to_mono();
            samples = mono_data.samples;
            debug!("Finished converting to mono channel");
        }

        if audio_data.config.sample_rate != 16000 {
            bail!(
                "Not compatible with whisper. Actual sample rate {}, expect 16kHz",
                audio_data.config.sample_rate
            );
        }

        Ok(samples)
    }

    /// Split audio data into chunks for chunked transcription to avoid timestamp drift
    /// Uses intelligent silence detection to split at natural pause points rather than cutting speech
    fn split_audio_into_chunks(&self, audio_data: &AudioData) -> Vec<AudioChunk> {
        let chunk_length_ms = self.config.chunk_length_ms.unwrap_or(60000);
        let overlap_ms = self.config.chunk_overlap_ms.unwrap_or(1000);

        let sample_rate = audio_data.config.sample_rate as f64;
        let total_samples = audio_data.samples.len();
        let total_duration_ms = (total_samples as f64 / sample_rate) * 1000.0;

        // If audio is shorter than chunk length, return single chunk
        if total_duration_ms <= chunk_length_ms as f64 {
            debug!(
                "Audio duration {:.2}s is shorter than chunk length {:.2}s, using single chunk",
                total_duration_ms / 1000.0,
                chunk_length_ms as f64 / 1000.0
            );
            return vec![AudioChunk {
                samples: audio_data.samples.clone(),
                start_offset_ms: 0,
            }];
        }

        let chunk_samples = (chunk_length_ms as f64 * sample_rate / 1000.0) as usize;
        let overlap_samples = (overlap_ms as f64 * sample_rate / 1000.0) as usize;

        let mut chunks = Vec::new();
        let mut current_start = 0;
        let mut chunk_index = 0;

        while current_start < total_samples {
            // Default chunk end at target length
            let ideal_chunk_end = std::cmp::min(current_start + chunk_samples, total_samples);

            // Try to find a better split point using silence detection
            let (chunk_end, is_split_in_slient_point) = if ideal_chunk_end < total_samples {
                self.find_silence_split_point(
                    &audio_data.samples,
                    ideal_chunk_end,
                    sample_rate,
                    30_000, // Search up to 30 seconds forward
                )
            } else {
                (ideal_chunk_end, false)
            };

            let chunk_samples_data = audio_data.samples[current_start..chunk_end].to_vec();
            let offset_ms = ((current_start as f64 / sample_rate) * 1000.0) as u64;

            chunks.push(AudioChunk {
                samples: chunk_samples_data,
                start_offset_ms: offset_ms,
            });

            debug!(
                "Created chunk {}: start={}, end={}, offset={:.2}s, samples={}, duration={:.2}s",
                chunk_index,
                current_start,
                chunk_end,
                offset_ms as f32 / 1000.0,
                chunk_end - current_start,
                (chunk_end - current_start) as f64 / sample_rate
            );

            chunk_index += 1;
            current_start = if is_split_in_slient_point {
                chunk_end
            } else {
                chunk_end - overlap_samples // Move to next chunk with overlap
            };

            if current_start >= total_samples - overlap_samples {
                break;
            }
        }

        debug!(
            "Split audio into {} chunks using silence detection",
            chunks.len()
        );
        chunks
    }

    fn find_silence_split_point(
        &self,
        samples: &[f32],
        target_end: usize,
        sample_rate: f64,
        max_search_ms: u64,
    ) -> (usize, bool) {
        let max_search_samples = (max_search_ms as f64 * sample_rate / 1000.0) as usize;
        let search_end = std::cmp::min(target_end + max_search_samples, samples.len());

        let search_samples = &samples[target_end..search_end];
        if search_samples.is_empty() {
            debug!("No samples to search for silence, using target split position");
            return (target_end, false);
        }

        // Create VAD with adaptive threshold based on the search region
        let rms_threshold = EnergyVAD::calculate_rms(search_samples) * 0.5; // Use 50% of RMS as threshold

        let vad = EnergyVAD::new(sample_rate as u32)
            .with_threshold(rms_threshold)
            .with_frame_size_ms(200)
            .with_frame_shift_ms(100);

        let frame_size = (sample_rate * vad.frame_size_ms as f64 / 1000.0) as usize;
        let frame_shift = (sample_rate * vad.frame_shift_ms as f64 / 1000.0) as usize;

        // Track silence segments
        let mut silence_start_offset: Option<usize> = None;
        let min_silence_duration_ms = 500u64; // Require 500ms of silence
        let min_silence_samples = (min_silence_duration_ms as f64 * sample_rate / 1000.0) as usize;

        for (offset, _) in (0..search_samples.len()).step_by(frame_shift).enumerate() {
            let frame_offset = offset * frame_shift;
            let frame_end = std::cmp::min(frame_offset + frame_size, search_samples.len());

            if frame_offset >= search_samples.len() {
                break;
            }

            let frame = &search_samples[frame_offset..frame_end];
            let has_speech = vad.contain_speech(frame);

            if !has_speech {
                if silence_start_offset.is_none() {
                    silence_start_offset = Some(frame_offset);
                }
            } else {
                // Found speech, check if we had enough silence before it
                if let Some(start_offset) = silence_start_offset {
                    let silence_duration_samples = frame_offset - start_offset;
                    if silence_duration_samples >= min_silence_samples {
                        // Found a suitable split point at the middle of the silence
                        let split_offset = start_offset + silence_duration_samples / 2;
                        let split_pos = target_end + split_offset;

                        let silence_duration_ms =
                            (silence_duration_samples as f64 / sample_rate * 1000.0) as u64;

                        debug!(
                            "Found silence split point at {:.2}s ({}ms silence, {} samples)",
                            split_pos as f64 / sample_rate,
                            silence_duration_ms,
                            silence_duration_samples
                        );

                        return (std::cmp::min(split_pos, samples.len()), true);
                    }
                }
                silence_start_offset = None;
            }
        }

        debug!(
            "No suitable silence found, using target split position at {:.2}s",
            target_end as f64 / sample_rate
        );
        (target_end, false)
    }

    fn extract_transcription_result(
        &self,
        state: &WhisperState,
        audio_duration: f64,
        start_time: std::time::Instant,
    ) -> Result<TranscriptionResult> {
        let audio_duration_ms = (audio_duration * 1000.0) as u64;

        let num_segments = state.full_n_segments();

        let mut segments = Vec::new();
        let mut full_text = String::new();

        for i in 0..num_segments {
            let Some(segment) = state.get_segment(i) else {
                continue;
            };

            let segment_text = segment.to_str().unwrap_or("").trim().to_string();

            if segment_text.is_empty() {
                continue;
            }

            let start_time = (segment.start_timestamp() as u64) * 10;
            let end_time = (segment.end_timestamp() as u64) * 10;
            let confidence = self.calculate_segment_confidence(state, i)?;

            segments.push(TranscriptionSegment {
                index: i as i32 + 1,
                start_time,
                end_time,
                text: segment_text.clone(),
                confidence,
            });

            if !full_text.is_empty() {
                full_text.push(' ');
            }
            full_text.push_str(&segment_text);
        }

        let processing_time = start_time.elapsed().as_millis() as u64;
        Ok(TranscriptionResult {
            text: full_text,
            language: self.config.language.clone(),
            segments,
            processing_time,
            audio_duration: audio_duration_ms,
        })
    }

    fn calculate_segment_confidence(
        &self,
        state: &WhisperState,
        segment_index: i32,
    ) -> Result<f32> {
        let Some(segment) = state.get_segment(segment_index) else {
            return Ok(0.0);
        };
        let token_count = segment.n_tokens();

        if token_count == 0 {
            return Ok(0.0);
        }

        let mut total_prob = 0.0;
        let mut valid_tokens = 0;

        for token_index in 0..token_count {
            if let Some(token) = segment.get_token(token_index) {
                total_prob += token.token_probability();
                valid_tokens += 1;
            }
        }

        if valid_tokens > 0 {
            Ok(total_prob / valid_tokens as f32)
        } else {
            Ok(0.5)
        }
    }
}

pub fn convert_to_compatible_audio(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    cancel: Arc<AtomicBool>,
    progress_cb: impl FnMut(i32) + 'static,
) -> Result<()> {
    is_valid_aduio_file(&output)?;
    ffmpeg::convert_to_whisper_compatible_audio(&input, &output, cancel, progress_cb)?;
    wav::is_whisper_compatible(&output)?;

    Ok(())
}

pub async fn transcribe_file(
    config: WhisperConfig,
    audio_path: impl AsRef<Path>,
    progress_cb: impl FnMut(i32) + 'static,
    segmemnt_cb: impl FnMut(SegmentCallbackData) + 'static,
    abort_cb: impl FnMut() -> bool + 'static,
) -> Result<TranscriptionResult> {
    let transcriber = WhisperTranscriber::new(config)?;
    transcriber
        .transcribe_file(audio_path, progress_cb, segmemnt_cb, abort_cb)
        .await
}

pub fn save_ggml_silero_vad_model(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    fs::write(&path, GGML_SILERO_VAD_MODEL)
        .with_context(|| format!("save {} failed", path.display()))?;

    Ok(())
}

fn is_valid_aduio_file(audio_path: impl AsRef<Path>) -> Result<()> {
    if !audio_path
        .as_ref()
        .to_str()
        .unwrap_or_default()
        .to_lowercase()
        .ends_with(".wav")
    {
        bail!("Only support wav format file");
    }

    Ok(())
}
