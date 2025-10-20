use anyhow::Result;
use transcribe::vad::estimate_rms_for_duration;

fn main() -> Result<()> {
    let wav_path = "./examples/data/test-20.wav";
    let duration_seconds = 180; // 3 minutes
    
    let rms = estimate_rms_for_duration(wav_path, duration_seconds)?;
    
    println!("WAV file: {}", wav_path);
    println!("RMS for first {} seconds: {:.6}", duration_seconds, rms);
    
    Ok(())
}