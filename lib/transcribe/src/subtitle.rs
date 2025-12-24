use super::whisper::{TranscriptionResult, TranscriptionSegment};
use anyhow::{Context, Result};
use chrono::{NaiveTime, Timelike};
use std::{fs, path::Path};
use unicode_segmentation::UnicodeSegmentation;
use whisper_rs::SegmentCallbackData;

#[derive(Debug, Clone, Default)]
pub struct Subtitle {
    pub index: i32,
    pub start_timestamp: u64,
    pub end_timestamp: u64,
    pub text: String,
}

impl From<SegmentCallbackData> for Subtitle {
    fn from(segment: SegmentCallbackData) -> Self {
        Subtitle {
            index: segment.segment + 1,
            start_timestamp: (segment.start_timestamp as u64) * 10,
            end_timestamp: (segment.end_timestamp as u64) * 10,
            text: segment.text,
        }
    }
}

impl From<&TranscriptionSegment> for Subtitle {
    fn from(segment: &TranscriptionSegment) -> Self {
        Subtitle {
            index: segment.index,
            start_timestamp: segment.start_time,
            end_timestamp: segment.end_time,
            text: segment.text.clone(),
        }
    }
}

pub fn transcription_to_subtitle(transcription: &TranscriptionResult) -> Vec<Subtitle> {
    let mut item = vec![];

    for segment in transcription.segments.iter() {
        item.push(segment.into());
    }

    item
}

pub fn ms_to_srt_timestamp(milliseconds: u64) -> String {
    ms_to_timestamp(milliseconds, ",")
}

pub fn ms_to_vtt_timestamp(milliseconds: u64) -> String {
    ms_to_timestamp(milliseconds, ".")
}

fn ms_to_timestamp(milliseconds: u64, ms_sep: &str) -> String {
    let total_seconds = milliseconds / 1000;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    let millis = milliseconds % 1000;

    format!(
        "{:02}:{:02}:{:02}{ms_sep}{:03}",
        hours, minutes, seconds, millis
    )
}

pub fn srt_timestamp_to_ms(timestamp: &str) -> Result<u64> {
    let time = NaiveTime::parse_from_str(timestamp, "%H:%M:%S,%f")
        .with_context(|| format!("Invalid srt timestamp {timestamp}"))?;

    Ok((time.hour() as u64 * 3600000)
        + (time.minute() as u64 * 60000)
        + (time.second() as u64 * 1000)
        // This's not a bug，chrono would parse ',%f' into nanosecond field
        + (time.nanosecond() as u64))
}

pub fn valid_srt_timestamp(timestamp: &str) -> bool {
    srt_timestamp_to_ms(timestamp).is_ok()
}

pub fn subtitle_to_srt(subtitle: &Subtitle) -> String {
    format!(
        "{}\n{} --> {}\n{}",
        subtitle.index,
        ms_to_srt_timestamp(subtitle.start_timestamp),
        ms_to_srt_timestamp(subtitle.end_timestamp),
        subtitle.text
    )
}

pub fn subtitle_to_vtt(subtitle: &Subtitle) -> String {
    format!(
        "{}\n{} --> {}\n{}",
        subtitle.index,
        ms_to_vtt_timestamp(subtitle.start_timestamp),
        ms_to_vtt_timestamp(subtitle.end_timestamp),
        subtitle.text
    )
}

pub fn subtitle_to_plain(subtitle: &Subtitle) -> String {
    format!("{}", subtitle.text)
}

pub fn save_as_srt(subtitle: &[Subtitle], path: impl AsRef<Path>) -> Result<()> {
    let contents = subtitle
        .iter()
        .map(|item| format!("{}\n\n", subtitle_to_srt(&item)))
        .collect::<String>();

    fs::write(path.as_ref(), contents)
        .with_context(|| format!("Save {} failed", path.as_ref().display()))?;

    Ok(())
}

pub fn save_as_vtt(subtitle: &[Subtitle], path: impl AsRef<Path>) -> Result<()> {
    let contents = subtitle
        .iter()
        .map(|item| format!("{}\n\n", subtitle_to_vtt(&item)))
        .collect::<String>();

    fs::write(path.as_ref(), contents)
        .with_context(|| format!("Save {} failed", path.as_ref().display()))?;

    Ok(())
}

pub fn save_as_txt(subtitle: &[Subtitle], path: impl AsRef<Path>) -> Result<()> {
    let contents = subtitle
        .iter()
        .map(|item| format!("{} ", subtitle_to_plain(&item)))
        .collect::<String>();

    fs::write(path.as_ref(), contents)
        .with_context(|| format!("Save {} failed", path.as_ref().display()))?;

    Ok(())
}

pub fn convert_traditional_to_simplified_chinese(text: &str) -> String {
    fast2s::convert(text)
}

pub fn split_subtitle_into_two(
    start_timestamp: u64,
    end_timestamp: u64,
    content: &str,
) -> Option<((u64, u64, String), (u64, u64, String))> {
    if content.is_empty() || content.trim().len() <= 1 {
        return None;
    }

    let delimiters = [' ', ',', '.', '，', '。'];
    let mut split_positions: Vec<usize> = Vec::new();

    for (i, c) in content.char_indices() {
        if delimiters.contains(&c) {
            let next_pos = i + c.len_utf8();
            if next_pos <= content.len() {
                split_positions.push(next_pos);
            }
        }
    }

    let (first_part, second_part) = if split_positions.is_empty() {
        let graphemes: Vec<&str> = content.graphemes(true).collect();
        let mid = graphemes.len() / 2;
        let first_part = graphemes[..mid].concat();
        let second_part = graphemes[mid..].concat();
        (first_part, second_part)
    } else {
        let target_split = content.len() / 2;
        let Some(best_split) = split_positions
            .iter()
            .min_by_key(|&&pos| (pos as isize - target_split as isize).abs())
        else {
            return None;
        };

        let first_part = content[..*best_split].trim().to_string();
        let second_part = content[*best_split..].trim().to_string();
        (first_part, second_part)
    };

    let total_chars = content.chars().count();
    let first_part_chars = first_part.chars().count();

    let duration = end_timestamp - start_timestamp;
    let split_time = start_timestamp + (duration * first_part_chars as u64) / total_chars as u64;

    Some((
        (start_timestamp, split_time, first_part),
        (split_time, end_timestamp, second_part),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_split_with_timestamps() {
        let ((start1, end1, part1), (start2, end2, part2)) =
            split_subtitle_into_two(0, 1000, "Hello, world!").unwrap();

        assert_eq!(part1, "Hello,");
        assert_eq!(part2, "world!");
        assert_eq!(start1, 0);
        assert_eq!(end1, start2);
        assert_eq!(end2, 1000);
        assert!(end1 > 0 && end1 < 1000);
    }

    #[test]
    fn test_chinese_split_with_timestamps() {
        let ((start1, end1, part1), (start2, end2, part2)) =
            split_subtitle_into_two(0, 1000, "你好，世界！").unwrap();

        assert_eq!(part1, "你好，");
        assert_eq!(part2, "世界！");
        assert_eq!(start1, 0);
        assert_eq!(end1, start2);
        assert_eq!(end2, 1000);
    }

    #[test]
    fn test_no_delimiters_with_timestamps() {
        let ((start1, end1, part1), (_start2, end2, part2)) =
            split_subtitle_into_two(0, 1000, "abcdefgh").unwrap();

        assert_eq!(part1, "abcd");
        assert_eq!(part2, "efgh");
        assert_eq!(start1, 0);
        assert_eq!(end1, 500);
        assert_eq!(end2, 1000);
    }

    #[test]
    fn test_empty_string() {
        assert!(split_subtitle_into_two(0, 1000, "").is_none());
    }

    #[test]
    fn test_single_character() {
        assert!(split_subtitle_into_two(0, 1000, "a").is_none());
    }

    #[test]
    fn test_time_calculation_proportion() {
        let ((_start1, end1, _part1), (start2, _end2, _part2)) =
            split_subtitle_into_two(0, 1000, "Hello world").unwrap();

        let total_chars = 11;
        let expected_split_time = (1000 * 5) / total_chars;

        assert_eq!(end1, expected_split_time);
        assert_eq!(start2, expected_split_time);
    }

    // cargo test test_complicate -- --no-capture
    #[test]
    fn test_complicate() {
        let s = "就來看下這個庫,的手用情況 就顯得是要支數是278次.給帶了兩個版本";
        let items = split_subtitle_into_two(0, 100, s).unwrap();

        println!("{items:?}");
    }
}
