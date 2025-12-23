use crate::slint_generatedAppWindow::{
    MediaType as UIMediaType, ModelEntry as UIModelEntry, ModelSource, ModelStatus,
    SubtitleEntry as UISubtitleEntry, SubtitleSetting as UISubtitleSetting,
    TextListEntry as UITextListEntry, TranscribeEntry as UITranscribeEntry,
};
use ffmpeg::MediaType;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use slint::{Model, ModelRc, VecModel};
use std::fmt;

pub const TRANSCRIBE_TABLE: &str = "transcribe";
pub const MODEL_TABLE: &str = "model";

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TextListEntry {
    id: String,
    text: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct SubtitleEntry {
    pub start_timestamp: String,
    pub end_timestamp: String,
    pub original_text: String,
    pub translation_text: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct SubtitleSetting {
    pub font_name: String,
    pub font_size: i32,
    pub is_white_font_color: bool,
    pub enable_background: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TranscribeEntry {
    pub id: String,
    pub file_path: String,
    pub media_type: MediaType,
    pub model_name: String,
    pub lang: String,
    pub sidebar_entry: TextListEntry,
    pub subtitle_entries: Vec<SubtitleEntry>,
    pub subtitle_setting: SubtitleSetting,
}

impl From<UITextListEntry> for TextListEntry {
    fn from(entry: UITextListEntry) -> Self {
        Self {
            id: entry.id.into(),
            text: entry.text.into(),
        }
    }
}

impl From<TextListEntry> for UITextListEntry {
    fn from(entry: TextListEntry) -> Self {
        Self {
            id: entry.id.into(),
            text: entry.text.into(),
            ..Default::default()
        }
    }
}

impl From<UISubtitleEntry> for SubtitleEntry {
    fn from(entry: UISubtitleEntry) -> Self {
        Self {
            start_timestamp: entry.start_timestamp.into(),
            end_timestamp: entry.end_timestamp.into(),
            original_text: entry.original_text.into(),
            translation_text: entry.translation_text.into(),
        }
    }
}

impl From<SubtitleEntry> for UISubtitleEntry {
    fn from(entry: SubtitleEntry) -> Self {
        Self {
            start_timestamp: entry.start_timestamp.into(),
            end_timestamp: entry.end_timestamp.into(),
            original_text: entry.original_text.into(),
            translation_text: entry.translation_text.into(),
            sound_data: ModelRc::new(VecModel::from_slice(&[])),
            ..Default::default()
        }
    }
}

impl From<UISubtitleSetting> for SubtitleSetting {
    fn from(entry: UISubtitleSetting) -> Self {
        Self {
            font_name: entry.font_name.into(),
            font_size: entry.font_size,
            is_white_font_color: entry.is_white_font_color,
            enable_background: entry.enable_background,
        }
    }
}

impl From<SubtitleSetting> for UISubtitleSetting {
    fn from(entry: SubtitleSetting) -> Self {
        Self {
            font_name: entry.font_name.into(),
            font_size: entry.font_size,
            is_white_font_color: entry.is_white_font_color,
            enable_background: entry.enable_background,
        }
    }
}

impl From<UITranscribeEntry> for TranscribeEntry {
    fn from(entry: UITranscribeEntry) -> Self {
        Self {
            id: entry.id.into(),
            file_path: entry.file_path.into(),
            model_name: entry.model_name.into(),
            media_type: entry.media_type.into(),
            lang: entry.lang.into(),
            sidebar_entry: entry.sidebar_entry.into(),
            subtitle_entries: entry
                .subtitle_entries
                .iter()
                .map(|item| item.into())
                .collect::<Vec<_>>(),
            subtitle_setting: entry.subtitle_setting.into(),
        }
    }
}

impl From<TranscribeEntry> for UITranscribeEntry {
    fn from(entry: TranscribeEntry) -> Self {
        Self {
            id: entry.id.into(),
            file_path: entry.file_path.into(),
            model_name: entry.model_name.into(),
            media_type: entry.media_type.into(),
            lang: entry.lang.into(),
            sidebar_entry: entry.sidebar_entry.into(),
            subtitle_entries: ModelRc::new(
                entry
                    .subtitle_entries
                    .into_iter()
                    .map(|item| item.into())
                    .collect::<VecModel<_>>(),
            ),
            subtitle_setting: entry.subtitle_setting.into(),
            ..Default::default()
        }
    }
}

impl From<MediaType> for UIMediaType {
    fn from(ty: MediaType) -> Self {
        match ty {
            MediaType::Video => UIMediaType::Video,
            MediaType::Audio => UIMediaType::Audio,
            MediaType::Unknown => UIMediaType::Unknown,
        }
    }
}

impl From<UIMediaType> for MediaType {
    fn from(ty: UIMediaType) -> Self {
        match ty {
            UIMediaType::Video => MediaType::Video,
            UIMediaType::Audio => MediaType::Audio,
            UIMediaType::Unknown => MediaType::Unknown,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub file_path: String,

    #[serde(default)]
    pub file_size: String,

    pub source: ModelSource,
    pub status: ModelStatus,
}

impl From<ModelEntry> for UIModelEntry {
    fn from(entry: ModelEntry) -> Self {
        Self {
            id: entry.id.into(),
            name: entry.name.into(),
            file_path: entry.file_path.into(),
            file_size: entry.file_size.into(),
            source: entry.source,
            status: entry.status,
            ..Default::default()
        }
    }
}

impl From<UIModelEntry> for ModelEntry {
    fn from(entry: UIModelEntry) -> Self {
        Self {
            id: entry.id.into(),
            name: entry.name.into(),
            file_path: entry.file_path.into(),
            file_size: entry.file_size.into(),
            source: entry.source,
            status: entry.status,
        }
    }
}

impl Serialize for ModelSource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            ModelSource::Network => serializer.serialize_str("Network"),
            ModelSource::Local => serializer.serialize_str("Local"),
        }
    }
}

impl<'de> Deserialize<'de> for ModelSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ModelSourceVisitor;

        impl<'de> Visitor<'de> for ModelSourceVisitor {
            type Value = ModelSource;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a string representing ModelSource ('Network' or 'Local')")
            }

            fn visit_str<E>(self, value: &str) -> Result<ModelSource, E>
            where
                E: de::Error,
            {
                match value {
                    "Network" => Ok(ModelSource::Network),
                    "Local" => Ok(ModelSource::Local),
                    _ => Err(E::custom(format!("unknown ModelSource variant: {}", value))),
                }
            }
        }

        deserializer.deserialize_str(ModelSourceVisitor)
    }
}

impl Serialize for ModelStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            ModelStatus::Downloading => serializer.serialize_str("Downloading"),
            ModelStatus::DownloadFailed => serializer.serialize_str("DownloadFailed"),
            ModelStatus::DownloadFinished => serializer.serialize_str("DownloadFinished"),
            ModelStatus::DownloadCancelled => serializer.serialize_str("DownloadCancelled"),
            ModelStatus::NoFound => serializer.serialize_str("NoFound"),
            ModelStatus::Import => serializer.serialize_str("Import"),
            ModelStatus::InvalidFormat => serializer.serialize_str("InvalidFormat"),
        }
    }
}

impl<'de> Deserialize<'de> for ModelStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ModelStatusVisitor;

        impl<'de> Visitor<'de> for ModelStatusVisitor {
            type Value = ModelStatus;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a string representing ModelStatus ('Downloading', 'DownloadFailed', 'Import', 'DownloadFinished', 'DownloadCancelled', 'NoFound' or 'InvalidFormat')")
            }

            fn visit_str<E>(self, value: &str) -> Result<ModelStatus, E>
            where
                E: de::Error,
            {
                match value {
                    "Downloading" => Ok(ModelStatus::Downloading),
                    "DownloadFailed" => Ok(ModelStatus::DownloadFailed),
                    "DownloadFinished" => Ok(ModelStatus::DownloadFinished),
                    "DownloadCancelled" => Ok(ModelStatus::DownloadCancelled),
                    "NoFound" => Ok(ModelStatus::NoFound),
                    "Import" => Ok(ModelStatus::Import),
                    "InvalidFormat" => Ok(ModelStatus::InvalidFormat),
                    _ => Err(E::custom(format!("unknown ModelStatus variant: {}", value))),
                }
            }
        }

        deserializer.deserialize_str(ModelStatusVisitor)
    }
}
