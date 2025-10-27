use crate::{
    config,
    db::{
        self,
        def::{TRANSCRIBE_TABLE as DB_TABLE, TranscribeEntry},
    },
    global_logic, global_store,
    logic::{
        toast::{self, async_toast_warn},
        tr::tr,
    },
    slint_generatedAppWindow::{
        AiHandleSubtitleSetting as UIAiHandleSubtitleSetting, AppWindow,
        ExportVideoSetting as UIExportVideoSetting, MediaType as UIMediaType, PopupIndex,
        ProgressType, SubtitleEntry as UISubtitleEntry, SubtitleSetting as UISubtitleSetting,
        SystemFontInfo as UISystemFontInfo, TextListEntry as UITextListEntry,
        TranscribeEntry as UITranscribeEntry, VideoPlayerSetting as UIVideoPlayerSetting,
    },
    toast_info, toast_success, toast_warn,
};
use anyhow::{Result, anyhow};
use async_openai::{
    Client,
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
};
use ffmpeg::{
    MediaType, SubtitleConfig, VideoExitStatus, VideoFramesIterConfig, VideoMetadata,
    VideoResolution,
};
use kittyaudio::{Mixer, Sound, SoundHandle};
use log::{debug, info, trace, warn};
use once_cell::sync::Lazy;
use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel, Weak};
use std::{
    fs,
    path::{Path, PathBuf},
    str::FromStr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
};
use tokio::{sync::mpsc, task::AbortHandle};
use transcribe::{
    SegmentCallbackData,
    subtitle::{self, Subtitle},
    whisper_lang::WhisperLang,
};
use uuid::Uuid;

static MEDIA_INC_NUM: AtomicU64 = AtomicU64::new(0);
static CACHE: Lazy<Mutex<Cache>> = Lazy::new(|| Mutex::new(Cache::default()));

#[macro_export]
macro_rules! store_system_font_infos {
    ($ui:expr) => {
        crate::global_store!($ui)
            .get_system_font_infos()
            .as_any()
            .downcast_ref::<VecModel<UISystemFontInfo>>()
            .expect("We know we set a VecModel<SystemFontInfo> earlier")
    };
}

#[macro_export]
macro_rules! store_whisper_langs {
    ($ui:expr) => {
        crate::global_store!($ui)
            .get_whisper_langs()
            .as_any()
            .downcast_ref::<VecModel<SharedString>>()
            .expect("We know we set a whisper lang VecModel<SharedString> earlier")
    };
}

#[macro_export]
macro_rules! store_transcribe_entries {
    ($ui:expr) => {
        crate::global_store!($ui)
            .get_transcribe_entries()
            .as_any()
            .downcast_ref::<VecModel<UITranscribeEntry>>()
            .expect("We know we set a VecModel<UITranscribeEntry> earlier")
    };
}

#[macro_export]
macro_rules! store_transcribe_entries_cache {
    ($ui:expr) => {
        crate::global_store!($ui)
            .get_transcribe_entries_cache()
            .as_any()
            .downcast_ref::<VecModel<UITranscribeEntry>>()
            .expect("We know we set a cache VecModel<UITranscribeEntry> earlier")
    };
}

#[macro_export]
macro_rules! store_transcribe_subtitle_entries {
    ($entry:expr) => {
        $entry
            .subtitle_entries
            .as_any()
            .downcast_ref::<VecModel<UISubtitleEntry>>()
            .expect("We know we set a VecModel<UISubtitleEntry> earlier")
    };
}

pub fn init(ui: &AppWindow) {
    inner_init(ui);

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_new_transcribe_entry(move || {
        let ui = ui_weak.unwrap();
        new_transcribe_entry(&ui);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_rename_transcribe_entry(move |index, text| {
        let ui = ui_weak.unwrap();
        let index = index as usize;

        let mut entry = store_transcribe_entries!(ui).row_data(index).unwrap();
        entry.sidebar_entry.text = text;
        store_transcribe_entries!(ui).set_row_data(index, entry.clone());
        global_logic!(ui).invoke_toggle_update_transcribe_sidebar_flag();
        global_logic!(ui).invoke_switch_popup(PopupIndex::None);
        toast_success!(ui, tr("Rename entry successfully"));

        update_db_entry(&ui, entry.into());
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_remove_transcribe_entry(move |index| {
        let ui = ui_weak.unwrap();

        let id = store_transcribe_entries!(ui)
            .remove(index as usize)
            .id
            .to_string();

        let selected_index = global_store!(ui).get_selected_transcribe_sidebar_index();
        if index == selected_index {
            global_store!(ui).set_selected_transcribe_sidebar_index(-1);
        }

        global_logic!(ui).invoke_toggle_update_transcribe_sidebar_flag();
        toast_success!(ui, tr("Remove entry successfully"));

        delete_db_entry(&ui, id);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_show_rename_transcribe_dialog(move |index| {
        let ui = ui_weak.unwrap();
        global_store!(ui).set_edit_transcribe_sidebar_index(index);
        global_store!(ui).set_current_popup_index(PopupIndex::TranscribeRename);
    });

    global_logic!(ui).on_gen_transcribe_sidebar_entries(move |_flag, entries| {
        let sidebar_entries = entries
            .iter()
            .map(|item| item.sidebar_entry)
            .collect::<Vec<UITextListEntry>>();

        ModelRc::new(VecModel::from_slice(&sidebar_entries))
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_search_sidebar(move |value| {
        let ui = ui_weak.unwrap();

        if value.is_empty() {
            let entries = store_transcribe_entries_cache!(ui)
                .iter()
                .collect::<Vec<UITranscribeEntry>>();
            store_transcribe_entries!(ui).set_vec(entries);
            store_transcribe_entries_cache!(ui).set_vec(vec![]);
        } else {
            if store_transcribe_entries_cache!(ui).row_count() == 0 {
                let entries = store_transcribe_entries!(ui)
                    .iter()
                    .collect::<Vec<UITranscribeEntry>>();
                store_transcribe_entries_cache!(ui).set_vec(entries);
            }

            let filter_entries = store_transcribe_entries_cache!(ui)
                .iter()
                .filter(|item| item.sidebar_entry.text.contains(value.as_str()))
                .collect::<Vec<UITranscribeEntry>>();
            store_transcribe_entries!(ui).set_vec(filter_entries);
        }

        global_logic!(ui).invoke_toggle_update_transcribe_sidebar_flag();
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_switch_sidebar_entry(move |old_index, new_index| {
        if old_index == new_index {
            return;
        }

        let ui = ui_weak.unwrap();
        switch_sidebar_entry(&ui, old_index, new_index);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_start_transcribe(move |entry| {
        let ui = ui_weak.unwrap();

        if get_progressing() {
            toast_warn!(ui, tr("Already runing whisper transcription"));
            return;
        }

        global_logic!(ui).invoke_switch_popup(PopupIndex::None);
        start_transcribe(&ui, entry);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_update_progress(move |id, progress| {
        let ui = ui_weak.unwrap();
        for (index, mut entry) in store_transcribe_entries!(ui).iter().enumerate() {
            if entry.id == id {
                entry.progress = progress;
                store_transcribe_entries!(ui).set_row_data(index, entry);
                return;
            }
        }
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_update_progress_type(move |id, ty| {
        let ui = ui_weak.unwrap();
        for (index, mut entry) in store_transcribe_entries!(ui).iter().enumerate() {
            if entry.id == id {
                entry.progress_type = ty;
                store_transcribe_entries!(ui).set_row_data(index, entry);
                return;
            }
        }
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_cancel_progress(move |id, ty| {
        cancel_progress(&ui_weak.unwrap(), id, ty);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_import_media_file(move || {
        import_media_file(&ui_weak.unwrap());
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_export_subtitles(move |ty| {
        let ui = ui_weak.unwrap();
        global_logic!(ui).invoke_switch_popup(PopupIndex::None);
        export_subtitles(&ui, ty.into());
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_export_video(move |setting| {
        let ui = ui_weak.unwrap();
        global_logic!(ui).invoke_switch_popup(PopupIndex::None);
        export_video(&ui, setting);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_refresh_subtitles(move || {
        let ui = ui_weak.unwrap();
        refresh_subtitles(&ui);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_show_ai_handle_subtitle_setting_dialog(move |ty| {
        let ui = ui_weak.unwrap();
        let mut setting = global_store!(ui).get_edit_ai_handle_subtitle_setting();

        if setting.chunk_size <= 0 {
            setting.chunk_size = 10;
        }

        if setting.lang.is_empty() {
            setting.lang = "English".to_string().into();
        }

        match ty.as_str() {
            "translate" => {
                setting.ty = ProgressType::Translate;
            }
            "correct" => {
                setting.ty = ProgressType::Correct;
            }
            _ => unreachable!(),
        }

        global_store!(ui).set_edit_ai_handle_subtitle_setting(setting);
        global_logic!(ui).invoke_switch_popup(PopupIndex::AiHandleSubtitleSetting);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_ai_translate_all_subtitles(move |setting| {
        let ui = ui_weak.unwrap();
        global_logic!(ui).invoke_switch_popup(PopupIndex::None);
        global_store!(ui).set_edit_ai_handle_subtitle_setting(setting.clone());
        ai_translate_all_subtitles(&ui, setting);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_remove_all_translated_subtitles(move || {
        remove_all_translated_subtitles(&ui_weak.unwrap());
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_ai_correct_all_subtitles(move |setting| {
        let ui = ui_weak.unwrap();
        global_logic!(ui).invoke_switch_popup(PopupIndex::None);
        global_store!(ui).set_edit_ai_handle_subtitle_setting(setting.clone());
        ai_correct_all_subtitles(&ui, setting);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_accept_all_corrected_subtitles(move || {
        accept_all_corrected_subtitles(&ui_weak.unwrap());
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_remove_all_corrected_subtitles(move || {
        remove_all_corrected_subtitles(&ui_weak.unwrap());
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_show_replace_subtitles_content_dialog(move || {
        let ui = ui_weak.unwrap();
        global_logic!(ui).invoke_switch_popup(PopupIndex::SubtitlesReplace);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_replace_subtitles_content(move |old_text, new_text| {
        if old_text.is_empty() {
            return;
        }

        let ui = ui_weak.unwrap();
        global_logic!(ui).invoke_switch_popup(PopupIndex::None);
        replace_subtitles_content(&ui, old_text, new_text);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_replace_subtitles_all_separator(move || {
        replace_subtitles_all_separator(&ui_weak.unwrap());
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_traditional_to_simple_chinese(move || {
        traditional_to_simple_chinese(&ui_weak.unwrap());
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_swap_all_original_and_translation(move || {
        swap_all_original_and_translation(&ui_weak.unwrap());
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_remove_all_subtitles(move || {
        remove_all_subtitles(&ui_weak.unwrap());
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_optimize_subtitles_timestamp(move || {
        optimize_subtitles_timestamp(&ui_weak.unwrap());
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_recover_subtitles_timestamp(move || {
        recover_subtitles_timestamp(&ui_weak.unwrap());
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_split_subtitle(move |index| {
        split_subtitle(&ui_weak.unwrap(), index as usize);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_merge_above_subtitle(move |index| {
        merge_above_subtitle(&ui_weak.unwrap(), index as usize);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_insert_above_subtitle(move |index| {
        insert_above_subtitle(&ui_weak.unwrap(), index as usize);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_insert_below_subtitle(move |index| {
        insert_below_subtitle(&ui_weak.unwrap(), index as usize);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_remove_subtitle(move |index| {
        remove_subtitle(&ui_weak.unwrap(), index as usize);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_save_subtitle(move |index, subtitle| {
        save_subtitle(&ui_weak.unwrap(), index as usize, subtitle);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_reject_subtitle_correction(move |index| {
        reject_subtitle_correction(&ui_weak.unwrap(), index as usize);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_accept_subtitle_correction(move |index| {
        accept_subtitle_correction(&ui_weak.unwrap(), index as usize);
    });

    global_logic!(ui)
        .on_is_valid_subtitle_timestamp(|timestamp| subtitle::valid_srt_timestamp(&timestamp));

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_video_player_start(move |timestamp| {
        let ui = ui_weak.unwrap();
        if video_player_is_playing() {
            video_player_stop(&ui, false);
        }

        video_player_start(&ui, timestamp, None);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_video_player_partial_play(move |start_timestamp, end_timestamp| {
        let ui = ui_weak.unwrap();
        if video_player_is_playing() {
            video_player_stop(&ui, false);
        }

        video_player_partial_play(&ui, start_timestamp, end_timestamp);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_video_player_stop(move || {
        video_player_stop(&ui_weak.unwrap(), true);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_before_change_video_player_position(move || {
        video_player_stop(&ui_weak.unwrap(), false);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_change_video_player_position(move |timestamp| {
        let ui = ui_weak.unwrap();
        global_logic!(ui).invoke_video_player_start(timestamp);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_change_video_player_sound(move |sound| {
        let ui = ui_weak.unwrap();
        let index = global_store!(ui).get_selected_transcribe_sidebar_index() as usize;

        let mut entry = store_transcribe_entries!(ui).row_data(index).unwrap();
        entry.video_player_setting.volume = sound;
        store_transcribe_entries!(ui).set_row_data(index, entry.clone());

        set_audio_volume(sound);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_update_video_subtitle_setting(move |setting| {
        let ui = ui_weak.unwrap();
        let index = global_store!(ui).get_selected_transcribe_sidebar_index() as usize;

        let mut entry = store_transcribe_entries!(ui).row_data(index).unwrap();
        entry.subtitle_setting = setting;
        store_transcribe_entries!(ui).set_row_data(index, entry.clone());

        update_db_entry(&ui, entry.into());
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_audio_player_start(move |timestamp| {
        audio_player_start(&ui_weak.unwrap(), timestamp, None);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_audio_player_partial_play(move |start_timestamp, end_timestamp| {
        audio_player_partial_play(&ui_weak.unwrap(), start_timestamp, end_timestamp);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_audio_player_stop(move |timestamp| {
        audio_player_stop(&ui_weak.unwrap(), timestamp);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_before_change_audio_player_position(move || {
        before_change_audio_player_position(&ui_weak.unwrap());
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_change_audio_player_position(move |timestamp| {
        change_audio_player_player_position(&ui_weak.unwrap(), timestamp);
    });

    let ui_weak = ui.as_weak();
    global_logic!(ui).on_change_audio_player_sound(move |sound| {
        let ui = ui_weak.unwrap();
        let index = global_store!(ui).get_selected_transcribe_sidebar_index() as usize;

        let mut entry = store_transcribe_entries!(ui).row_data(index).unwrap();
        entry.video_player_setting.volume = sound;
        store_transcribe_entries!(ui).set_row_data(index, entry.clone());

        set_audio_volume(sound);
    });

    global_logic!(ui).on_media_is_finished(|| match get_audio_player_handle() {
        None => true,
        Some(handle) => handle.finished(),
    });

    global_logic!(ui).on_srt_timestamp_to_ms_second(|timestamp| {
        subtitle::srt_timestamp_to_ms(&timestamp).unwrap_or_default() as f32
    });

    global_logic!(ui).on_ai_available(move || {
        let setting = config::model();
        !setting.api_base_url.is_empty()
            && !setting.model_name.is_empty()
            && !setting.api_key.is_empty()
    });

    global_logic!(ui).on_get_current_subtitle(move |subtitles, current_time, _flag| {
        let current_time = (current_time * 1000.0) as u64;
        get_current_subtitle(subtitles, current_time)
    });

    global_logic!(ui).on_system_font_names(move |infos, _flag| {
        let names = infos
            .iter()
            .map(|item| item.name.clone())
            .collect::<VecModel<SharedString>>();

        ModelRc::new(names)
    });

    global_logic!(ui).on_system_font_family(move |name, infos, _flag1| {
        if let Some(item) = infos.iter().find(|item| item.name == name) {
            item.family.clone()
        } else {
            Default::default()
        }
    });

    // END
}

fn inner_init(ui: &AppWindow) {
    set_whisper_langs(&ui);

    store_system_font_infos!(ui).set_vec(vec![]);
    store_transcribe_entries!(ui).set_vec(vec![]);
    global_store!(ui).set_selected_transcribe_sidebar_index(-1);
    global_store!(ui).set_ffmpeg_is_installed(ffmpeg::is_installed());

    let ui = ui.as_weak();
    tokio::spawn(async move {
        let entries = match db::entry::select_all(DB_TABLE).await {
            Ok(items) => items
                .into_iter()
                .filter_map(|item| serde_json::from_str::<TranscribeEntry>(&item.data).ok())
                .collect(),

            Err(e) => {
                log::warn!("{:?}", e);
                vec![]
            }
        };

        _ = ui.clone().upgrade_in_event_loop(move |ui| {
            let entries = entries
                .into_iter()
                .rev()
                .map(|entry| {
                    let mut entry: UITranscribeEntry = entry.into();
                    entry.video_player_setting.volume = 1.0;
                    entry.is_file_exist = PathBuf::from_str(&entry.file_path)
                        .unwrap_or_default()
                        .exists();
                    entry
                })
                .collect::<Vec<UITranscribeEntry>>();

            store_transcribe_entries!(ui).set_vec(entries);
            global_logic!(ui).invoke_toggle_update_transcribe_sidebar_flag();
        });

        let (mut chinese_font_infos, mut none_chinese_font_infos) = font::system_fonts();
        chinese_font_infos.sort_by(|a, b| a.0.cmp(&b.0));
        none_chinese_font_infos.sort_by(|a, b| a.0.cmp(&b.0));
        chinese_font_infos.extend(none_chinese_font_infos.into_iter());

        let font_infos = chinese_font_infos
            .into_iter()
            .map(|item| UISystemFontInfo {
                name: item.0.into(),
                family: item.1.into(),
            })
            .collect::<Vec<UISystemFontInfo>>();

        _ = ui.clone().upgrade_in_event_loop(move |ui| {
            store_system_font_infos!(ui).set_vec(font_infos);
        });
    });
}

fn set_whisper_langs(ui: &AppWindow) {
    let entries = WhisperLang::all_languages()
        .into_iter()
        .map(|item| item.2.to_string().into())
        .collect::<Vec<SharedString>>();

    store_whisper_langs!(ui).set_vec(entries);
}

fn new_transcribe_entry(ui: &AppWindow) {
    let ui = ui.as_weak();

    tokio::spawn(async move {
        let id = Uuid::new_v4().to_string();

        let Some(media_file) = picker_file(ui.clone(), &tr("Choose a media file")) else {
            return;
        };

        let Some(file_name) = file_name(ui.clone(), &media_file) else {
            return;
        };

        let Some(media_type) = media_type(ui.clone(), &media_file) else {
            return;
        };

        let screenshot_path = video_screenshot(&id, &media_file, media_type.clone());
        let media_duration = media_duration(&media_file, media_type.clone());

        // TODO:
        _ = slint::invoke_from_event_loop(move || {
            let ui = ui.unwrap();
            let mut entry = UITranscribeEntry::default();
            entry.id = id.clone().into();
            entry.file_path = media_file.as_path().to_string_lossy().to_string().into();
            entry.is_file_exist = true;
            entry.media_type = media_type.into();
            entry.lang = "Auto detect".into();
            entry.subtitle_entries = ModelRc::new(VecModel::from_slice(&vec![]));
            entry.video_player_setting.volume = 1.0;

            entry.sidebar_entry = UITextListEntry {
                id: id.clone().into(),
                text: file_name.into(),
                ..Default::default()
            };

            entry.subtitle_setting = UISubtitleSetting {
                font_name: store_system_font_infos!(ui)
                    .row_data(0)
                    .unwrap_or_default()
                    .name,
                font_size: 20,
                is_white_font_color: true,
                enable_background: false,
            };

            set_video_player_setting(
                &ui,
                &mut entry.video_player_setting,
                screenshot_path,
                media_duration,
            );

            store_transcribe_entries!(ui).insert(0, entry.clone());
            global_logic!(ui).invoke_toggle_update_transcribe_sidebar_flag();
            global_store!(ui).set_selected_transcribe_sidebar_index(0);
            toast_success!(ui, &tr("Add entry successfully"));

            add_db_entry(&ui, entry.clone().into());

            // convert to whisper compatiable audio
            let (ui_weak, id) = (ui.as_weak(), entry.id.clone().to_string());
            let (input_media_path, output_audio_path, output_audio_path_tmp) =
                get_convert_to_audio_paths(&entry);

            tokio::spawn(async move {
                convert_to_whisper_compatible_audio(
                    ui_weak,
                    id,
                    &input_media_path,
                    &output_audio_path,
                    &output_audio_path_tmp,
                );
            });
        });
    });
}

fn add_db_entry(ui: &AppWindow, entry: TranscribeEntry) {
    let ui = ui.as_weak();
    tokio::spawn(async move {
        let data = serde_json::to_string(&entry).unwrap();
        match db::entry::insert(DB_TABLE, &entry.id, &data).await {
            Err(e) => toast::async_toast_warn(
                ui,
                format!("{}. {}: {e}", tr("insert entry failed"), tr("Reason")),
            ),
            _ => (),
        }
    });
}

fn update_db_entry(ui: &AppWindow, entry: TranscribeEntry) {
    let ui = ui.as_weak();
    tokio::spawn(async move {
        let data = serde_json::to_string(&entry).unwrap();
        match db::entry::update(DB_TABLE, &entry.id, &data).await {
            Err(e) => toast::async_toast_warn(
                ui,
                format!("{}. {}: {e}", tr("Update entry failed"), tr("Reason")),
            ),
            _ => (),
        }
    });
}

fn delete_db_entry(ui: &AppWindow, id: String) {
    let ui = ui.as_weak();
    tokio::spawn(async move {
        match db::entry::delete(DB_TABLE, &id).await {
            Err(e) => toast::async_toast_warn(
                ui,
                format!("{}. {}: {e:?}", tr("Remove entry failed"), tr("Reason")),
            ),
            _ => (),
        }
    });
}

pub fn picker_file(ui: Weak<AppWindow>, title: &str) -> Option<PathBuf> {
    let result = native_dialog::DialogBuilder::file()
        .set_title(title)
        .open_single_file()
        .show();

    match result {
        Ok(Some(path)) => Some(path),
        Err(e) => {
            toast::async_toast_warn(
                ui,
                format!("{}. {}: {}", tr("Choose file failed"), tr("Reason"), e),
            );
            None
        }
        _ => None,
    }
}

pub fn picker_directory(ui: Weak<AppWindow>, title: &str, filename: &str) -> Option<PathBuf> {
    let result = native_dialog::DialogBuilder::file()
        .set_title(title)
        .set_filename(filename)
        .open_single_dir()
        .show();

    match result {
        Ok(Some(path)) => Some(path),
        Err(e) => {
            toast::async_toast_warn(
                ui,
                format!("{}. {}: {}", tr("Choose directory failed"), tr("Reason"), e),
            );
            None
        }
        _ => None,
    }
}

fn file_name(ui: Weak<AppWindow>, media_file: impl AsRef<Path>) -> Option<String> {
    match media_file.as_ref().file_name() {
        Some(v) => Some(v.to_string_lossy().to_string()),
        _ => {
            toast::async_toast_warn(
                ui,
                tr(&format!(
                    "{}. {}.",
                    tr("can't paree filename"),
                    media_file.as_ref().display(),
                )),
            );
            None
        }
    }
}

fn media_type(ui: Weak<AppWindow>, media_file: impl AsRef<Path>) -> Option<MediaType> {
    match ffmpeg::media_type(&media_file) {
        Ok(ty) => {
            if ty == MediaType::Unknown {
                toast::async_toast_warn(
                    ui,
                    tr(&format!(
                        "{} {}",
                        media_file.as_ref().display(),
                        tr("is not a media file")
                    )),
                );

                return None;
            }

            Some(ty)
        }
        Err(e) => {
            toast::async_toast_warn(
                ui,
                tr(&format!(
                    "{} {}. {}: {e}",
                    tr("detect media file failed!"),
                    media_file.as_ref().display(),
                    tr("Reason"),
                )),
            );

            None
        }
    }
}

fn video_screenshot(id: &str, path: impl AsRef<Path>, media_type: MediaType) -> Option<PathBuf> {
    let save_path = config::cache_dir().join(format!("{id}.png"));

    match media_type {
        MediaType::Video => match ffmpeg::video_screenshots(path, 1) {
            Ok(imgs) => {
                let Some(img) = imgs.first() else {
                    return None;
                };

                match img.save(&save_path) {
                    Ok(_) => Some(save_path),
                    Err(e) => {
                        warn!(
                            "save {} failed. error: {e:?}",
                            save_path.as_path().display()
                        );

                        None
                    }
                }
            }
            Err(e) => {
                warn!("get video screenshot failed. error: {e:?}");
                None
            }
        },

        _ => None,
    }
}

fn media_duration(path: impl AsRef<Path>, media_type: MediaType) -> Option<f64> {
    match media_type {
        MediaType::Video => {
            match ffmpeg::video_metadata(path.as_ref().to_str().unwrap_or_default()) {
                Ok(info) => Some(info.duration),
                Err(e) => {
                    warn!(
                        "get video file {} duration failed. error: {e}",
                        path.as_ref().display()
                    );
                    None
                }
            }
        }
        MediaType::Audio => {
            match ffmpeg::audio_metadata(path.as_ref().to_str().unwrap_or_default()) {
                Ok(info) => Some(info.duration),
                Err(e) => {
                    warn!(
                        "get audio file {} duration failed. error: {e}",
                        path.as_ref().display()
                    );
                    None
                }
            }
        }

        _ => None,
    }
}

fn set_video_player_setting(
    ui: &AppWindow,
    setting: &mut UIVideoPlayerSetting,
    screenshot_path: Option<PathBuf>,
    duration: Option<f64>,
) {
    if let Some(duration) = duration {
        setting.end_time = duration as f32;
    }

    if let Some(path) = screenshot_path {
        match slint::Image::load_from_path(&path) {
            Ok(img) => {
                setting.img_width = img.size().width as i32;
                setting.img_height = img.size().height as i32;
                setting.img = img;
            }
            Err(e) => warn!(
                "load img from {} faild. error: {e}",
                path.as_path().display()
            ),
        }
    } else {
        let img = global_logic!(ui).invoke_default_audio_player_screenshot();
        setting.img_width = img.size().width as i32;
        setting.img_height = img.size().height as i32;
        setting.img = img;
    }
}

fn switch_sidebar_entry(ui: &AppWindow, old_index: i32, new_index: i32) {
    if get_progressing() {
        toast_warn!(
            ui,
            tr("Can't switch to new entry. Please wait for finishing processing")
        );
        return;
    }

    if old_index >= 0 {
        let old_index = old_index as usize;
        let entry = store_transcribe_entries!(ui).row_data(old_index).unwrap();
        if entry.video_player_setting.is_playing {
            match entry.media_type {
                UIMediaType::Video => global_logic!(ui).invoke_video_player_stop(),
                UIMediaType::Audio => global_logic!(ui)
                    .invoke_audio_player_stop(entry.video_player_setting.current_time),
                _ => (),
            }
        }
    }

    let new_index = new_index as usize;
    let entry = store_transcribe_entries!(ui).row_data(new_index).unwrap();
    match entry.media_type {
        UIMediaType::Video => {
            update_audio_player_setting_when_switch(&ui, &entry);

            if entry.video_player_setting.img_width <= 0 {
                update_video_player_setting_when_switch(ui.as_weak(), &entry, new_index);
            }
        }
        UIMediaType::Audio => {
            update_audio_player_setting_when_switch(&ui, &entry);
        }
        _ => (),
    }

    global_store!(ui).set_selected_transcribe_sidebar_index(new_index as i32);
}

fn update_video_player_setting_when_switch(
    ui: Weak<AppWindow>,
    entry: &UITranscribeEntry,
    index: usize,
) {
    let id = entry.id.clone();
    let video_path = entry.file_path.clone();
    let file_path = config::cache_dir().join(format!("{id}.png"));

    if file_path.exists() {
        tokio::spawn(async move {
            let Ok(metadata) = ffmpeg::video_metadata(&video_path) else {
                return;
            };

            async_update_video_player_setting_when_switch(ui, id, file_path, index, metadata);
        });
    } else {
        tokio::spawn(async move {
            let Ok(metadata) = ffmpeg::video_metadata(&video_path) else {
                return;
            };

            let Ok(imgs) = ffmpeg::video_screenshots(&video_path, 1) else {
                return;
            };

            let Some(img) = imgs.first() else {
                return;
            };

            if let Err(e) = img.save(&file_path) {
                warn!("save {} failed. error: {e}", file_path.display());
                return;
            }

            async_update_video_player_setting_when_switch(ui, id, file_path, index, metadata);
        });
    }
}

fn async_update_video_player_setting_when_switch(
    ui: Weak<AppWindow>,
    id: SharedString,
    file_path: PathBuf,
    index: usize,
    metadata: VideoMetadata,
) {
    _ = slint::invoke_from_event_loop(move || {
        let ui = ui.unwrap();
        let selected_index = global_store!(ui).get_selected_transcribe_sidebar_index();
        if index != selected_index as usize {
            return;
        }

        match slint::Image::load_from_path(&file_path) {
            Ok(img) => {
                let mut entry = store_transcribe_entries!(ui).row_data(index).unwrap();
                if entry.id != id {
                    return;
                }

                entry.video_player_setting.img_width = img.size().width as i32;
                entry.video_player_setting.img_height = img.size().height as i32;
                entry.video_player_setting.img = img;
                entry.video_player_setting.end_time = metadata.duration as f32;
                store_transcribe_entries!(ui).set_row_data(index, entry);

                global_logic!(ui).invoke_toggle_update_video_player_flag();
            }
            Err(e) => {
                _ = fs::remove_file(&file_path);
                warn!("load {} failed. error: {e}", file_path.display());
            }
        }
    });
}

fn update_audio_player_setting_when_switch(ui: &AppWindow, entry: &UITranscribeEntry) {
    let (input_media_path, output_audio_path, output_audio_path_tmp) =
        get_convert_to_audio_paths(&entry);

    if !input_media_path.exists() {
        warn!("{} not exists", input_media_path.display());
        return;
    }

    let ui_weak = ui.as_weak();
    let id = entry.id.clone().to_string();
    let is_media_audio = entry.media_type == UIMediaType::Audio;

    if is_media_audio && entry.video_player_setting.end_time <= 0.0 && output_audio_path.exists() {
        let audio_path = output_audio_path.as_path().to_string_lossy().to_string();
        let (ui, id) = (ui_weak.clone(), id.clone());
        tokio::spawn(async move {
            async_update_audio_player_setting_when_switch(ui, id, audio_path);
        });
    }

    if !output_audio_path.exists() {
        tokio::spawn(async move {
            set_progressing(true);

            convert_to_whisper_compatible_audio(
                ui_weak.clone(),
                id.clone(),
                &input_media_path,
                &output_audio_path,
                &output_audio_path_tmp,
            );

            set_progressing(false);

            if is_media_audio && output_audio_path.exists() {
                async_update_audio_player_setting_when_switch(
                    ui_weak.clone(),
                    id,
                    output_audio_path.as_path().to_string_lossy().to_string(),
                );
            }
        });
    }
}

fn async_update_audio_player_setting_when_switch(
    ui: Weak<AppWindow>,
    id: String,
    audio_path: String,
) {
    match media_duration(&audio_path, MediaType::Audio) {
        Some(duration) => {
            _ = slint::invoke_from_event_loop(move || {
                let ui = ui.unwrap();
                let index = global_store!(ui).get_selected_transcribe_sidebar_index() as usize;
                let mut entry = global_logic!(ui).invoke_current_transcribe_entry();

                if entry.id != id {
                    return;
                }

                entry.video_player_setting.end_time = duration as f32;
                entry.video_player_setting.is_playing = false;

                store_transcribe_entries!(ui).set_row_data(index, entry);
                global_logic!(ui).invoke_toggle_update_audio_player_flag();
            });
        }
        _ => (),
    }
}

fn start_transcribe(ui: &AppWindow, entry: UITranscribeEntry) {
    let ui_weak = ui.as_weak();
    let id = entry.id.to_string();

    let Some(lang) = WhisperLang::get_code_from_long_name(&entry.lang) else {
        toast_warn!(
            ui,
            format!("{}: {}", tr("Unsupport whisper language"), entry.lang)
        );
        return;
    };

    let Some((model_path, input_media_path, output_audio_path, output_audio_path_tmp)) =
        velify_transcribe_files(ui, &entry)
    else {
        return;
    };

    let index = global_store!(ui).get_selected_transcribe_sidebar_index();
    store_transcribe_subtitle_entries!(entry).set_vec(vec![]);
    store_transcribe_entries!(ui).set_row_data(index as usize, entry.clone());
    update_db_entry(ui, entry.into());

    tokio::spawn(async move {
        set_progressing(true);
        set_progress_cancel_signal(false);

        if !output_audio_path.exists()
            && !convert_to_whisper_compatible_audio(
                ui_weak.clone(),
                id.clone(),
                &input_media_path,
                &output_audio_path,
                &output_audio_path_tmp,
            )
        {
            set_progressing(false);
            return;
        }

        if !progress_cancelled() {
            transcribe(ui_weak, id, &model_path, &output_audio_path, lang).await;
        }

        set_progressing(false);
    });
}

fn velify_transcribe_files(
    ui: &AppWindow,
    entry: &UITranscribeEntry,
) -> Option<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    let model_path = super::model::get_model_path(&ui, &entry.model_name);
    let input_media_path = PathBuf::from_str(&entry.file_path).unwrap();
    let output_audio_path = config::cache_dir().join(format!("{}.wav", entry.id));
    let output_audio_path_tmp = config::cache_dir().join(format!("{}.tmp.wav", entry.id));

    if model_path.is_none() {
        toast_warn!(ui, tr("Can't find modle"));
        return None;
    }

    let model_path = PathBuf::from_str(&model_path.unwrap()).unwrap();
    if !model_path.exists() {
        toast_warn!(
            ui,
            format!("{}: {}", tr("Can't find modle"), model_path.display())
        );
        return None;
    }

    if !input_media_path.exists() {
        toast_warn!(
            ui,
            format!("{}: {}", tr("Can't find file"), input_media_path.display())
        );
        return None;
    }

    debug!("model_path: {}", model_path.display());
    debug!("input_media_path: {}", input_media_path.display());
    debug!("output_audio_path: {}", output_audio_path.display());
    debug!("output_audio_path_tmp: {}", output_audio_path_tmp.display());

    Some((
        model_path,
        input_media_path,
        output_audio_path,
        output_audio_path_tmp,
    ))
}

fn get_convert_to_audio_paths(entry: &UITranscribeEntry) -> (PathBuf, PathBuf, PathBuf) {
    let input_media_path = PathBuf::from_str(&entry.file_path).unwrap();
    let output_audio_path = config::cache_dir().join(format!("{}.wav", entry.id));
    let output_audio_path_tmp = config::cache_dir().join(format!("{}.tmp.wav", entry.id));

    (input_media_path, output_audio_path, output_audio_path_tmp)
}

fn convert_to_whisper_compatible_audio(
    ui_weak: Weak<AppWindow>,
    id: String,
    input_media_path: &PathBuf,
    output_audio_path: &PathBuf,
    output_audio_path_tmp: &PathBuf,
) -> bool {
    debug!("Convert to whisper compatible audio file...");

    let (ui, id_duplicate) = (ui_weak.clone(), id.clone());
    _ = slint::invoke_from_event_loop(move || {
        let ui = ui.unwrap();
        update_progress(&ui, id_duplicate, Some(ProgressType::ConvertToAduio), 0.0);
    });

    let ui_cb = ui_weak.clone();
    match transcribe::whisper::convert_to_compatible_audio(
        &input_media_path,
        &output_audio_path_tmp,
        get_progress_cancel_signal(),
        move |v| {
            debug!("convert to auido progress: {v}%");

            let ui = ui_cb.clone();
            _ = slint::invoke_from_event_loop(move || {
                let ui = ui.unwrap();
                let id = global_logic!(ui)
                    .invoke_current_transcribe_entry()
                    .id
                    .into();

                update_progress(&ui, id, None, v as f32 / 100.0);
            });
        },
    ) {
        Err(e) => {
            toast::async_toast_warn(ui_weak.clone(), e.to_string());
            return false;
        }
        _ => {
            if progress_cancelled() {
                toast::async_toast_info(
                    ui_weak.clone(),
                    tr("Cancelled converting to whisper compatible audio"),
                );
                return false;
            } else {
                _ = fs::rename(&output_audio_path_tmp, &output_audio_path);

                let (ui, id_duplicate) = (ui_weak.clone(), id.clone());
                _ = slint::invoke_from_event_loop(move || {
                    let ui = ui.unwrap();
                    update_progress(
                        &ui,
                        id_duplicate,
                        Some(ProgressType::ConvertToAduioFinished),
                        1.0,
                    );
                });
            }
        }
    }

    true
}

async fn transcribe(
    ui_weak: Weak<AppWindow>,
    id: String,
    model_path: &PathBuf,
    audio_path: &PathBuf,
    lang: String,
) {
    debug!("start transcribe. lang: {lang}");

    let (ui, id_duplicate) = (ui_weak.clone(), id.clone());
    _ = slint::invoke_from_event_loop(move || {
        let ui = ui.unwrap();
        update_progress(&ui, id_duplicate, Some(ProgressType::Transcribe), 0.0);
    });

    let config = transcribe::whisper::WhisperConfig::new(model_path).with_language(lang);

    let (ui_progress, ui_segement) = (ui_weak.clone(), ui_weak.clone());
    match transcribe::whisper::transcribe_file(
        config,
        &audio_path,
        move |v: i32| {
            debug!("whisper transcribe progress: {v}");

            let ui = ui_progress.clone();
            _ = slint::invoke_from_event_loop(move || {
                let ui = ui.unwrap();
                let id = global_logic!(ui)
                    .invoke_current_transcribe_entry()
                    .id
                    .into();
                update_progress(&ui, id, None, v as f32 / 100.0);
            });
        },
        move |segment: SegmentCallbackData| {
            let ui = ui_segement.clone();
            let segment: Subtitle = segment.into();

            _ = slint::invoke_from_event_loop(move || {
                let ui = ui.unwrap();
                let entry = global_logic!(ui).invoke_current_transcribe_entry();

                #[cfg(debug_assertions)]
                {
                    let contents = transcribe::subtitle::subtitle_to_srt(&segment);
                    println!("{contents}\n");
                }

                store_transcribe_subtitle_entries!(entry).push(segment.into());
            });
        },
        || progress_cancelled(),
    )
    .await
    {
        Ok(_) => {
            let (ui, id_duplicate) = (ui_weak.clone(), id.clone());
            _ = slint::invoke_from_event_loop(move || {
                let ui = ui.unwrap();
                update_progress(
                    &ui,
                    id_duplicate,
                    Some(ProgressType::TranscribeFinished),
                    1.0,
                );

                let entry = global_logic!(ui).invoke_current_transcribe_entry();
                update_db_entry(&ui, entry.into());
            });
        }
        Err(e) => {
            if !progress_cancelled() {
                toast::async_toast_warn(ui_weak.clone(), e.to_string());
            } else {
                toast::async_toast_info(ui_weak.clone(), tr("Cancelled transcribing"));
            }
        }
    }
}

fn cancel_progress(ui: &AppWindow, id: SharedString, ty: ProgressType) {
    set_progress_cancel_signal(true);

    if ty == ProgressType::Translate || ty == ProgressType::Correct {
        if let Some(abort_handles) = get_partial_abort_handles() {
            for handle in abort_handles.into_iter() {
                handle.abort();
            }
        }
    }

    for (index, mut entry) in store_transcribe_entries!(ui).iter().enumerate() {
        if entry.id == id {
            entry.progress_type = ProgressType::None;
            entry.progress = 0.0;
            store_transcribe_entries!(ui).set_row_data(index, entry);
            global_logic!(ui).invoke_toggle_update_transcribe_flag();

            return;
        }
    }
}

fn import_media_file(ui: &AppWindow) {
    let ui = ui.as_weak();
    tokio::spawn(async move {
        let Some(media_file) = picker_file(ui.clone(), &tr("Choose a media file")) else {
            return;
        };

        let Some(media_type) = media_type(ui.clone(), &media_file) else {
            return;
        };

        debug!("import {}", media_file.display());

        _ = slint::invoke_from_event_loop(move || {
            let ui = ui.unwrap();
            let index = global_store!(ui).get_selected_transcribe_sidebar_index();

            let mut entry = global_logic!(ui).invoke_current_transcribe_entry();
            entry.file_path = media_file.to_string_lossy().to_string().into();
            entry.is_file_exist = true;
            entry.media_type = media_type.into();
            store_transcribe_entries!(ui).set_row_data(index as usize, entry);

            global_logic!(ui).invoke_toggle_update_transcribe_flag();
            global_logic!(ui).invoke_toggle_update_video_player_flag();
        });
    });
}

fn export_subtitles(ui: &AppWindow, ty: String) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();
    let mut filename = cutil::fs::file_name_without_ext(&entry.file_path);
    filename.push_str(&format!(".{ty}"));

    let Some(items) = to_subtitles(ui) else {
        return;
    };

    let ui = ui.as_weak();
    tokio::spawn(async move {
        let Some(path) = picker_directory(ui.clone(), &tr("Export Subtitle"), &filename) else {
            return;
        };

        let path = path.join(filename);
        let ret = match ty.as_str() {
            "srt" => subtitle::save_as_srt(&items, path),
            "vtt" => subtitle::save_as_vtt(&items, path),
            "txt" => subtitle::save_as_txt(&items, path),
            _ => unreachable!("Unsupport subtitle type"),
        };

        match ret {
            Err(e) => toast::async_toast_warn(ui, format!("{}. {e}", "save subtitle failed")),
            _ => toast::async_toast_success(ui, tr("save subtitle successfully")),
        }
    });
}

fn export_video(ui: &AppWindow, setting: UIExportVideoSetting) {
    let Some(subtitles) = to_subtitles(&ui) else {
        return;
    };

    let subtitle_save_path = config::cache_dir().join(format!("{}.srt", setting.id));
    if let Err(e) = subtitle::save_as_srt(&subtitles, &subtitle_save_path) {
        toast_warn!(ui, format!("{}. {e}", tr("save subtitle failed.")));
        return;
    }

    let ui_weak = ui.as_weak();
    tokio::spawn(async move {
        let filename = cutil::fs::file_name(&setting.file_path);
        let Some(path) = picker_directory(ui_weak.clone(), &tr("Export Video"), "") else {
            return;
        };

        let adjust_volume_output_path = path.join(format!("adjust_volume_{filename}"));
        let add_subtitle_output_path = path.join(format!("output_{filename}"));
        let add_subtitle_input_path = if setting.is_adjust_volume {
            adjust_volume_output_path.clone()
        } else {
            PathBuf::from_str(&setting.file_path).unwrap()
        };

        set_progressing(true);
        set_progress_cancel_signal(false);

        if setting.is_adjust_volume
            && !adjust_normalized_voice(ui_weak.clone(), &setting, &adjust_volume_output_path)
        {
            set_progressing(false);
            _ = fs::remove_file(&adjust_volume_output_path);
            return;
        }

        if !progress_cancelled() {
            add_subtitle(
                ui_weak.clone(),
                &setting,
                &subtitle_save_path,
                &add_subtitle_input_path,
                &add_subtitle_output_path,
            );
        }

        set_progressing(false);
        _ = fs::remove_file(&adjust_volume_output_path);
    });
}

fn refresh_subtitles(ui: &AppWindow) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();
    let subtitles = store_transcribe_subtitle_entries!(entry)
        .iter()
        .collect::<Vec<UISubtitleEntry>>();

    store_transcribe_subtitle_entries!(entry).set_vec(subtitles);
    toast_success!(ui, tr("Refresh successfully"));
}

fn adjust_normalized_voice(
    ui_weak: Weak<AppWindow>,
    setting: &UIExportVideoSetting,
    output_path: &PathBuf,
) -> bool {
    let (ui, id) = (ui_weak.clone(), setting.id.clone().to_string());
    _ = slint::invoke_from_event_loop(move || {
        update_progress(&ui.unwrap(), id, Some(ProgressType::AdjustVoice), 0.0);
    });

    let ui_cb = ui_weak.clone();
    match ffmpeg::adjust_normalized_voice(
        &setting.file_path,
        &output_path,
        setting.adjust_volume_times,
        get_progress_cancel_signal(),
        move |v| {
            debug!("adjust normalized voice progress: {v}%");

            let ui = ui_cb.clone();
            _ = slint::invoke_from_event_loop(move || {
                let ui = ui.unwrap();
                let id = global_logic!(ui)
                    .invoke_current_transcribe_entry()
                    .id
                    .into();

                update_progress(&ui, id, None, v as f32 / 100.0);
            });
        },
    ) {
        Err(e) => {
            set_progressing(false);
            async_toast_warn(ui_weak.clone(), e.to_string());
            return false;
        }
        _ => {
            if progress_cancelled() {
                toast::async_toast_info(
                    ui_weak.clone(),
                    tr("Cancelled adjusting normalized voice"),
                );
                return false;
            } else {
                let (ui, id) = (ui_weak.clone(), setting.id.clone().to_string());
                _ = slint::invoke_from_event_loop(move || {
                    update_progress(
                        &ui.unwrap(),
                        id,
                        Some(ProgressType::AdjustVoiceFinished),
                        1.0,
                    );
                });
            }
        }
    }

    true
}

fn add_subtitle(
    ui_weak: Weak<AppWindow>,
    setting: &UIExportVideoSetting,
    subtitle_save_path: &PathBuf,
    input_path: &PathBuf,
    output_path: &PathBuf,
) -> bool {
    let config = SubtitleConfig::new(subtitle_save_path)
        .with_font_name(&setting.inner.font_name)
        .with_font_size((setting.inner.font_size as u32).max(1))
        .with_is_white_font_color(setting.inner.is_white_font_color)
        .with_enable_background(setting.inner.enable_background)
        .with_is_embedded(setting.is_embedded);

    let (ui, id) = (ui_weak.clone(), setting.id.clone().to_string());
    _ = slint::invoke_from_event_loop(move || {
        update_progress(&ui.unwrap(), id, Some(ProgressType::AddSubtitle), 0.0);
    });

    let ui_cb = ui_weak.clone();
    match ffmpeg::add_subtitle(
        &input_path,
        &output_path,
        config,
        get_progress_cancel_signal(),
        move |v| {
            trace!("adjust add embedded subtitle progress: {v}%");

            let ui = ui_cb.clone();
            _ = slint::invoke_from_event_loop(move || {
                let ui = ui.unwrap();
                let id = global_logic!(ui)
                    .invoke_current_transcribe_entry()
                    .id
                    .into();

                update_progress(&ui, id, None, v as f32 / 100.0);
            });
        },
    ) {
        Err(e) => {
            async_toast_warn(ui_weak.clone(), e.to_string());
            return false;
        }
        _ => {
            if progress_cancelled() {
                toast::async_toast_info(ui_weak.clone(), tr("Cancelled adding subtitle"));
                return false;
            } else {
                let (ui, id) = (ui_weak.clone(), setting.id.clone().to_string());
                _ = slint::invoke_from_event_loop(move || {
                    update_progress(
                        &ui.unwrap(),
                        id,
                        Some(ProgressType::AddSubtitleFinished),
                        1.0,
                    );
                });
            }
        }
    }

    true
}

fn ai_translate_all_subtitles(ui: &AppWindow, mut setting: UIAiHandleSubtitleSetting) {
    setting.prompt.push_str(
        r#"\n
<Input format>
["text1", "text2", "test3", ...]
</Input format>

<Output format>
["translation1", "translation2", "translation3", ...]
</Output format>
"#,
    );

    handle_partial_subtitle(&ui, setting);
}

fn ai_correct_all_subtitles(ui: &AppWindow, mut setting: UIAiHandleSubtitleSetting) {
    setting.prompt.push_str(
        r#"\n
<Input format>
["text1", "text2", "test3", ...]
</Input format>

<Output format>
["correction1", "correction2", "correction3", ...]
</Output format>
"#,
    );

    handle_partial_subtitle(&ui, setting);
}

fn handle_partial_subtitle(ui: &AppWindow, setting: UIAiHandleSubtitleSetting) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();

    let original_subtitles = Arc::new(
        store_transcribe_subtitle_entries!(entry)
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                if setting.ty == ProgressType::Translate && entry.translation_text.is_empty() {
                    Some((index, entry.original_text.into()))
                } else if setting.ty == ProgressType::Correct && entry.correction_text.is_empty() {
                    Some((index, entry.original_text.into()))
                } else {
                    None
                }
            })
            .collect::<Vec<(usize, String)>>(),
    );

    let original_subtitles_len = original_subtitles.len();
    if original_subtitles_len == 0 {
        toast_info!(ui, "Already handled all subtitles");
        return;
    }

    let ui_weak = ui.as_weak();
    update_progress(&ui, entry.id.to_string(), Some(setting.ty), 0.0);

    tokio::spawn(async move {
        set_progressing(true);
        let (tx, mut rx) = mpsc::channel(1024);
        let mut current_index = 0;
        let mut abort_handles = vec![];
        let valid_indexs = Arc::new(AtomicUsize::new(0));

        let original_subtitle_chunks =
            cutil::vec::chunk_with_merge(&original_subtitles, setting.chunk_size.max(1) as usize);

        for chunk in original_subtitle_chunks.into_iter() {
            let original_subtitles = original_subtitles.clone();
            let (ui, tx) = (ui_weak.clone(), tx.clone());
            let setting = setting.clone();
            let chunk_size = chunk.len();
            let start_index = current_index;
            current_index += chunk_size;

            let valid_indexs = valid_indexs.clone();
            let handle = tokio::spawn(async move {
                let subtitle_chunk = chunk
                    .into_iter()
                    .map(|item| item.1)
                    .collect::<Vec<String>>();

                let resp = ask_ai(&subtitle_chunk, &setting.prompt).await;

                match resp {
                    Err(e) => {
                        toast::async_toast_warn(
                            ui.clone(),
                            format!("{}. {e}", tr("Handle subtitles chunk failed")),
                        );
                    }
                    Ok(resp_items) => {
                        if chunk_size != resp_items.len() {
                            toast::async_toast_warn(
                                ui.clone(),
                                format!(
                                    "{} {}. {} {}",
                                    tr("Chunk size"),
                                    resp_items.len(),
                                    tr("Expect chunk size"),
                                    chunk_size
                                ),
                            );

                            return;
                        }

                        let progress_type = setting.ty.clone();
                        _ = slint::invoke_from_event_loop(move || {
                            let ui = ui.unwrap();
                            let entry = global_logic!(ui).invoke_current_transcribe_entry();
                            let mut ui_subtitles = store_transcribe_subtitle_entries!(entry)
                                .iter()
                                .collect::<Vec<UISubtitleEntry>>();

                            for (idx, original_subtitle) in original_subtitles
                                [start_index..start_index + chunk_size]
                                .iter()
                                .enumerate()
                            {
                                let index = original_subtitle.0;
                                if index >= ui_subtitles.len() {
                                    toast_warn!(
                                        ui,
                                        format!(
                                            "{} {}. {} {}",
                                            tr("Insert index"),
                                            index,
                                            tr("Expect index"),
                                            ui_subtitles.len()
                                        )
                                    );
                                    return;
                                }

                                if progress_type == ProgressType::Translate {
                                    ui_subtitles[index].translation_text =
                                        resp_items[idx].clone().into();
                                } else if progress_type == ProgressType::Correct {
                                    ui_subtitles[index].correction_text =
                                        resp_items[idx].clone().into();
                                } else {
                                    unreachable!();
                                };
                            }

                            store_transcribe_subtitle_entries!(entry).set_vec(ui_subtitles);

                            let valid_indexs =
                                valid_indexs.fetch_add(chunk_size, Ordering::Relaxed) + chunk_size;

                            let ty = if valid_indexs == original_subtitles_len {
                                if setting.ty == ProgressType::Translate {
                                    Some(ProgressType::TranslateFinished)
                                } else if setting.ty == ProgressType::Correct {
                                    Some(ProgressType::CorrectFinished)
                                } else {
                                    unreachable!();
                                }
                            } else {
                                None
                            };

                            let progress = valid_indexs as f32 / original_subtitles_len as f32;
                            update_progress(&ui, entry.id.to_string(), ty, progress);
                        });
                    }
                }

                _ = tx.send(()).await;
            });

            abort_handles.push(handle.abort_handle());
        }

        set_partial_abort_handles(abort_handles);
        drop(tx);

        while let Some(_) = rx.recv().await {}

        let ui = ui_weak.clone();
        _ = slint::invoke_from_event_loop(move || {
            let ui = ui.unwrap();
            let entry = global_logic!(ui).invoke_current_transcribe_entry();
            let valid_indexs = valid_indexs.load(Ordering::Relaxed);

            if valid_indexs != original_subtitles_len {
                update_progress(
                    &ui,
                    entry.id.to_string(),
                    Some(ProgressType::PartiallyFinished),
                    entry.progress,
                );
            }

            update_db_entry(&ui, entry.into());
        });

        set_progressing(false);
    });
}

async fn ask_ai(subtitles: &[String], prompt: &str) -> Result<Vec<String>> {
    let model_setting = config::model();
    if model_setting.api_key.is_empty()
        || model_setting.model_name.is_empty()
        || model_setting.api_base_url.is_empty()
    {
        return Err(anyhow!(tr("Please configure model setting firstly")));
    }

    debug!("prompt:\n{prompt}");

    let config = async_openai::config::OpenAIConfig::new()
        .with_api_key(&model_setting.api_key)
        .with_api_base(&model_setting.api_base_url);

    let client = Client::with_config(config);
    let user_message = serde_json::to_string(subtitles).unwrap();
    let request = CreateChatCompletionRequestArgs::default()
        .temperature(1.0)
        .model(model_setting.model_name)
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content(prompt)
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_message)
                .build()?
                .into(),
        ])
        .build()?;

    debug!("{}", serde_json::to_string(&request).unwrap());

    let response = client.chat().create(request).await?;

    let content = response
        .choices
        .iter()
        .next()
        .ok_or(anyhow!("No response content"))?
        .message
        .content
        .clone()
        .ok_or(anyhow!("No response content"))?;

    debug!("\nResponse:\n{}", content);

    if content.len() > 0 {
        Ok(serde_json::from_str::<Vec<String>>(&content)?)
    } else {
        return Err(anyhow!("No response content"));
    }
}

fn accept_all_corrected_subtitles(ui: &AppWindow) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();

    let subtitles = store_transcribe_subtitle_entries!(entry)
        .iter()
        .map(|mut entry| {
            if !entry.correction_text.is_empty() {
                entry.original_text = entry.correction_text.clone();
                entry.correction_text = Default::default();
            }

            entry
        })
        .collect::<Vec<UISubtitleEntry>>();

    store_transcribe_subtitle_entries!(entry).set_vec(subtitles);
    toast_success!(ui, tr("accept all subtitles successfully"));

    update_db_entry(&ui, entry.into());
}

fn remove_all_translated_subtitles(ui: &AppWindow) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();

    let subtitles = store_transcribe_subtitle_entries!(entry)
        .iter()
        .map(|mut entry| {
            entry.translation_text = SharedString::default();
            entry
        })
        .collect::<Vec<UISubtitleEntry>>();

    store_transcribe_subtitle_entries!(entry).set_vec(subtitles);
    toast_success!(ui, tr("remove all subtitles successfully"));

    update_db_entry(&ui, entry.into());
}

fn remove_all_corrected_subtitles(ui: &AppWindow) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();

    let subtitles = store_transcribe_subtitle_entries!(entry)
        .iter()
        .map(|mut entry| {
            entry.correction_text = SharedString::default();
            entry
        })
        .collect::<Vec<UISubtitleEntry>>();

    store_transcribe_subtitle_entries!(entry).set_vec(subtitles);
    toast_success!(ui, tr("remove all correction successfully"));

    update_db_entry(&ui, entry.into());
}

fn replace_subtitles_content(ui: &AppWindow, old_text: SharedString, new_text: SharedString) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();

    let subtitles = store_transcribe_subtitle_entries!(entry)
        .iter()
        .map(|mut entry| {
            entry.original_text = entry
                .original_text
                .replace(old_text.as_str(), new_text.as_str())
                .into();
            entry
        })
        .collect::<Vec<UISubtitleEntry>>();

    store_transcribe_subtitle_entries!(entry).set_vec(subtitles);
    toast_success!(ui, tr("replace content of subtitles successfully"));

    update_db_entry(&ui, entry.into());
}

fn replace_subtitles_all_separator(ui: &AppWindow) {
    let seps = [',', '，', '。'];
    let entry = global_logic!(ui).invoke_current_transcribe_entry();

    let subtitles = store_transcribe_subtitle_entries!(entry)
        .iter()
        .map(|mut entry| {
            entry.original_text =
                cutil::str::replace_multiple_chars(&entry.original_text, &seps, ' ').into();
            entry.translation_text =
                cutil::str::replace_multiple_chars(&entry.translation_text, &seps, ' ').into();
            entry
        })
        .collect::<Vec<UISubtitleEntry>>();

    store_transcribe_subtitle_entries!(entry).set_vec(subtitles);
    toast_success!(ui, tr("replace separators of subtitles successfully"));

    update_db_entry(&ui, entry.into());
}

fn traditional_to_simple_chinese(ui: &AppWindow) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();

    let subtitles = store_transcribe_subtitle_entries!(entry)
        .iter()
        .map(|mut entry| {
            entry.original_text =
                subtitle::convert_traditional_to_simplified_chinese(&entry.original_text).into();
            entry.translation_text =
                subtitle::convert_traditional_to_simplified_chinese(&entry.translation_text).into();
            entry
        })
        .collect::<Vec<UISubtitleEntry>>();

    store_transcribe_subtitle_entries!(entry).set_vec(subtitles);
    toast_success!(ui, tr("replace separators of subtitles successfully"));

    update_db_entry(&ui, entry.into());
}

fn swap_all_original_and_translation(ui: &AppWindow) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();

    let subtitles = store_transcribe_subtitle_entries!(entry)
        .iter()
        .map(|mut entry| {
            let original_text = entry.original_text.clone();
            entry.original_text = entry.translation_text;
            entry.translation_text = original_text;
            entry
        })
        .collect::<Vec<UISubtitleEntry>>();

    store_transcribe_subtitle_entries!(entry).set_vec(subtitles);
    toast_success!(ui, tr("swap original and translation successfully"));

    update_db_entry(&ui, entry.into());
}

fn remove_all_subtitles(ui: &AppWindow) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();
    store_transcribe_subtitle_entries!(entry).set_vec(vec![]);
    toast_success!(ui, tr("remove subtitles successfully"));
    update_db_entry(&ui, entry.into());
}

fn optimize_subtitles_timestamp(ui: &AppWindow) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();
    let id = entry.id.clone().to_string();
    let mut timestamps = vec![];

    for item in store_transcribe_subtitle_entries!(entry).iter() {
        let start_timestamp = transcribe::subtitle::srt_timestamp_to_ms(&item.start_timestamp);
        let end_timestamp = transcribe::subtitle::srt_timestamp_to_ms(&item.end_timestamp);

        if start_timestamp.is_err() || end_timestamp.is_err() {
            toast_warn!(
                ui,
                format!(
                    "{}: {} -> {}",
                    tr("invalid timestamp"),
                    item.start_timestamp,
                    item.end_timestamp
                )
            );
        }

        timestamps.push((start_timestamp.unwrap(), end_timestamp.unwrap()));
    }

    if timestamps.is_empty() {
        return;
    }

    // debug!("{timestamps:?}");

    let audio_path = config::cache_dir().join(format!("{id}.wav"));
    if !audio_path.exists() {
        toast_warn!(ui, format!("{} {}", tr("no found"), audio_path.display()));
        return;
    }

    update_progress(ui, id.clone(), Some(ProgressType::OptimizeTimestamp), 0.0);

    let ui_weak = ui.as_weak();

    tokio::spawn(async move {
        let (ui_weak_duplicate, id_duplicate) = (ui_weak.clone(), id.clone());
        match transcribe::vad::trim_start_slient_duration_of_audio(
            &audio_path,
            &timestamps,
            0.5,
            get_progress_cancel_signal(),
            move |v| {
                let (ui_weak, id_duplicate) = (ui_weak_duplicate.clone(), id_duplicate.clone());
                _ = slint::invoke_from_event_loop(move || {
                    update_progress(
                        &ui_weak.clone().unwrap(),
                        id_duplicate.clone(),
                        None,
                        v as f32 / 100.0,
                    );
                });
            },
        ) {
            Ok((optimize_timestamps, status)) => {
                let (ui_weak, id) = (ui_weak.clone(), id.clone());
                _ = slint::invoke_from_event_loop(move || {
                    let ui = ui_weak.unwrap();
                    let entry = global_logic!(ui).invoke_current_transcribe_entry();
                    let counts = store_transcribe_subtitle_entries!(entry).row_count();

                    if optimize_timestamps.len() != counts {
                        toast_warn!(
                            &ui,
                            format!(
                                "{}. {}: {} {}: {}",
                                tr("optimize timestamps failed."),
                                tr("expect"),
                                counts,
                                tr("found"),
                                optimize_timestamps.len()
                            )
                        );
                        update_progress(
                            &ui,
                            id.clone(),
                            Some(ProgressType::PartiallyFinished),
                            entry.progress,
                        );

                        return;
                    }

                    match status {
                        transcribe::ProgressStatus::Finished => {
                            let entry = global_logic!(ui).invoke_current_transcribe_entry();
                            let subtitles = store_transcribe_subtitle_entries!(entry)
                                .iter()
                                .enumerate()
                                .map(|(index, mut item)| {
                                    if item.start_timestamp_cache.is_empty() {
                                        item.start_timestamp_cache = item.start_timestamp.clone();
                                        item.end_timestamp_cache = item.end_timestamp.clone();
                                    }

                                    item.start_timestamp =
                                        transcribe::subtitle::ms_to_srt_timestamp(
                                            optimize_timestamps[index].0,
                                        )
                                        .into();

                                    item.end_timestamp = transcribe::subtitle::ms_to_srt_timestamp(
                                        optimize_timestamps[index].1,
                                    )
                                    .into();

                                    item
                                })
                                .collect::<Vec<UISubtitleEntry>>();

                            store_transcribe_subtitle_entries!(entry).set_vec(subtitles);
                            update_db_entry(&ui, entry.into());

                            update_progress(
                                &ui,
                                id.clone(),
                                Some(ProgressType::OptimizeTimestampFinished),
                                1.0,
                            );
                        }
                        transcribe::ProgressStatus::Cancelled => {
                            update_progress(&ui, id.clone(), Some(ProgressType::None), 0.0)
                        }
                    }
                });
            }
            Err(e) => {
                let (ui_weak, id) = (ui_weak.clone(), id.clone());
                _ = slint::invoke_from_event_loop(move || {
                    let ui = ui_weak.unwrap();
                    update_progress(&ui, id.clone(), Some(ProgressType::None), 0.0);
                    toast_warn!(ui, format!("{}. {e}", tr("optimize timestamps failed")));
                })
            }
        }
    });
}

fn recover_subtitles_timestamp(ui: &AppWindow) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();
    let subtitles = store_transcribe_subtitle_entries!(entry)
        .iter()
        .map(|mut item| {
            if !item.start_timestamp_cache.is_empty() && !item.end_timestamp_cache.is_empty() {
                item.start_timestamp = item.start_timestamp_cache.clone();
                item.end_timestamp = item.end_timestamp_cache.clone();
                item.start_timestamp_cache = SharedString::default();
                item.end_timestamp_cache = SharedString::default();
            }

            item
        })
        .collect::<Vec<UISubtitleEntry>>();

    store_transcribe_subtitle_entries!(entry).set_vec(subtitles);
    update_db_entry(&ui, entry.into());
}

fn split_subtitle(ui: &AppWindow, index: usize) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();
    let subtitles_len = store_transcribe_subtitle_entries!(entry).row_count();
    let subtitle = store_transcribe_subtitle_entries!(entry)
        .row_data(index)
        .unwrap();

    let start_timestamp_ms = transcribe::subtitle::srt_timestamp_to_ms(&subtitle.start_timestamp);
    let end_timestamp_ms = transcribe::subtitle::srt_timestamp_to_ms(&subtitle.end_timestamp);
    if start_timestamp_ms.is_err() || end_timestamp_ms.is_err() {
        toast_warn!(
            ui,
            format!(
                "{}. {} -> {}",
                tr("invalid timestamp"),
                subtitle.start_timestamp,
                subtitle.end_timestamp
            )
        );
        return;
    }

    let Some((first_part, second_part)) = transcribe::subtitle::split_subtitle_into_two(
        start_timestamp_ms.unwrap(),
        end_timestamp_ms.unwrap(),
        &subtitle.original_text,
    ) else {
        toast_warn!(ui, tr("split subtitle failed"));
        return;
    };

    let current_subtitle = UISubtitleEntry {
        start_timestamp: transcribe::subtitle::ms_to_srt_timestamp(first_part.0).into(),
        end_timestamp: transcribe::subtitle::ms_to_srt_timestamp(first_part.1).into(),
        original_text: first_part.2.into(),
        ..Default::default()
    };

    let next_subtitle = UISubtitleEntry {
        start_timestamp: transcribe::subtitle::ms_to_srt_timestamp(second_part.0).into(),
        end_timestamp: transcribe::subtitle::ms_to_srt_timestamp(second_part.1).into(),
        original_text: second_part.2.into(),
        ..Default::default()
    };

    store_transcribe_subtitle_entries!(entry).set_row_data(index, current_subtitle);
    if index == subtitles_len - 1 {
        store_transcribe_subtitle_entries!(entry).push(next_subtitle);
    } else {
        store_transcribe_subtitle_entries!(entry).insert(index + 1, next_subtitle);
    }

    update_db_entry(&ui, entry.into());
}

fn merge_above_subtitle(ui: &AppWindow, index: usize) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();
    let subtitles_len = store_transcribe_subtitle_entries!(entry).row_count();

    if subtitles_len < 2 || index == 0 {
        return;
    }

    let mut prev_subtitle = store_transcribe_subtitle_entries!(entry)
        .row_data(index - 1)
        .unwrap();

    let current_subtitle = store_transcribe_subtitle_entries!(entry)
        .row_data(index)
        .unwrap();

    prev_subtitle.end_timestamp = current_subtitle.end_timestamp.clone();
    prev_subtitle.end_timestamp_cache = current_subtitle.end_timestamp.clone();

    if !prev_subtitle.original_text.is_empty() {
        prev_subtitle.original_text.push_str(" ");
    }
    prev_subtitle
        .original_text
        .push_str(&current_subtitle.original_text);

    if !prev_subtitle.correction_text.is_empty() {
        prev_subtitle.correction_text.push_str(" ");
    }
    prev_subtitle
        .correction_text
        .push_str(&current_subtitle.correction_text);

    if !prev_subtitle.translation_text.is_empty() {
        prev_subtitle.translation_text.push_str(" ");
    }
    prev_subtitle
        .translation_text
        .push_str(&current_subtitle.translation_text);

    store_transcribe_subtitle_entries!(entry).set_row_data(index - 1, prev_subtitle);
    store_transcribe_subtitle_entries!(entry).remove(index);

    update_db_entry(&ui, entry.into());
}

fn insert_above_subtitle(ui: &AppWindow, index: usize) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();
    let subtitles_len = store_transcribe_subtitle_entries!(entry).row_count();

    if index >= subtitles_len || subtitles_len == 0 {
        return;
    }

    if index == 0 {
        let first_subtitle = store_transcribe_subtitle_entries!(entry)
            .row_data(0)
            .unwrap();

        let subtitle = UISubtitleEntry {
            start_timestamp: transcribe::subtitle::ms_to_srt_timestamp(0).into(),
            end_timestamp: first_subtitle.start_timestamp,
            original_text: tr("click and edit").into(),
            ..Default::default()
        };

        store_transcribe_subtitle_entries!(entry).insert(index, subtitle);
    } else {
        let prev_subtitle = store_transcribe_subtitle_entries!(entry)
            .row_data(index - 1)
            .unwrap();
        let next_subtitle = store_transcribe_subtitle_entries!(entry)
            .row_data(index)
            .unwrap();

        let subtitle = UISubtitleEntry {
            start_timestamp: prev_subtitle.end_timestamp,
            end_timestamp: next_subtitle.start_timestamp,
            original_text: tr("click and edit").into(),
            ..Default::default()
        };

        store_transcribe_subtitle_entries!(entry).insert(index, subtitle);
    }
}

fn insert_below_subtitle(ui: &AppWindow, index: usize) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();
    let subtitles_len = store_transcribe_subtitle_entries!(entry).row_count();

    if index >= subtitles_len || subtitles_len == 0 {
        return;
    }

    if index == subtitles_len - 1 {
        let last_subtitle = store_transcribe_subtitle_entries!(entry)
            .row_data(subtitles_len - 1)
            .unwrap();

        let subtitle = UISubtitleEntry {
            start_timestamp: last_subtitle.end_timestamp.clone(),
            end_timestamp: last_subtitle.end_timestamp,
            original_text: tr("click and edit").into(),
            ..Default::default()
        };

        store_transcribe_subtitle_entries!(entry).push(subtitle);
    } else {
        let prev_subtitle = store_transcribe_subtitle_entries!(entry)
            .row_data(index)
            .unwrap();
        let next_subtitle = store_transcribe_subtitle_entries!(entry)
            .row_data(index + 1)
            .unwrap();

        let subtitle = UISubtitleEntry {
            start_timestamp: prev_subtitle.end_timestamp,
            end_timestamp: next_subtitle.start_timestamp,
            original_text: tr("click and edit").into(),
            ..Default::default()
        };

        store_transcribe_subtitle_entries!(entry).insert(index + 1, subtitle);
    }
}

fn remove_subtitle(ui: &AppWindow, index: usize) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();
    let subtitles_len = store_transcribe_subtitle_entries!(entry).row_count();

    if index >= subtitles_len || subtitles_len == 0 {
        return;
    }

    store_transcribe_subtitle_entries!(entry).remove(index);
    toast_success!(ui, tr("remove subtitle successfully"));

    update_db_entry(&ui, entry.into());
}

fn save_subtitle(ui: &AppWindow, index: usize, subtitle: UISubtitleEntry) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();
    let subtitles_len = store_transcribe_subtitle_entries!(entry).row_count();

    if index >= subtitles_len || subtitles_len == 0 {
        return;
    }

    store_transcribe_subtitle_entries!(entry).set_row_data(index, subtitle);
    toast_success!(ui, tr("save subtitle successfully"));

    update_db_entry(&ui, entry.into());
}

fn reject_subtitle_correction(ui: &AppWindow, index: usize) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();
    let subtitles_len = store_transcribe_subtitle_entries!(entry).row_count();

    if index >= subtitles_len || subtitles_len == 0 {
        return;
    }

    let mut subtitle = store_transcribe_subtitle_entries!(entry)
        .row_data(index)
        .unwrap();
    subtitle.correction_text = Default::default();
    store_transcribe_subtitle_entries!(entry).set_row_data(index, subtitle);
}

fn accept_subtitle_correction(ui: &AppWindow, index: usize) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();
    let subtitles_len = store_transcribe_subtitle_entries!(entry).row_count();

    if index >= subtitles_len || subtitles_len == 0 {
        return;
    }

    let mut subtitle = store_transcribe_subtitle_entries!(entry)
        .row_data(index)
        .unwrap();

    subtitle.original_text = subtitle.correction_text.clone();
    subtitle.correction_text = Default::default();
    store_transcribe_subtitle_entries!(entry).set_row_data(index, subtitle);

    update_db_entry(&ui, entry.into());
}

fn video_player_start(ui: &AppWindow, timestamp: f32, duration: Option<f32>) {
    let entry = global_logic!(ui).invoke_current_transcribe_entry();
    let path = entry.file_path.to_string();

    if !PathBuf::from_str(&path).unwrap_or_default().exists() {
        toast_warn!(ui, format!("{} {}", tr("No found"), &path));
        return;
    }

    let ui_weak = ui.as_weak();
    tokio::spawn(async move {
        let metadata = match ffmpeg::video_metadata(&path) {
            Ok(metadata) => metadata,
            Err(e) => {
                toast::async_toast_warn(
                    ui_weak.clone(),
                    format!("{}. {e}", tr("get video metadata failed")),
                );
                return;
            }
        };

        debug!("{metadata:?}");

        let media_num = MEDIA_INC_NUM.load(Ordering::Relaxed);
        let config = VideoFramesIterConfig::default()
            .with_offset_ms((timestamp * 1000.0) as u64)
            .with_resolution(if metadata.height > 480 {
                VideoResolution::P480
            } else {
                VideoResolution::Origin
            })
            .with_fps(metadata.fps);

        let config = if let Some(duration) = duration {
            config.with_duration_ms((duration * 1000.0) as u64)
        } else {
            config
        };

        // FIXME: low efficiency
        match ffmpeg::video_frames_iter(
            &path,
            config,
            get_video_player_cancel_signal(),
            |img, inner_timestamp, inner_index| {
                if MEDIA_INC_NUM.load(Ordering::Relaxed) != media_num || !video_player_is_playing()
                {
                    return;
                }

                let ui = ui_weak.clone();
                _ = slint::invoke_from_event_loop(move || {
                    if MEDIA_INC_NUM.load(Ordering::Relaxed) != media_num
                        || !video_player_is_playing()
                    {
                        return;
                    }

                    let ui = ui.unwrap();
                    let index = global_store!(ui).get_selected_transcribe_sidebar_index() as usize;
                    let mut entry = global_logic!(ui).invoke_current_transcribe_entry();

                    if inner_index == 0 {
                        let audio_path = config::cache_dir().join(format!("{}.wav", &entry.id));
                        info!("start play audio: {}", audio_path.display());
                        play_audio(&ui, &audio_path);
                        seek_audio(timestamp);
                        set_audio_volume(entry.video_player_setting.volume);
                    }

                    let buffer = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::clone_from_slice(
                        img.as_raw(),
                        img.width(),
                        img.height(),
                    );

                    entry.video_player_setting.img = slint::Image::from_rgb8(buffer);
                    entry.video_player_setting.img_width = metadata.width as i32;
                    entry.video_player_setting.img_height = metadata.height as i32;
                    entry.video_player_setting.current_time = timestamp + inner_timestamp;
                    entry.video_player_setting.end_time = metadata.duration as f32;
                    entry.video_player_setting.is_playing = true;

                    store_transcribe_entries!(ui).set_row_data(index, entry);
                    global_logic!(ui).invoke_toggle_update_video_player_flag();
                })
            },
        ) {
            Err(e) => toast::async_toast_warn(
                ui_weak.clone(),
                format!("{}. {e}", tr("play video frames failed")),
            ),
            Ok(status) => {
                if MEDIA_INC_NUM.load(Ordering::Relaxed) != media_num {
                    return;
                }

                if matches!(status, VideoExitStatus::Finished) {
                    if duration.is_none() {
                        drop_audio_player_handle();
                    } else {
                        stop_audio();
                    }
                }

                let ui = ui_weak.clone();
                _ = slint::invoke_from_event_loop(move || {
                    global_logic!(&ui.unwrap()).invoke_video_player_stop();
                });
            }
        }
    });
}

fn video_player_partial_play(ui: &AppWindow, start_timestamp: f32, end_timestamp: f32) {
    if start_timestamp >= end_timestamp {
        return;
    }

    let duration = end_timestamp - start_timestamp;
    video_player_start(ui, start_timestamp, Some(duration));
}

fn video_player_stop(ui: &AppWindow, update_ui: bool) {
    set_video_player_cancel_signal(true);
    MEDIA_INC_NUM.fetch_add(1, Ordering::Relaxed);

    stop_audio();

    if update_ui {
        let index = global_store!(ui).get_selected_transcribe_sidebar_index() as usize;
        let mut entry = global_logic!(ui).invoke_current_transcribe_entry();
        entry.video_player_setting.is_playing = false;

        store_transcribe_entries!(ui).set_row_data(index, entry);
        global_logic!(ui).invoke_toggle_update_video_player_flag();
    }
}

fn audio_player_start(ui: &AppWindow, timestamp: f32, segment_duration: Option<f32>) {
    let index = global_store!(ui).get_selected_transcribe_sidebar_index() as usize;
    let mut entry = global_logic!(ui).invoke_current_transcribe_entry();
    let (_, audio_path, _) = get_convert_to_audio_paths(&entry);

    if !audio_path.exists() {
        toast_warn!(ui, format!("{} {}", tr("No found"), audio_path.display()));
        return;
    }

    play_audio(&ui, audio_path);
    seek_audio(timestamp);
    set_audio_volume(entry.video_player_setting.volume);

    let (duration, audio_total_index) = if let Some(handle) = get_audio_player_handle() {
        let duration = handle.duration_seconds();
        let sample_rate = handle.sample_rate();
        (duration as f32, (duration * sample_rate as f64) as u64)
    } else {
        return;
    };

    update_audio_progress_background(ui.as_weak(), duration, audio_total_index, segment_duration);

    entry.video_player_setting.current_time = timestamp;
    entry.video_player_setting.end_time = duration;
    entry.video_player_setting.is_playing = true;

    store_transcribe_entries!(ui).set_row_data(index, entry);
    global_logic!(ui).invoke_toggle_update_audio_player_flag();
}

fn update_audio_progress_background(
    ui_weak: Weak<AppWindow>,
    duration: f32,
    audio_total_index: u64,
    segment_duration: Option<f32>,
) {
    let media_index = MEDIA_INC_NUM.load(Ordering::Relaxed);

    tokio::spawn(async move {
        let ms_step = 20;
        let mut segment_duration_ms = if let Some(segment_duration) = segment_duration {
            Some((segment_duration * 1000.0) as i64)
        } else {
            None
        };

        loop {
            tokio::time::sleep(tokio::time::Duration::from_millis(ms_step)).await;

            if media_index != MEDIA_INC_NUM.load(Ordering::Relaxed) {
                return;
            }

            let handle = match get_audio_player_handle() {
                Some(handle) => {
                    if handle.finished() {
                        async_update_audio_progress_background(
                            ui_weak.clone(),
                            media_index,
                            duration,
                            audio_total_index,
                        );
                        return;
                    }
                    handle
                }
                None => return,
            };

            if handle.paused() {
                segment_duration_ms = None;
            } else {
                if let Some(ms) = segment_duration_ms {
                    if ms <= 0 {
                        let ui_weak = ui_weak.clone();
                        _ = slint::invoke_from_event_loop(move || {
                            let ui = ui_weak.unwrap();
                            let setting = global_logic!(ui)
                                .invoke_current_transcribe_entry()
                                .video_player_setting;
                            global_logic!(ui).invoke_audio_player_stop(setting.current_time);
                        });

                        return;
                    } else {
                        segment_duration_ms = Some(ms - ms_step as i64);
                    }
                }
            }

            if !handle.paused() {
                async_update_audio_progress_background(
                    ui_weak.clone(),
                    media_index,
                    duration,
                    audio_total_index,
                );
            }
        }
    });
}

fn async_update_audio_progress_background(
    ui: Weak<AppWindow>,
    media_index: u64,
    duration: f32,
    audio_total_index: u64,
) {
    _ = slint::invoke_from_event_loop(move || {
        let ui = ui.unwrap();

        if media_index != MEDIA_INC_NUM.load(Ordering::Relaxed) {
            return;
        }

        let index = global_store!(ui).get_selected_transcribe_sidebar_index();
        if index < 0 {
            return;
        }

        let (current_time, is_playing) = match get_audio_player_handle() {
            Some(handle) => {
                if handle.finished() {
                    (duration, false)
                } else {
                    let current_time =
                        handle.index() as f64 / audio_total_index as f64 * duration as f64;
                    (current_time as f32, true)
                }
            }
            None => return,
        };

        let mut entry = global_logic!(ui).invoke_current_transcribe_entry();
        entry.video_player_setting.current_time = current_time;
        entry.video_player_setting.is_playing = is_playing;

        store_transcribe_entries!(ui).set_row_data(index as usize, entry);
        global_logic!(ui).invoke_toggle_update_audio_player_flag();
    });
}

fn audio_player_partial_play(ui: &AppWindow, start_timestamp: f32, end_timestamp: f32) {
    if start_timestamp >= end_timestamp {
        return;
    }

    audio_player_start(ui, start_timestamp, Some(end_timestamp - start_timestamp));
}

fn audio_player_stop(ui: &AppWindow, timestamp: f32) {
    stop_audio();
    MEDIA_INC_NUM.fetch_add(1, Ordering::Relaxed);

    let index = global_store!(ui).get_selected_transcribe_sidebar_index() as usize;
    let mut entry = global_logic!(ui).invoke_current_transcribe_entry();
    entry.video_player_setting.current_time = timestamp;
    entry.video_player_setting.is_playing = false;

    store_transcribe_entries!(ui).set_row_data(index, entry);
    global_logic!(ui).invoke_toggle_update_audio_player_flag();
}

fn before_change_audio_player_position(_ui: &AppWindow) {
    stop_audio();
}

fn change_audio_player_player_position(ui: &AppWindow, timestamp: f32) {
    global_logic!(ui).invoke_audio_player_start(timestamp);
}

fn get_current_subtitle(
    subtitles: ModelRc<UISubtitleEntry>,
    current_time: u64,
) -> ModelRc<SharedString> {
    let subtitles = subtitles
        .iter()
        .filter_map(|item| {
            let start_timestamp = transcribe::subtitle::srt_timestamp_to_ms(&item.start_timestamp);
            let end_timestamp = transcribe::subtitle::srt_timestamp_to_ms(&item.end_timestamp);

            let texts = if item.translation_text.is_empty() {
                [item.original_text.clone(), Default::default()]
            } else {
                [item.original_text.clone(), item.translation_text.clone()]
            };

            if start_timestamp.is_err() || end_timestamp.is_err() {
                None
            } else {
                Some((start_timestamp.unwrap(), end_timestamp.unwrap(), texts))
            }
        })
        .collect::<Vec<(u64, u64, [SharedString; 2])>>();

    let index = subtitles.partition_point(|(start, _, _)| *start <= current_time);
    if index > 0 {
        let (start, end, texts) = &subtitles[index - 1];
        if *start <= current_time && current_time <= *end {
            return ModelRc::new(VecModel::from_slice(texts));
        }
    }

    ModelRc::new(VecModel::from_slice(&vec![
        Default::default(),
        Default::default(),
    ]))
}

fn play_audio(ui: &AppWindow, path: impl AsRef<Path>) -> Option<SoundHandle> {
    stop_audio();
    drop_audio_player_handle();

    match Sound::from_path(path.as_ref()) {
        Ok(data) => {
            let mut mixer = Mixer::new();
            mixer.init();

            let handle = mixer.play(data);
            set_audio_player_handle(handle.clone());
            Some(handle)
        }
        Err(e) => {
            toast_warn!(
                ui,
                format!(
                    "{}. {}. {}",
                    tr("play aduio file fialed"),
                    path.as_ref().display(),
                    e
                )
            );
            return None;
        }
    }
}

fn stop_audio() {
    if let Some(handle) = get_audio_player_handle() {
        if !handle.paused() {
            handle.pause();
        }
    }
}

fn seek_audio(timestamp: f32) {
    if let Some(handle) = get_audio_player_handle() {
        handle.seek_to(timestamp as f64);
    }
}

fn set_audio_volume(v: f32) {
    if let Some(handle) = get_audio_player_handle() {
        handle.set_volume(v);
    }
}

fn get_progressing() -> bool {
    CACHE.lock().unwrap().progressing
}

fn set_progressing(v: bool) {
    let mut cache = CACHE.lock().unwrap();
    cache.progressing = v;
}

fn get_progress_cancel_signal() -> Arc<AtomicBool> {
    let cancel_sig = CACHE.lock().unwrap().progress_cancel_signal.clone();
    cancel_sig.store(false, Ordering::Relaxed);
    cancel_sig
}

fn set_progress_cancel_signal(v: bool) {
    CACHE
        .lock()
        .unwrap()
        .progress_cancel_signal
        .store(v, Ordering::Relaxed);
}

fn progress_cancelled() -> bool {
    CACHE
        .lock()
        .unwrap()
        .progress_cancel_signal
        .load(Ordering::Relaxed)
}

fn get_audio_player_handle() -> Option<SoundHandle> {
    CACHE.lock().unwrap().audio_player_handle.clone()
}

fn set_audio_player_handle(handle: SoundHandle) {
    let mut cache = CACHE.lock().unwrap();
    cache.audio_player_handle = Some(handle);
}

fn drop_audio_player_handle() {
    CACHE.lock().unwrap().audio_player_handle.take();
}

fn get_video_player_cancel_signal() -> Arc<AtomicBool> {
    let cancel_sig = CACHE.lock().unwrap().video_player_cancel_signal.clone();
    cancel_sig.store(false, Ordering::Relaxed);
    cancel_sig
}

fn set_video_player_cancel_signal(v: bool) {
    CACHE
        .lock()
        .unwrap()
        .video_player_cancel_signal
        .store(v, Ordering::Relaxed);
}

fn video_player_is_playing() -> bool {
    !CACHE
        .lock()
        .unwrap()
        .video_player_cancel_signal
        .load(Ordering::Relaxed)
}

fn get_partial_abort_handles() -> Option<Vec<AbortHandle>> {
    CACHE.lock().unwrap().partial_abort_handles.take()
}

fn set_partial_abort_handles(handles: Vec<AbortHandle>) {
    let mut cache = CACHE.lock().unwrap();
    cache.partial_abort_handles = Some(handles);
}

fn update_progress(ui: &AppWindow, id: String, ty: Option<ProgressType>, progress: f32) {
    if let Some(ty) = ty {
        global_logic!(ui).invoke_update_progress_type(id.clone().into(), ty);
    }

    global_logic!(ui).invoke_update_progress(id.into(), progress);
    global_logic!(ui).invoke_toggle_update_transcribe_flag();
}

fn to_subtitle(index: i32, entry: &UISubtitleEntry) -> Result<Subtitle> {
    Ok(Subtitle {
        index,
        start_timestamp: subtitle::srt_timestamp_to_ms(&entry.start_timestamp)?,
        end_timestamp: subtitle::srt_timestamp_to_ms(&entry.end_timestamp)?,
        text: if entry.translation_text.is_empty() {
            entry.original_text.to_string()
        } else {
            format!("{}\n{}", entry.original_text, entry.translation_text)
        },
    })
}

fn to_subtitles(ui: &AppWindow) -> Option<Vec<Subtitle>> {
    let mut items = vec![];
    let entry = global_logic!(ui).invoke_current_transcribe_entry();

    for (index, item) in store_transcribe_subtitle_entries!(entry).iter().enumerate() {
        match to_subtitle(index as i32 + 1, &item) {
            Ok(item) => items.push(item),
            Err(e) => {
                toast_warn!(ui, format!("{e}"));
                return None;
            }
        }
    }

    Some(items)
}

struct Cache {
    progressing: bool,
    partial_abort_handles: Option<Vec<AbortHandle>>,

    progress_cancel_signal: Arc<AtomicBool>,
    audio_player_handle: Option<SoundHandle>,
    video_player_cancel_signal: Arc<AtomicBool>,
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            progressing: false,
            partial_abort_handles: None,
            audio_player_handle: None,
            progress_cancel_signal: Arc::new(AtomicBool::new(false)),
            video_player_cancel_signal: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl From<Subtitle> for UISubtitleEntry {
    fn from(sub: Subtitle) -> Self {
        UISubtitleEntry {
            start_timestamp: transcribe::subtitle::ms_to_srt_timestamp(sub.start_timestamp).into(),
            end_timestamp: transcribe::subtitle::ms_to_srt_timestamp(sub.end_timestamp).into(),
            original_text: sub.text.into(),
            ..Default::default()
        }
    }
}
