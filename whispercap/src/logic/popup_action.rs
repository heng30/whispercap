use crate::{
    global_logic, global_store,
    logic::tr::tr,
    slint_generatedAppWindow::{AppWindow, ConfirmDialogSetting, PopupActionSetting},
};
use slint::{ComponentHandle, SharedString};

pub fn init(ui: &AppWindow) {
    let ui_handle = ui.as_weak();
    ui.global::<PopupActionSetting>()
        .on_action(move |action, user_data| {
            let ui = ui_handle.unwrap();

            #[allow(clippy::single_match)]
            match action.as_str() {
                "remove-caches" => {
                    global_logic!(ui).invoke_remove_caches();
                }

                // ============= trancribe sidebar ================ //
                "show-rename-transcribe-dialog" => {
                    let index = user_data.parse::<i32>().unwrap_or_default();
                    global_logic!(ui).invoke_show_rename_transcribe_dialog(index);
                }
                "remove-transcribe-entry" => {
                    let index = user_data.parse::<i32>().unwrap_or_default();
                    global_logic!(ui).invoke_remove_transcribe_entry(index);
                }

                // ============= trancribe ================ //
                "show-ai-handle-subtitle-setting-dialog" => {
                    global_logic!(ui).invoke_show_ai_handle_subtitle_setting_dialog(user_data);
                }
                "accept-all-corrected-subtitles" => {
                    global_logic!(ui).invoke_accept_all_corrected_subtitles();
                }
                "remove-all-corrected-subtitles" => {
                    ui.global::<ConfirmDialogSetting>().invoke_set(
                        true,
                        tr("Warning").into(),
                        tr("Remove all corrections or not?").into(),
                        "remove-all-corrected-subtitles".to_string().into(),
                        SharedString::default(),
                    );
                }
                "remove-all-translated-subtitles" => {
                    ui.global::<ConfirmDialogSetting>().invoke_set(
                        true,
                        tr("Warning").into(),
                        tr("Remove all translations or not?").into(),
                        "remove-all-translated-subtitles".to_string().into(),
                        SharedString::default(),
                    );
                }
                "show-replace-subtitles-content-dialog" => {
                    global_logic!(ui).invoke_show_replace_subtitles_content_dialog();
                }
                "subtitles-to-lowercase" => {
                    global_logic!(ui).invoke_subtitles_to_lowercase();
                }
                "replace-subtitles-all-separator" => {
                    global_logic!(ui).invoke_replace_subtitles_all_separator();
                }
                "traditional-to-simple-chinese" => {
                    global_logic!(ui).invoke_traditional_to_simple_chinese();
                }
                "swap-all-original-and-translation" => {
                    global_logic!(ui).invoke_swap_all_original_and_translation();
                }
                "remove-all-subtitles" => {
                    ui.global::<ConfirmDialogSetting>().invoke_set(
                        true,
                        tr("Warning").into(),
                        tr("Remove all subtitles or not?").into(),
                        "remove-all-subtitles".to_string().into(),
                        SharedString::default(),
                    );
                }
                "optimize-subtitles-timestamp" => {
                    global_logic!(ui).invoke_optimize_subtitles_timestamp();
                }
                "recover-subtitles-timestamp" => {
                    global_logic!(ui).invoke_recover_subtitles_timestamp();
                }
                "adjust-overlap-timestamp" => {
                    global_logic!(ui).invoke_adjust_overlap_timestamp();
                }

                // ============= subtitle entry ================ //
                "split-subtitle" => {
                    let index = user_data.parse::<i32>().unwrap_or_default();
                    global_logic!(ui).invoke_split_subtitle(index);
                }
                "merge-above-subtitle" => {
                    let index = user_data.parse::<i32>().unwrap_or_default();
                    global_logic!(ui).invoke_merge_above_subtitle(index);
                }
                "show-shift-subtitles-timestamp" => {
                    let index = user_data.parse::<i32>().unwrap_or_default();
                    global_store!(ui).set_subtitles_shift_timestamp_index(index);
                    global_logic!(ui)
                        .invoke_switch_popup(crate::PopupIndex::SubtitlesShiftTimestamp);
                }
                "insert-above-subtitle" => {
                    let index = user_data.parse::<i32>().unwrap_or_default();
                    global_logic!(ui).invoke_insert_above_subtitle(index);
                }
                "insert-below-subtitle" => {
                    let index = user_data.parse::<i32>().unwrap_or_default();
                    global_logic!(ui).invoke_insert_below_subtitle(index);
                }
                "remove-subtitle" => {
                    ui.global::<ConfirmDialogSetting>().invoke_set(
                        true,
                        tr("Warning").into(),
                        tr("Remove subtitle or not?").into(),
                        "remove-subtitle".to_string().into(),
                        user_data,
                    );
                }

                "download-model" => {
                    let model_name = user_data;
                    global_logic!(ui).invoke_download_model(model_name);
                }
                _ => (),
            }
        });
}
