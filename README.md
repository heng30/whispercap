<div style="display: flex, margin: 8px">
    <img src="./screenshot/1-en.png"/>
    <img src="./screenshot/2-en.png"/>
</div>

[中文文档](./README.zh-CN.md)

[Introduction video](https://youtu.be/j8Qip68JvDI)

### Introduction
This is a desktop application that uses the Whisper model to transcribe videos or audio into subtitles. It supports Linux, Windows, and MacOS platforms.

### Features
- Transcribe audio and video into subtitles
- Download Whisper models
- Edit and modify subtitles
- AI-powered subtitle translation
- AI-assisted subtitle correction
- Play audio and video to assist in subtitle correction
- Export subtitles and video
- Adjust timestamps for audio waveform images
- Transcribe in chunks to prevent severe timestamp drift in long videos

### How to build?
- Install `Rust` and `Cargo`
- Run `make desktop-debug` to run it on desktop platform
- Run `make desktop-build-release` to build a release version desktop application
- Refer to [Makefile](./Makefile) for more information

### Troubleshooting
- Using the `Qt backend` can resolve the issue of fuzzy fonts on the Windows platform. It is also recommended to prioritize the `Qt backend` to maintain a consistent build environment with the developers.
- Since the program uses ffmpeg to handle audio and video format conversion, [ffmpeg](https://ffmpeg.org/) needs to be installed. On the `Windows` platform, you need to install `ffmpeg` to the **system path**.
- On the `Linux` platform, `Zenity` or `Kdialog` must be installed to open the file selection dialog.

### Reference
- [Slint Language Documentation](https://slint-ui.com/releases/1.0.0/docs/slint/)
- [slint::android](https://snapshots.slint.dev/master/docs/rust/slint/android/#building-and-deploying)
- [Running In A Browser Using WebAssembly](https://releases.slint.dev/1.7.0/docs/slint/src/quickstart/running_in_a_browser)
- [github/slint-ui](https://github.com/slint-ui/slint)
- [Viewer for Slint](https://github.com/slint-ui/slint/tree/master/tools/viewer)
- [LSP (Language Server Protocol) Server for Slint](https://github.com/slint-ui/slint/tree/master/tools/lsp)
- [developer.android.com](https://developer.android.com/guide)
- [How to Deploy Rust Binaries with GitHub Actions](https://dzfrias.dev/blog/deploy-rust-cross-platform-github-actions/)
