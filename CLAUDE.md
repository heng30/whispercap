# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Whispercap is a cross-platform desktop application that uses the Whisper model to transcribe videos or audio into subtitles. Built with Rust and the Slint UI framework, it supports Linux, Windows, macOS, Android, and Web platforms.

## Architecture

### Workspace Structure
- **whispercap/**: Main application crate with desktop, mobile, and web targets
- **lib/**: Shared libraries:
  - **cutil/**: Common utilities (crypto, fs, http, time, etc.)
  - **ffmpeg/**: Audio/video processing utilities
  - **font/**: Font management
  - **sqldb/**: Database operations
  - **transcribe/**: Whisper model integration and subtitle processing
- **tr-helper/**: Translation helper tool

### UI Framework
- Uses **Slint** for cross-platform UI development
- UI files located in `whispercap/ui/` with `.slint` extension
- Three main window types: desktop, android, and web
- Component-based architecture with reusable UI components in `ui/base/`

### Platform Support
- **Desktop**: Windows, Linux, macOS (features: `desktop`, `database`, `qrcode`, `center-window`)
- **Mobile**: Android (features: `mobile`, `android`)
- **Web**: WebAssembly (features: `web`)

## Development Commands

### Building
```bash
# Desktop debug build
make desktop-debug

# Desktop release build
make desktop-build-release

# Android build
make android-build

# Web build
make web-build
```

### Testing & Quality
```bash
# Run tests
make test

# Run clippy
make clippy

# Check compilation
make check
```

### Development Tools
```bash
# Run Slint viewer for desktop UI
make slint-viewer-desktop

# Run Slint viewer for Android UI
make slint-viewer-android

# Run Slint viewer for web UI
make slint-viewer-web

# Run translation helper
make tr
```

### Packaging
```bash
# Create Debian package
make deb

# Package for different platforms
make packing-linux
make packing-windows
make packing-android
make packing-web
```

## Key Dependencies

- **slint**: UI framework
- **whisper-rs**: Whisper model integration
- **ffmpeg-sidecar**: Audio/video processing
- **async-openai**: AI-powered subtitle correction and translation
- **sqlx**: Database operations
- **kittyaudio**: Audio playback

## Configuration

- Uses **stable** Rust toolchain
- Platform-specific dependencies managed via Cargo features
- Android configuration in `whispercap/Cargo.toml` metadata
- Desktop platforms require ffmpeg installed in system path
- Linux requires Zenity or Kdialog for file dialogs

## UI Development

- Slint files use declarative syntax for UI definition
- UI logic separated into `logic.slint` and `store.slint`
- Components organized in `ui/base/` for reusability
- Platform-specific panels in `ui/panel/desktop/`, `ui/panel/mobile/`, `ui/panel/web/`

## Troubleshooting

- On Windows: Use Qt backend to resolve font rendering issues
- Ensure ffmpeg is installed and in system PATH
- Linux: Install Zenity or Kdialog for file dialogs
- Web: Requires wasm-pack for WebAssembly builds