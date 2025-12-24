<div style="display: flex, margin: 8px">
    <img src="./screenshot/1-cn.png"/>
    <img src="./screenshot/2-cn.png"/>
</div>

[English Documentation](./README.md)

[介绍视频](https://www.bilibili.com/video/BV1N6nuzzE7X)

### 简介
这是一个桌面应用程序，使用**whisper模型**将视频或音频转录为字幕。支持`Linux`、`Windows`和`MacOS`平台。

### 功能
- 转录音频和视频为字幕
- 下载whisper模型
- 编辑和修改字幕
- AI翻译字幕
- AI矫正字幕
- 播放音频和视频辅助矫正字幕
- 导出字幕和视频
- 声波图像调整时间戳
- 分块转录，避免长视频时间戳严重漂移

### 如何构建?
- 安装 `Rust` 和 `Cargo`
- 运行 `make desktop-debug` 调试桌面平台程序
- 运行 `make desktop-build-release` 编译桌面平台程序
- 参考 [Makefile](./Makefile) 了解更多信息

### 问题排查
- 使用`Qt后端`能解决windows平台字体发虚的问题。也推荐优先使用`Qt后端`保持和开发者相同的构建环境
- 因为程序使用`ffmpeg`处理音频和视频格式转换，所以需要安装[ffmpeg](https://ffmpeg.org/)。`Windows`平台，需要将`ffmpeg`安装到**系统路径**
- `Linux`平台需要安装`Zenity`或者`Kdialog`，才能打开文件选择框

### 参考
- [Slint Language Documentation](https://slint-ui.com/releases/1.0.0/docs/slint/)
- [slint::android](https://snapshots.slint.dev/master/docs/rust/slint/android/#building-and-deploying)
- [Running In A Browser Using WebAssembly](https://releases.slint.dev/1.7.0/docs/slint/src/quickstart/running_in_a_browser)
- [github/slint-ui](https://github.com/slint-ui/slint)
- [Viewer for Slint](https://github.com/slint-ui/slint/tree/master/tools/viewer)
- [LSP (Language Server Protocol) Server for Slint](https://github.com/slint-ui/slint/tree/master/tools/lsp)
- [developer.android.com](https://developer.android.com/guide)
- [color4bg](https://www.color4bg.com/zh-hans/)
- [How to Deploy Rust Binaries with GitHub Actions](https://dzfrias.dev/blog/deploy-rust-cross-platform-github-actions/)
