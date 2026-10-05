# 灰泽满注入链路：Remotion 样片

这份工程把 [`docs/架构.md`](../docs/架构.md)（判别 → 检索 → 注入这条链路）做成 16:9 的视觉说明视频。

当前版本：

- `1920x1080`
- `30fps`
- `92s`
- 8 个场景
- 无配音也能渲染，画面内置逐句字幕

## 本地运行

```bash
cd video
npm install --registry=https://registry.npmmirror.com
npm run studio
```

渲染预览：

```bash
npm run render:preview
```

渲染 1080p：

```bash
npm run render
```

## 配音时间轴

旁白数据在 `src/narration.json`，每次修改后重新导出：

```bash
npm run captions
```

会生成：

```text
narration.csv
public/subtitles.srt
```

`narration.csv` 的每一行就是一条待合成语音：

```text
scene_id,line_id,start_ms,end_ms,duration_ms,text
hook,hook-1,0,3600,3600,你可能以为，机器人看到的是很多文件。
```

## 接入灰泽满 TTS

把每个 `line_id` 合成成同名文件，放进：

```text
video/public/audio/
```

例如：

```text
hook-1.wav
hook-2.wav
file-1.wav
...
```

然后渲染带配音版本：

```bash
npm run render:voice
```

`render:voice` 通过 `render-props-voice.json` 打开配音轨，避免 Windows
命令行对内联 JSON 引号的转义问题。

配乐、环境音或额外拼接仍然可以在这版无声视频之后继续做。

## 使用灰泽满 Speak 模型

模型目录已经有一组对应权重：

```text
GPT_weights_v2/HZM-SPEAK-e10.ckpt
SoVITS_weights_v2/HZM-SPEAK_e15_s525.pth
```

启动独立 API：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-speak-api.ps1
```

批量合成 `narration.csv` 中的 16 句：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/generate-voiceover.ps1
```

合成结果会放回 `public/audio/`，并生成 `voiceover-report.json`，
其中包含每句的实际音频时长和是否超出时间槽。

如果实际音频长于原时间槽，按真实音频顺延时间轴：

```powershell
npm run fit-voice
npm run captions
npm run render:voice
```
