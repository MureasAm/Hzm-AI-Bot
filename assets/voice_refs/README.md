# 灰泽满语音参考音频（GPT-SoVITS）

参考音频决定 bot 语音的**音色和语气**。**放一个 `ref_voice.wav` 就能跑**：

| 文件 | 作用 |
|---|---|
| `ref_voice.wav` | 主参考（必放）。灰泽满一段干净的说话声，3~10 秒 |
| `ref_voice.txt` | **同名的转写**：她在这段里说的话（一字不差，GPT-SoVITS 对齐用）。放音频时一起放 |
| `ref_happy.wav` / `ref_lazy.wav` / `ref_serious.wav`（可选） | 按回复情绪粗粒度切换的情绪档。缺档时自动回落 `ref_voice.wav` |

怎么换/加：把新音频拷进来、写个同名 `.txt` 转写即可，不用改代码。放好后再把 `.env.prod` 的 `VOICE_ENABLED` 改成 `1`，重启 bot 生效。

（2026-09 从 Hzm_Reading 训练切片 `output/slicer_opt/` 预置了一份：`ref_voice.wav` 5.5s，转写=“都高高兴兴的干,每天早上从剁,剁堆积如山的洋葱头开始.”——先用它验证链路，之后想换更贴聊天语气的素材再替换。）
