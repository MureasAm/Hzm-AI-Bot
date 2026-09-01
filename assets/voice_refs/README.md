# 灰泽满语音参考音频（GPT-SoVITS）

语音情绪靠参考音频粗粒度切换（细粒度"小声/心虚"做不到，走文字）。放 3 个灰泽满的参考音频：

| 文件名 | 情绪 | 说明 |
|---|---|---|
| `ref_happy.wav` | 开心 | 带笑/活泼的素材 |
| `ref_lazy.wav` | 慵懒 | 困/累/慢吞吞的素材 |
| `ref_serious.wav` | 认真 | 正常说话/认真的素材 |

- 来源：灰泽满直播切片 / 语音素材（见 `assets/transcripts/`，先转音频）
- 时长建议 5~15s，清晰无杂音、情绪单一
- 对应的参考文本（prompt_text）在 `src/plugins/chatbot/voice.py` 的 `EMOTION_REFS` 里
- 文件就位后把 `.env.prod` 的 `VOICE_ENABLED` 改成 `1`，重启 bot 生效
