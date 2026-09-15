"""语音模块单元测试：should_voice 触发规则 / 文本清洗 / 情绪选参考 / 开关解析 / 静音裁剪 / 发送路径。"""
import array
import math
import wave
from pathlib import Path

import pytest

from src.plugins.chatbot import config as cfg
from src.plugins.chatbot import voice as v


class TestToQqVoiceUrl:
    """wav → `file:///` URL：**必须是正斜杠**。

    踩坑：`Path.resolve()` 在 Windows 上给反斜杠，直接拼出 `file:///D:\\a\\b.wav`
    ——那不是合法 file URL。NapCat（Node）会把反斜杠当路径里的普通字符，找不到文件，
    语音就发不出去（而接入方案「三.4」当初试通的是正斜杠版）。
    """

    def test_uses_forward_slashes(self, tmp_path):
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"x")
        url = v._to_qq_voice(str(wav))
        assert url.startswith("file:///")
        assert "\\" not in url, f"反斜杠的 file URL 会让 NapCat 找不到文件：{url}"
        assert url.endswith("/a.wav")

    def test_absolute_path(self):
        url = v._to_qq_voice(str(Path("data/voice_cache/x.wav")))
        # 必须是绝对路径（相对路径 NapCat 无从解析）
        assert ":" in url.split("file:///", 1)[1][:3], f"路径该是绝对的：{url}"

    def test_empty_path_returns_none(self):
        assert v._to_qq_voice("") is None

    def test_real_cache_file_shape(self, tmp_path, monkeypatch):
        # 贴近真实形态：项目里的 data/voice_cache/*.wav
        monkeypatch.chdir(tmp_path)
        d = tmp_path / "data" / "voice_cache"
        d.mkdir(parents=True)
        wav = d / "voice_abc.wav"
        wav.write_bytes(b"x")
        url = v._to_qq_voice(str(wav))
        assert url.count("/") >= 4 and "\\" not in url
        assert url.endswith("data/voice_cache/voice_abc.wav")


class TestShouldVoice:
    """成句(≥30字)才朗读语音；短敷衍词/含内心戏括号/链接/超长 → 打字。"""

    def test_empty_no(self):
        assert v.should_voice("") is False

    def test_short_ack_below_min_no(self):
        assert v.should_voice("在呢") is False            # 2 字敷衍词 → 打字
        assert v.should_voice("今天好开心呀") is False     # 7 字短句 < 20 → 打字

    def test_min_boundary(self):
        assert v.should_voice("今" * 29) is False              # 29 字 < 30
        assert v.should_voice("今" * 29, min_len=15) is True    # 若下调到 15 档则会语音
        assert v.should_voice("今" * 30) is True               # 正好 30 字成句 → 语音

    def test_in_range_sentence_yes(self):
        assert v.should_voice("灰泽满今天直播聊得特别开心，下次有空我们再一起好好玩呀，明天也要来哦") is True

    def test_inner_voice_paren_no(self):
        assert v.should_voice("今天直播聊得特别开心（小声）下次有空再一起玩") is False
        assert v.should_voice("（捂脸）今天真的不知道说什么好了完全") is False

    def test_super_long_no(self):
        assert v.should_voice("今天直播聊得特别开心" * 13) is False  # 130 字 > 120 → 文字分段

    def test_max_boundary(self):
        assert v.should_voice("长" * 120) is True
        assert v.should_voice("长" * 121) is False

    def test_digits_url_no(self):
        assert v.should_voice("我们约在10点半的咖啡店见面聊那个新企划吧") is False
        assert v.should_voice("链接 http://x.com 的这个企划你帮我看看行吗") is False

    def test_short_greeting_yes(self):
        assert v.should_voice("晚安") is True            # 2 字但寒暄命中 → 语音
        assert v.should_voice("晚安呀～") is True
        assert v.should_voice("早上好呀！") is True
        assert v.should_voice("太晚了先睡吧 晚安") is True   # 短句里带晚安也读

    def test_plain_ack_still_text(self):
        assert v.should_voice("在呢") is False          # 非寒暄的短敷衍词仍打字
        assert v.should_voice("好哒") is False
        assert v.should_voice("哈哈哈") is False

    def test_greeting_word_in_long_sentence_not_voice(self):
        # 28 字陈述句顺带提到"还没睡"，不是寒暄 → 不突破下限(回归: 曾误语音)
        s = "灰泽满确实还没睡……明天没课，就放纵了一下。你咋也没睡？"
        assert v.should_voice(s) is False
        assert v.should_voice("还没睡呀") is True    # 真·短应酬仍语音

    def test_greeting_cap_keeps_short_greetings(self):
        assert v.should_voice("晚安") is True
        assert v.should_voice("太晚了先睡吧 晚安") is True   # 9字仍算寒暄

    def test_greeting_inner_paren_still_text(self):
        assert v.should_voice("晚安（心虚）") is False   # 寒暄也打不过内心戏括号铁则

    def test_paren_tail_not_counted_toward_length(self):
        # 长度看的是"实际会读出来的文本"，不是原文——否则带长括号尾巴的短句会被误判成语音。
        # 括号内容要够长，才能让"按原文判长度"真的会误判（否则原文本身就 <30，测不出东西）
        reply = "要啊，灰泽满明天还得早起呢。（不然上学要迟到了，昨天就已经迟到十分钟了）"
        assert len(reply) >= 30                 # 原文 ≥30：按原文判长度就会误触发
        assert len(v._tts_text(reply)) < 30     # 剥掉括号后其实只有 14 字
        assert v.should_voice(reply) is False   # 所以仍该走文字


class TestTtsText:
    def test_laugh_paren_converted(self):
        assert v._tts_text("（笑）你今天真棒") == "哈哈哈你今天真棒"

    def test_inner_paren_stripped(self):
        assert v._tts_text("（揉眼睛）我好困") == "我好困"

    def test_whitespace_removed(self):
        assert v._tts_text(" 好了  就这样 ") == "好了就这样"


class TestRefResolution:
    """参考音频：情绪档选择 + 缺档回落默认/任一 .wav + 同名 .txt 转写读取。"""

    def test_pick_tier(self):
        assert v._pick_tier("哈哈好好笑") == "ref_happy"
        assert v._pick_tier("好困啊……") == "ref_lazy"
        assert v._pick_tier("我们聊聊这个") == "ref_serious"

    def test_prefers_tier_file(self, monkeypatch, tmp_path):
        (tmp_path / "ref_happy.wav").write_bytes(b"x")
        (tmp_path / "ref_voice.wav").write_bytes(b"x")
        monkeypatch.setattr(v, "SOVITS_REF_DIR", tmp_path)
        assert v._resolve_ref("哈哈好好笑").name == "ref_happy.wav"

    def test_falls_back_to_default(self, monkeypatch, tmp_path):
        # 只放一个 ref_voice.wav 就能跑：任何情绪都回落它
        (tmp_path / "ref_voice.wav").write_bytes(b"x")
        monkeypatch.setattr(v, "SOVITS_REF_DIR", tmp_path)
        assert v._resolve_ref("哈哈好好笑").name == "ref_voice.wav"
        assert v._resolve_ref("我们聊聊这个").name == "ref_voice.wav"

    def test_falls_back_to_any_wav(self, monkeypatch, tmp_path):
        (tmp_path / "mysample.wav").write_bytes(b"x")
        monkeypatch.setattr(v, "SOVITS_REF_DIR", tmp_path)
        assert v._resolve_ref("我们聊聊这个").name == "mysample.wav"

    def test_no_ref_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(v, "SOVITS_REF_DIR", tmp_path)
        assert v._resolve_ref("我们聊聊这个") is None

    def test_read_prompt_text_from_sibling_txt(self, tmp_path):
        (tmp_path / "ref_voice.txt").write_text("晚安呀", encoding="utf-8")
        assert v._read_prompt_text(tmp_path / "ref_voice.wav") == "晚安呀"
        assert v._read_prompt_text(tmp_path / "nope.wav") == ""  # 无 .txt → 空转写


class TestGetVoiceEnabled:
    def test_default_off(self, monkeypatch):
        monkeypatch.setattr(cfg, "get_config", lambda *a, **k: "")
        assert cfg.get_voice_enabled() is False

    def test_zero_off(self, monkeypatch):
        # NoneBot 把 "0" 当字符串读入，bool("0") 会误判 True —— 必须显式按字符串解析
        monkeypatch.setattr(cfg, "get_config", lambda *a, **k: "0")
        assert cfg.get_voice_enabled() is False

    def test_one_on(self, monkeypatch):
        monkeypatch.setattr(cfg, "get_config", lambda *a, **k: "1")
        assert cfg.get_voice_enabled() is True

    def test_true_on(self, monkeypatch):
        monkeypatch.setattr(cfg, "get_config", lambda *a, **k: "true")
        assert cfg.get_voice_enabled() is True


class TestTrimSilence:
    """_trim_silence：裁掉 wav 首尾静音（防"几个字+长空尾"的坏合成）。"""

    def _make_wav(self, path, rate=16000, lead_s=0.3, tone_s=0.5, tail_s=0.4):
        """前 lead 静音 + 中 tone 方波(8000) + 后 tail 静音 的单声道 16bit wav。"""
        n = int(rate * (lead_s + tone_s + tail_s))
        a = array.array("h", [0]) * n
        for i in range(int(rate * lead_s), int(rate * (lead_s + tone_s))):
            a[i] = 8000 if (i // 20) % 2 else -8000
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(a.tobytes())
        return n / rate

    def test_trims_leading_and_trailing_silence(self, tmp_path):
        p = tmp_path / "x.wav"
        orig = self._make_wav(p)
        v._trim_silence(p)
        with wave.open(str(p)) as w:
            dur = w.getnframes() / w.getframerate()
        # 原 1.2s → 裁后只剩 音段(0.5s) + 两侧边距(~0.1s)，远小于原长、也大于纯音段
        assert dur < orig - 0.5
        assert dur >= 0.5

    def test_all_silence_untouched(self, tmp_path):
        p = tmp_path / "x.wav"
        rate, dur = 16000, 1.0
        a = array.array("h", [0]) * int(rate * dur)
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(a.tobytes())
        v._trim_silence(p)  # 全静音：不裁，防裁空
        with wave.open(str(p)) as w:
            assert w.getnframes() / w.getframerate() == dur

    def test_short_wav_ok(self, tmp_path):
        p = tmp_path / "short.wav"
        self._make_wav(p, tone_s=0.08, lead_s=0.0, tail_s=0.0)  # 全语音、几乎无空白
        v._trim_silence(p)  # 不应抛异常
        with wave.open(str(p)) as w:
            assert w.getnframes() > 0


class TestSpeechSeconds:
    """_speech_seconds：量有效语音时长（校验截断用）。"""

    def _mk(self, path, rate=16000, tone_s=0.5):
        n = int(rate * tone_s)
        a = array.array("h", [0]) * n
        for i in range(n):
            a[i] = 8000 if (i // 20) % 2 else -8000
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
            w.writeframes(a.tobytes())

    def test_measures_tone_length(self, tmp_path):
        p = tmp_path / "t.wav"
        self._mk(p, tone_s=0.5)
        sec = v._speech_seconds(p)
        assert 0.4 <= sec <= 0.7   # ~0.5s 的纯音

    def test_silence_counts_zero(self, tmp_path):
        p = tmp_path / "s.wav"
        a = array.array("h", [0]) * (16000)   # 1s 纯静音
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(a.tobytes())
        assert v._speech_seconds(p) == 0.0
