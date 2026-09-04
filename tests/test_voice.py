"""语音模块单元测试：should_voice 触发规则 / 文本清洗 / 情绪选参考 / 开关解析 / 静音裁剪。"""
import array
import math
import wave

from src.plugins.chatbot import config as cfg
from src.plugins.chatbot import voice as v


class TestShouldVoice:
    """成句(≥20字)才朗读语音；短敷衍词/含内心戏括号/链接/超长 → 打字。"""

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

    def test_greeting_inner_paren_still_text(self):
        assert v.should_voice("晚安（心虚）") is False   # 寒暄也打不过内心戏括号铁则

    def test_paren_tail_not_counted_toward_length(self):
        # 原文带括号尾巴够 20 字、但剥括号后实际读出来不足 20 字 → 仍文字
        reply = "要啊，灰泽满明天还得早起呢。（不然上学要迟到了）"
        assert len(reply) >= 20               # 原文 ≥20 会误触发旧逻辑
        assert v.should_voice(reply) is False  # 读出来只有 14 字 → 不该语音


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
