"""语音模块单元测试：should_voice 触发规则 / 文本清洗 / 情绪选参考 / 开关解析。"""
from src.plugins.chatbot import config as cfg
from src.plugins.chatbot import voice as v


class TestShouldVoice:
    def test_short_no_inner_paren_yes(self):
        assert v.should_voice("今天好开心呀！") is True

    def test_empty_no(self):
        assert v.should_voice("") is False

    def test_long_reply_no(self):
        assert v.should_voice("今天的直播真的特别特别顺利，感觉大家也都玩得很开心呀") is False

    def test_inner_voice_paren_no(self):
        assert v.should_voice("嗯（小声）") is False
        assert v.should_voice("（捂脸）我不行了") is False

    def test_digits_url_no(self):
        assert v.should_voice("记得看时间，10点半") is False
        assert v.should_voice("链接 http://x.com") is False

    def test_max_len_threshold(self):
        assert v.should_voice("今天好开心呀", max_len=20) is True
        assert v.should_voice("今天好开心呀", max_len=5) is False  # 超过阈值不语音


class TestTtsText:
    def test_laugh_paren_converted(self):
        assert v._tts_text("（笑）你今天真棒") == "哈哈哈你今天真棒"

    def test_inner_paren_stripped(self):
        assert v._tts_text("（揉眼睛）我好困") == "我好困"

    def test_whitespace_removed(self):
        assert v._tts_text(" 好了  就这样 ") == "好了就这样"


class TestPickRef:
    def test_happy(self):
        assert v._pick_ref("哈哈好好笑")[0] == "ref_happy.wav"

    def test_lazy(self):
        assert v._pick_ref("好困啊……")[0] == "ref_lazy.wav"

    def test_serious_default(self):
        assert v._pick_ref("我们聊聊这个")[0] == "ref_serious.wav"


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
