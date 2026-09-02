"""语音模块单元测试：should_voice 触发规则 / 文本清洗 / 情绪选参考 / 开关解析。"""
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
        assert v.should_voice("今" * 19) is False              # 19 字 < 20
        assert v.should_voice("今" * 19, min_len=15) is True    # 若下调到 15 档则会语音
        assert v.should_voice("今" * 20) is True               # 正好 20 字成句 → 语音

    def test_in_range_sentence_yes(self):
        assert v.should_voice("今天直播聊得特别开心，下次有空我们再一起好好玩呀") is True

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
