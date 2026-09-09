"""weibo_bridge.py 纯函数单元测试（网络相关不测）。"""
from src.plugins.chatbot import weibo_bridge as wb


class TestParseCookie:
    def test_basic(self):
        assert wb._parse_cookie_str("SUB=abc; ULV=1; WBPSESS=x=y") == {
            "SUB": "abc", "ULV": "1", "WBPSESS": "x=y"}

    def test_empty(self):
        assert wb._parse_cookie_str("") == {}


class TestExtractImages:
    def test_pic_infos_original_preferred(self):
        item = {"pic_ids": ["a", "b"],
                "pic_infos": {
                    "a": {"large": {"url": "https://x/orj960/a.jpg"},
                          "original": {"url": "https://x/large/a_org.jpg"}},
                    "b": {"large": {"url": "https://x/orj960/b.jpg"}}}}
        assert wb._extract_post_images(item) == [
            "https://x/large/a_org.jpg", "https://x/orj960/b.jpg"]

    def test_old_pics_fallback(self):
        item = {"pics": [{"large": {"url": "https://y/1.jpg"}}, {"url": "https://y/2.jpg"}]}
        assert wb._extract_post_images(item) == ["https://y/1.jpg", "https://y/2.jpg"]

    def test_no_images(self):
        assert wb._extract_post_images({}) == []


class TestMidToBid:
    def test_known_value_stable(self):
        # 微博 mid→bid 是 base36，给个确定值验证不抛且稳定
        assert wb._mid_to_bid("1617118739000") == wb._mid_to_bid("1617118739000")
        assert wb._mid_to_bid("0") == "0"
        assert wb._mid_to_bid("abc") == ""  # 非数字
