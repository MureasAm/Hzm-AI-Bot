"""B站动态推送模板（结构化通知）的单元测试。"""
from src.plugins.chatbot import bili_bridge as bb


def _monitor():
    m = bb.BiliMonitor()
    m.uid = "1298779265"
    return m


class TestFormatDynamicPush:
    def test_text_dynamic(self):
        out = _monitor()._format_dynamic_push("唱拉了 关上门悄悄听")
        assert out == (
            "灰泽满刚刚发了动态哦！\n\n"
            "动态内容：唱拉了 关上门悄悄听\n\n"
            "https://space.bilibili.com/1298779265/dynamic"
        )

    def test_image_only_dynamic(self):
        out = _monitor()._format_dynamic_push("")
        assert "动态内容：（图片动态）" in out
        assert "https://space.bilibili.com/1298779265/dynamic" in out

    def test_strips_text_whitespace(self):
        out = _monitor()._format_dynamic_push("  有内容的动态  ")
        assert "动态内容：有内容的动态" in out
