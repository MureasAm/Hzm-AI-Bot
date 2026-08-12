"""会话级记忆（session_memory）单元测试。

测纯逻辑：事件累计、转话题清空、冷场判定、上下文构建、短 query 判定；
LLM 路径用 mock 测降级与解析。
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import src.plugins.chatbot.session_memory as sm


@pytest.fixture(autouse=True)
def _isolate_file(tmp_path, monkeypatch):
    """把 SESSION_MEMORY_FILE 指向临时文件，避免污染线上数据。"""
    f = tmp_path / "session_memory.json"
    monkeypatch.setattr(sm, "SESSION_MEMORY_FILE", f)
    yield f


class TestGetSession:
    def test_no_session_returns_empty(self):
        assert sm.get_session("u1") == {"topic": "", "events": [], "last_active": ""}

    def test_stale_session_resets(self, _isolate_file):
        old = datetime.now() - timedelta(hours=24)
        _isolate_file.write_text(json.dumps({
            "u1": {"topic": "旧话题", "events": ["旧事件"],
                   "last_active": old.isoformat()}
        }, ensure_ascii=False), encoding="utf-8")
        assert sm.get_session("u1") == {"topic": "", "events": [], "last_active": ""}

    def test_fresh_session_kept(self, _isolate_file):
        now = datetime.now().isoformat()
        _isolate_file.write_text(json.dumps({
            "u1": {"topic": "香水", "events": ["聊了檀香"], "last_active": now}
        }, ensure_ascii=False), encoding="utf-8")
        sess = sm.get_session("u1")
        assert sess["topic"] == "香水"


class TestProbeSession:
    async def test_empty_msg_returns_early(self):
        assert await sm.probe_session("u1", "  ", "历史", None) == ""

    async def test_llm_failure_degrades(self, tmp_path, monkeypatch):
        """LLM 失败返回空 → probe_session 静默降级，原消息返回，不写文件。"""
        f = tmp_path / "s.json"
        monkeypatch.setattr(sm, "SESSION_MEMORY_FILE", f)

        async def fake_llm(client, prompt, max_tokens=250, temperature=0.2):
            return ""

        monkeypatch.setattr(sm, "_llm", fake_llm)
        out = await sm.probe_session("u1", "你好", "历史", object())
        assert out == "你好"  # 原样返回
        assert f.exists() is False  # 无有效结果不写文件

    async def test_new_event_appended_and_topic_updated(self, tmp_path, monkeypatch):
        f = tmp_path / "s.json"
        monkeypatch.setattr(sm, "SESSION_MEMORY_FILE", f)
        # 预置旧会话：话题"香水"，有事件
        f.write_text(json.dumps({
            "u1": {"topic": "香水", "events": ["聊了檀香"],
                   "last_active": datetime.now().isoformat()}
        }, ensure_ascii=False), encoding="utf-8")

        # mock LLM：延续话题，新增一个事件
        async def fake_llm(client, prompt, max_tokens=250, temperature=0.2):
            return json.dumps({
                "topic": "香水",
                "topic_changed": False,
                "new_event": "被夸香水好闻",
                "expanded_query": None,
            }, ensure_ascii=False)

        monkeypatch.setattr(sm, "_llm", fake_llm)
        out = await sm.probe_session("u1", "你好", "历史", object())

        data = json.loads(f.read_text(encoding="utf-8"))
        sess = data["u1"]
        assert sess["topic"] == "香水"
        assert "聊了檀香" in sess["events"]
        assert "被夸香水好闻" in sess["events"]
        assert out == "你好"  # 长消息原样返回

    async def test_topic_change_clears_events(self, tmp_path, monkeypatch):
        f = tmp_path / "s.json"
        monkeypatch.setattr(sm, "SESSION_MEMORY_FILE", f)
        f.write_text(json.dumps({
            "u1": {"topic": "香水", "events": ["聊了檀香"],
                   "last_active": datetime.now().isoformat()}
        }, ensure_ascii=False), encoding="utf-8")

        async def fake_llm(client, prompt, max_tokens=250, temperature=0.2):
            return json.dumps({
                "topic": "聊起游泳",
                "topic_changed": True,
                "new_event": "说想去游泳",
                "expanded_query": None,
            }, ensure_ascii=False)

        monkeypatch.setattr(sm, "_llm", fake_llm)
        await sm.probe_session("u1", "你会游泳吗", "历史", object())

        data = json.loads(f.read_text(encoding="utf-8"))
        sess = data["u1"]
        assert sess["topic"] == "聊起游泳"
        assert sess["events"] == ["说想去游泳"]  # 旧事件被清空

    async def test_referential_message_expands_in_probe(self, tmp_path, monkeypatch):
        """指代性消息（>4字，如"能读给我听听吗"）也补全检索 query——否则检索不到任何记忆。"""
        f = tmp_path / "s.json"
        monkeypatch.setattr(sm, "SESSION_MEMORY_FILE", f)
        f.write_text(json.dumps({
            "u1": {"topic": "聊灰泽满写的小说乌色月",
                   "events": ["用户提起她写过小说"],
                   "last_active": datetime.now().isoformat()}
        }, ensure_ascii=False), encoding="utf-8")

        async def fake_llm(client, prompt, max_tokens=250, temperature=0.2):
            return json.dumps({
                "topic": "聊灰泽满写的小说乌色月",
                "topic_changed": False,
                "new_event": None,
                "expanded_query": "能读给我听听吗（用户让灰泽满念她写的小说乌色月）",
            }, ensure_ascii=False)

        monkeypatch.setattr(sm, "_llm", fake_llm)
        out = await sm.probe_session("u1", "能读给我听听吗", "历史", object())
        assert "乌色月" in out  # 指代性消息返回补全后的完整句（>4字也能扩充）

    async def test_short_query_expands_in_probe(self, tmp_path, monkeypatch):
        """短 query 的扩充与话题探测在同一次 LLM 调用里完成。"""
        f = tmp_path / "s.json"
        monkeypatch.setattr(sm, "SESSION_MEMORY_FILE", f)
        f.write_text(json.dumps({
            "u1": {"topic": "用户在撒娇，灰泽满在傲娇推拉",
                   "events": ["用户发比爱心"],
                   "last_active": datetime.now().isoformat()}
        }, ensure_ascii=False), encoding="utf-8")

        async def fake_llm(client, prompt, max_tokens=250, temperature=0.2):
            assert "咋这样" in prompt
            return json.dumps({
                "topic": "用户在撒娇，灰泽满在傲娇推拉",
                "topic_changed": False,
                "new_event": None,
                "expanded_query": "用户向灰泽满撒娇说咋这样，意思是为什么这么冷淡",
            }, ensure_ascii=False)

        monkeypatch.setattr(sm, "_llm", fake_llm)
        out = await sm.probe_session("u1", "咋这样", "用户：比爱心\n灰泽满：不吃这套", object())
        assert "撒娇" in out
        assert "咋这样" in out


class TestBuildSessionContext:
    def test_empty_session_no_context(self):
        assert sm.build_session_context("nobody") == ""

    def test_builds_topic_and_events(self, tmp_path, monkeypatch):
        f = tmp_path / "s.json"
        monkeypatch.setattr(sm, "SESSION_MEMORY_FILE", f)
        f.write_text(json.dumps({
            "u1": {"topic": "香水", "events": ["聊了檀香", "被夸"],
                   "last_active": datetime.now().isoformat()}
        }, ensure_ascii=False), encoding="utf-8")
        ctx = sm.build_session_context("u1")
        assert "香水" in ctx
        assert "聊了檀香" in ctx


