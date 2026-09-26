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

    def test_missing_last_active_treated_as_stale(self, _isolate_file):
        """**缺时间戳的老记录不能再当【当前会话】**。

        旧写法是 `if last:` 才做判定 —— 缺 last_active 的记录直接跳过检查、
        **永久免疫**，永远被当"你们这一场对话"注入。线上 106 条目前都有该字段，
        所以还没炸；但那是运气：手改/老数据随时会踩。
        """
        _isolate_file.write_text(json.dumps({
            "u1": {"topic": "很久以前聊的", "events": ["旧事件"]}
        }, ensure_ascii=False), encoding="utf-8")
        assert sm.get_session("u1")["topic"] == ""

    def test_broken_timestamp_treated_as_stale(self, _isolate_file):
        _isolate_file.write_text(json.dumps({
            "u1": {"topic": "话题", "events": [], "last_active": "不是时间"}
        }, ensure_ascii=False), encoding="utf-8")
        assert sm.get_session("u1")["topic"] == ""


class TestSessionGapSeconds:
    def test_no_record_returns_none(self):
        assert sm.session_gap_seconds("nobody") is None

    def test_no_timestamp_returns_none(self, _isolate_file):
        _isolate_file.write_text(json.dumps({"u1": {"topic": "x"}}, ensure_ascii=False),
                                 encoding="utf-8")
        assert sm.session_gap_seconds("u1") is None

    def test_hours_gap(self, _isolate_file):
        _isolate_file.write_text(json.dumps({
            "u1": {"topic": "x",
                   "last_active": (datetime.now() - timedelta(hours=3)).isoformat()}
        }, ensure_ascii=False), encoding="utf-8")
        gap = sm.session_gap_seconds("u1")
        assert 3 * 3600 - 60 <= gap <= 3 * 3600 + 60


class TestPreviousSessionNote:
    """过期会话**不再静默丢掉**，而是带着"隔了多久"降级成「上次聊过」。

    背景：12 小时那道闸原来是静默的——过了就整个丢掉，她连"上次你不是说要加粉丝群吗"
    都说不出来。时间戳只决定了"丢不丢"，没决定"怎么定性"，这里补上后半句。
    """

    def _write_stale(self, f, hours, topic="香水", events=("聊了檀香",)):
        f.write_text(json.dumps({
            "u1": {"topic": topic, "events": list(events),
                   "last_active": (datetime.now() - timedelta(hours=hours)).isoformat()}
        }, ensure_ascii=False), encoding="utf-8")

    def test_fresh_session_returns_empty(self, _isolate_file):
        # 还在有效期内 → 走正常的【当前会话】注入，别重复说一遍
        self._write_stale(_isolate_file, 1)
        assert sm.previous_session_note("u1") == ""

    def test_no_record_returns_empty(self):
        assert sm.previous_session_note("nobody") == ""

    def test_stale_carries_time_topic_and_caution(self, _isolate_file):
        self._write_stale(_isolate_file, 72 + 1)   # 3 天多 → humanize_gap 给"3天前"
        note = sm.previous_session_note("u1")
        assert "3天前" in note
        assert "香水" in note and "聊了檀香" in note
        assert "不是现在正在聊的话题" in note   # 关键：定性，不然她会当成现在

    def test_missing_timestamp_notes_without_time(self, _isolate_file):
        # 缺时间戳 → 照样提示"上次聊过"，只是说不出隔了多久
        _isolate_file.write_text(json.dumps({"u1": {"topic": "香水", "events": []}},
                                            ensure_ascii=False), encoding="utf-8")
        note = sm.previous_session_note("u1")
        assert "香水" in note and "上次聊过" in note
        assert "天前" not in note and "小时前" not in note

    def test_events_capped(self, _isolate_file):
        # 只带最近的几条：整场搬进来会挤上下文
        self._write_stale(_isolate_file, 72, events=("e1", "e2", "e3", "e4"))
        note = sm.previous_session_note("u1")
        assert "e4" in note and "e3" in note
        assert "e1" not in note and "e2" not in note

    async def test_must_be_called_before_probe(self, _isolate_file, monkeypatch):
        """**必须在 probe_session 之前调用**——probe 会把 last_active 改写成"现在"。

        这条用例就是把这个调用顺序钉死：谁把它挪到 probe 之后，这里立刻红。
        """
        self._write_stale(_isolate_file, 72)
        assert sm.previous_session_note("u1") != ""

        async def fake_llm(client, prompt, max_tokens=250, temperature=0.2):
            return json.dumps({"topic": "新话题", "topic_changed": True,
                               "new_event": None, "expanded_query": None}, ensure_ascii=False)

        monkeypatch.setattr(sm, "_llm", fake_llm)
        await sm.probe_session("u1", "你好", "历史", object())
        assert sm.previous_session_note("u1") == ""   # 已过期记录被改写 → 不再是"上次"


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


