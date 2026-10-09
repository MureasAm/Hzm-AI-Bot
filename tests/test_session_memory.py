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
    """整个会话记忆指到临时目录（2026-10-09 起**按会话分文件**）。

    `use_storage` 一次改齐目录/旧单文件路径/迁移标志——只改文件路径的话，
    用例里那条 `SESSION_MEMORY_FILE = ...` 就成了"半隔离"，写会落到真实
    `user_memory/session/` 去。yield 的那个路径是**旧单文件**，
    用例仍可以用它 preload 老格式数据（正好顺带测迁移）。
    """
    sm.use_storage(tmp_path)
    yield tmp_path / "session.json"


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

    def test_between_decay_and_note_thresholds_no_note(self, _isolate_file):
        """两个阈值已拆开：话题 3h 就消散，但没隔够 12h 不发「上次聊过」。"""
        self._write_stale(_isolate_file, 5)
        assert sm.get_session("u1")["topic"] == ""       # 话题已消散
        assert sm.previous_session_note("u1") == ""      # 但还不到念"上次聊过"的时候

    def test_clearly_old_session_gets_note(self, _isolate_file):
        self._write_stale(_isolate_file, 13)
        assert "上次聊过" in sm.previous_session_note("u1")

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

        sess = json.loads(sm._file_for("u1").read_text(encoding="utf-8"))   # 按会话分文件
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

        sess = json.loads(sm._file_for("u1").read_text(encoding="utf-8"))
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


class TestTopicAge:
    """话题要带**年龄**；冷场兜底按"距上一条消息"，3 小时（2026-10-10 改）。

    2026-10-05 曾按"话题年龄 topic_since、24h"兜底，理由是"last_active 每轮刷新 →
    话题永不老化"。但那把"话题跨天延续"当常态了——真实对话是一阵一阵的，
    停几小时再回来就是新的一场。所以改回"几小时内没消息 → 这场过去了"（3h）。
    """

    def _write(self, tmp_path, monkeypatch, **sess):
        f = tmp_path / "s.json"
        monkeypatch.setattr(sm, "SESSION_MEMORY_FILE", f)
        f.write_text(json.dumps({"u1": sess}, ensure_ascii=False), encoding="utf-8")

    def test_age_is_stated(self, tmp_path, monkeypatch):
        self._write(tmp_path, monkeypatch, topic="香水", events=[],
                    topic_since=(datetime.now() - timedelta(hours=2)).isoformat(),
                    last_active=datetime.now().isoformat())
        ctx = sm.build_session_context("u1")
        assert "这个话题是2小时前开始的" in ctx
        # 不再断言"当前/本场"（旧措辞在断言"这是正在进行的对话"）
        assert "当前话题" not in ctx
        assert "本场发生" not in ctx

    def test_fresh_topic_has_no_age_label(self, tmp_path, monkeypatch):
        """刚开始（<90 秒）不加年龄——humanize_gap 返回空串。"""
        self._write(tmp_path, monkeypatch, topic="香水", events=[],
                    topic_since=datetime.now().isoformat(),
                    last_active=datetime.now().isoformat())
        ctx = sm.build_session_context("u1")
        assert "香水" in ctx and "前开始的" not in ctx

    def test_stale_by_silence(self, tmp_path, monkeypatch):
        """停够久没说话（>3h）→ 这场过去了（按 last_active 判）。"""
        self._write(tmp_path, monkeypatch, topic="香水", events=[],
                    topic_since=datetime.now().isoformat(),
                    last_active=(datetime.now() - timedelta(hours=4)).isoformat())
        assert sm.get_session("u1") == {"topic": "", "events": [], "last_active": ""}
        assert sm.build_session_context("u1") == ""

    def test_long_running_topic_with_recent_message_is_alive(self, tmp_path, monkeypatch):
        """话题开了很久，但**一直在聊**（last_active 刚刷新）→ 仍然算活着。

        这正是推翻"按话题年龄判"的原因：连续聊一下午不该因为话题"年龄"大而被判死。
        """
        self._write(tmp_path, monkeypatch, topic="香水", events=[],
                    topic_since=(datetime.now() - timedelta(hours=30)).isoformat(),
                    last_active=datetime.now().isoformat())
        assert "香水" in sm.build_session_context("u1")

    def test_old_record_without_topic_since_falls_back(self, tmp_path, monkeypatch):
        """老记录没有 topic_since → 靠 last_active 判；30 小时前 → 冷场。"""
        self._write(tmp_path, monkeypatch, topic="香水", events=[],
                    last_active=(datetime.now() - timedelta(hours=30)).isoformat())
        assert sm.build_session_context("u1") == ""




class TestMood:
    """当下情绪住在会话记忆里，和话题同寿命 3 小时（2026-10-10 从长期记忆搬来）。"""

    def test_no_mood_returns_empty(self, _isolate_file):
        assert sm.mood_note("nobody") == ""

    def test_set_and_read_mood(self, _isolate_file):
        sm.set_mood("u1", "因为考试失利而失落")
        note = sm.mood_note("u1")
        assert "考试失利" in note
        assert "此刻的情绪" in note

    def test_mood_expires_after_three_hours(self, _isolate_file):
        sm.set_mood("u1", "失落", now=datetime.now() - timedelta(hours=4))
        assert sm.mood_note("u1") == ""

    def test_mood_gone_when_session_goes_stale(self, _isolate_file):
        """mood 和 last_active 同一次写入 → 整场冷场后情绪也读不到（一起消散）。"""
        sm.set_mood("u1", "失落", now=datetime.now() - timedelta(hours=5))
        assert sm.get_session("u1")["topic"] == ""
        assert sm.mood_note("u1") == ""

    async def test_probe_records_mood(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sm, "SESSION_MEMORY_FILE", tmp_path / "s.json")

        async def fake_llm(client, prompt, max_tokens=250, temperature=0.2):
            return json.dumps({
                "topic": "聊考试", "topic_changed": True,
                "new_event": None, "mood": "因为考砸了很难过",
                "expanded_query": None,
            }, ensure_ascii=False)

        monkeypatch.setattr(sm, "_llm", fake_llm)
        await sm.probe_session("u1", "我考砸了，好难过", "历史", object())

        sess = json.loads(sm._file_for("u1").read_text(encoding="utf-8"))
        assert sess["mood"] == "因为考砸了很难过"
        assert sess["mood_at"]

    async def test_probe_null_mood_keeps_previous(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sm, "SESSION_MEMORY_FILE", tmp_path / "s.json")
        sm.set_mood("u1", "有点失落")

        async def fake_llm(client, prompt, max_tokens=250, temperature=0.2):
            return json.dumps({
                "topic": "香水", "topic_changed": False,
                "new_event": None, "mood": None, "expanded_query": None,
            }, ensure_ascii=False)

        monkeypatch.setattr(sm, "_llm", fake_llm)
        await sm.probe_session("u1", "嗯嗯", "历史", object())

        sess = json.loads(sm._file_for("u1").read_text(encoding="utf-8"))
        assert sess["mood"] == "有点失落"


class TestPerSessionFiles:
    """会话记忆也按会话分文件（2026-10-09，与短期记忆/聊天落盘同一套路）。"""

    def test_each_session_own_file(self, _isolate_file):
        sm._write_json(sm._file_for("111"), {"topic": "A", "events": [], "last_active": ""})
        sm._write_json(sm._file_for("222"), {"topic": "B", "events": [], "last_active": ""})
        assert sm._raw_session("111")["topic"] == "A"
        assert sm._raw_session("222")["topic"] == "B"

    def test_new_user_gets_a_file_on_first_write(self, _isolate_file):
        # 新用户第一次来：目录/文件都应该自动建出来，不能抛错
        assert not sm.SESSION_DIR.exists()
        sm._write_json(sm._file_for("999999"), {"topic": "新来的", "events": []})
        assert sm._file_for("999999").exists()
        assert sm._raw_session("999999")["topic"] == "新来的"

    def test_legacy_single_file_is_split_once(self, _isolate_file, monkeypatch):
        # fixture 里的 use_storage 关掉了迁移（临时目录没老文件可拆）；这里把它打开
        monkeypatch.setattr(sm, "_migrated", False)
        _isolate_file.write_text(json.dumps({
            "u1": {"topic": "老话题", "events": ["老事件"], "last_active": datetime.now().isoformat()}
        }, ensure_ascii=False), encoding="utf-8")
        assert sm._raw_session("u1")["topic"] == "老话题"      # 触发迁移
        assert sm._file_for("u1").exists()
        assert _isolate_file.with_suffix(".json.migrated").exists()
