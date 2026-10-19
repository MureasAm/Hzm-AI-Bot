# -*- coding: utf-8 -*-
"""只读周报（`scripts/weekly_review.py`）的测试。

它是 `docs/数据迭代.md` §8 第一项：从 `chat_log` + `injection_trace` 里**确定性**地
挑出"该人看的轮次"，不调模型、不改任何数据。所以硬要求：
1. 配对要对（同一轮的轨迹总在聊天记录之前；没有轨迹的轮次不能认领别人的）；
2. 判据宁可窄（纠正/听不懂/opener 重复），候选带得出原始证据；
3. 产物落 `outputs/review/` 那种目录，指标和候选都对得上。
"""
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import weekly_review as wr  # noqa: E402

BASE = 1_800_000_000.0   # 固定时间，避免用例跨午夜/时区抖动


def _turn(t, user, reply, session="u1", kind="private"):
    return {"t": t, "session": session, "kind": kind, "user": user, "reply": reply}


def _trace(t, session="u1", items=None, blocks=None, used=False):
    return {"t": t, "session": session, "route": "私",
            "blocks": blocks if blocks is not None else ["now"],
            "items": items or {}, "reply_used_nickname": used}


class TestPairing:
    def test_trace_less_turn_does_not_steal_neighbour(self):
        """经典梗硬路由 / 群聊不接的轮次没有轨迹——不能把下一轮的轨迹认领过来。"""
        turns = [_turn(BASE + 10, "a", "A"), _turn(BASE + 20, "b", "B"),
                 _turn(BASE + 30, "c", "C")]
        traces = [_trace(BASE + 9), _trace(BASE + 29)]   # 第 2 轮没有轨迹
        paired = wr.pair_traces(turns, traces)
        assert [p and p["t"] for p in paired] == [BASE + 9, None, BASE + 29]

    def test_sessions_pair_independently(self):
        turns = [_turn(BASE + 10, "a", "A", session="u1"),
                 _turn(BASE + 11, "b", "B", session="g1")]
        traces = [_trace(BASE + 9, session="u1"), _trace(BASE + 10, session="g1")]
        paired = wr.pair_traces(turns, traces)
        assert [p["session"] for p in paired] == ["u1", "g1"]


class TestDeterministicCandidates:
    def test_confused_reply_becomes_terms_candidate(self):
        res = wr.analyse([_turn(BASE, "富区是啥", "听不懂，你说的是哪个区的富区")], [])
        assert res["metrics"]["confused_reply_count"] == 1
        cand = [c for c in res["candidates"] if c["kind"] == "confused"]
        assert cand and cand[0]["file"] == "terms" and cand[0]["proposal"] is None

    def test_next_turn_correction_flags_previous(self):
        turns = [_turn(BASE, "讲个笑话", "今天天气不错"),
                 _turn(BASE + 30, "不是这个，我说的是笑话", "哦…")]
        res = wr.analyse(turns, [])
        assert res["metrics"]["correction_count"] == 1
        cand = [c for c in res["candidates"] if c["kind"] == "correction"]
        assert cand and "今天天气不错" in cand[0]["evidence"][0]["reply"]
        assert "不是这个" in cand[0]["evidence"][1]["user"]

    def test_correction_far_later_is_not_counted(self):
        turns = [_turn(BASE, "讲个笑话", "今天天气不错"),
                 _turn(BASE + 3600, "我说的是笑话", "哦")]
        assert wr.analyse(turns, [])["metrics"]["correction_count"] == 0

    def test_repeated_opener_consecutive(self):
        turns = [_turn(BASE, "a", "灰泽满才不呢，你听错了"),
                 _turn(BASE + 5, "b", "灰泽满才不呢，别乱说")]
        res = wr.analyse(turns, [])
        assert res["metrics"]["repeated_opener_count"] == 1
        assert res["metrics"]["repeated_opener_rate"] > 0
        assert any(c["kind"] == "repeat_opener" for c in res["candidates"])

    def test_empty_retrieval_uses_trace_items(self):
        turns = [_turn(BASE + 10, "在吗", "在")]
        empty = wr.analyse(turns, [_trace(BASE + 9, items={"terms": ["绿冻"]})])
        assert empty["metrics"]["empty_retrieval_turns"] == 1
        assert any(c["kind"] == "empty_retrieval" for c in empty["candidates"])

        filled = wr.analyse(turns, [_trace(BASE + 9, items={"corpus": ["42"]})])
        assert filled["metrics"]["empty_retrieval_turns"] == 0

    def test_turn_without_trace_is_not_counted_as_empty(self):
        turns = [_turn(BASE + 10, "a", "A")]
        assert wr.analyse(turns, [])["metrics"]["empty_retrieval_turns"] == 0


class TestMetrics:
    def test_block_fire_rate_and_nickname_usage(self):
        turns = [_turn(BASE + 10, "a", "A"), _turn(BASE + 20, "b", "B")]
        traces = [
            _trace(BASE + 9, blocks=["terms", "nickname", "now"],
                   items={"corpus": ["1", "2"]}, used=True),
            _trace(BASE + 19, blocks=["terms", "now"], items={}),
        ]
        m = wr.analyse(turns, traces)["metrics"]
        assert m["window_turns"] == 2 and m["turns_with_trace"] == 2
        assert m["block_fire_rate"]["terms"] == 1.0
        assert m["block_fire_rate"]["nickname"] == 0.5
        assert m["avg_items_per_turn"]["corpus"] == 2.0
        assert m["nickname_hint_turns"] == 1 and m["nickname_used_turns"] == 1

    def test_metrics_without_any_trace(self):
        m = wr.analyse([_turn(BASE, "在吗", "在")], [])["metrics"]
        assert m["trace_coverage"] == 0.0
        assert m["block_fire_rate"] == {} and m["avg_items_per_turn"] == {}


class TestRunWritesReport:
    def test_writes_markdown_and_json(self, tmp_path, monkeypatch):
        chat_dir = tmp_path / "chat_log"
        chat_dir.mkdir()
        (chat_dir / "u1.jsonl").write_text(
            json.dumps(_turn(BASE + 10, "富区是啥", "听不懂你说的富区"), ensure_ascii=False) + "\n",
            encoding="utf-8")
        trace_file = tmp_path / "injection_trace.jsonl"
        trace_file.write_text(json.dumps(_trace(BASE + 9, items={"terms": ["富区"]}),
                                         ensure_ascii=False) + "\n", encoding="utf-8")
        monkeypatch.setattr(wr, "CHAT_DIR", chat_dir)
        monkeypatch.setattr(wr, "TRACE_FILE", trace_file)

        result = wr.run(days=0, out_dir=tmp_path / "review", date="2026-10-20")

        assert result["turns"] == 1
        md = Path(result["md"]).read_text(encoding="utf-8")
        data = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
        assert "确定性指标" in md and "候选" in md
        assert data["metrics"]["window_turns"] == 1
        assert data["candidates"]

    def test_empty_window_returns_zero_turns(self, tmp_path, monkeypatch):
        chat_dir = tmp_path / "chat_log"
        chat_dir.mkdir()
        monkeypatch.setattr(wr, "CHAT_DIR", chat_dir)
        monkeypatch.setattr(wr, "TRACE_FILE", tmp_path / "none.jsonl")
        assert wr.run(days=0, out_dir=tmp_path / "review")["turns"] == 0
