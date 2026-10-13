# -*- coding: utf-8 -*-
"""LLM 用量分账的测试。

这套东西是**分析用的旁路**（2026-10-13，用户要量"群聊到底占多少花费"），
所以硬要求跟 chatlog 一样：
1. 记的账对（prompt/命中/未命中/输出 + **哪条通道**）
2. **记账绝不能影响聊天**（写不进去就静默）
3. 关得掉（LLM_USAGE=0），且跑用例不会把测试记录混进真实账本
4. 聚合读得回来（脚本 `llm_cost.py` 的口径与写侧一致）
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import llm_cost  # noqa: E402

from src.plugins.chatbot import config  # noqa: E402
from src.plugins.chatbot.config import log_cache_usage, set_llm_route  # noqa: E402


def _usage(prompt=1000, hit=600, miss=400, out=20):
    """造一个跟 OpenAI 兼容响应形状一致的 usage 对象。"""
    return SimpleNamespace(prompt_tokens=prompt, prompt_cache_hit_tokens=hit,
                           prompt_cache_miss_tokens=miss, completion_tokens=out)


def _resp(usage):
    return SimpleNamespace(usage=usage)


@pytest.fixture(autouse=True)
def _tmp_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "USAGE_LOG_FILE", tmp_path / "llm_usage.jsonl")
    monkeypatch.delenv("LLM_USAGE", raising=False)
    set_llm_route("")          # contextvar 是全局的，用例之间别互相串
    yield tmp_path / "llm_usage.jsonl"
    set_llm_route("")


def _read(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class TestRecord:
    def test_writes_one_line_per_call(self, _tmp_ledger):
        log_cache_usage(_resp(_usage()), "主回复")
        recs = _read(_tmp_ledger)
        assert len(recs) == 1
        assert recs[0]["tag"] == "主回复"
        assert (recs[0]["prompt"], recs[0]["hit"], recs[0]["miss"], recs[0]["output"]) \
            == (1000, 600, 400, 20)

    def test_route_is_stamped_from_contextvar(self, _tmp_ledger):
        """通道靠 contextvar 带下来——调用点有 12 处，不该挨个加参数。"""
        set_llm_route("群")
        log_cache_usage(_resp(_usage()), "接话门")
        set_llm_route("私")
        log_cache_usage(_resp(_usage()), "主回复")
        assert [r["route"] for r in _read(_tmp_ledger)] == ["群", "私"]

    def test_route_blank_when_unset(self, _tmp_ledger):
        log_cache_usage(_resp(_usage()), "主回复")
        assert _read(_tmp_ledger)[0]["route"] == ""

    def test_print_carries_route_and_output(self, capsys, _tmp_ledger):
        """日志行要能一眼看出通道和输出量（stdout 老格式两者都没有）。"""
        set_llm_route("群")
        log_cache_usage(_resp(_usage()), "主回复")
        line = capsys.readouterr().out
        assert "主回复·群" in line and "输出=20" in line and "60%" in line

    def test_appends_across_calls(self, _tmp_ledger):
        log_cache_usage(_resp(_usage()), "主回复")
        log_cache_usage(_resp(_usage()), "主回复")
        assert len(_read(_tmp_ledger)) == 2

    def test_no_usage_field_is_skipped(self, _tmp_ledger):
        """上游没给 usage（异常形状）→ 不记账也不抛。"""
        log_cache_usage(SimpleNamespace(usage=None), "主回复")
        assert not _tmp_ledger.exists()

    def test_missing_cache_fields_still_recorded(self, _tmp_ledger, capsys):
        """网关不给 prompt_cache_* 时：**不打印**那行，但账照记。

        只按"有没有缓存字段"决定记不记的话，换个模型就等于账本凭空少一半——
        分账是要拿它对花费的，宁可记下来。
        """
        u = SimpleNamespace(prompt_tokens=500, completion_tokens=10)   # 无 cache 字段
        log_cache_usage(_resp(u), "主回复")
        assert capsys.readouterr().out == ""
        assert _read(_tmp_ledger)[0]["prompt"] == 500

    def test_survives_unwritable_path(self, monkeypatch, tmp_path):
        """旁路：写不进去也**绝不能**影响聊天。"""
        bad = tmp_path / "not_a_dir"
        bad.write_text("我是文件不是目录", encoding="utf-8")
        monkeypatch.setattr(config, "USAGE_LOG_FILE", bad / "llm_usage.jsonl")
        log_cache_usage(_resp(_usage()), "主回复")   # 不该抛


class TestToggle:
    def test_enabled_by_default(self):
        assert config.usage_log_enabled() is True

    def test_disabled_records_nothing(self, monkeypatch, _tmp_ledger):
        monkeypatch.setenv("LLM_USAGE", "0")
        log_cache_usage(_resp(_usage()), "主回复")
        assert not _tmp_ledger.exists()


class TestAggregate:
    """聚合口径：`(prompt, hit, miss, output)` 与 `_yen`。"""

    def test_tokens_passthrough(self):
        assert llm_cost._tokens({"prompt": 100, "hit": 60, "miss": 40, "output": 5}) \
            == (100, 60, 40, 5)

    def test_missing_cache_fields_count_as_miss(self):
        """没有缓存字段的老记录：prompt 全算未命中——**不能凭空从账单里消失**。"""
        assert llm_cost._tokens({"prompt": 100, "output": 5}) == (100, 0, 100, 5)

    def test_by_route_splits_channels(self, tmp_path, capsys):
        ledger = tmp_path / "llm_usage.jsonl"
        ledger.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
            {"t": 1760300000.0, "tag": "主回复", "route": "私", "prompt": 8000, "hit": 6000,
             "miss": 2000, "output": 60},
            {"t": 1760300001.0, "tag": "主回复", "route": "群", "prompt": 12000, "hit": 7000,
             "miss": 5000, "output": 55},
            {"t": 1760300002.0, "tag": "批量归纳", "route": "群", "prompt": 300, "hit": 0,
             "miss": 300, "output": 40},
        ]) + "\n", encoding="utf-8")
        assert llm_cost.main(["--file", str(ledger)]) == 0
        out = capsys.readouterr().out
        assert "群" in out and "私" in out and "批量归纳" in out
        assert "记录 3 条" in out

    def test_days_filter_drops_old(self, tmp_path, capsys):
        ledger = tmp_path / "llm_usage.jsonl"
        import time as _t
        ledger.write_text("\n".join(json.dumps(r) for r in [
            {"t": _t.time() - 10 * 86400, "tag": "主回复", "route": "私", "prompt": 1, "output": 1},
            {"t": _t.time(), "tag": "主回复", "route": "群", "prompt": 2, "output": 1},
        ]) + "\n", encoding="utf-8")
        assert llm_cost.main(["--file", str(ledger), "--days", "1"]) == 0
        out = capsys.readouterr().out
        assert "记录 1 条" in out and "群" in out

    def test_empty_ledger_says_so(self, tmp_path, capsys):
        assert llm_cost.main(["--file", str(tmp_path / "nope.jsonl")]) == 0
        assert "没有可用记录" in capsys.readouterr().out

    def test_broken_lines_skipped(self, tmp_path, capsys):
        ledger = tmp_path / "llm_usage.jsonl"
        ledger.write_text("{坏行\n" + json.dumps({"t": 1, "tag": "x", "route": "私", "prompt": 1}) + "\n",
                          encoding="utf-8")
        assert llm_cost.main(["--file", str(ledger)]) == 0
        assert "记录 1 条" in capsys.readouterr().out

    def test_json_output_parses(self, tmp_path, capsys):
        ledger = tmp_path / "llm_usage.jsonl"
        ledger.write_text(json.dumps({"t": 1760300000.0, "tag": "主回复", "route": "群",
                                      "prompt": 100, "hit": 50, "miss": 50, "output": 3}) + "\n",
                          encoding="utf-8")
        assert llm_cost.main(["--file", str(ledger), "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["calls"] == 1 and data["prompt"] == 100
        assert data["by_route"]["群"][0] == 1


class TestWiring:
    """链路级：`_flush` **有没有把这一轮标对通道**——整张分账表都建立在这个标记上。

    记账靠入口设一次 contextvar（调用点 12 处不各加参数），所以"设在哪、设没设上"
    就是这套东西唯一的脆弱点：设晚了（比如设在 handle_chat 里），
    群聊多出来的那两次调用就会漏成"未标"，而它们恰恰是要量的东西。
    """

    async def _flush_once(self, monkeypatch, tmp_path, is_group):
        from src.plugins.chatbot import chat_window as cw

        monkeypatch.setattr(config, "USAGE_LOG_FILE", tmp_path / "llm_usage.jsonl")
        resp = _resp(_usage())

        async def fake_summarize(msgs):
            log_cache_usage(resp, "批量归纳")
            return ""

        async def fake_gate(client, hist, batch):
            log_cache_usage(resp, "接话门")
            return True

        async def fake_handle(uid, text, vision_desc="", batch_summary="", is_group=False):
            log_cache_usage(resp, "主回复")
            return "在"

        async def noop(*a, **k):
            return None

        async def fake_voice(*a, **k):
            return False

        async def fake_name(bot, group_id, user_id):
            return "小王"

        monkeypatch.setattr(cw, "summarize_batch", fake_summarize)
        monkeypatch.setattr(cw, "should_reply_in_group", fake_gate)
        monkeypatch.setattr(cw, "handle_chat", fake_handle)
        monkeypatch.setattr(cw, "send_voice", fake_voice)
        monkeypatch.setattr(cw, "_maybe_send_sticker", noop)
        monkeypatch.setattr(cw, "_send", noop)
        monkeypatch.setattr(cw, "_get_sender_name", fake_name)
        # 别碰真实群记忆（conftest 没隔离它，写进去就是线上数据）
        monkeypatch.setattr(cw.group_memory, "upsert_members", lambda *a, **k: None)

        win = cw._UserWindow("123456" if is_group else "u1")
        win.is_private = not is_group
        win.bot = object()
        win.pending = [("u1", "在吗", "", ""), ("u1", "在不在", "", "")]
        await cw._flush(win)
        return _read(tmp_path / "llm_usage.jsonl")

    async def test_group_turn_tagged_group(self, monkeypatch, tmp_path):
        recs = await self._flush_once(monkeypatch, tmp_path, is_group=True)
        assert [r["tag"] for r in recs] == ["批量归纳", "接话门", "主回复"]
        assert {r["route"] for r in recs} == {"群"}

    async def test_private_turn_tagged_private(self, monkeypatch, tmp_path):
        recs = await self._flush_once(monkeypatch, tmp_path, is_group=False)
        assert recs and {r["route"] for r in recs} == {"私"}

