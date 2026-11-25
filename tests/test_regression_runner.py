"""回归工具必须在同一事件循环复用真实异步客户端；测试不联网。"""
import asyncio
import json
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import _common
import regression_test as regression


def test_check_reuses_one_event_loop_across_repeats_and_cases(monkeypatch, tmp_path):
    loops = []

    class Core:
        async def handle_chat(self, *args, **kwargs):
            loop = asyncio.get_running_loop()
            if loops:
                assert loop is loops[0], "cached async clients need their original event loop"
            loops.append(loop)
            await asyncio.sleep(0)
            return "这样啊"

    cases = [{"id": name, "user": "你好", "runs": 2} for name in ("first", "second")]
    monkeypatch.setattr(regression, "_init", lambda: Core())
    monkeypatch.setattr(regression, "_load_cases", lambda _: cases)
    monkeypatch.setattr(regression, "_judge", lambda case, result: [])
    assert regression.check(out_dir=tmp_path) == 0
    assert len(loops) == 4


def test_historical_opener_reader_imports_its_own_dependency(monkeypatch):
    # A caller importing _common must not be required to inject a hidden global.
    monkeypatch.delattr(regression, "_common", raising=False)
    monkeypatch.setattr(regression, "_BASELINE_OPENERS", None)
    replies = [chr(0x4e00 + i) + "好" for i in range(101)]
    monkeypatch.setattr(_common, "chatlog_lines", lambda: [
        json.dumps({"session": "123", "reply": reply}, ensure_ascii=False) for reply in replies
    ])
    assert regression._baseline_openers() == set(replies)
