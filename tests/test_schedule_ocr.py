# -*- coding: utf-8 -*-
"""周表 OCR 的解析/写入（不联网，不碰真实 schedule.json）。"""
import json

import pytest

from src.plugins.chatbot import persona, schedule_ocr


def test_clean_weekly_fills_missing_days_and_aliases():
    arr = [{"day": "星期六", "time": "19:00", "topic": "杂谈", "note": "加播"}]
    out = schedule_ocr.clean_weekly(arr)
    assert len(out) == 7
    assert out[5] == {"day": "周六", "time": "19:00", "topic": "杂谈", "note": "加播"}
    assert out[0] == {"day": "周一", "time": "无", "topic": "", "note": ""}


def test_clean_weekly_sorts_monday_first():
    arr = [{"day": "周日", "time": "19:00"}, {"day": "周一", "time": "20:00"}]
    out = schedule_ocr.clean_weekly(arr)
    assert [x["day"] for x in out] == ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    assert out[0]["time"] == "20:00"


def test_clean_weekly_rejects_no_valid_time():
    with pytest.raises(ValueError):
        schedule_ocr.clean_weekly([{"day": "周一", "time": "无"}])
    with pytest.raises(ValueError):
        schedule_ocr.clean_weekly([])


def test_write_schedule_weekly_preserves_other_fields(tmp_path):
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps({"_readme": "keep", "weekly": []}, ensure_ascii=False),
                    encoding="utf-8")
    persona._schedule_cache = {"weekly": [{"day": "周一", "time": "旧"}]}

    schedule_ocr.write_schedule_weekly(
        [{"day": "周一", "time": "20:00", "topic": "", "note": ""}], path=path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["_readme"] == "keep"
    assert data["weekly"] == [{"day": "周一", "time": "20:00", "topic": "", "note": ""}]
    assert persona._schedule_cache is None, "写完必须清缓存，否则运行中的 bot 还用旧表"
