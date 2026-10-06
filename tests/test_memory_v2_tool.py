import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import memory_v2_tool as tool


def _state():
    return {
        "schema_version": 2,
        "users": {
            "10001": {
                "updated_at": "2026-10-06T20:00:00",
                "memories": [{
                    "id": "m1",
                    "kind": "fact",
                    "key": "preferred_name",
                    "value": "小明",
                    "status": "confirmed",
                    "source": {
                        "role": "user",
                        "excerpt": "我叫小明",
                        "session_id": "scope-a",
                        "observed_at": "2026-10-06T20:00:00",
                    },
                }],
            },
            "20002": {
                "updated_at": "2026-10-06T21:00:00",
                "memories": [{
                    "id": "m2",
                    "kind": "interaction_preference",
                    "key": "support_style",
                    "value": "comfort",
                    "status": "candidate",
                    "source": {
                        "role": "user",
                        "excerpt": "先安慰我",
                        "session_id": "scope-b",
                        "observed_at": "2026-10-06T21:00:00",
                    },
                }],
            },
        },
    }


def test_aggregate_audit_contains_counts_but_no_raw_ids_or_values():
    report = tool.build_audit_report(_state())
    rendered = json.dumps(report, ensure_ascii=False)

    assert report["users"] == 2
    assert report["memories"] == 2
    assert report["by_kind"] == {"fact": 1, "interaction_preference": 1}
    assert report["by_status"] == {"candidate": 1, "confirmed": 1}
    assert "10001" not in rendered
    assert "20002" not in rendered
    assert "小明" not in rendered
    assert "comfort" not in rendered


def test_user_detail_requires_exact_user_and_redacts_source_by_default():
    report = tool.build_audit_report(_state(), user_id="10001")

    assert report["user"] == tool.redact_user_id("10001")
    assert report["memories"][0]["value"] == "小明"
    assert report["memories"][0]["source"]["excerpt"] == "[redacted]"


def test_user_detail_can_explicitly_show_source():
    report = tool.build_audit_report(_state(), user_id="10001", show_source=True)
    assert report["memories"][0]["source"]["excerpt"] == "我叫小明"


def test_unknown_user_returns_empty_partition():
    report = tool.build_audit_report(_state(), user_id="missing")
    assert report["memories"] == []


def test_iter_private_turns_filters_groups_and_respects_limit(tmp_path):
    path = tmp_path / "chat.jsonl"
    rows = [
        {"t": 1, "session": "u1", "kind": "private", "user": "一", "reply": "甲"},
        {"t": 2, "session": "g1", "kind": "group", "user": "群消息", "reply": "不处理"},
        {"t": 3, "session": "u2", "kind": "private", "user": "二", "reply": "乙"},
        {"t": 4, "session": "u3", "kind": "private", "user": "三", "reply": "丙"},
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    turns = list(tool.iter_private_turns([path], limit=2))

    assert [turn["session"] for turn in turns] == ["u1", "u2"]
    assert all(turn["kind"] == "private" for turn in turns)


def test_iter_private_turns_ignores_bad_and_empty_rows(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text(
        "{broken\n"
        + json.dumps({"t": 1, "session": "u1", "kind": "private", "user": "", "reply": "x"})
        + "\n"
        + json.dumps({"t": 2, "session": "u2", "kind": "private", "user": "有效", "reply": "回复"}),
        encoding="utf-8",
    )
    assert [turn["session"] for turn in tool.iter_private_turns([path], limit=10)] == ["u2"]


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_positive_limit_rejects_unbounded_or_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        tool.positive_limit(value)


def test_positive_limit_accepts_bounded_value():
    assert tool.positive_limit("100") == 100


def test_run_tool_registers_developer_memory_commands():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_tool.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0
    assert "memory-audit" in result.stdout
    assert "memory-replay" in result.stdout
