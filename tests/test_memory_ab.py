import json

from scripts import memory_ab


def test_v1_gate_matches_production_density_gate():
    assert memory_ab.v1_gate("在吗") is False          # 太短
    assert memory_ab.v1_gate("哈哈哈") is False
    assert memory_ab.v1_gate("[表情：可怜]") is False
    assert memory_ab.v1_gate("今天好累啊") is True
    assert memory_ab.v1_gate("   ") is False


def test_iter_turns_by_user_caps_per_user_and_skips_group(tmp_path):
    path = tmp_path / "10001.jsonl"
    rows = []
    for i in range(5):
        rows.append({"kind": "private", "session": "10001", "user": f"消息{i}", "reply": "回", "t": i})
        rows.append({"kind": "group", "session": "20002", "user": "群消息", "reply": "回", "t": i})
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    turns = list(memory_ab.iter_turns_by_user([path], limit=100, per_user=3))

    assert [t["user"] for t in turns] == ["消息0", "消息1", "消息2"]  # 每人上限 3，群聊跳过


def test_order_paths_by_activity_prefers_chatty_sessions(tmp_path):
    quiet = tmp_path / "111.jsonl"
    chatty = tmp_path / "222.jsonl"
    other = tmp_path / "u4.jsonl"  # 非 QQ 号会话排到最后
    quiet.write_text(
        json.dumps({"kind": "private", "session": "111", "user": "你好呀", "reply": "回"}, ensure_ascii=False),
        encoding="utf-8",
    )
    chatty.write_text(
        "\n".join(
            json.dumps({"kind": "private", "session": "222", "user": f"消息{i}", "reply": "回"}, ensure_ascii=False)
            for i in range(5)
        ),
        encoding="utf-8",
    )
    other.write_text(
        "\n".join(
            json.dumps({"kind": "private", "session": "u4", "user": f"x{i}", "reply": "回"}, ensure_ascii=False)
            for i in range(20)
        ),
        encoding="utf-8",
    )

    ordered = memory_ab.order_paths_by_activity([quiet, other, chatty])

    assert [p.name for p in ordered] == ["222.jsonl", "111.jsonl", "u4.jsonl"]


def test_iter_turns_skips_synthetic_qa_sessions(tmp_path):
    path = tmp_path / "10001.jsonl"
    path.write_text("\n".join([
        json.dumps({"kind": "private", "session": "10001", "user": "问题1:你爱不爱绿冻？", "reply": "回"},
                   ensure_ascii=False),
        json.dumps({"kind": "private", "session": "10001", "user": "回答:早就说过了", "reply": "回"},
                   ensure_ascii=False),
        json.dumps({"kind": "private", "session": "10001", "user": "今天好累啊", "reply": "回"},
                   ensure_ascii=False),
    ]), encoding="utf-8")

    turns = list(memory_ab.iter_turns_by_user([path], limit=10, per_user=10))

    assert [t["user"] for t in turns] == ["今天好累啊"]


def test_build_report_counts_and_noise_scan():
    v1_cards = {
        "u1": {
            "user_facts": [{"fact": "在澳洲"}],
            "impressions": [{"tag": "夜猫子"}],
            "promises": [{"promise": "下次唱一段"}],
        },
        "u2": {"user_facts": [{"fact": "灰泽满是最棒的机器人"}]},
        "u3": {},
    }
    v2_state = {"users": {
        "u1": {"memories": [
            {"kind": "fact", "key": "location", "value": "在澳洲",
             "status": "confirmed", "valid_for": "none"},
            {"kind": "fact", "key": "exam_status", "value": "在准备考试",
             "status": "confirmed", "valid_for": "14d"},
        ]},
        "u2": {"memories": [
            {"kind": "fact", "key": "bot_version", "value": "V8.0", "status": "confirmed"},
        ]},
    }}

    report = memory_ab.build_report(
        v1_cards, v2_state,
        v1_contexts={"u1": "abc", "u2": "", "u3": ""},
        v2_contexts={"u1": "xy", "u2": "", "u3": ""},
        samples=[], turns=9,
    )

    assert report["users"] == {"v1": 3, "v2": 2}
    assert report["v1"]["cards_with_content"] == 2
    assert report["v1"]["empty_cards"] == 1
    assert report["v1"]["totals"]["user_facts"] == 2
    assert report["v1"]["bot_self_noise"] == 1          # u2 的 fact 以"灰泽满"开头
    assert report["v2"]["by_kind"] == {"fact": 3}
    assert report["v2"]["bot_self_noise"] == 1          # bot_version 被扫出
    assert report["v2"]["timebound_facts"] == 1         # 只有 exam_status 带期限
    assert report["v2"]["injection_chars"]["max"] == 2
