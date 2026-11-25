"""QQ 已确认发送的回复原话，只能在同一个私聊内召回。"""
import json

from src.plugins.chatbot import chatlog, reply_evidence as evidence


def test_groups_confirmed_parts_and_deduplicates_repeated_writes():
    incoming = [{"message_id": "qq-1", "ts": 100, "text": "记得姐姐吗？"}]
    assert evidence.record_part("u1", "turn-1", 0, "记得你叫过姐姐", user_text="记得姐姐吗？",
                                incoming=incoming, message_id="qq-2")
    assert evidence.record_part("u1", "turn-1", 2, "不过只限今天", message_id="qq-4")
    assert evidence.record_part("u1", "turn-1", 0, "这份重复记录不应替换原话")

    [turn] = evidence.recall_turns("u1", turn_ids=["turn-1"])
    assert turn["verified"] is True
    assert turn["user"] == "记得姐姐吗？"
    assert turn["reply"] == "记得你叫过姐姐\n不过只限今天"
    assert [part["part_index"] for part in turn["parts"]] == [0, 2]
    assert turn["parts"][0]["message_id"] == "qq-2"
    assert turn["incoming"] == [{"message_id": "qq-1", "t": 100.0, "text": "记得姐姐吗？"}]
    assert evidence.recall_turns("u1", turn_ids=["qq-1"])[0]["turn_id"] == "turn-1"


def test_exact_reference_wins_before_keyword_and_stays_in_private_scope():
    assert evidence.record_part("u1", "older", 0, "旧引用")
    assert evidence.record_part("u1", "recent", 0, "她叫姐姐")
    assert evidence.record_part("u2", "other", 0, "别人的姐姐")

    selected = evidence.recall_turns("u1", turn_ids=["older"], terms=["姐姐"])
    assert [turn["turn_id"] for turn in selected] == ["older", "recent"]
    assert evidence.recall_turns("u2", turn_ids=["older"]) == []
    assert not evidence.record_part("../u1", "invalid", 0, "不可写入其他目录")


def test_legacy_requires_explicit_opt_in_and_is_never_verified():
    chatlog.LOG_DIR.mkdir()
    rows = [
        {"t": 1, "session": "u1", "kind": "private", "user": "姐姐", "reply": "这个是旧日志"},
        {"t": 2, "session": "u1", "kind": "group", "user": "姐姐", "reply": "群聊不能取"},
        {"t": 3, "session": "u2", "kind": "private", "user": "姐姐", "reply": "别人的也不能取"},
    ]
    (chatlog.LOG_DIR / "u1.123.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    assert evidence.recall_turns("u1", terms=["姐姐"]) == []
    [legacy] = evidence.recall_turns("u1", terms=["姐姐"], include_legacy=True)
    assert legacy["reply"] == "这个是旧日志"
    assert legacy["parts"] == []
    assert legacy["verified"] is False


def test_read_across_rotation_and_respects_chatlog_switch(monkeypatch):
    monkeypatch.setattr(evidence, "MAX_BYTES", 1)
    assert evidence.record_part("u1", "turn", 0, "第一段")
    assert evidence.record_part("u1", "turn", 1, "第二段", modality="voice")
    [turn] = evidence.recall_turns("u1", turn_ids=["turn"])
    assert turn["reply"] == "第一段\n第二段"
    assert turn["parts"][1]["modality"] == "voice"

    monkeypatch.setenv("CHATLOG", "0")
    assert not evidence.record_part("u1", "turned-off", 0, "关闭后不记录")
    assert evidence.recall_turns("u1", turn_ids=["turn"]) == []


def test_unanswered_incoming_is_recallable_only_as_received_user_text():
    incoming = [
        {"message_id": "100", "t": 100.0, "text": "凌晨说的"},
        {"message_id": "101", "t": 200.0, "text": "傍晚买了胶片相机"},
    ]
    assert evidence.record_incoming("u1", "unanswered", "凌晨说的\n傍晚买了胶片相机", incoming=incoming)
    assert evidence.recall_turns("u1", role="assistant", terms=["胶片相机"]) == []
    assert evidence.recall_turns("u2", role="user", terms=["胶片相机"]) == []
    assert evidence.recall_turns("u1", role="user", terms=["胶片相机"], since=100, until=200) == []
    [turn] = evidence.recall_turns("u1", role="user", terms=["胶片相机"], since=200, until=201)
    assert turn["incoming"] == incoming
    assert turn["received"] is True
    assert turn["verified"] is False
    assert turn["reply"] == ""
    assert turn["parts"] == []

    assert evidence.record_part("u1", "unanswered", 0, "这台相机挺适合拍夜景", incoming=incoming)
    [replied] = evidence.recall_turns("u1", turn_ids=["101"])
    assert replied["received"] is replied["verified"] is True
    assert replied["incoming"] == incoming


def test_explicit_time_and_speaker_bound_archive_search():
    assert evidence.record_part("u1", "older", 0, "我们聊的是滑雪板", user_text="我想买滑雪镜")
    assert evidence.record_part("u1", "recent", 0, "出了新版滑雪镜", user_text="我想换手套")
    assert evidence.recall_turns("u1", use_recent=False) == []
    [user_turn] = evidence.recall_turns("u1", terms=["滑雪镜"], role="user")
    [bot_turn] = evidence.recall_turns("u1", terms=["滑雪镜"], role="assistant")
    assert user_turn["turn_id"] == "older"
    assert bot_turn["turn_id"] == "recent"
    assert evidence.recall_turns("u1", turn_ids=["older"], since=user_turn["t"], role="user") == []
    stamp = bot_turn["parts"][0]["t"]
    assert evidence.recall_turns("u1", turn_ids=["recent"], since=stamp, role="assistant") == [bot_turn]
    assert evidence.recall_turns("u1", turn_ids=["recent"], until=stamp, role="assistant") == []
    assert evidence.recall_turns("u1", since=stamp, role="assistant", use_recent=False) == [bot_turn]


def test_conservative_fuzzy_match_keeps_literal_first_and_rejects_single_bigram():
    assert evidence.record_part("u1", "literal", 0, "我也在研究蓝牙耳机")
    assert evidence.record_part("u1", "related", 0, "耳机我会选蓝牙款")
    assert evidence.record_part("u1", "distractor", 0, "你说过耳机坏了")
    assert evidence.record_part("u1", "filler", 0, "你说过，你也记得")

    exact = evidence.recall_turns("u1", role="assistant", terms=["蓝牙耳机"], limit=4)
    assert [turn["turn_id"] for turn in exact] == ["literal"]
    ranked = evidence.recall_turns("u1", role="assistant", terms=["你以前跟我说过的那个蓝牙耳机"], fuzzy=True, limit=4)
    assert [turn["turn_id"] for turn in ranked] == ["literal", "related"]
