import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.plugins.chatbot import chatlog, event_memory, memory_recall, memory_v2, reply_evidence


def _store(tmp_path):
    return memory_v2.ShadowMemoryStore(
        state_file=tmp_path / "memory.json", audit_file=tmp_path / "audit.jsonl",
    )


def test_recall_uses_nickname_response_turn_even_after_later_uses(tmp_path):
    store = _store(tmp_path)
    user = "可以叫你姐姐吗？"
    reply = "今天让你叫一次姐姐。"
    fields = {"session_id": "session-a", "turn_id": "accept", "user_text": user,
              "assistant_text": reply}
    assert store.ingest("111", store.nickname_observations("111", user, reply), source=fields)
    assert reply_evidence.record_part("111", "accept", 0, reply, user_text=user)

    later = "姐姐，晚安"
    fields.update(turn_id="later", user_text=later, assistant_text="晚安啊")
    assert store.ingest("111", store.nickname_observations("111", later, "晚安啊"), source=fields)
    assert reply_evidence.record_part("111", "later", 0, "晚安啊", user_text=later)
    assert reply_evidence.record_part("222", "accept", 0, "另一个人的回话", user_text=user)

    query = "你以前答应我叫你姐姐了吗？"
    names, refs = [], []
    hint = store.nickname_hint("111", query, matched_terms=names, source_turn_ids=refs)
    assert "仅当时" in hint and refs == ["accept"]
    context = memory_recall.build_recall_context(store, "111", query, nicknames=names,
                                                nickname_turn_ids=refs)
    assert reply in context and "QQ 已确认发送" in context
    assert "另一个人的回话" not in context and "晚安啊" not in context


def test_commitment_recall_separates_original_reply_from_delivery_status(tmp_path):
    store = _store(tmp_path)
    item = {"kind": "commitment", "key": "take_medicine", "value": "吃药", "actor": "assistant",
            "explicit": True, "confidence": 0.99, "due_at": "2026-11-26T19:30:00",
            "reply_quote": "好，到点提醒你吃药。"}
    source = {"session_id": "session-a", "turn_id": "meds", "user_text": "晚上提醒我吃药",
              "assistant_text": item["reply_quote"]}
    now = datetime(2026, 11, 26, 18)
    [record] = store.ingest("111", [item], source=source, now=now)
    assert reply_evidence.record_part("111", "meds", 0, item["reply_quote"], user_text=source["user_text"])
    assert store.commitment_history("111", "吃药", now=now)[0]["status"] == "active"
    assert store.commitment_history("111", "吃药", now=datetime(2026, 12, 1))[0]["status"] == "expired"
    assert reply_evidence.record_part("111", "reminder-sent", 0, "到点了，记得吃药。")
    assert store.mark_reminder_delivered("111", record["id"], now=datetime(2026, 11, 26, 19, 30),
                                        turn_id="reminder-sent")

    query = "你上次答应几点提醒我吃药？"
    recalled = memory_recall.build_recall_context(store, "111", query)
    assert item["reply_quote"] in recalled
    assert "到点了，记得吃药" not in recalled
    assert "2026-11-26 19:30" in recalled and "已发送提醒" in recalled
    assert "回查本身不会续约" in recalled
    assert "吃药" in memory_recall.build_recall_context(store, "111", "你以前答应提醒我什么？")
    assert "吃药" not in memory_recall.build_recall_context(store, "111", "你之前答应提醒我带钥匙吗？")
    delivery = memory_recall.build_recall_context(store, "111", "吃药的提醒发了吗？")
    assert "到点了，记得吃药" in delivery and "已发送提醒" in delivery


async def test_commitment_needs_the_acknowledged_quote(tmp_path):
    store = _store(tmp_path)
    item = {"kind": "commitment", "key": "take_medicine", "value": "吃药", "actor": "assistant",
            "explicit": True, "confidence": 0.99, "reply_quote": "我会提醒你吃药。"}
    source = {"session_id": "session-a", "turn_id": "partial", "user_text": "晚上提醒我吃药",
              "assistant_text": "我在呢。"}

    assert store.ingest("111", [item], source=source) == []
    source.pop("turn_id")
    source["delivery_verified"] = True
    assert store.ingest("111", [item], source=source) == []
    assert store.snapshot("111")["memories"] == []

    async def extract(**kwargs):
        message = SimpleNamespace(content=json.dumps({"items": [item]}))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=extract)))
    assert await memory_v2.extract_and_ingest(
        client=client, model="test", store=store, user_id="111",
        user_text="你上次说几点提醒我吃药？", assistant_text=item["reply_quote"],
        session_id="session-a", source_turn_id="historical-question", delivery_verified=True,
    ) == []


def test_old_quotes_are_only_hints_and_new_unconfirmed_drafts_are_excluded(tmp_path):
    store = _store(tmp_path)
    chatlog.log_turn("111", False, "可以叫你姐姐吗", "旧日志里的回复。")
    chatlog.log_turn("111", False, "可以叫你姐姐吗", "这句还没发送。", reply_evidence_turn_id="draft")
    chatlog.log_turn("111", True, "姐姐", "这句是群聊。")
    recalled = memory_recall.build_recall_context(store, "111", "你以前答应我叫你姐姐了吗？")
    assert "旧日志里的回复" in recalled and "发送未确认" in recalled
    assert "未找到能确认发送" in recalled
    assert "这句还没发送" not in recalled and "这句是群聊" not in recalled

    chatlog.log_turn("222", False, "可以叫你灰宝吗", "这个称呼还挺顺口的。")
    recalled = memory_recall.build_recall_context(store, "222", "你以前答应我叫你「灰宝」了吗？")
    assert "这个称呼还挺顺口的" in recalled and "发送未确认" in recalled


def test_only_explicit_history_queries_read_evidence(monkeypatch, tmp_path):
    def unexpected_recall(*args, **kwargs):
        raise AssertionError("普通对话不应读取原话记录")

    monkeypatch.setattr(reply_evidence, "recall_turns", unexpected_recall)
    store = _store(tmp_path)
    assert not memory_recall.build_recall_context(store, "111", "姐姐在吗", nicknames=["姐姐"])
    assert not memory_recall.build_recall_context(store, "111", "明天提醒我带钥匙")
    assert memory_recall.is_commitment_question("你上次说几点提醒我带钥匙？")
    assert memory_recall.is_commitment_question("吃药的提醒发了吗？")
    assert not memory_recall.is_commitment_question("那你能提醒我吗？")
    assert not memory_recall.is_commitment_question("你上次说几点？明天再提醒我带钥匙吧")
    assert not memory_recall.is_commitment_question("你上次说几点提醒我带钥匙？改成八点")
    assert not memory_recall.build_recall_context(store, "111", "现在几点了？")
    assert not memory_recall.build_recall_context(store, "111", "你记得我吗？")
    assert not memory_recall.build_recall_context(store, "111", "你以前去哪里旅游？")
    assert not memory_recall.build_recall_context(store, "111", "你上次直播里说过什么？")


def test_general_recall_checks_the_speaker_for_untyped_chat(tmp_path):
    store = _store(tmp_path)
    assert reply_evidence.record_part("111", "asked", 0, "我没提到银灰色。",
                                     user_text="你说的颜色是银灰色吗")
    assert reply_evidence.record_part("111", "confirmed", 0, "我喜欢银灰色配深蓝。",
                                     user_text="你会怎么配色")
    assert reply_evidence.record_part("222", "other", 0, "别人的银灰色搭配")

    result = memory_recall.build_recall_context(
        store, "111", "你之前说过「银灰色配深蓝」吗？",
    )
    assert "我喜欢银灰色配深蓝" in result
    assert "我没提到银灰色" not in result and "别人的银灰色搭配" not in result


def test_history_time_filter_uses_original_user_messages(tmp_path, monkeypatch):
    store = _store(tmp_path)
    cst = timezone(timedelta(hours=8))
    now = datetime(2026, 11, 26, 18, tzinfo=cst)
    morning = now.replace(hour=9).timestamp()
    afternoon = now.replace(hour=16).timestamp()
    monkeypatch.setattr(reply_evidence.time, "time", lambda: now.timestamp())
    assert reply_evidence.record_incoming("111", "unanswered", "早上说了新钥匙",
                                         incoming=[{"t": morning, "text": "早上说了新钥匙"}])
    assert reply_evidence.record_part("111", "later", 0, "六点才回",
                                      user_text="下午说了另一件事",
                                      incoming=[{"t": afternoon, "text": "下午说了另一件事"}])

    text = memory_recall.build_recall_context(store, "111", "今天上午我跟你说了什么？", now=now)
    assert "09:00" in text and "早上说了新钥匙" in text
    assert "下午说了另一件事" not in text and "六点才回" not in text
    assert "QQ 已确认发送" not in text

    assert reply_evidence.record_incoming("111", "evening", "这是傍晚的话",
                                         incoming=[{"t": now.replace(hour=18, minute=40).timestamp(),
                                                    "text": "这是傍晚的话"}])
    assert reply_evidence.record_incoming("111", "night", "这是凌晨一点半的话",
                                         incoming=[{"t": now.replace(hour=1, minute=30).timestamp(),
                                                    "text": "这是凌晨一点半的话"}])
    precise = memory_recall.build_recall_context(
        store, "111", "今天下午六点四十左右我对你说了什么？", now=now.replace(hour=19),
    )
    assert "18:40" in precise and "这是傍晚的话" in precise
    assert "凌晨一点半" not in precise and "下午说了另一件事" not in precise
    conversation = memory_recall.build_recall_context(store, "111", "帮我翻今天的聊天记录", now=now)
    assert "六点才回" in conversation


def test_recall_uses_structured_source_for_a_fact_and_its_changed_state(tmp_path):
    store = _store(tmp_path)
    fact = {"kind": "fact", "key": "city", "value": "住在广州", "explicit": True,
            "confidence": 0.99, "temporal_scope": "current"}
    source = {"session_id": "session-a", "turn_id": "city-a",
              "user_text": "我家在广州塔附近。", "assistant_text": "我记下了。"}
    assert store.ingest("111", [fact], source=source)
    assert reply_evidence.record_part("111", "city-a", 0, source["assistant_text"],
                                     user_text=source["user_text"])
    fact["value"] = "现在住深圳"
    source.update(turn_id="city-b", user_text="我现在搬去深圳了。")
    assert store.ingest("111", [fact], source=source)
    assert reply_evidence.record_part("111", "city-b", 0, source["assistant_text"],
                                     user_text=source["user_text"])

    result = memory_recall.build_recall_context(store, "111", "我以前跟你说我住广州吗？")
    assert "我家在广州塔附近" in result and "已被更新或撤回" in result
    assert "我现在搬去深圳了" not in result


def test_event_label_locates_an_original_utterance_with_different_words(tmp_path, monkeypatch):
    monkeypatch.setattr(event_memory, "EVENT_DIR", tmp_path / "events")
    store = _store(tmp_path)
    assert event_memory.add_events(
        "111", [{"what": "他正在准备铁人三项比赛", "type": "other", "subject": "user",
                 "status": "ongoing", "entities": ["铁人三项比赛"]}],
        source={"turn_id": "sports-plan"},
    )
    assert reply_evidence.record_incoming("111", "sports-plan", "我打算把游泳单车长跑放一起练",
                                         incoming=[{"t": 1, "text": "我打算把游泳单车长跑放一起练"}])

    text = memory_recall.build_recall_context(store, "111", "我以前怎么跟你说铁人三项比赛的？")
    assert "我打算把游泳单车长跑放一起练" in text


def test_explicit_qq_quote_selects_just_the_referenced_part(tmp_path):
    store = _store(tmp_path)
    assert reply_evidence.record_part("111", "topic", 0, "同一轮的另一句", message_id="1111")
    assert reply_evidence.record_part("111", "topic", 1, "被引用的回复", message_id="1112")
    assert reply_evidence.record_part("222", "other", 0, "另一个人的回复", message_id="1112")

    text = memory_recall.build_recall_context(
        store, "111", "这句是什么意思？", reference_ids=("1112",),
    )
    assert "被引用的回复" in text and "QQ 已确认发送" in text
    assert "同一轮的另一句" not in text and "另一个人的回复" not in text
    assert not memory_recall.needs_recall("[引用灰泽满自己说的：「真的吗？」] 哈哈哈",
                                          reference_ids=("1112",))


def test_topicless_recall_returns_the_last_utterance(tmp_path):
    store = _store(tmp_path)
    assert reply_evidence.record_part("111", "latest", 0, "下次一起看电影。")
    result = memory_recall.build_recall_context(store, "111", "你上次说的话是什么？")
    assert "下次一起看电影" in result
