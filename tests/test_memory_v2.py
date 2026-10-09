import json
from datetime import datetime, timedelta

import pytest

from src.plugins.chatbot.memory_v2 import (
    ShadowMemoryStore,
    build_extraction_prompt,
    extract_and_ingest,
    injection_enabled,
    parse_extraction_payload,
    shadow_enabled,
)


@pytest.fixture
def clock():
    return datetime(2026, 10, 6, 20, 0, 0)


@pytest.fixture
def store(tmp_path):
    return ShadowMemoryStore(
        state_file=tmp_path / "memory_v2_shadow.json",
        audit_file=tmp_path / "memory_v2_audit.jsonl",
        context_budget_chars=700,
    )


def _source(session_id="session-a", now=None, user="测试用户消息", reply="测试回复"):
    now = now or datetime(2026, 10, 6, 20, 0, 0)
    return {
        "session_id": session_id,
        "observed_at": now.isoformat(),
        "user_text": user,
        "assistant_text": reply,
    }


def test_explicit_fact_is_confirmed_with_bounded_provenance(store, clock):
    accepted = store.ingest(
        "u1",
        [{
            "kind": "fact",
            "key": "preferred_name",
            "value": "小明",
            "explicit": True,
            "confidence": 0.98,
        }],
        source=_source(now=clock, user="我叫小明，以后叫我小明"),
        now=clock,
    )

    assert len(accepted) == 1
    assert accepted[0]["status"] == "confirmed"
    assert accepted[0]["source"]["role"] == "user"
    assert accepted[0]["source"]["excerpt"] == "我叫小明，以后叫我小明"
    assert len(accepted[0]["source"]["excerpt"]) <= 240


def test_inferred_fact_is_rejected_instead_of_becoming_profile(store, clock):
    accepted = store.ingest(
        "u1",
        [{
            "kind": "fact",
            "key": "occupation",
            "value": "上班族",
            "explicit": False,
            "confidence": 0.8,
        }],
        source=_source(now=clock, user="刚下班，累死了"),
        now=clock,
    )

    assert accepted == []
    assert store.snapshot("u1")["memories"] == []


def test_inferred_preference_needs_three_observations_across_two_sessions(store, clock):
    item = {
        "kind": "interaction_preference",
        "key": "banter_tolerance",
        "value": "high",
        "explicit": False,
        "confidence": 0.75,
    }

    first = store.ingest("u1", [item], source=_source("session-a", clock), now=clock)[0]
    second = store.ingest(
        "u1", [item], source=_source("session-a", clock + timedelta(minutes=5)),
        now=clock + timedelta(minutes=5),
    )[0]
    third = store.ingest(
        "u1", [item], source=_source("session-b", clock + timedelta(days=1)),
        now=clock + timedelta(days=1),
    )[0]

    assert first["status"] == "candidate"
    assert second["status"] == "candidate"
    assert third["status"] == "confirmed"
    assert third["evidence_count"] == 3
    assert third["session_count"] == 2


def test_replaying_same_evidence_does_not_inflate_preference_confidence(store, clock):
    item = {
        "kind": "interaction_preference",
        "key": "banter_tolerance",
        "value": "high",
        "explicit": False,
        "confidence": 0.75,
    }
    source = _source("session-a", clock, user="用户继续和灰泽满互怼")

    first = store.ingest("u1", [item], source=source, now=clock)
    replayed = store.ingest("u1", [item], source=source, now=clock)

    assert first[0]["evidence_count"] == 1
    assert replayed == []
    memory = store.snapshot("u1")["memories"][0]
    assert memory["evidence_count"] == 1
    assert memory["status"] == "candidate"


def test_explicit_interaction_preference_confirms_immediately(store, clock):
    accepted = store.ingest(
        "u1",
        [{
            "kind": "interaction_preference",
            "key": "support_style",
            "value": "comfort",
            "explicit": True,
            "confidence": 0.95,
        }],
        source=_source(now=clock, user="我难受的时候不想听建议，先安慰我就好"),
        now=clock,
    )

    assert accepted[0]["status"] == "confirmed"


def test_timebound_fact_expires_but_stable_fact_survives(store, clock):
    """时效性近况：`valid_for` 给期限，到期后不再注入；长期事实不受影响。"""
    store.ingest(
        "u1",
        [
            {"kind": "fact", "key": "exam_status", "value": "正在准备考试",
             "temporal_scope": "current", "valid_for": "3d",
             "explicit": True, "confidence": 0.9},
            {"kind": "fact", "key": "location", "value": "在曼谷",
             "temporal_scope": "timeless", "valid_for": "none",
             "explicit": True, "confidence": 0.9},
        ],
        source=_source(now=clock, user="我在曼谷，这几天在准备考试"),
        now=clock,
    )

    assert "准备考试" in store.build_context("u1", "你考试准备得怎么样了", now=clock)
    later = clock + timedelta(days=3, seconds=1)
    context = store.build_context("u1", "你考试准备得怎么样了", now=later)
    assert "准备考试" not in context        # 近况到期
    assert "曼谷" in context                # 长期事实 + always-on key 仍在


def test_past_facts_accumulate_instead_of_superseding(store, clock):
    """经历（past）累积——后来的经历不能顶掉先前的（通用 key 冲突曾真丢过数据）。"""
    store.ingest(
        "u1",
        [{"kind": "fact", "key": "past_experience", "value": "小时候被霸凌",
          "temporal_scope": "past", "explicit": True, "confidence": 0.9}],
        source=_source(now=clock), now=clock,
    )
    store.ingest(
        "u1",
        [{"kind": "fact", "key": "past_experience", "value": "高考前遭遇车祸",
          "temporal_scope": "past", "explicit": True, "confidence": 0.9}],
        source=_source(now=clock + timedelta(minutes=5)), now=clock + timedelta(minutes=5),
    )

    kept = [m["value"] for m in store.snapshot("u1")["memories"] if m["status"] != "superseded"]
    assert any("霸凌" in v for v in kept)
    assert any("车祸" in v for v in kept)


def test_current_facts_supersede_same_key(store, clock):
    """当前状态（current）是单值槽：同一 key 的新值顶掉旧的（换城市）。"""
    store.ingest(
        "u1",
        [{"kind": "fact", "key": "location", "value": "在广州",
          "temporal_scope": "current", "explicit": True, "confidence": 0.9}],
        source=_source(now=clock), now=clock,
    )
    store.ingest(
        "u1",
        [{"kind": "fact", "key": "location", "value": "搬到深圳了",
          "temporal_scope": "current", "explicit": True, "confidence": 0.9}],
        source=_source(now=clock + timedelta(minutes=5)), now=clock + timedelta(minutes=5),
    )

    kept = [m["value"] for m in store.snapshot("u1")["memories"] if m["status"] != "superseded"]
    assert kept == ["搬到深圳了"]


def test_past_fact_is_not_rendered_as_current_core_profile(store, clock):
    store.ingest(
        "u1",
        [{
            "kind": "fact",
            "key": "city",
            "value": "以前住在广州，现在已经搬走",
            "temporal_scope": "past",
            "explicit": True,
            "confidence": 0.98,
        }],
        source=_source(now=clock, user="我以前住广州，现在已经搬走了"),
        now=clock,
    )

    memory = store.snapshot("u1")["memories"][0]
    assert memory["temporal_scope"] == "past"
    assert "广州" not in store.build_context("u1", "今天天气怎么样", now=clock)
    assert "过去事实" in store.build_context("u1", "我以前住哪里", now=clock)
    assert "广州" in store.build_context("u1", "我以前住哪里", now=clock)


def test_bot_self_facts_are_rejected_and_user_facts_survive(store, clock):
    accepted = store.ingest(
        "u1",
        [
            {"kind": "fact", "key": "bot_version", "value": "V8.0",
             "explicit": True, "confidence": 0.99},
            {"kind": "fact", "key": "identity", "value": "灰泽满自己是澳洲留学生",
             "explicit": True, "confidence": 0.9},
            {"kind": "fact", "key": "preferred_name", "value": "小明",
             "explicit": True, "confidence": 0.99},
        ],
        source=_source(now=clock, user="你更新了 V8.0，我叫小明"),
        now=clock,
    )

    assert [m["key"] for m in accepted] == ["preferred_name"]


def test_user_plan_cannot_be_stored_as_assistant_commitment(store, clock):
    accepted = store.ingest(
        "u1",
        [{
            "kind": "commitment",
            "key": "promise",
            "value": "周六直播",
            "actor": "user",
            "explicit": True,
            "confidence": 0.99,
        }],
        source=_source(now=clock, user="我打算周六直播", reply="那你加油"),
        now=clock,
    )

    assert accepted == []


def test_assistant_commitment_uses_assistant_provenance(store, clock):
    accepted = store.ingest(
        "u1",
        [{
            "kind": "commitment",
            "key": "promise",
            "value": "下次给用户唱一段",
            "actor": "assistant",
            "explicit": True,
            "confidence": 0.96,
        }],
        source=_source(now=clock, user="下次能唱吗", reply="行，下次给你唱一段"),
        now=clock,
    )

    assert accepted[0]["status"] == "active"
    assert accepted[0]["source"]["role"] == "assistant"
    assert accepted[0]["source"]["excerpt"] == "行，下次给你唱一段"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("phone", "13800138000"),
        ("email", "someone@example.com"),
        ("exact_address", "某市某路 12 号 301 室"),
        ("medical_diagnosis", "确诊了某种疾病"),
    ],
)
def test_sensitive_facts_fail_closed(store, clock, key, value):
    accepted = store.ingest(
        "u1",
        [{
            "kind": "fact",
            "key": key,
            "value": value,
            "explicit": True,
            "confidence": 1.0,
        }],
        source=_source(now=clock),
        now=clock,
    )

    assert accepted == []


def test_sensitive_data_is_redacted_from_otherwise_valid_source(store, clock):
    accepted = store.ingest(
        "u1",
        [{
            "kind": "fact",
            "key": "preferred_name",
            "value": "小明",
            "explicit": True,
            "confidence": 0.99,
        }],
        source=_source(now=clock, user="我叫小明，手机号是13800138000"),
        now=clock,
    )

    assert accepted[0]["source"]["excerpt"] == "我叫小明，手机号是[redacted]"


def test_invalid_preference_value_is_rejected(store, clock):
    accepted = store.ingest(
        "u1",
        [{
            "kind": "interaction_preference",
            "key": "banter_tolerance",
            "value": "随便编一个值",
            "explicit": True,
            "confidence": 0.9,
        }],
        source=_source(now=clock),
        now=clock,
    )

    assert accepted == []


def test_context_turns_preferences_into_behavior_without_erasing_persona(store, clock):
    store.ingest(
        "u1",
        [
            {
                "kind": "interaction_preference",
                "key": "support_style",
                "value": "comfort",
                "explicit": True,
                "confidence": 0.95,
            },
            {
                "kind": "interaction_preference",
                "key": "banter_tolerance",
                "value": "high",
                "explicit": True,
                "confidence": 0.95,
            },
            {
                "kind": "emotional_state",
                "key": "current_emotion",
                "value": "因为考试结果很失落",
                "explicit": True,
                "confidence": 0.95,
            },
        ],
        source=_source(now=clock),
        now=clock,
    )

    context = store.build_context("u1", "陪我说说话", now=clock)

    assert "先接住情绪" in context
    assert "普通闲聊可以更强地互怼" in context
    assert "明显低落时暂停" in context
    assert "不改变灰泽满的核心性格、事实和边界" in context
    assert len(context) <= 700


def test_context_budget_is_hard_cap(store, clock):
    for index in range(20):
        store.ingest(
            "u1",
            [{
                "kind": "fact",
                "key": f"hobby_{index}",
                "value": f"长期喜欢第{index}种兴趣" + "很具体" * 20,
                "explicit": True,
                "confidence": 0.95,
            }],
            source=_source(f"session-{index}", clock + timedelta(minutes=index)),
            now=clock + timedelta(minutes=index),
        )

    context = store.build_context("u1", "最近喜欢什么", now=clock + timedelta(hours=1))

    assert len(context) <= 700


class TestPreferenceDimensions:
    """偏好维度从 4 个扩到 8 个（2026-10-10）——每个维度都是"枚举值 → 行为句"。"""

    NEW_DIMENSIONS = [
        ("address_style", "plain"),
        ("reply_length", "short"),
        ("topic_preference", "avoid"),
        ("self_reference", "first_person"),
    ]

    def test_new_dimensions_validate_and_render_as_behavior(self, store, clock):
        accepted = store.ingest(
            "u1",
            [
                {"kind": "interaction_preference", "key": key, "value": value,
                 "explicit": True, "confidence": 0.9}
                for key, value in self.NEW_DIMENSIONS
            ],
            source=_source(now=clock),
            now=clock,
        )

        assert len(accepted) == 4
        assert all(item["status"] == "confirmed" for item in accepted)

        context = store.build_preference_context("u1", "在吗", now=clock)
        assert "别用亲昵称谓" in context
        assert "回复尽量短" in context
        assert "被带到就自然岔开" in context
        assert "自称时用'我'" in context

    def test_unknown_value_is_rejected_for_new_dimensions(self, store, clock):
        accepted = store.ingest(
            "u1",
            [{"kind": "interaction_preference", "key": "reply_length", "value": "epic",
              "explicit": True, "confidence": 0.9}],
            source=_source(now=clock),
            now=clock,
        )

        assert accepted == []

    def test_all_dimension_values_have_behavior_sentences(self):
        from src.plugins.chatbot import memory_v2

        assert len(memory_v2._PREFERENCE_VALUES) == 8
        for key, values in memory_v2._PREFERENCE_VALUES.items():
            for value in values:
                behavior = memory_v2._PREFERENCE_BEHAVIORS.get((key, value))
                assert behavior, f"缺少行为句：{key}={value}"


class TestContextSegments:
    """`build_context` 拆成两段：靠前相处方式 / 中段约定（情绪已搬去会话记忆）。"""

    def _seed(self, store, clock):
        store.ingest(
            "u1",
            [
                {"kind": "interaction_preference", "key": "support_style", "value": "comfort",
                 "explicit": True, "confidence": 0.95},
                {"kind": "fact", "key": "preferred_name", "value": "小明",
                 "explicit": True, "confidence": 0.99},
                {"kind": "commitment", "key": "promise", "value": "下次给用户唱一段",
                 "actor": "assistant", "explicit": True, "confidence": 0.96},
            ],
            source=_source(now=clock, reply="行，下次给你唱一段"),
            now=clock,
        )

    def test_preference_segment_holds_behaviors_and_facts_only(self, store, clock):
        self._seed(store, clock)

        profile = store.build_preference_context("u1", "陪我说说话", now=clock)

        assert "先接住情绪" in profile
        assert "小明" in profile
        assert "未完成约定" not in profile
        assert "考试很失落" not in profile

    def test_commitment_segment_only_when_query_related(self, store, clock):
        self._seed(store, clock)

        assert "唱一段" in store.build_commitment_context("u1", "下次能唱吗，唱一段", now=clock)
        assert store.build_commitment_context("u1", "今天天气怎么样", now=clock) == ""

    def test_emotion_lives_in_session_memory_not_here(self, store, clock):
        """情绪不再是 v2 的 kind：抽到 emotional_state 一律拒收（它住在会话记忆）。"""
        accepted = store.ingest(
            "u1",
            [{"kind": "emotional_state", "key": "current_emotion", "value": "很失落",
              "explicit": True, "confidence": 0.9}],
            source=_source(now=clock),
            now=clock,
        )
        assert accepted == []

    def test_aggregate_context_includes_both_segments(self, store, clock):
        self._seed(store, clock)

        context = store.build_context("u1", "下次能唱吗，唱一段", now=clock)

        assert "先接住情绪" in context
        assert "唱一段" in context
        assert "考试很失落" not in context
        assert len(context) <= 700


class TestInjectionFlag:
    def test_injection_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("MEMORY_V2_INJECT", raising=False)
        assert injection_enabled() is True

    def test_injection_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("MEMORY_V2_INJECT", "0")
        assert injection_enabled() is False


def test_audit_log_is_jsonl_and_does_not_escape_user_partition(store, clock, tmp_path):
    store.ingest(
        "u1",
        [{
            "kind": "fact",
            "key": "preferred_name",
            "value": "小明",
            "explicit": True,
            "confidence": 0.98,
        }],
        source=_source(now=clock),
        now=clock,
    )

    audit_path = tmp_path / "memory_v2_audit.jsonl"
    entries = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert entries[0]["user_id"] == "u1"
    assert entries[0]["decision"] == "accepted"
    assert "state_file" not in entries[0]


def test_delete_user_removes_state_but_keeps_other_users(store, clock):
    item = {
        "kind": "fact",
        "key": "preferred_name",
        "value": "小明",
        "explicit": True,
        "confidence": 0.98,
    }
    store.ingest("u1", [item], source=_source(now=clock), now=clock)
    store.ingest("u2", [item | {"value": "小红"}], source=_source(now=clock), now=clock)

    store.delete_user("u1", now=clock)

    assert store.snapshot("u1")["memories"] == []
    assert store.snapshot("u2")["memories"][0]["value"] == "小红"


class TestShadowExtraction:
    def test_shadow_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("MEMORY_V2_SHADOW", raising=False)
        assert shadow_enabled() is True

    def test_shadow_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("MEMORY_V2_SHADOW", "0")
        assert shadow_enabled() is False

    def test_parser_accepts_strict_json_or_json_fence(self):
        payload = '{"items":[{"kind":"fact","key":"city","value":"广州"}]}'
        assert parse_extraction_payload(payload)["items"][0]["key"] == "city"
        assert parse_extraction_payload(f"```json\n{payload}\n```")["items"][0]["value"] == "广州"

    @pytest.mark.parametrize("content", ["", "null", "[]", "{broken", '{"items":"no"}'])
    def test_parser_fails_closed(self, content):
        assert parse_extraction_payload(content) == {"items": []}

    def test_prompt_separates_user_plan_from_assistant_commitment(self, store, clock):
        prompt = build_extraction_prompt(
            user_text="我打算周六直播",
            assistant_text="那你加油",
            current_summary="（无）",
            session_id="session-a",
            observed_at=clock,
        )

        assert "承诺只从灰泽满回复中提取" in prompt
        assert "用户自己的计划不是灰泽满的承诺" in prompt
        assert '"actor": "assistant"' in prompt

    def test_prompt_carries_admission_rule_and_counter_examples(self, store, clock):
        prompt = build_extraction_prompt(
            user_text="你更新了 V8.0 吗",
            assistant_text="嗯更新了",
            current_summary="（无）",
            session_id="session-a",
            observed_at=clock,
        )

        assert "会不会改变灰泽满下一次回复" in prompt
        assert "反例清单" in prompt
        assert "版本号" in prompt  # bot_version 那类误记的正面拦截
        assert "感知模块的活" in prompt  # 时间/天气交给 context_probe
        # 8 个偏好维度都要在提示词里（否则模型不会产出新维度）
        for key in ("address_style", "reply_length", "topic_preference", "self_reference"):
            assert key in prompt

    async def test_extract_and_ingest_uses_strict_schema(self, store, clock):
        captured = {}

        class FakeCompletions:
            async def create(self, **kwargs):
                captured.update(kwargs)
                content = json.dumps({
                    "items": [{
                        "kind": "interaction_preference",
                        "key": "support_style",
                        "value": "comfort",
                        "action": "upsert",
                        "explicit": True,
                        "confidence": 0.98,
                    }]
                }, ensure_ascii=False)
                message = type("Message", (), {"content": content})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        client = type(
            "Client", (),
            {"chat": type("Chat", (), {"completions": FakeCompletions()})()},
        )()

        accepted = await extract_and_ingest(
            client=client,
            model="test-model",
            store=store,
            user_id="u1",
            user_text="我难受时你先安慰我就好",
            assistant_text="知道了",
            session_id="session-a",
            observed_at=clock,
        )

        assert accepted[0]["status"] == "confirmed"
        assert captured["temperature"] == 0.1
        assert captured["max_tokens"] >= 400
        assert captured["extra_body"]["thinking"]["type"] == "disabled"

    async def test_extract_failure_does_not_write(self, store, clock):
        class BrokenCompletions:
            async def create(self, **kwargs):
                raise RuntimeError("upstream unavailable")

        client = type(
            "Client", (),
            {"chat": type("Chat", (), {"completions": BrokenCompletions()})()},
        )()

        accepted = await extract_and_ingest(
            client=client,
            model="test-model",
            store=store,
            user_id="u1",
            user_text="我叫小明",
            assistant_text="知道了",
            session_id="session-a",
            observed_at=clock,
        )

        assert accepted == []
        assert store.snapshot("u1")["memories"] == []
