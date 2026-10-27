import json
from datetime import datetime, timedelta

import pytest

from src.plugins.chatbot import memory_v2
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


def test_single_valued_fact_supersedes_even_when_timeless(store, clock):
    """名字/地点/时区这类**天然单值**的键，模型标成 timeless 也必须顶替。

    实测（2026-10-11）：某用户存了 **3 份** `preferred_name`（都标 timeless → 走"累积"），
    于是每轮注入 3 次名字，还叠上"她每句都叫名字"的正反馈。
    """
    for i, value in enumerate(("小明", "阿明")):
        store.ingest(
            "u1",
            [{"kind": "fact", "key": "preferred_name", "value": value,
              "temporal_scope": "timeless", "explicit": True, "confidence": 0.9}],
            source=_source(now=clock + timedelta(minutes=i),
                           user="我叫小明" if i == 0 else "以后叫我阿明"),
            now=clock + timedelta(minutes=i),
        )

    kept = [m["value"] for m in store.snapshot("u1")["memories"] if m["status"] != "superseded"]
    assert kept == ["阿明"]


def test_multi_valued_timeless_fact_still_accumulates(store, clock):
    """但 `identity` / `background` 这类**会有多条**的，不能跟着单值键一起顶替（会真丢信息）。"""
    for i, value in enumerate(("是大学生", "在澳洲留学")):
        store.ingest(
            "u1",
            [{"kind": "fact", "key": "background", "value": value,
              "temporal_scope": "timeless", "explicit": True, "confidence": 0.9}],
            source=_source(now=clock + timedelta(minutes=i)), now=clock + timedelta(minutes=i),
        )

    kept = [m["value"] for m in store.snapshot("u1")["memories"] if m["status"] != "superseded"]
    assert sorted(kept) == ["在澳洲留学", "是大学生"]


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


def test_generic_fact_key_is_kept_but_flagged(store, clock, tmp_path):
    """「什么都能装」的栏目名**只记不拦**（2026-10-16）。

    为什么不是直接拒：退掉 ≠ 清理干净，退掉 = **这件事压根没被记住**。
    实测那 5 条 `fact/moment` 里有一条是"用户向灰泽满求婚"——退了她以后就完全不知道有这回事。
    要的是**逼它改个具体名字**，不是**丢事**。所以先收下、打标记、写审计，跑几天看命中率。
    """
    accepted = store.ingest(
        "u1",
        [
            {"kind": "fact", "key": "moment", "value": "用户向灰泽满求婚",
             "explicit": True, "confidence": 0.9},
            {"kind": "fact", "key": "proposal", "value": "用户向灰泽满求婚",
             "explicit": True, "confidence": 0.9},
        ],
        source=_source(now=clock, user="请嫁给我"),
        now=clock,
    )
    # 两条都收下了——拦的话第一条就没了
    assert [m["key"] for m in accepted] == ["moment", "proposal"]

    rows = [json.loads(line) for line in
            (tmp_path / "memory_v2_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    flagged = {r["key"]: r.get("generic_key")
               for r in rows if r.get("decision") == "accepted"}
    assert flagged["moment"] is True, "笼统栏目名要在审计里留痕，才能攒命中率"
    assert flagged.get("proposal") is None, "具体栏目名不该被标记"


def test_generic_key_set_covers_the_names_actually_observed():
    """名单要盯着**真实出现过的**名字，不是凭想象列的——`moment` 是实测攒了 5 条的那个。"""
    from src.plugins.chatbot import memory_v2
    assert "moment" in memory_v2._GENERIC_FACT_KEYS
    assert "info" in memory_v2._GENERIC_FACT_KEYS
    assert "preferred_name" not in memory_v2._GENERIC_FACT_KEYS
    assert "location" not in memory_v2._GENERIC_FACT_KEYS


def test_prompt_pins_the_direction_of_address_style():
    """`address_style` 填的是「**他希望灰泽满怎么称呼他自己**」，方向不能反。

    实测踩坑 2026-10-16：同一个句式"我想叫你宝宝"，两条记录分别被填成 `nickname` 和 `plain`
    两个相反的值——不管哪个对，至少有一个是错的。
    """
    prompt = build_extraction_prompt(
        user_text="u", assistant_text="a", current_summary="",
        session_id="s", observed_at=datetime(2026, 10, 6, 20, 0, 0))
    assert "他希望灰泽满怎么称呼他自己" in prompt
    assert "我想叫你宝宝" in prompt, "光说「方向」不够，要有一条能被认出来的反例"


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


@pytest.mark.parametrize(("value", "hours"), [
    ("明天给用户发语音", 48),                     # 一次性措辞
    ("以后少说用户两句", 90 * 24),                 # 长期态度
    ("晚上陪用户学习", 14 * 24),                   # 默认承诺
])
def test_commitment_gets_ttl_by_wording(store, clock, value, hours):
    """2026-10-18：承诺必须有 TTL——旧数据里"明天…"会永久 active 被反复注入。"""
    accepted = store.ingest(
        "u1",
        [{"kind": "commitment", "key": "promise", "value": value,
          "actor": "assistant", "explicit": True, "confidence": 0.95}],
        source=_source(now=clock, reply=value),
        now=clock,
    )
    assert accepted[0]["expires_at"] == (clock + timedelta(hours=hours)).isoformat()


def test_expired_commitment_is_not_injected(store, clock):
    store.ingest(
        "u1",
        [{"kind": "commitment", "key": "promise", "value": "明天给用户发语音",
          "actor": "assistant", "explicit": True, "confidence": 0.95}],
        source=_source(now=clock, reply="明天给用户发语音"),
        now=clock,
    )
    assert "发语音" in store.build_commitment_context("u1", "明天能发语音吗", now=clock)
    assert store.build_commitment_context(
        "u1", "明天能发语音吗", now=clock + timedelta(days=3)) == ""


def test_same_key_commitment_supersedes_old_value(store, clock):
    """同一承诺 key 只留最新一条（旧实现会累积 3 条"明天发语音"）。"""
    for i, value in enumerate(("明天给用户发语音", "明天给用户发语音，说到做到")):
        store.ingest(
            "u1",
            [{"kind": "commitment", "key": "send_voice_tomorrow", "value": value,
              "actor": "assistant", "explicit": True, "confidence": 0.95}],
            source=_source(now=clock + timedelta(minutes=i), reply=value),
            now=clock + timedelta(minutes=i),
        )
    active = [m for m in store.snapshot("u1")["memories"]
              if m["kind"] == "commitment" and m["status"] == "active"]
    assert len(active) == 1
    assert active[0]["value"] == "明天给用户发语音，说到做到"


def test_her_nickname_is_always_on_fact(store, clock):
    """用户给她起的外号：存 her_nickname，且不因 query 不相关而丢。"""
    accepted = store.ingest(
        "u1",
        [{"kind": "fact", "key": "her_nickname", "value": "灰宝",
          "explicit": True, "confidence": 0.9}],
        source=_source(now=clock, user="我就叫你灰宝吧"),
        now=clock,
    )
    assert accepted and accepted[0]["key"] == "her_nickname"
    assert "her_nickname：灰宝" in store.build_preference_context("u1", "今天天气怎么样", now=clock)


def test_nickname_hint_only_when_user_calls_it(store, clock):
    """第一档：只有用户这一轮叫了外号才提示一次；默认仍自称灰泽满。"""
    store.ingest(
        "u1",
        [{"kind": "fact", "key": "her_nickname", "value": "灰宝",
          "explicit": True, "confidence": 0.9}],
        source=_source(now=clock, user="以后我就叫你灰宝了"),
        now=clock,
    )
    hint = store.nickname_hint("u1", "灰宝在吗", now=clock)
    assert "灰宝" in hint and "灰泽满" in hint
    assert store.nickname_hint("u1", "在吗", now=clock) == ""
    assert store.nickname_hint("u2", "灰宝在吗", now=clock) == ""


def test_nickname_hint_cooldown(store, clock):
    """用户连叫外号时，10 分钟内只提示一次，避免她每句都自称灰宝。"""
    store.ingest(
        "u1",
        [{"kind": "fact", "key": "her_nickname", "value": "灰宝",
          "explicit": True, "confidence": 0.9}],
        source=_source(now=clock, user="以后我就叫你灰宝了"),
        now=clock,
    )
    assert store.nickname_hint("u1", "灰宝在吗", now=clock)
    assert store.nickname_hint("u1", "灰宝", now=clock + timedelta(minutes=1)) == ""
    assert store.nickname_hint("u1", "灰宝", now=clock + timedelta(minutes=11))


def test_self_reference_dimension_is_removed(store, clock):
    """2026-10-19：整个 self_reference 维度删除——第三人称自称是核心人设，不能按用户改。"""
    for value in ("first_person", "nickname"):
        accepted = store.ingest(
            "u1",
            [{"kind": "interaction_preference", "key": "self_reference", "value": value,
              "explicit": True, "confidence": 0.9}],
            source=_source(now=clock), now=clock,
        )
        assert accepted == []


def test_preferred_name_requires_explicit_request(store, clock):
    """2026-10-19：'云子'是网络用语（嫌她不懂/很云），不是名字——没有称呼请求就拒收。"""
    rejected = store.ingest(
        "u1",
        [{"kind": "fact", "key": "preferred_name", "value": "云子",
          "explicit": True, "confidence": 0.9}],
        source=_source(now=clock, user="云子", reply="灰泽满才没有要睡觉呢"),
        now=clock,
    )
    assert rejected == []

    accepted = store.ingest(
        "u1",
        [{"kind": "fact", "key": "preferred_name", "value": "云子",
          "explicit": True, "confidence": 0.9}],
        source=_source(now=clock, user="我叫云子，记住了吗", reply="记住了"),
        now=clock,
    )
    assert accepted and accepted[0]["value"] == "云子"


def test_preferred_name_rejects_fan_identity_and_sentences(store, clock):
    for value, user_text in [
        ("绿冻", "我是绿冻"),
        ("用户希望被称呼为草草子", "你叫我草草子可以吗"),
    ]:
        accepted = store.ingest(
            "u1",
            [{"kind": "fact", "key": "preferred_name", "value": value,
              "explicit": True, "confidence": 0.9}],
            source=_source(now=clock, user=user_text, reply="嗯"),
            now=clock,
        )
        assert accepted == [], value


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
    """偏好维度从 4 个扩到 8 个，2026-10-19 删掉 self_reference → 7 个。"""

    NEW_DIMENSIONS = [
        ("address_style", "plain"),
        ("reply_length", "short"),
        ("topic_preference", "avoid"),
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

        assert len(accepted) == 3
        assert all(item["status"] == "confirmed" for item in accepted)

        context = store.build_preference_context("u1", "在吗", now=clock)
        assert "别用亲昵称谓" in context
        assert "回复尽量短" in context
        assert "被带到就自然岔开" in context
        assert "自称时用用户给她的昵称" not in context

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
        from src.plugins.chatbot import injection, memory_v2

        assert len(memory_v2._PREFERENCE_VALUES) == 7
        for key, values in memory_v2._PREFERENCE_VALUES.items():
            for value in values:
                behavior = injection.behavior_sentence(key, value)
                assert behavior, f"缺少行为句：{key}={value}"


class TestRitualPreference:
    """固定仪式（2026-10-23）：晚安仪式这类长期要求，只在触发场景出现时注入。"""

    def _ingest(self, store, clock, value="晚安：晚安+喜欢你+爱绿冻+mua"):
        return store.ingest(
            "u1",
            [{"kind": "interaction_preference", "key": "ritual_goodnight", "value": value,
              "explicit": True, "confidence": 0.95}],
            source=_source(now=clock), now=clock,
        )

    def test_ritual_validates(self, store, clock):
        accepted = self._ingest(store, clock)
        assert len(accepted) == 1
        assert accepted[0]["key"] == "ritual_goodnight"

    def test_ritual_without_cue_is_rejected(self, store, clock):
        # 没有「场景：内容」就无法按场景触发，直接拒收
        assert self._ingest(store, clock, value="晚安喜欢你") == []

    def test_injected_only_at_cue(self, store, clock):
        self._ingest(store, clock)
        night = store.build_preference_context("u1", "晚安哦", now=clock)
        assert "【他的固定期待】" in night and "喜欢你" in night
        day = store.build_preference_context("u1", "今天中午吃什么", now=clock)
        assert "【他的固定期待】" not in day

    def test_used_ids_only_when_injected(self, store, clock):
        self._ingest(store, clock)
        night_ids, day_ids = [], []
        store.build_preference_context("u1", "晚安", now=clock, used_ids=night_ids)
        store.build_preference_context("u1", "吃了吗", now=clock, used_ids=day_ids)
        assert night_ids and not day_ids

    def test_extract_prompt_teaches_ritual(self):
        """提取提示词必须留着 ritual 规则：单次“晚安”不记，固定仪式要记。"""
        from src.plugins.chatbot import memory_v2
        prompt = memory_v2.MEMORY_V2_EXTRACT_PROMPT
        assert "ritual_goodnight" in prompt
        assert "固定仪式" in prompt
        assert "仪式感的晚安" in prompt

    def test_extract_prompt_teaches_structured_habit(self):
        """习惯要带 type/cadence/period/cues，而不是只写一句“有…习惯”。"""
        from src.plugins.chatbot import memory_v2
        prompt = memory_v2.MEMORY_V2_EXTRACT_PROMPT
        assert '"habit"' in prompt
        for field in ("cadence", "period", "cues", "routine", "interaction", "once"):
            assert field in prompt


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
            source=_source(now=clock, user="我叫小明", reply="行，下次给你唱一段"),
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

    def test_used_ids_reports_only_injected_memories(self, store, clock):
        """条目级注入轨迹要的 id：**注入了才报**，没命中/没注入就不能报。"""
        self._seed(store, clock)
        known = {m["id"] for m in store.snapshot("u1")["memories"]}

        ids: list = []
        profile = store.build_preference_context("u1", "陪我说说话", now=clock, used_ids=ids)
        assert profile and ids and set(ids) <= known

        commit_ids: list = []
        assert store.build_commitment_context("u1", "下次能唱吗，唱一段", now=clock,
                                              used_ids=commit_ids)
        assert commit_ids and set(commit_ids) <= known

        miss_ids: list = []
        assert store.build_commitment_context("u1", "今天天气怎么样", now=clock,
                                              used_ids=miss_ids) == ""
        assert miss_ids == [], "没注入的不该出现在轨迹里"


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
        source=_source(now=clock, user="我叫小明"),
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
    store.ingest("u1", [item], source=_source(now=clock, user="我叫小明"), now=clock)
    store.ingest("u2", [item | {"value": "小红"}], source=_source(now=clock, user="我叫小红"), now=clock)

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
        # ⚠️ 契约里 **`daily` 也一起回空**：它跟 items 走同一次调用（2026-10-17），
        #    解析失败时两边都不能漏出半个结果。
        assert parse_extraction_payload(content) == {"items": [], "daily": []}

    def test_parser_carries_daily_events(self):
        """每日记忆搭同一次调用：`daily` 收成字符串列表，一轮最多 3 条。"""
        payload = ('{"items":[],"daily":[{"text":"中午吃了炸鸡"},{"text":"下午的课取消了"},'
                   '{"text":"晚上去跑步"},{"text":"第四条要被截掉"}]}')
        assert parse_extraction_payload(payload)["daily"] == [
            "中午吃了炸鸡", "下午的课取消了", "晚上去跑步"]

    def test_parser_tolerates_missing_or_dirty_daily(self):
        """老输出里没有 daily（或塞了脏东西）时，items 照常，daily 回空列表。"""
        assert parse_extraction_payload('{"items":[]}')["daily"] == []
        assert parse_extraction_payload('{"items":[],"daily":"中午吃了炸鸡"}')["daily"] == []
        assert parse_extraction_payload('{"items":[],"daily":[{"text":"  "},"裸字符串"]}')["daily"] == []

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

    def test_prompt_rejects_copypasta(self, store, clock):
        """网上抄来的定型文/段子/角色扮演 → 一条都不该记。

        2026-10-12 实测：某用户发"丫头，我重新介绍一下自己。我今年四十，河南周口人，
        现跑豫粤线大货…"——被当真存成 occupation/location/age/income/savings 五条，
        之后她张口就是"明天还得握方向盘的人"。加了这条反例后同样输入抽 0 条。
        """
        prompt = build_extraction_prompt(
            user_text="丫头，我重新介绍一下自己",
            assistant_text="……",
            current_summary="（无）",
            session_id="session-a",
            observed_at=clock,
        )
        assert "定型文" in prompt and "角色扮演" in prompt
        assert "一篇文章/一个模板" in prompt  # 判据：读起来像文章而不像他在说自己

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
        # 7 个偏好维度都要在提示词里（self_reference 已于 2026-10-19 删除）
        for key in ("address_style", "reply_length", "topic_preference"):
            assert key in prompt
        assert "her_nickname" in prompt
        assert "self_reference" not in prompt

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


_NAME_ITEM = {"kind": "fact", "key": "preferred_name", "value": "小明",
              "explicit": True, "confidence": 0.98}


def test_empty_ingest_does_not_rewrite_existing_user(store, clock):
    """提取为空的轮次不该把全体用户的状态文件整份重写（只会改一个没人读的 updated_at）。"""
    store.ingest("u1", [_NAME_ITEM], source=_source(now=clock, user="我叫小明"), now=clock)
    before = store.state_file.read_bytes()
    later = clock + timedelta(hours=1)
    assert store.ingest("u1", [], source=_source(now=later), now=later) == []
    assert store.state_file.read_bytes() == before
    assert store.snapshot("u1")["memories"][0]["value"] == "小明"


def test_empty_ingest_still_creates_new_user(store, clock):
    """新用户第一次（哪怕没提取到东西）照旧建档——离线工具按 users 数统计，口径不变。"""
    store.ingest("u2", [], source=_source(now=clock), now=clock)
    state = json.loads(store.state_file.read_text(encoding="utf-8"))
    assert state["users"]["u2"]["memories"] == []


class TestFactRelevanceHygiene:
    """扩写句的泛词不能给无关事实蹭分（2026-10-20，用户 2151983431）。"""

    MORNING = {"kind": "fact", "key": "morning_running",
               "value": "每天早上要去跑早操（校园跑），学校要求的"}
    TIME_DIFF = {"kind": "fact", "key": "time_diff",
                 "value": "与灰泽满时差约3小时（用户当地21点多时）"}
    GAOKAO = {"kind": "fact", "key": "gaokao_regret", "value": "从高考完一直怪自己没考好"}

    def test_expanded_query_matches_routine_only(self):
        q = "用户说自己去跑步了"
        assert memory_v2._fact_relevance(q, self.MORNING) > 0
        assert memory_v2._fact_relevance(q, self.TIME_DIFF) == 0, "“用户”不能算相关"
        assert memory_v2._fact_relevance(q, self.GAOKAO) == 0, "“自己”不能算相关"

    def test_long_query_does_not_use_single_char_fallback(self):
        assert memory_v2._fact_relevance("用户说自己去跑步了", self.TIME_DIFF) == 0

    def test_short_query_still_uses_char_fallback(self):
        m = {"key": "morning_running", "value": "每天早上要跑步"}
        assert memory_v2._fact_relevance("早安", m) > 0

    def test_unrelated_short_query_stays_zero(self):
        assert memory_v2._fact_relevance("在吗", self.MORNING) == 0


class TestTimePeriodFacts:
    """当前时段的固定安排（2026-10-20）：和查询无关也要进 profile 块。"""

    def _seed(self, store, clock):
        store.ingest("u1", [
            {"kind": "fact", "key": "morning_running",
             "value": "每天早上要去跑早操（校园跑），学校要求的",
             "temporal_scope": "timeless", "explicit": True, "confidence": 0.95},
            {"kind": "fact", "key": "evening_study_hall",
             "value": "晚上七点要上晚自习",
             "temporal_scope": "current", "explicit": True, "confidence": 0.95},
            {"kind": "fact", "key": "old_morning_routine",
             "value": "以前每天早上要跑步",
             "temporal_scope": "past", "explicit": True, "confidence": 0.9},
        ], source=_source(now=clock), now=clock)

    def test_morning_injects_morning_routine_only(self, store, clock):
        self._seed(store, clock)
        ctx = store.build_preference_context("u1", "你忘了吗",
                                             now=clock.replace(hour=6, minute=30))
        assert "每天早上要去跑早操" in ctx
        assert "晚自习" not in ctx
        assert "以前每天早上要跑步" not in ctx, "过去事实不算当前时段"

    def test_evening_injects_study_hall(self, store, clock):
        self._seed(store, clock)
        ctx = store.build_preference_context("u1", "你忘了吗", now=clock)   # 20:00
        assert "晚自习" in ctx
        assert "跑早操" not in ctx

    def test_query_selected_fact_not_duplicated(self, store, clock):
        self._seed(store, clock)
        ctx = store.build_preference_context("u1", "跑步", now=clock.replace(hour=6, minute=30))
        assert ctx.count("每天早上要去跑早操") == 1

    def test_time_fact_id_goes_into_used_ids(self, store, clock):
        self._seed(store, clock)
        ids: list = []
        store.build_preference_context("u1", "你忘了吗",
                                       now=clock.replace(hour=6, minute=30), used_ids=ids)
        mem = next(m for m in store.snapshot("u1")["memories"]
                   if m["key"] == "morning_running")
        assert mem["id"] in ids


class TestHabitHint:
    """结构化习惯：只对拥有这条记忆的用户、在真正相关的情况下出现（2026-10-26）。

    旧版靠“消息短 + value 里有早上”猜；这套测试锁住升级后的边界：
    - 作息（routine）才可能按时段召回，互动（interaction）只按线索召回；
    - 每周/一次性计划不冒充“每天的习惯”；
    - “在吗/早上吃什么”不算打招呼；
    - 否定/取消/最近已提过 → 不再追着问。
    """

    def _seed(self, store, clock, key="habit_morning_run",
              value="每天早上起床后都要跑步/跑操（学校要求）", habit=None):
        item = {
            "kind": "fact", "key": key, "value": value,
            "action": "upsert", "temporal_scope": "timeless", "valid_for": "none",
            "explicit": True, "confidence": 1.0,
        }
        if habit is not None:
            item["habit"] = habit
        return store.ingest("u1", [item], source=_source(now=clock), now=clock)

    def test_hint_only_for_owner_and_greeting(self, store, clock):
        self._seed(store, clock)
        morning = clock.replace(hour=7)
        hint = store.build_habit_hint("u1", "早", now=morning)
        # 只念 value，不把英文 key 塞给模型；话术不再写死“今天也去吗”
        assert "他每天早上起床后都要跑步/跑操" in hint
        assert "今天还照不照常" in hint
        assert "habit_morning_run" not in hint
        assert "今天也去" not in hint
        # 别人没有这条记忆 → 不出现
        assert store.build_habit_hint("u2", "早", now=morning) == ""
        # 同一个人的午饭话题 → 不把早操塞进去
        assert store.build_habit_hint("u1", "今天中午吃什么", now=morning) == ""
        # 短消息不等于问候：以前“在吗/早上吃什么”也会触发
        assert store.build_habit_hint("u1", "在吗", now=morning) == ""
        assert store.build_habit_hint("u1", "早上吃什么", now=morning) == ""

    def test_morning_template_does_not_assume_going_out(self, store, clock):
        """话术要适用于喝咖啡/背单词，不能写死“今天也去”。"""
        self._seed(store, clock, key="habit_morning_coffee",
                   value="每天早上都要喝一杯咖啡")
        hint = store.build_habit_hint("u1", "早", now=clock.replace(hour=7))
        assert "他每天早上都要喝一杯咖啡" in hint
        assert "今天还照不照常" in hint
        assert "今天也去" not in hint

    def test_interaction_habit_not_triggered_by_time(self, store, clock):
        """表情包/撒娇是互动习惯，不该在晚安时按“夜间时段”自动翻出来。"""
        self._seed(store, clock, key="habit_expression_sticker",
                   value="用户习惯用表情包撒娇、求语音",
                   habit={"type": "interaction", "cadence": "daily",
                          "period": "night", "cues": ["表情包", "撒娇", "语音"]})
        night = clock.replace(hour=22)
        assert store.build_habit_hint("u1", "晚安", now=night) == ""
        hint = store.build_habit_hint("u1", "发个表情包", now=night)
        assert "他习惯用表情包" in hint
        assert "不是这一轮必须执行" in hint

    def test_weekly_habit_not_triggered_by_morning_greeting(self, store, clock):
        """每周三次的 3 公里不是“每天早上”，不能在每个“早”里都提。"""
        self._seed(store, clock, key="habit_run_3km_weekly",
                   value="除早上的日常跑外，每周还有三次 3 公里跑")
        morning = clock.replace(hour=7)
        assert store.build_habit_hint("u1", "早", now=morning) == ""
        hint = store.build_habit_hint("u1", "今天跑了3公里", now=morning)
        assert "3 公里" in hint

    def test_one_off_plan_is_not_a_fixed_habit(self, store, clock):
        """今晚熬夜看比赛是一次性计划，不能当成每天的习惯。"""
        self._seed(store, clock, key="habit_stay_up_late",
                   value="今晚要熬夜看比赛")
        night = clock.replace(hour=3)
        assert store.build_habit_hint("u1", "又熬夜了", now=night) == ""
        assert store.build_habit_hint("u1", "看比赛吗", now=night) == ""

    def test_negative_or_cancelled_routine_is_not_prompted(self, store, clock):
        """他说“不跑了/取消了”时，提示不能还叫他“今天也去”。"""
        self._seed(store, clock)
        morning = clock.replace(hour=7)
        assert store.build_habit_hint("u1", "今天不跑操了", now=morning) == ""
        assert store.build_habit_hint("u1", "跑操取消了", now=morning) == ""

    def test_negative_check_does_not_swallow_sleep_state(self, store, clock):
        """"还没睡"是夜间状态的线索，不是“今天不熬夜”的例外。"""
        self._seed(store, clock, key="habit_late_night",
                   value="每天深夜凌晨还没睡")
        night = clock.replace(hour=3)
        assert "每天深夜" in store.build_habit_hint("u1", "还没睡", now=night)
        assert store.build_habit_hint("u1", "今天不熬夜了", now=night) == ""

    def test_recent_bot_mention_suppresses_repeat(self, store, clock):
        """最近几轮她已经提过这条习惯 → 这轮不再重复提示。"""
        self._seed(store, clock)
        morning = clock.replace(hour=7)
        assert store.build_habit_hint(
            "u1", "早", now=morning, recent_bot_text="今天也去跑操吗？") == ""
        assert store.build_habit_hint(
            "u1", "早", now=morning, recent_bot_text="今天天气不错") != ""

    def test_habit_metadata_is_derived_and_stored(self, store, clock):
        accepted = self._seed(store, clock)
        meta = accepted[0]["habit"]
        assert meta["type"] == "routine"
        assert meta["cadence"] == "daily"
        assert meta["period"] == "morning"
        assert "跑操" in meta["cues"]
        assert "跑步" in meta["cues"]

    def test_legacy_habit_without_meta_is_derived_on_read(self):
        meta = memory_v2._habit_meta(
            {"key": "habit_late_night", "value": "每天深夜凌晨还没睡"})
        assert meta["type"] == "routine"
        assert meta["cadence"] == "daily"
        assert meta["period"] == "night"

    def test_habit_meta_accepts_chinese_aliases(self):
        meta = memory_v2.normalize_habit_metadata(
            {"type": "互动", "cadence": "每周", "period": "晚上",
             "cues": "表情包，撒娇"},
            key="habit_x", value="随便")
        assert meta["type"] == "interaction"
        assert meta["cadence"] == "weekly"
        assert meta["period"] == "evening"
        assert meta["cues"] == ["表情包", "撒娇"]

    def test_same_habit_key_new_value_supersedes_old(self, store, clock):
        self._seed(store, clock, value="每天早上跑操")
        self._seed(store, clock, value="每天早上跑操（学校要求）")
        active = store._active_memories("u1", now=clock)
        values = [m["value"] for m in active if m["key"] == "habit_morning_run"]
        assert values == ["每天早上跑操（学校要求）"]

    def test_habit_excluded_from_generic_facts(self, store, clock):
        self._seed(store, clock)
        active = store._active_memories("u1", now=clock)
        selected = store._select_facts(active, "早")
        assert all(not str(m.get("key", "")).startswith("habit_") for m in selected)

    def test_hint_reaches_user_message_tail(self, store, clock, monkeypatch):
        from src.plugins.chatbot import core

        monkeypatch.setattr(core.context_probe, "get_now_context", lambda city="": "")
        monkeypatch.setattr(core, "load_schedule", lambda: {})
        msgs = core.build_message_list(
            "早", [], [], habit_hint="【你记得的事】测试")
        assert msgs[-1]["role"] == "user"
        assert msgs[-2]["content"] == "【你记得的事】测试"
