#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""注入 v3 的统一登记表。

一个 kind 一份 KindSpec：主体、有效期、权威、policy、取数 adapter、渲染来源、预算、
失败方向。**“什么时候注”只在这里定义**；build_frame / recall 只执行 policy。

文案仍归 `persona/injection/<NN>/block.json`；这里只登记“这块是什么、归谁、何时可见”。
"""
from __future__ import annotations

from dataclasses import dataclass

POLICIES = {"always", "context", "conditional", "tool", "guard"}
SUBJECTS = {"user", "her", "world", "conversation"}
VALIDITIES = {"stable", "week", "session", "day", "now", "past", "turn"}
AUTHORITIES = {"canon", "user_stated", "generated", "api"}
FAIL_MODES = {"inject", "omit", "empty"}


@dataclass(frozen=True)
class KindSpec:
    kind: str
    label: str
    subject: str
    validity: str
    authority: str
    policy: str
    adapter: str
    render: str              # block_id/key；tool 类可以是空串
    budget: int = 0
    fail_mode: str = "omit"
    block_ids: tuple = ()    # 这个 kind 对应的 block.json id（可多个）
    notes: str = ""

    def __post_init__(self):
        assert self.policy in POLICIES, f"{self.kind}: bad policy {self.policy}"
        assert self.subject in SUBJECTS, f"{self.kind}: bad subject {self.subject}"
        assert self.validity in VALIDITIES, f"{self.kind}: bad validity {self.validity}"
        assert self.authority in AUTHORITIES, f"{self.kind}: bad authority {self.authority}"
        assert self.fail_mode in FAIL_MODES, f"{self.kind}: bad fail_mode {self.fail_mode}"


# 顺序 = 现有注入顺序（缓存/语义顺序，别随便动）。tool/guard 不在 frame 里渲染。
FRAME_ORDER = (
    "scene", "identity", "schedule",
    "interaction_preference", "name_budget", "request", "preference", "core_story",
    "group", "group_thread", "group_address", "terms", "association",
    "session", "prev_session", "behaviors", "corpus", "commitment",
    "history", "same_request", "batch", "query_hint",
    "frame_now", "session_time_hint", "mood", "nickname_hint", "daily", "habit",
)

REGISTRY: dict[str, KindSpec] = {s.kind: s for s in (
    KindSpec("identity", "骨架", "world", "stable", "canon", "always",
             "persona.system_prompt", "identity/raw", 2802, "inject", ("identity",)),
    KindSpec("scene", "群聊场景", "conversation", "session", "canon", "always",
             "core.is_group", "scene/group", 200, "omit", ("scene",)),
    KindSpec("schedule", "周表", "her", "week", "canon", "always",
             "persona.schedule", "schedule/main", 200, "omit", ("schedule",)),
    KindSpec("interaction_preference", "相处方式/行为句", "user", "stable", "user_stated",
             "conditional", "memory_v2.interaction_preference", "profile/behaviors", 360,
             "omit", ("profile",),
             "它是“怎么对她”，不是可查事实；默认 conditional，recall 可补充"),
    KindSpec("name_budget", "称呼预算", "user", "session", "generated", "conditional",
             "core.name_budget_note", "profile/name_budget", 80, "omit", ("profile",)),
    KindSpec("request", "长期请求", "user", "stable", "user_stated", "conditional",
             "memory_v2.request", "profile/request", 240, "omit", ("profile",),
             "用户希望她以后怎么对待自己；背景事实，不是本轮待办；频率交给输出预算"),
    KindSpec("preference", "全局偏好", "her", "stable", "canon", "tool",
             "retrieval.retrieve_preferences", "preferences/main", 200, "empty", ("preferences",)),
    KindSpec("core_story", "核心记忆", "her", "past", "canon", "tool",
             "retrieval.retrieve_core_stories", "core_stories/main", 400, "empty", ("core_stories",)),
    KindSpec("group", "群聊现场", "conversation", "session", "generated", "context",
             "group_memory.get_group", "group/main", 0, "omit", ("group",)),
    KindSpec("group_thread", "群聊对话线", "conversation", "turn", "generated", "context",
             "core.summarize_group_batch", "group/thread_note", 0, "empty", ("group",)),
    KindSpec("group_address", "群聊点名", "conversation", "turn", "generated", "context",
             "core.group_address_note", "group/address_note", 0, "empty", ("group",)),
    KindSpec("terms", "名词库", "world", "stable", "canon", "conditional",
             "core.build_terms_note", "terms/main", 200, "omit", ("terms",),
             "绿冻常驻；其余 keyword/LLM 确认；recall(kind=term) 补充"),
    KindSpec("association", "关联背景", "world", "session", "canon", "conditional",
             "retrieval.resolve_associations", "terms/associations", 0, "omit", ("terms",)),
    KindSpec("session", "会话记忆", "conversation", "session", "generated", "context",
             "session_memory.build_session_context", "session/main", 0, "omit", ("session",)),
    KindSpec("prev_session", "上次聊过", "conversation", "past", "generated", "context",
             "session_memory.previous_session_note", "prev_session/main", 0, "omit", ("prev_session",)),
    KindSpec("behaviors", "行为指令", "conversation", "turn", "canon", "conditional",
             "routing.classify_l3", "behaviors/main", 650, "omit", ("behaviors",),
             "风格一致性不赌模型是否查工具"),
    KindSpec("corpus", "直播记忆", "her", "past", "canon", "tool",
             "retrieval.corpus+judge", "corpus/main", 400, "empty", ("corpus",)),
    KindSpec("commitment", "未完成约定", "conversation", "day", "user_stated", "conditional",
             "memory_v2.commitment", "commitment/main", 220, "omit", ("commitment",),
             "漏了是关系错误，不赌模型自觉"),
    KindSpec("history", "最近对话", "conversation", "session", "generated", "context",
             "memory.get_user_history_timed", "history/main", 0, "omit", ("history",)),
    KindSpec("same_request", "连续索要", "conversation", "session", "generated", "context",
             "routing.classify_l3", "same_request/main", 80, "omit", ("same_request",)),
    KindSpec("batch", "批次归纳", "conversation", "turn", "generated", "context",
             "chat_window.summarize_batch", "batch/main", 120, "empty", ("batch",)),
    KindSpec("query_hint", "短句语境", "conversation", "turn", "generated", "context",
             "session_memory.probe_session", "query_hint/short|emoji", 80, "omit", ("query_hint",)),
    KindSpec("frame_now", "日期/星期/直播状态", "world", "now", "api", "always",
             "context_probe.frame_now", "now/frame", 80, "omit", ("now",)),
    KindSpec("session_time_hint", "本场时段", "conversation", "session", "api", "context",
             "context_probe.session_time_hint", "now/time_hint", 40, "omit", ("now",),
             "开场/问候/作息触发，不每轮"),
    KindSpec("mood", "当下情绪", "user", "session", "generated", "context",
             "session_memory.mood_note", "mood/main", 80, "omit", ("mood",)),
    KindSpec("nickname_hint", "外号提示", "user", "turn", "generated", "conditional",
             "memory_v2.nickname_hint", "profile/nickname_hint", 60, "omit", ("profile",)),
    KindSpec("daily", "今天的日常", "user", "day", "user_stated", "tool",
             "daily_memory.recall_items", "daily/main", 120, "empty", ("daily",)),
    KindSpec("habit", "个人习惯", "user", "stable", "user_stated", "conditional",
             "memory_v2.build_habit_hint", "profile/habit_hint", 120, "omit", ("profile",),
             "问候/时段/cues 命中；recall(kind=habit) 补充"),
    KindSpec("fact", "用户事实", "user", "past", "user_stated", "tool",
             "memory_v2.fact", "profile/fact", 300, "empty", ("profile",)),
    KindSpec("weather", "天气", "user", "now", "api", "tool",
             "context_probe.weather_now", "now/weather", 80, "empty", ("now",),
             "owner/time 必须；未知不返回"),
    KindSpec("perception", "精确时间/农历", "world", "now", "api", "tool",
             "context_probe.perception_now", "now/perception", 80, "empty", ("now",)),
    KindSpec("guard", "输出侧守卫", "conversation", "turn", "canon", "guard",
             "reply_style/core", "", 0, "omit", ()),
)}


def spec(kind: str) -> KindSpec:
    try:
        return REGISTRY[kind]
    except KeyError:
        raise KeyError(f"没有这个 kind：{kind}（现有：{', '.join(REGISTRY)}）") from None


def policy(kind: str) -> str:
    return spec(kind).policy


def frame_specs() -> list[KindSpec]:
    """按现有注入顺序返回 frame 里出现的 kind（tool/guard 跳过）。"""
    return [spec(k) for k in FRAME_ORDER
            if spec(k).policy in {"always", "context", "conditional"}]


def tool_kinds() -> list[str]:
    return [k for k, s in REGISTRY.items() if s.policy == "tool"]


def validate_blocks(block_ids: set[str]) -> list[str]:
    """测试用：每个 block.json 至少被一个 kind 认领；返回缺失的 block id。"""
    covered = set()
    for s in REGISTRY.values():
        covered.update(s.block_ids)
    return sorted(set(block_ids) - covered)


__all__ = ["KindSpec", "REGISTRY", "FRAME_ORDER", "spec", "policy",
           "frame_specs", "tool_kinds", "validate_blocks"]