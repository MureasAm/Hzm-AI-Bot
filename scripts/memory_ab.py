"""Offline A/B for the memory layer: same real private turns → v1 card vs v2 store.

Not a production path.  It replays historical private chats through **both**
extractors — v1 (``memory_manager`` free-text card) and v2 (``memory_v2`` typed
items) — and reports what each ends up holding and what it would inject.

Everything is derived and written under the gitignored ``outputs/memory_ab``
directory.  Raw source excerpts only ever appear in that local output.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterator

try:  # direct script via run_tool.py
    import _common
    import memory_v2_tool as _v2tool
except ModuleNotFoundError:  # imported as scripts.memory_ab in tests
    from . import _common
    from . import memory_v2_tool as _v2tool

# 回放要 import `src.plugins.chatbot.*`；run_tool 只把 scripts/ 加进 sys.path，
# 项目根得自己补（和 memory_v2_tool 一样）。
import sys as _sys  # noqa: E402

if str(_common.PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_common.PROJECT_ROOT))


DEFAULT_OUT_DIR = _common.OUTPUTS_DIR / "memory_ab"

# v1 在 core.update_memory_task 里给提示词追加的强制规则（此处原样复刻，保证回放忠实）。
V1_FORCED_RULE = (
    "\n【本轮的强制规则】new_self_fact 一律返回 null。只提取关于用户的信息"
    "（new_impression / new_user_fact），不要从灰泽满的回复中提取任何自我披露内容。"
)

V1_CONTENT_FIELDS = (
    "impressions", "user_facts", "significant_moments", "promises", "user_name", "weather_city",
)
V1_ITEM_FIELDS = ("impressions", "user_facts", "significant_moments", "promises")

# 与 memory_v2._BOT_SELF_VALUE_SUBJECTS 对齐：v1 没有类型校验，只能用文本粗判噪声。
_BOT_SELF_SUBJECTS = ("灰泽满", "hzm", "机器人", "assistant")


def _response_text(response) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(getattr(part, "text", ""))
            for part in content
        )
    return ""


def v1_gate(msg: str) -> bool:
    """v1 的密度门控：太短或纯表情的消息不提取（core.update_memory_task 同款）。"""
    text = (msg or "").strip()
    if not text or len(text) < 4:
        return False
    if text.startswith("[表情：") and text.endswith("]"):
        return False
    return True


# 语料里混着 QA/联调脚本造的"假会话"（`问题1:…\n回答:…`、`人物1:…`、session 叫 `u4`）。
# 它们条数还特别多，按活跃度排序会先被捞出来，把 A/B 带偏，直接过滤掉。
_SYNTHETIC_RE = re.compile(r"^\s*(?:问题|人物)\s*\d|^\s*回答\s*[:：]|[\r\n]\s*(?:问题|人物)\s*\d")


def looks_synthetic(text: str) -> bool:
    return bool(_SYNTHETIC_RE.search(text or ""))


def iter_turns_by_user(paths, *, limit: int, per_user: int) -> Iterator[dict]:
    """按会话轮转取历史私聊：每人最多 ``per_user`` 条，铺开到多人（默认回放只吃前几个文件）。

    一个会话一个文件（2026-10-09 起），按 mtime 顺序轮转，保证 A/B 覆盖多个人而不是一个人。
    只保留真实真人会话：session 必须是 QQ 号（纯数字），且不是 QA/联调脚本造的假会话。
    """
    yielded = 0
    for path in paths:
        if yielded >= limit:
            return
        if not Path(path).stem.isdigit():  # 跳过 `u4` 这类联调/测试会话
            continue
        try:
            handle = Path(path).open("r", encoding="utf-8")
        except OSError:
            continue
        taken = 0
        with handle:
            for line in handle:
                if yielded >= limit or taken >= per_user:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict) or record.get("kind") != "private":
                    continue
                session = str(record.get("session") or "").strip()
                user_text = str(record.get("user") or "").strip()
                reply = str(record.get("reply") or "").strip()
                if not session or not user_text or not reply:
                    continue
                if looks_synthetic(user_text):
                    continue
                yielded += 1
                taken += 1
                yield record


def order_paths_by_activity(paths) -> list[Path]:
    """按私聊条数降序排会话文件——记忆层对"常来的用户"才有意义。

    默认（mtime / 文件名）顺序会把统计摊在一堆只聊过几句的人身上，两路都抽不出东西、
    对比全是噪声。先取最活跃的会话，A/B 才有信息量。
    """
    def private_lines(path: Path) -> int:
        if not Path(path).stem.isdigit():
            return -1  # 非 QQ 号会话（联调/测试）排到最后
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                return sum(1 for line in handle if '"private"' in line)
        except OSError:
            return 0

    return sorted(paths, key=private_lines, reverse=True)


def _v1_noise_scan(cards: dict) -> list[str]:
    """v1 没有类型守卫：扫出明显"讲机器人自己"的条目，当噪声统计。"""
    hits: list[str] = []
    for card in cards.values():
        for field in ("user_facts", "impressions", "significant_moments"):
            for entry in card.get(field) or []:
                text = entry.get("fact") if isinstance(entry, dict) else None
                if text is None and isinstance(entry, dict):
                    text = entry.get("tag") or entry.get("summary")
                if not isinstance(text, str):
                    text = str(entry)
                if text.strip().startswith(_BOT_SELF_SUBJECTS):
                    hits.append(f"{field}: {text[:60]}")
    return hits


def _v2_noise_scan(state: dict) -> list[str]:
    hits: list[str] = []
    for user in (state.get("users") or {}).values():
        for memory in user.get("memories") or []:
            if memory.get("kind") != "fact":
                continue
            key = str(memory.get("key") or "")
            value = str(memory.get("value") or "")
            if key.startswith(("bot", "hzm")) or value.strip().startswith(_BOT_SELF_SUBJECTS):
                hits.append(f"{key}: {value[:60]}")
    return hits


def _card_items(card: dict) -> int:
    return sum(len(card.get(f) or []) for f in V1_ITEM_FIELDS)


def _injection_stats(sizes: list[int]) -> dict:
    if not sizes:
        return {"n": 0, "median": 0, "mean": 0, "max": 0}
    return {
        "n": len(sizes),
        "median": int(statistics.median(sizes)),
        "mean": round(statistics.mean(sizes), 1),
        "max": max(sizes),
    }


def build_report(v1_cards: dict, v2_state: dict, *, v1_contexts: dict, v2_contexts: dict,
                 samples: list[dict], turns: int) -> dict:
    cards_with_content = {uid: c for uid, c in v1_cards.items() if any(c.get(f) for f in V1_CONTENT_FIELDS)}
    v1_totals = {field: sum(len(c.get(field) or []) for c in v1_cards.values()) for field in V1_ITEM_FIELDS}
    v1_totals["user_name"] = sum(1 for c in v1_cards.values() if c.get("user_name"))
    v1_totals["weather_city"] = sum(1 for c in v1_cards.values() if c.get("weather_city"))

    by_kind: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    timebound_facts = 0
    for user in (v2_state.get("users") or {}).values():
        for memory in user.get("memories") or []:
            by_kind[str(memory.get("kind") or "unknown")] += 1
            by_status[str(memory.get("status") or "unknown")] += 1
            if memory.get("kind") == "fact" and str(memory.get("valid_for") or "none") != "none":
                timebound_facts += 1

    v2_total = sum(by_kind.values())
    return {
        "turns_replayed": turns,
        "users": {"v1": len(v1_cards), "v2": len(v2_state.get("users") or {})},
        "v1": {
            "cards_with_content": len(cards_with_content),
            "empty_cards": len(v1_cards) - len(cards_with_content),
            "totals": v1_totals,
            "items": sum(v1_totals[f] for f in V1_ITEM_FIELDS),
            "injection_chars": _injection_stats([len(x) for x in v1_contexts.values()]),
            "bot_self_noise": len(_v1_noise_scan(v1_cards)),
            "bot_self_samples": _v1_noise_scan(v1_cards)[:8],
        },
        "v2": {
            "memories": v2_total,
            "by_kind": dict(sorted(by_kind.items())),
            "by_status": dict(sorted(by_status.items())),
            "injection_chars": _injection_stats([len(x) for x in v2_contexts.values()]),
            "timebound_facts": timebound_facts,
            "bot_self_noise": len(_v2_noise_scan(v2_state)),
            "bot_self_samples": _v2_noise_scan(v2_state)[:8],
        },
        "samples": samples,
    }


async def run_ab(*, paths, limit: int, per_user: int, out_dir: Path) -> dict:
    from openai import AsyncOpenAI

    import nonebot

    try:
        nonebot.get_driver()
    except ValueError:
        nonebot.init()

    import memory_manager as mm
    from src.plugins.chatbot import memory_v2
    from src.plugins.chatbot.constants import MEMORY_EXTRACT_TEMPERATURE, THINKING_DISABLED
    from src.plugins.chatbot.routing import _repair_llm_json

    api_key = _common.get_api_key("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("未检测到 OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key, base_url=_common.get_openai_base_url())
    model = _common.get_model_name()

    paths = order_paths_by_activity(paths)

    out_dir = _v2tool._safe_replay_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("ab_v2_shadow.json", "ab_v2_audit.jsonl"):
        path = out_dir / stale
        if path.exists():
            path.unlink()
    store = memory_v2.ShadowMemoryStore(
        state_file=out_dir / "ab_v2_shadow.json",
        audit_file=out_dir / "ab_v2_audit.jsonl",
    )

    v1_cards: dict[str, dict] = {}

    async def v1_turn(card: dict, user_msg: str, reply: str) -> dict:
        prompt = mm.MEMORY_EXTRACT_PROMPT.format(
            current_summary=mm._format_profile_summary(card),
            user_msg=user_msg,
            reply=reply,
        ) + V1_FORCED_RULE
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=MEMORY_EXTRACT_TEMPERATURE,
                max_tokens=100,
                **THINKING_DISABLED,
            )
        except Exception as exc:
            print(f"[AB v1] 提取调用失败: {exc!r}")
            return card
        content = (_response_text(resp) or "").strip()
        if not content or content == "null":
            return card
        try:
            updates = json.loads(_repair_llm_json(content))
        except Exception:
            return card
        if not isinstance(updates, dict):
            return card
        return mm.merge_memory_card(card, updates)

    turns = 0
    last_query: dict[str, str] = {}
    last_obs: dict[str, datetime] = {}
    for record in iter_turns_by_user(paths, limit=limit, per_user=per_user):
        user_msg = str(record["user"])
        reply = str(record["reply"])
        uid = str(record["session"])
        if not v1_gate(user_msg):
            continue
        turns += 1
        try:
            timestamp = float(record.get("t") or 0)
        except (TypeError, ValueError):
            timestamp = 0
        observed_at = datetime.fromtimestamp(timestamp) if timestamp > 0 else datetime.now()
        scope = f"{uid}:{int(timestamp // (6 * 3600)) if timestamp else turns}"

        card = await v1_turn(v1_cards.get(uid, {}), user_msg, reply)
        # merge_memory_card 把 last_seen 写成 now()；这里改回**历史时刻**，
        # 否则回放里"距上次跟TA说话"永远显示成"几分钟前"（假精度）。
        card["last_seen"] = observed_at.isoformat()
        v1_cards[uid] = card
        last_query[uid] = user_msg
        last_obs[uid] = observed_at
        try:
            await memory_v2.extract_and_ingest(
                client=client, model=model, store=store, user_id=uid,
                user_text=user_msg, assistant_text=reply, session_id=scope,
                observed_at=observed_at,
            )
        except Exception as exc:
            print(f"[AB v2] 提取调用失败: {exc!r}")

    await client.close()

    state = _v2tool.load_state(out_dir / "ab_v2_shadow.json")

    # 每个用户"会被注入什么"：v1 整卡 vs v2 三段聚合（用他最后一条消息当话题）。
    # ⚠️ v2 的 now 要用**该用户最后一轮的历史时刻**，不能用 wall-clock：
    #    回放的是几周前的对话，拿"现在"去算 TTL，情绪层会全部判过期（对比就假了）。
    v1_contexts = {uid: mm.build_memory_context(card) for uid, card in v1_cards.items()}
    v2_contexts = {
        uid: store.build_context(uid, last_query.get(uid, ""), now=last_obs.get(uid))
        for uid in v1_cards
    }

    ranked = sorted(
        v1_cards,
        key=lambda u: (_card_items(v1_cards[u]), len(last_query.get(u, ""))),
        reverse=True,
    )
    samples = []
    for uid in ranked[:5]:
        samples.append({
            "user": _v2tool.redact_user_id(uid),
            "v1": v1_contexts[uid][:600],
            "v2": v2_contexts[uid][:600],
        })

    report = build_report(
        v1_cards, state,
        v1_contexts=v1_contexts, v2_contexts=v2_contexts,
        samples=samples, turns=turns,
    )
    (out_dir / "ab_v1_cards.json").write_text(
        json.dumps({_v2tool.redact_user_id(k): v for k, v in v1_cards.items()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "ab_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_ab_cli(args) -> None:
    paths = _v2tool._resolve_inputs(args.input)
    report = asyncio.run(run_ab(
        paths=paths, limit=args.limit, per_user=args.per_user,
        out_dir=Path(args.out_dir),
    ))
    print(json.dumps(report, ensure_ascii=False, indent=2))


__all__ = [
    "run_ab", "run_ab_cli", "build_report", "iter_turns_by_user",
    "order_paths_by_activity", "looks_synthetic", "v1_gate", "V1_FORCED_RULE",
]
