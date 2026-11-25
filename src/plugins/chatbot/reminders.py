# -*- coding: utf-8 -*-
"""定时提醒：读 memory_v2 里带 due_at 的 commitment，到点用她的口吻私聊发出。

提醒的内容和时间由一条 commitment 保存，发出后标为 done，错过补发期限时标为 missed。
实际发出的那句话通过私聊回复档案回查。
识别在 `routing.classify_l3` 的 `reminder` 槽，落库在 `memory_v2` 的 commitment 提取。

调度用项目现有的 `on_startup` + `spawn` 轮询，不引入 APScheduler/Redis：
单进程、量小，每 20 秒扫一次足够。
"""
import asyncio
import datetime
import os
import re
from uuid import uuid4

from nonebot import get_bot, get_driver
from nonebot.adapters.onebot.v11 import Message, MessageSegment

from .bg import spawn
from . import context_probe, daily_memory, injection as inj, reply_evidence
from .memory import get_user_history_timed
from .session_memory import mood_note
from .reply_style import planning_leak_reason

CHECK_SEC = 20
LATE_SKIP_HOURS = 6


def _naive(dt: datetime.datetime) -> datetime.datetime:
    return dt.astimezone().replace(tzinfo=None) if dt.tzinfo else dt


def _task_text(value: str) -> str:
    """commitment 的 value 可能是「提醒他扔垃圾」；提醒话术里只需要「扔垃圾」。"""
    raw = (value or "").strip()
    return re.sub(r"^(?:提醒|叫)(?:他|她|对方|你)", "", raw).strip() or raw


async def _render(item: dict, now: datetime.datetime) -> str:
    """走和聊天同一套 build_message_list：人格块 + 用户记忆 + 最近对话 + 时间。

    之前只用一条极简 system prompt，生成出来是「到点了，该扔垃圾了」这种客服腔；
    这里改成完整注入，让她知道对方是谁、刚才聊过什么、自己的日程和相处方式。
    """
    from .core import _MEMORY_V2_STORE, _FALLBACK_REPLY, build_message_list, generate_reply

    user_id = str(item.get("user_id") or "")
    task = _task_text(str(item.get("value") or ""))
    fallback = f"到点了，{task}别忘了。"
    if not user_id:
        return fallback
    try:
        state_stable = ""
        card = _MEMORY_V2_STORE.build_state_card(user_id)
        if card:
            state_stable = inj.T("state", "stable", lines=card)
        commitment_context = _MEMORY_V2_STORE.build_commitment_context(
            user_id, task, used_ids=[])
        daily_context = daily_memory.context(user_id) if daily_memory.enabled() else ""
        try:
            emotion_context = mood_note(user_id)
        except Exception:
            emotion_context = ""
        try:
            history = get_user_history_timed(user_id)
        except Exception:
            history = []
        try:
            now_context = context_probe.frame_now()
        except Exception:
            now_context = ""
        prompt = (
            f"【系统提醒】现在是 {now:%H:%M}，到点了。之前你答应过提醒他：{task}。\n"
            "请你主动私聊发一句提醒。结合你们最近聊的内容、你对他的称呼和相处方式，"
            "像平时一样自然地说一句；不要解释这是提醒系统，不要写旁白、不要列方案。"
        )
        messages = build_message_list(
            prompt, [], history,
            commitment_context=commitment_context,
            daily_context=daily_context,
            emotion_context=emotion_context,
            state_stable=state_stable,
            now_context=now_context,
        )
        out = await generate_reply(messages)
        out = (out or "").strip()
        if out and out != _FALLBACK_REPLY and not planning_leak_reason(out):
            return out.split("\n")[0].strip() or fallback
    except Exception as e:
        print(f"⚠️ 提醒话术生成失败（用模板）: {type(e).__name__}: {e}")
    return fallback


async def tick(now: datetime.datetime | None = None) -> int:
    """扫一次到期 commitment；返回本次成功发出的条数。"""
    from .core import _MEMORY_V2_STORE

    now = _naive(now or datetime.datetime.now())
    due = _MEMORY_V2_STORE.due_commitments(now)
    if not due:
        return 0
    try:
        bot = get_bot()
    except Exception:
        return 0  # 还没连上 QQ，下一轮再试
    sent = 0
    for item in due:
        user_id = str(item.get("user_id") or "")
        if not user_id.isdigit():
            continue
        try:
            due_at = datetime.datetime.fromisoformat(str(item.get("due_at")))
            due_at = _naive(due_at)
        except (TypeError, ValueError):
            due_at = now
        if now - due_at > datetime.timedelta(hours=LATE_SKIP_HOURS):
            print(f"[提醒] 过期超过 {LATE_SKIP_HOURS} 小时，跳过: {item.get('value')}")
            _MEMORY_V2_STORE.mark_reminder_missed(user_id, item.get("id"), now)
            continue
        msg = await _render(item, now)
        message = Message(MessageSegment.text(msg))
        try:
            receipt = await bot.send_private_msg(user_id=int(user_id), message=message)
        except Exception as e:
            print(f"⚠️ 提醒发送失败（下轮重试）: {type(e).__name__}: {e}")
            continue
        turn_id = uuid4().hex
        message_id = receipt.get("message_id") if isinstance(receipt, dict) else None
        if not reply_evidence.record_part(user_id, turn_id, 0, msg, message_id=str(message_id or "")):
            turn_id = ""
        _MEMORY_V2_STORE.mark_reminder_delivered(user_id, item.get("id"), now, turn_id=turn_id)
        # 和主动发言一样：她主动发出去的话不进事件循环，必须手动写进短期记忆，
        # 否则对方顺着提醒回她时，她的【最近对话记录】里没有这一句。
        try:
            from .memory import append_bot_message
            append_bot_message(user_id, msg)
        except Exception:
            pass
        sent += 1
        print(f"[提醒] 已发送 -> {user_id}: {msg}")
    return sent


async def _loop() -> None:
    while True:
        try:
            await tick()
        except Exception as e:
            print(f"⚠️ 提醒循环异常（忽略）: {type(e).__name__}: {e}")
        await asyncio.sleep(CHECK_SEC)


@get_driver().on_startup
async def _start_reminders() -> None:
    if os.environ.get("REMINDERS", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    spawn(_loop())
