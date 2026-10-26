#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""原生 `recall` 工具轮（注入 v3 的可选路径）。

- 只在能力白名单/探测通过的模型上带 tools；不支持的层直接不带。
- 每轮最多 1 个工具轮（≤2 次模型调用）、最多 2 个 recall、总执行预算 3s。
- 第一次调用如果没有 tool_calls，它本身就是主回复（零额外调用）。
- 工具失败/超时 → 返回一条“查不到”的 tool 结果，模型继续生成；不发异常给用户。
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Awaitable, Callable

from .config import (_get_clients, _get_model_name, extract_chat_content,
                     get_flag, log_cache_usage)
from .constants import THINKING_DISABLED
from . import recall as recall_mod

_MAX_TOOL_CALLS = 2
_RECALL_TIMEOUT = 1.5
_WEATHER_TIMEOUT = 2.5
_TOTAL_TIMEOUT = 3.0

_DEFAULT_CAPABLE = {
    "gemini-2.5-flash-lite", "gemini-3.1-flash-lite", "deepseek-flash",
}
_capability_cache: dict[str, bool] = {}


def tools_enabled() -> bool:
    return get_flag("RECALL_TOOLS", "0").strip().lower() not in ("0", "false", "no", "off")


def _capable_models() -> set[str]:
    raw = get_flag("TOOL_CAPABLE_MODELS", "").strip()
    if raw:
        return {m.strip() for m in raw.split(",") if m.strip()}
    return set(_DEFAULT_CAPABLE)


def tool_capable(model: str | None = None) -> bool:
    model = (model or _get_model_name()).strip()
    if model in _capability_cache:
        return _capability_cache[model]
    ok = model in _capable_models()
    # 没有探测的层按白名单；探测失败/关闭不额外发请求。
    _capability_cache[model] = ok
    return ok


_PING_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "测试工具调用能力；收到请求时调用它，参数 ok=true。",
        "parameters": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                       "required": ["ok"]},
    },
}


async def ensure_tool_capable(client=None, model: str | None = None) -> bool:
    """一次性探测（TOOL_PROBE=1）；默认只查白名单。探测结果按模型缓存到进程。"""
    model = (model or _get_model_name()).strip()
    if model in _capability_cache:
        return _capability_cache[model]
    probe = get_flag("TOOL_PROBE", "0").strip().lower() in ("1", "true", "on", "yes")
    if not probe:
        return tool_capable(model)
    client = client or _get_clients()[0]
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "请调用 ping 工具，参数 ok=true。"}],
            tools=[_PING_TOOL], tool_choice="auto", max_tokens=20,
            **THINKING_DISABLED,
        )
        ok = bool(getattr(_message_of(resp), "tool_calls", None))
    except Exception:
        ok = False
    _capability_cache[model] = ok
    return ok


def tool_defs() -> list[dict]:
    return [recall_mod.TOOL_SCHEMA]


def _message_of(resp):
    try:
        return resp.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return None


def _tool_call_dict(tc) -> dict:
    fn = getattr(tc, "function", None)
    return {
        "id": getattr(tc, "id", "") or "",
        "type": "function",
        "function": {
            "name": getattr(fn, "name", "") or "",
            "arguments": getattr(fn, "arguments", "") or "{}",
        },
    }


async def _run_one(name: str, arguments: str, call_id: str,
                   executor: Callable[[str, str, str], Awaitable[str]]) -> dict:
    # corpus 要过一次 LLM 判（宁可漏不可错），给它和天气同档的 2.5s；
    # 其它本地检索 1.5s 足够。
    slow = '"weather"' in (arguments or "") or '"corpus"' in (arguments or "")
    timeout = _WEATHER_TIMEOUT if slow else _RECALL_TIMEOUT
    try:
        content = await asyncio.wait_for(executor(name, arguments, call_id), timeout=timeout)
    except Exception:
        content = "没有查到；不要编造。"
    return {"role": "tool", "tool_call_id": call_id, "content": content}


async def generate_with_tools(
    messages: list,
    *,
    executor: Callable[[str, str, str], Awaitable[str]],
    model: str | None = None,
    temperature: float,
    frequency_penalty: float,
    max_tokens: int,
) -> str:
    """带 tools 生成；返回最终正文（失败返回空串，由调用方兜底）。"""
    client, _ = _get_clients()
    model = model or _get_model_name()
    kwargs = dict(model=model, messages=messages, temperature=temperature,
                  frequency_penalty=frequency_penalty, max_tokens=max_tokens,
                  **THINKING_DISABLED)
    try:
        resp = await client.chat.completions.create(
            **kwargs, tools=tool_defs(), tool_choice="auto")
    except Exception:
        # 这一层/供应商不认 tools：退回无工具生成，别让整轮变哑巴。
        resp = await client.chat.completions.create(**kwargs)

    msg = _message_of(resp)
    calls = list(getattr(msg, "tool_calls", None) or [])
    if not calls:
        log_cache_usage(resp, "主回复")
        return extract_chat_content(resp)

    log_cache_usage(resp, "工具选择")
    calls = calls[:_MAX_TOOL_CALLS]
    tasks = []
    for tc in calls:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", "") or ""
        args = getattr(fn, "arguments", "") or "{}"
        tasks.append(_run_one(name, args, getattr(tc, "id", "") or "", executor))
    try:
        tool_msgs = await asyncio.wait_for(asyncio.gather(*tasks), timeout=_TOTAL_TIMEOUT)
    except Exception:
        tool_msgs = [{"role": "tool", "tool_call_id": getattr(tc, "id", "") or "",
                      "content": "没有查到；不要编造。"} for tc in calls]

    # ⚠️ **不把 assistant(tool_calls)+tool 消息回给上游**（2026-10-27 实测）：
    # Gemini 走中转站的 OpenAI 兼容层时，`role:"tool"` 的结果会被忽略——
    # 工具明明查到了 26℃，最终回复仍说“哪知道几度”。改成把结果作为一条 system 消息
    # 放在用户消息前：所有 provider 都看得见，也不需要伪造 tool_call 配对。
    result_text = "【工具结果】" + "\n".join(
        str(m.get("content") or "") for m in tool_msgs
    ) + "\n（这是你刚查到的；直接用，查不到不要编。）"
    result_msg = {"role": "system", "content": result_text[:1200]}
    if messages and messages[-1].get("role") == "user":
        messages2 = list(messages[:-1]) + [result_msg, messages[-1]]
    else:
        messages2 = list(messages) + [result_msg]

    # 第二轮不再给 tools：max_rounds=1，避免工具循环；它只负责把工具结果织进回复。
    try:
        final = await client.chat.completions.create(**{**kwargs, "messages": messages2})
    except Exception:
        return (getattr(msg, "content", "") or "").strip()
    log_cache_usage(final, "主回复")
    return extract_chat_content(final)


__all__ = ["tools_enabled", "tool_capable", "ensure_tool_capable", "tool_defs",
           "generate_with_tools"]