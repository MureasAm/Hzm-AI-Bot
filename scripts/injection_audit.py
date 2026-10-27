#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""注入审计：按用户/一条消息，打印真正进入提示词的关系块与提醒块。

只读：真实用户的短期/会话/长期数据复制到临时目录；不写线上、不生成回复。
用法：
    python scripts/injection_audit.py --user 2758175283 --msg "你还记得我叫什么吗"
    python scripts/injection_audit.py --user ... --msg ... --old
"""
import argparse
import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import nonebot  # noqa: E402
try:
    nonebot.get_driver()
except ValueError:
    nonebot.init()

import _common  # noqa: E402
from src.plugins.chatbot import core, daily_memory, memory_v2  # noqa: E402


def _prepare(uid: str) -> Path:
    root = Path(tempfile.mkdtemp(prefix="inject_audit_"))
    for sub in ("short_term", "session", "daily"):
        (root / sub).mkdir()
    src = ROOT / "user_memory" / "memory_v2_shadow.json"
    if src.exists():
        shutil.copy(src, root / "memory_v2_shadow.json")
    for sub in ("short_term", "session", "daily"):
        p = ROOT / "user_memory" / sub / f"{uid}.json"
        if p.exists():
            shutil.copy(p, root / sub / f"{uid}.json")
    _common.isolate_user_stores(root)
    daily_memory._DEFAULT_STORE.base_dir = root / "daily"
    return root


def _header(text: str) -> str:
    first = (text or "").splitlines()[0] if text else ""
    return first[:80]


async def run(uid: str, msg: str, v3: bool = True) -> None:
    root = _prepare(uid)
    os.environ["INJECTION_V3"] = "1" if v3 else "0"
    os.environ["RECALL_TOOLS"] = "0"
    os.environ["MEMORY_V2_SHADOW"] = "0"
    os.environ["SELF_SPEECH_ANNOTATE"] = "0"   # 审计只看注入结构，不花钱做转述

    captured, traces = [], []
    orig_gen = core.generate_reply
    orig_trace = core.log_injection_trace
    orig_block = core.log_block_hits
    orig_corpus = core.log_corpus_hits
    orig_route = core.get_llm_route

    async def fake_gen(messages, tool_executor=None):
        captured.append(messages)
        return "[audit]"

    core.generate_reply = fake_gen
    core.log_injection_trace = lambda t: traces.append(t)
    core.log_block_hits = lambda *a, **k: None
    core.log_corpus_hits = lambda *a, **k: None
    core.get_llm_route = lambda: "私"
    try:
        await core.handle_chat(uid, msg)
    finally:
        core.generate_reply = orig_gen
        core.log_injection_trace = orig_trace
        core.log_block_hits = orig_block
        core.log_corpus_hits = orig_corpus
        core.get_llm_route = orig_route

    msgs = captured[-1] if captured else []
    trace = traces[-1] if traces else {}
    systems = [m.get("content", "") for m in msgs if m.get("role") == "system"]
    joined = "\n".join(systems)

    print("=" * 78)
    print(f"用户 {uid} ｜ 消息：{msg} ｜ 模式：{'v3' if v3 else 'old'}")
    print(f"消息条数：{len(msgs)}（system {len(systems)}）")
    print("=" * 78)
    print("\n【真正进入提示词的 system 段】")
    for i, text in enumerate(systems, 1):
        print(f"{i:02d}. {_header(text)}")
        for line in (text or "").splitlines()[1:8]:
            print("    " + line[:100])

    print("\n【注入轨迹】")
    print("blocks:", ", ".join(trace.get("blocks") or []) or "（空）")
    print("items:", trace.get("items") or {})
    print("values:", {k: (v[:120] if isinstance(v, str) else v)
                      for k, v in (trace.get("values") or {}).items()})

    print("\n【关系候选：active / 是否出现在提示词里】")
    snap = core._MEMORY_V2_STORE.snapshot(uid)
    used_ids = set(str(x) for x in (trace.get("items", {}) or {}).get("memory", []))
    from src.plugins.chatbot import injection as inj
    for m in snap.get("memories", []):
        kind = m.get("kind")
        if kind not in {"interaction_preference", "request", "commitment"}:
            continue
        value = str(m.get("value") or "")
        status = m.get("status")
        behavior = inj.behavior_sentence(m.get("key"), value) if kind == "interaction_preference" else ""
        included = (str(m.get("id")) in used_ids
                    or bool(behavior and behavior in joined)
                    or bool(value and value[:20] in joined))
        print(f"- [{status}] {kind}/{m.get('key')}={value[:80]}")
        print(f"    {'✅ 出现在提示词里' if included else '— 未出现（可能未触发/过期/被预算截断）'}")

    print("\n【工具层】")
    print("facts/preferences/story/corpus/daily/weather 不在常驻提示词；需要时走 recall。")
    print("active facts:", len([m for m in snap.get("memories", [])
                               if m.get("kind") == "fact"
                               and m.get("status") in {"confirmed", "active"}]))
    print("\n（临时数据目录：%s，退出后可删）" % root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--msg", required=True)
    ap.add_argument("--old", action="store_true", help="审计旧链路（INJECTION_V3=0）")
    args = ap.parse_args()
    asyncio.run(run(args.user, args.msg, v3=not args.old))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()