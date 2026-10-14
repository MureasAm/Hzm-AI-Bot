#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EP01 Phase 0-A：三板对照真实生成（裸模型 / 仅基础提示词 / 基础提示词 + behavior）。

规格：video/EP01_PERSONA_INJECTION_PRODUCTION_SPEC.md §4

## 实现原则（与 scripts/trace_chain.py 同一条）

**用真实的 system prompt、真实的 behavior 格式化、真实的模型与参数**，不另造一套。
唯一"造"的东西是 messages 的组装方式——因为裸模型这一版线上不存在
（线上永远是完整链路），必须手工构造三种上下文来对照。

## 采集量

每种版本跑 `--runs` 次（默认 5），全部原始输出保存；同时保存清洗后的版本。
`selectedRun` 的选择只按规格 §4.3 的规则"选一个有代表性的、不刻意挑最好/最差"，
理由写进 selectionReason。

## 用法

    python video/scripts/capture-persona-comparison.py
    python video/scripts/capture-persona-comparison.py --runs 5
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402

nonebot.init()

from openai import AsyncOpenAI  # noqa: E402

from scripts import _common  # noqa: E402
from src.plugins.chatbot import constants as C  # noqa: E402
from src.plugins.chatbot.persona import _format_behavior_rule, load_persona_rules  # noqa: E402
from src.plugins.chatbot.reply_style import clean_reply  # noqa: E402

OUT_FILE = ROOT / "video" / "data" / "ep01" / "persona-comparison.json"

INPUT = "你今天唱得特别好，我真的很喜欢。"
BEHAVIOR_NAME = "被夸时嘴硬否认"


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_behavior_text() -> tuple[str, dict]:
    """取真实 behavior 条目并走真实的 _format_behavior_rule（**不手写注入文本**）。"""
    for b in load_persona_rules():
        if b.get("name") == BEHAVIOR_NAME:
            return _format_behavior_rule(b), b
    raise SystemExit(f"behaviors.json 里找不到 {BEHAVIOR_NAME}")


def load_system_prompt() -> str:
    return C.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")


async def run_variant(client, model: str, messages: list, runs: int) -> list[dict]:
    out = []
    for i in range(runs):
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=C.CHAT_TEMPERATURE,
            frequency_penalty=C.CHAT_FREQUENCY_PENALTY,
            max_tokens=C.CHAT_MAX_TOKENS,
            **C.THINKING_DISABLED,
        )
        raw = (resp.choices[0].message.content or "").strip()
        out.append({
            "run": i,
            "raw": raw,
            "cleaned": clean_reply(raw),
            "finish_reason": resp.choices[0].finish_reason,
        })
        print(f"    run {i + 1}/{runs}: {raw[:56]}")
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    _common.ensure_utf8_stdout()

    system_prompt = load_system_prompt()
    behavior_text, behavior_raw = load_behavior_text()
    model = _common.get_model_name()
    api_key = _common.get_api_key("OPENAI_API_KEY")
    base_url = _common.get_openai_base_url()
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    print(f"模型 = {model}")
    print(f"参数 = temperature={C.CHAT_TEMPERATURE} "
          f"frequency_penalty={C.CHAT_FREQUENCY_PENALTY} max_tokens={C.CHAT_MAX_TOKENS} "
          f"thinking=disabled")
    print(f"system_prompt = {len(system_prompt)} 字")
    print(f"behavior = {BEHAVIOR_NAME}（格式化后 {len(behavior_text)} 字）")
    print(f"输入 = {INPUT}\n")

    # 三栏共用同一个 model 标识与同一份用户消息；唯一变量是上下文
    variants = [
        {
            "id": "bare",
            "label": "裸模型",
            "sublabel": "只给用户消息",
            "messages": [{"role": "user", "content": INPUT}],
        },
        {
            "id": "system_prompt",
            "label": "仅基础提示词",
            "sublabel": "system_prompt + user",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": INPUT},
            ],
        },
        {
            "id": "system_prompt_behavior",
            "label": "基础提示词 + behavior",
            "sublabel": "system_prompt + behavior + user",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"【当前情境下的行为指令】\n{behavior_text}"},
                {"role": "user", "content": INPUT},
            ],
        },
    ]

    for v in variants:
        print(f"[{v['id']}] {v['label']}")
        v["runs"] = await run_variant(client, model, v["messages"], args.runs)

    # 选择规则（规格 §4.3）：不用 LLM judge 判"更像"，也不为某栏挑最好/最差。
    # 统一取"该版本 5 次里长度最接近该版本中位数"的那一次——随机性最小、最可复现。
    for v in variants:
        lens = sorted(len(r["cleaned"]) for r in v["runs"])
        median = lens[len(lens) // 2]
        best = min(v["runs"], key=lambda r: (abs(len(r["cleaned"]) - median), r["run"]))
        v["selectedRun"] = best["run"]
        v["selectionReason"] = (
            f"取长度最接近本版本中位数（{median} 字）的一次，避免按好坏挑选；"
            f"该次 {len(best['cleaned'])} 字。"
        )

    payload = {
        "fictional": False,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "commit": _git_commit(),
        "model": model,
        "params": {
            "temperature": C.CHAT_TEMPERATURE,
            "frequency_penalty": C.CHAT_FREQUENCY_PENALTY,
            "max_tokens": C.CHAT_MAX_TOKENS,
            "thinking": "disabled",
        },
        "input": INPUT,
        "behaviorSource": {
            "file": "persona/behavior/behaviors.json",
            "name": BEHAVIOR_NAME,
            "trigger": behavior_raw.get("trigger", ""),
            "response": behavior_raw.get("response", ""),
            "sampleCount": len(behavior_raw.get("samples", []) or []),
            "formatted": behavior_text,
        },
        "systemPromptSource": {
            "file": "persona/core/system_prompt.txt",
            "chars": len(system_prompt),
        },
        "note": ("三种版本使用同一个模型与相同生成参数；唯一变量是模型当时看到的上下文。"
                 "raw 为上游原始输出，cleaned 为项目现有 clean_reply 的结果。未做任何人工改写。"),
        "variants": variants,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n共保存 {len(variants) * args.runs} 条原始输出")
    print(f"✅ {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
