#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EP01：从真实 persona 文件抽出画面要用的结构数据。

规格 §12.2：组件不得硬编码。system_prompt 的章节表与 behaviors 的
情景/反应/例句都从真实文件抽出来，写进 video/data/ep01/persona-structure.json，
Remotion 只读消费。

    python video/scripts/build-persona-structure.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SYSTEM_PROMPT = ROOT / "persona" / "core" / "system_prompt.txt"
BEHAVIORS = ROOT / "persona" / "behavior" / "behaviors.json"
OUT = ROOT / "video" / "data" / "ep01" / "persona-structure.json"

# 画面里要展示的那条 behavior（开场三板与 behavior 段共用同一条，保持单一真值）
FOCUS_BEHAVIOR = "被夸时嘴硬否认"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    text = SYSTEM_PROMPT.read_text(encoding="utf-8")
    # 抽真实的二级标题（## 【xxx】），保持文件顺序
    headings = re.findall(r"^##\s*【([^】]+)】", text, re.M)
    # 顶层标题行（# 开头的第一行）
    title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")), "")

    # 每章的字符数（让画面能显示"它并不靠写长取胜"）
    sections = []
    parts = re.split(r"^##\s*【([^】]+)】", text, flags=re.M)
    # parts = [前言, 章名1, 正文1, 章名2, 正文2, ...]
    for i in range(1, len(parts), 2):
        name = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append({"name": name, "chars": len(body.strip())})

    beh = json.loads(BEHAVIORS.read_text(encoding="utf-8"))["behaviors"]
    focus = next((b for b in beh if b.get("name") == FOCUS_BEHAVIOR), None)
    if focus is None:
        raise SystemExit(f"找不到 behavior: {FOCUS_BEHAVIOR}")

    payload = {
        "systemPrompt": {
            "file": "persona/core/system_prompt.txt",
            "title": title,
            "chars": len(text),
            "headings": headings,
            "sections": sections,
            "role": "稳定骨架：决定她始终是谁，以及表达的边界",
        },
        "behaviors": {
            "file": "persona/behavior/behaviors.json",
            "count": len(beh),
            "names": [b.get("name", "") for b in beh],
            "focus": {
                "name": focus["name"],
                "trigger": focus.get("trigger", ""),
                "response": focus.get("response", ""),
                "samples": focus.get("samples", []),
            },
        },
        "historyNote": (
            "旧 traits/styles/voice_samples/phrases 已于 2026-10-04 拆分："
            "稳定内容并进 core/system_prompt.txt，真人示范按情景并进 behaviors.json。"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"system_prompt: {len(text)} 字，{len(headings)} 个章节")
    for s in sections:
        print(f"  【{s['name']}】{s['chars']} 字")
    print(f"\nbehaviors: {len(beh)} 条；focus = {focus['name']}（{len(focus.get('samples', []))} 条例句）")
    print(f"\n✅ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
