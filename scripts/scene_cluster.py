#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情景聚类（第 2 层）：把多场直播的「情景片段」归并成一份处境清单。

为什么要这一层：单场直播切出 20~30 个片段，其中大量是**同一处境的不同名字**
（"被弹幕打断发言" / "被反复打断/抢话"）或**同一处境的不同次出现**。
直接交给人审会看不过来。

做法：把片段（label + other_party + her_move）按批喂给模型，让它
  · 合并同一处境（名字不同但处境相同 → 归一条）
  · 给每条一个**泛化名**（换个话题也成立）
  · 判断是"处境"还是"纯事件"（纯事件标 discard，不给人看）
  · 统计覆盖：出现在哪些片段里、大概多少段
输出：处境清单 + 每个处境下的候选片段（带出处，供人回查）

用法：
    python scripts/scene_cluster.py outputs/scene_segments/*_scenes.json
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import AsyncOpenAI  # noqa: E402
import scripts._common as _common  # noqa: E402

MODEL = "deepseek-flash"
THINKING_DISABLED = {"extra_body": {"thinking": {"type": "disabled"}}}
OUT = ROOT / "outputs" / "scene_clusters.json"
BATCH = 45

PROMPT = """你在整理一个虚拟主播（灰泽满）的直播「情景片段」，要归并成一份**处境清单**。

下面是若干片段，每条有：编号 / 情景名（模型标的，可能不一致）/ 对方在干什么 / 她怎么应对。

# 任务

把**同一个处境**的片段合并成一条。判据：

> **"换个话题、换个人，这套应对还成立吗？"** 成立 = 同一处境。
> 名字不同但处境相同（"被弹幕打断发言" / "被反复打断抢话"）→ **必须合并**。

同时把**纯事件**挑出来丢掉：只有因为那件具体事才成立、离开它这套应对毫无意义的，
标 `"discard": true` 并说明原因。

# 输出格式（只输出 JSON）

{{
  "scenes": [
    {{
      "label": "处境名（泛化到换个话题也成立，8~14 字）",
      "other_party": "对方在干什么（一句话，写行为）",
      "her_move": "她怎么应对（一句话，写姿态）",
      "member_ids": [片段编号…],
      "discard": false,
      "discard_reason": ""
    }}
  ]
}}

# 铁律

1. **合并要狠**：一份清单里相近的处境必须并成一条。目标是让人一眼看完（通常 8~20 条）。
2. `her_move` 只写姿态，**不要抄片段里的原话**。
3. `member_ids` 必须真实来自下面的编号，不许编。
4. 判断不了的（信息太少）→ `discard: true`。

# 片段

{text}
"""


def load_segments(paths):
    segs = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        src = Path(d.get("source", p)).stem
        for s in d.get("segments", []):
            segs.append({**s, "src": src})
    return segs


def render(segs, lo, hi):
    lines = []
    for i in range(lo, hi):
        s = segs[i]
        lines.append(f"[{i}] 情景名：{s['label']}\n"
                     f"    对方：{s['other_party']}\n"
                     f"    她：{s['her_move']}")
    return "\n".join(lines)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    segs = load_segments(args.files)
    print(f"载入 {len(segs)} 个片段，来自 {len(args.files)} 场直播\n")
    key = _common.get_api_key("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=key, base_url=_common.get_openai_base_url("https://api.deepseek.com/v1"))

    all_scenes = []
    for lo in range(0, len(segs), BATCH):
        hi = min(lo + BATCH, len(segs))
        prompt = PROMPT.format(text=render(segs, lo, hi))
        print(f"  批次 {lo}-{hi} …", end="")
        try:
            resp = await client.chat.completions.create(
                model=_common.get_model_name(MODEL),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=4000, **THINKING_DISABLED)
            c = resp.choices[0].message.content.strip()
            if "```json" in c:
                c = c.split("```json")[1].split("```")[0].strip()
            elif "```" in c:
                c = c.split("```")[1].split("```")[0].strip()
            scenes = json.loads(c).get("scenes", [])
        except Exception as e:
            print(f" 失败 {e}")
            continue
        print(f" → {len(scenes)} 条")
        # 回填片段出处（模型只给编号）
        for sc in scenes:
            ids = [i for i in (sc.get("member_ids") or []) if isinstance(i, int) and lo <= i < hi]
            sc["members"] = [
                {"src": segs[i]["src"], "range": f"{segs[i]['start']}-{segs[i]['end']}",
                 "label": segs[i]["label"], "events": segs[i].get("events", "")}
                for i in ids
            ]
            sc["n_members"] = len(sc["members"])
            sc.pop("member_ids", None)
        all_scenes.extend(scenes)

    keep = [s for s in all_scenes if not s.get("discard")]
    drop = [s for s in all_scenes if s.get("discard")]
    print(f"\n{'='*78}\n处境清单（{len(keep)} 条，另有 {len(drop)} 条判为纯事件已丢弃）\n{'='*78}")
    for s in sorted(keep, key=lambda x: -x.get("n_members", 0)):
        print(f"\n【{s['label']}】  出现在 {s['n_members']} 段")
        print(f"   对方：{s['other_party']}")
        print(f"   她：  {s['her_move']}")
        for m in s.get("members", [])[:5]:
            print(f"     · {m['src']} [{m['range']}] {m.get('events','')[:40]}")

    if not args.dry:
        OUT.write_text(json.dumps({"scenes": keep, "discarded": drop},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已保存 → {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
