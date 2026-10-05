#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最终归并（第 3 层）：把分批产出的处境清单合成一份。

为什么要这一步：第 2 层按 45 条分批跑，**批次之间不知道彼此**，
于是同一处境会以不同名字重复出现（"被质疑言行真实性" vs "被质疑说法真实性"），
或者同一处境被拆成好几个具体的（"被调侃酒量与醉态" 其实是 "被起哄加码" 的一个实例）。

做法：把所有处境（只有 label + other_party + her_move + 段数）一次性喂进去，
让它做**全局归并**，并且区分：
  · 已被现有 behaviors 覆盖  → covered
  · 现在没有的              → gap（这才是我们要的）

用法：
    python scripts/scene_merge.py outputs/scene_clusters.json
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

# 这里不能 import src.plugins.chatbot.routing（会拉起 NoneBot 初始化）。
# 与 routing._repair_llm_json 同一套规则，只够本脚本用。
def _repair_json(text: str) -> str:
    import re
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start = min((i for i in (t.find("{"), t.find("[")) if i != -1), default=-1)
    end = max(t.rfind("}"), t.rfind("]"))
    if start != -1 and end > start:
        t = t[start:end + 1]
    t = re.sub(r"([{,]\s*)'([^']+)'(\s*:)", r'\1"\2"\3', t)
    t = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', t)
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return t


MODEL = "deepseek-flash"
THINKING_DISABLED = {"extra_body": {"thinking": {"type": "disabled"}}}
BATCH = 40   # 一次喂多少条候选；超过就层级归并（实测 149 条一次喂会被 max_tokens 截断）

EXISTING = [
    "被夸时嘴硬否认", "被质疑戳穿时心虚辩解", "被说低俗或擦边时冷脸挡回",
    "被关系向越界时不当回事地略过", "立Flag后立刻给自己留退路",
    "被推着要求时打太极", "说了真心话就立刻缩回", "被真心对待时别扭缩回",
    "失约被催时认栽滑跪", "能嘴熟络同期是亲近", "被套话点名排名他人时端水",
]

PROMPT = """你在整理一个虚拟主播（灰泽满）的「人际处境清单」。

下面是从多场直播里抽出来的候选处境（名字由机器分批产生，**批次之间没有互相合并**，
所以同一个处境可能以不同名字重复出现，也可能被拆得太具体）。

# 现有项目里已经有的 11 个处境（用来判断"缺不缺"）

{existing}

# 你的任务

1. **全局归并**：把同一个处境的条目合成一条。判据：
   > **"换个话题、换个人，这套应对还成立吗？"** 成立 = 同一处境。
   名字不同但处境相同（"被质疑言行真实性" / "被质疑说法真实性"）→ **必须合并**。
   太具体的（"被调侃酒量与醉态"）→ 归到它所属的泛化处境（"被起哄加码调侃"）。

2. **标注是否已被现有 11 条覆盖**：
   · 和现有某条是同一个处境 → `covered: "那条的名字"`
   · 现有 11 条里没有 → `covered: ""`（**这是缺口，最重要**）

2b. ★ **可用性筛子**：这些素材要用在**一对一私聊**里。归并时判断：
   > **"把'弹幕/观众/投稿/直播'从这段里去掉，处境还成立吗？"**
   不成立（只有直播才有：念投稿点评二创 / 游戏实况解说 / 答谢打赏 / 念弹幕刷屏 /
   展示形象换装 / 直播设备操作 / 下播催播周表）→ 标 `"drop": true` 并写 `drop_reason`。

3. **合并要狠**：`drop` 为 false 的条目，**总条数控制在 10~22 条**。

# 输出格式（只输出 JSON）

{{
  "scenes": [
    {{
      "label": "处境名（8~14 字，泛化到换个话题也成立）",
      "other_party": "对方在干什么（一句话，写行为）",
      "her_move": "她怎么应对（一句话，写姿态）",
      "covered": "现有 11 条里的名字，或空字符串",
      "drop": false,
      "drop_reason": "只有直播才成立时写原因",
      "merged_from": ["被合并掉的原始名字…"],
      "total_segments": 合并后的总段数（整数）
    }}
  ]
}}

# 铁律

1. `her_move` 只写姿态，不要抄原文。
2. `merged_from` 必须真实来自下面的原始名字。
3. 归并后每个处境只能出现一次——**不许有两条意思相近的**。
4. 不要新增下面没出现过的处境。

# 候选处境（原始名 / 对方 / 她 / 段数）

{text}
"""


async def merge_once(client, scenes, existing_txt) -> list:
    """把一批候选交给模型做一次归并。"""
    text = "\n".join(
        f"- 【{s['label']}】对方：{s['other_party']} ｜ 她：{s['her_move']} ｜ {s.get('total_segments') or s.get('n_members') or 0} 段"
        for s in scenes
    )
    resp = await client.chat.completions.create(
        model=_common.get_model_name(MODEL),
        messages=[{"role": "user", "content": PROMPT.format(existing=existing_txt, text=text)}],
        temperature=0.2, max_tokens=8000, **THINKING_DISABLED)
    c = resp.choices[0].message.content.strip()
    try:
        return json.loads(c)["scenes"]
    except Exception:
        # 输出被截断/格式坏：修一次再试
        return json.loads(_repair_json(c))["scenes"]


async def merge_hierarchical(client, scenes, existing_txt, depth=0):
    """层级归并：一批塞得下就一次并完；塞不下就先分批并、再并结果。

    为什么需要（实测）：149 条一次喂进去，模型输出超过 max_tokens 被截断
    （`Unterminated string starting at: line 659`）。
    """
    if not scenes:
        return []
    pad = "  " * depth
    if len(scenes) <= BATCH:
        out = await merge_once(client, scenes, existing_txt)
        print(f"{pad}  一次归并 {len(scenes)} → {len(out)}", flush=True)
        return out
    merged = []
    for i in range(0, len(scenes), BATCH):
        chunk = scenes[i:i + BATCH]
        out = await merge_once(client, chunk, existing_txt)
        print(f"{pad}  批次 {i}-{i+len(chunk)}：{len(chunk)} → {len(out)}", flush=True)
        merged.extend(out)
    print(f"{pad}  轮 {depth+1} 合计 {len(merged)}，继续归并…", flush=True)
    if len(merged) == len(scenes):     # 没并动，停（防死循环）
        return merged
    return await merge_hierarchical(client, merged, existing_txt, depth + 1)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cluster_json")
    args = ap.parse_args()

    d = json.loads(Path(args.cluster_json).read_text(encoding="utf-8"))
    scenes = d["scenes"]
    print(f"载入 {len(scenes)} 条分批产出，做层级归并\n")

    key = _common.get_api_key("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=key, base_url=_common.get_openai_base_url("https://api.deepseek.com/v1"))
    existing_txt = "\n".join(f"- {e}" for e in EXISTING)
    out = await merge_hierarchical(client, scenes, existing_txt)

    gaps = [s for s in out if not s.get("covered") and not s.get("drop")]
    have = [s for s in out if s.get("covered") and not s.get("drop")]
    dropped = [s for s in out if s.get("drop")]
    print(f"{'='*78}\n⭐ 缺口处境（现有 11 条没有的）：{len(gaps)} 条\n{'='*78}")
    for s in sorted(gaps, key=lambda x: -int(x.get("total_segments") or 0)):
        print(f"\n【{s['label']}】  约 {s.get('total_segments')} 段")
        print(f"   对方：{s['other_party']}")
        print(f"   她：  {s['her_move']}")
        if s.get("merged_from"):
            print(f"   由这些并成：{'、'.join(s['merged_from'][:6])}")

    print(f"\n{'='*78}\n✅ 已被现有覆盖（{len(have)} 条）\n{'='*78}")
    for s in have:
        print(f"  【{s['label']}】 → {s['covered']}")

    if dropped:
        print(f"\n{'='*78}\n🚫 判为直播专属、已丢弃（{len(dropped)} 条）\n{'='*78}")
        for s in dropped:
            print(f"  【{s['label']}】{s.get('drop_reason','')}")

    p = ROOT / "outputs" / "scene_final.json"
    p.write_text(json.dumps({"gaps": gaps, "covered": have, "dropped": dropped},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 → {p}")


asyncio.run(main())

