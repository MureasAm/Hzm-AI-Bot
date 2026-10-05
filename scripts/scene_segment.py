#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情景切分：从连续转写里标出「情景」（第 1 层）。

## 为什么重写

旧的 `extract_persona.py` 吃的是**已切好的单轮问答对**（user_situation + reply），
而同一句提问会被复制给她的多个碎语，导致模型看到的是"同一情境配不同回答"——
只能得出"没有稳定模式"。877 对 → 2 条就是这么来的。

**情景的本质是「连续几轮推进出来的」，单轮对在结构上表达不了它。**

## 这个工具做什么

输入：`outputs/clean/*.json`（带 start/end 的连续话轮）
做法：按时间轴分窗（带重叠）喂给模型，让它标出
        · 片段的起止话轮（**连续区间**，不是切成对）
        · 情景名（自由命名，附候选清单但不强制）+ 置信度
        · 对方在干什么 / 她怎么应对 / 一句话局面
        · 为什么这算情景（判据）

## 铁律（写死在提示词里）

1. **模型只许"标区间"，不许写正文**——所有原文按话轮号回填，模型不产出任何字。
2. **情景名开放**，候选清单只是提示；新情景本来就该不断长出来。
3. 判据是"她在应付一类人际处境"，不是"她回答了一个问题"。

用法：
    python scripts/scene_segment.py outputs/clean/cleaned_百日.json
    python scripts/scene_segment.py <file> --window 40 --overlap 8 --dry
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
OUT_DIR = ROOT / "outputs" / "scene_segments"

# 候选情景名：**只是提示，不强制**（旧工具强制 8 类，已过时且会归错）
CANDIDATE_SCENES = [
    "被夸/被认可", "被质疑/被戳穿", "被催/被问责", "被要求表态或做事",
    "索要能力外的东西", "索要亲密称呼或动作", "被反复追问同一件事",
    "对方道歉/自我否定", "对方指责她冷淡", "对方情绪崩溃/自伤施压",
    "对方讲自己的事（陪聊）", "对方只发表情/图片", "对方骂她", "对方造谣",
    "问她自己的事", "对方看不懂的梗/黑话", "对方开玩笑调戏", "其他",
]

PROMPT = """你在给一个虚拟主播（灰泽满）的直播转写做「情景标注」。

下面是一段连续的直播话轮，每条带行号和秒数。

# 第一步（最重要）：分清「事件」和「处境」

**事件** = 这场直播里具体发生了什么（"弹幕起哄怀孕""追问孩子他爸""质疑周表真假"）。
**处境** = 一类人际处境，换个话题、换个人、甚至换到私聊，**照样会发生**。

⚠️ **你要标的是「处境」，不是「事件」。** 判据是一句话：

> **"把这件事换成别的话题，这套应对还成立吗？"**
> 成立 → 是处境，标它。不成立（只因为那件具体事才成立）→ 是事件，**降级处理**。

举例（同一个直播里可能同时出现多个"事件"，但它们其实是**同一个处境**）：

```
事件：弹幕起哄怀孕生子 / 追问孩子他爸 / 编造她买贵包
  → 处境都是「被弹幕编造故事加码调侃」→ **合并成一段，只标一个处境**

事件：质疑周表真实性 / 质疑在场证明 / 质疑直播间风气
  → 处境都是「被质疑说法真实性」→ **合并成一段**

处境样例（正确的粒度）：
  被夸时嘴硬否认 ｜ 被质疑戳穿时心虚辩解 ｜ 被反复索要同一样东西
  被索要亲密称呼或动作 ｜ 被推着要求表态或做事 ｜ 被真心对待时别扭缩回
```

# 第二步：标区间

情景通常**跨若干条连续话轮**（不是一问一答就完）。

# 什么不算情景（见下）

# 第三步：可用性筛子（★ 不过这一关的直接不要标）

**这是一场直播，但我们要的素材是要用在「一对一私聊」里的。** 所以要问：

> **"这段的处境，如果搬到一个只有两个人（她 + 对面一个人）的私聊里，还会发生吗？"**

```
❌ 只有直播才有、一对一里不会发生的 → **不要标**：
   · 念投稿 / 点评二创 / 看粉丝视频           （要"一堆人投稿"这个形式）
   · 游戏实况、边玩边解说                     （要"在直播打游戏"）
   · 答谢礼物、念打赏、欢迎舰长               （要"打赏"这个机制）
   · 念弹幕、回应刷屏、被弹幕打断              （要"弹幕"这个形式）
   · 展示形象 / 换装 / 3D 回 / 直播设备操作     （要"直播画面"）
   · 下播催播、直播流程、周表、开播时间        （要"直播"这件事本身）

✅ 一对一里照样会发生、照样需要她应对的 → 标：
   · 对方夸她 / 质疑她 / 催她 / 戳穿她
   · 对方反复要同一样东西、追问私事
   · 对方索要亲密称呼或动作、逼她表态
   · 对方道歉 / 自我否定 / 情绪崩溃 / 骂她 / 造谣
   · 对方真诚感谢她、关心她（真心对待）
   · 对方讲自己的事（倒苦水）
   · 对方说听不懂的词 / 只发表情
```

**判据一句话**：把"弹幕/观众/投稿/直播"这些词从这段里去掉，**这段处境还成立吗？** 不成立 → 不要标。

# 什么不算情景（不要标）

- 她在陈述事实、讲自己的经历（那是"记忆"，归 corpus）
- 单纯回答一个问题（"你吃什么""你住哪"）
- 寒暄、打招呼、念弹幕、答谢礼物、事务性对话
- 只有一条话轮、看不出"处境"的
- **任何"只因为那件具体事才成立"的东西**（那是事件，见第一步）

# 候选处境名（**只是提示，不强制**；没有合适的自己起名）

{candidates}

# 输出格式（只输出 JSON）

{{
  "segments": [
    {{
      "start": 起始行号（整数）,
      "end": 结束行号（整数，含）,
      "label": "处境名（**必须泛化到换个话题也成立**，如'被反复索要同一样东西'）",
      "events": "这一段里发生的具体事件（可以很短，供人核对用）",
      "transferable": true,
      "confidence": 0.0~1.0,
      "other_party": "对方在干什么（一句话，写行为不写台词）",
      "her_move": "她怎么应对（一句话，写姿态；**不要抄原文台词**）",
      "why": "为什么这算一个处境（一句话判据）"
    }}
  ]
}}

# 铁律（违反即作废）

1. **你必须只标区间，不要写任何原文**。所有原文由程序按行号回填，`her_move` 里**不许出现原句**。
2. 区间必须**连续**（start ≤ end），且区间之间**不要重叠**。
3. **同一个处境只标一段**：如果同一处境断断续续出现，取它**最完整的那次**，不要拆成多段。
4. 拿不准 / 置信度 < 0.6 的干脆别写。
5. 允许"这一段没有情景"——那就返回 `{{"segments": []}}`。
6. 不要为了凑数把同一个处境拆成好几段。

# 待标注话轮

{text}
"""


def load_turns(path: Path) -> list:
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, list):
        raise SystemExit(f"❌ {path} 不是话轮列表")
    return [t for t in d if str(t.get("text", "")).strip()]


def render(turns: list, lo: int, hi: int) -> str:
    lines = []
    for i in range(lo, hi):
        t = turns[i]
        lines.append(f"[{i}] {t.get('start')}s {t['text'].strip()}")
    return "\n".join(lines)


def make_client():
    key = _common.get_api_key("OPENAI_API_KEY")
    if not key:
        raise SystemExit("❌ 没有 OPENAI_API_KEY")
    return AsyncOpenAI(api_key=key, base_url=_common.get_openai_base_url("https://api.deepseek.com/v1"))


async def segment_window(client, turns, lo, hi) -> list:
    prompt = PROMPT.format(
        candidates="\n".join(f"- {c}" for c in CANDIDATE_SCENES),
        text=render(turns, lo, hi),
    )
    try:
        resp = await client.chat.completions.create(
            model=_common.get_model_name(MODEL),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=4000, **THINKING_DISABLED,
        )
        c = resp.choices[0].message.content.strip()
        if "```json" in c:
            c = c.split("```json")[1].split("```")[0].strip()
        elif "```" in c:
            c = c.split("```")[1].split("```")[0].strip()
        segs = json.loads(c).get("segments", [])
    except Exception as e:
        print(f"  ⚠️ 窗口 {lo}-{hi} 失败: {e}")
        return []
    out = []
    for s in segs:
        try:
            a, b = int(s["start"]), int(s["end"])
        except Exception:
            continue
        a, b = max(a, lo), min(b, hi - 1)
        if b < a:
            continue
        if float(s.get("confidence") or 0) < 0.6:
            continue
        # 可用性筛子：模型自评"搬到一对一私聊还成立吗"；显式 false 的直接丢
        if s.get("transferable") is False:
            continue
        out.append({
            "start": a, "end": b,
            "label": str(s.get("label", "")).strip(),
            "events": str(s.get("events", "")).strip(),
            "transferable": bool(s.get("transferable", True)),
            "confidence": float(s.get("confidence") or 0),
            "other_party": str(s.get("other_party", "")).strip(),
            "her_move": str(s.get("her_move", "")).strip(),
            "why": str(s.get("why", "")).strip(),
            # 原文由程序回填，模型碰不到
            "turns": [{"i": i, "start": turns[i].get("start"), "text": turns[i]["text"].strip()}
                      for i in range(a, b + 1)],
        })
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--window", type=int, default=40, help="每窗话轮数")
    ap.add_argument("--overlap", type=int, default=8, help="窗口重叠话轮数")
    ap.add_argument("--dry", action="store_true", help="只打印标注结果，不写文件")
    args = ap.parse_args()

    client = make_client()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for f in args.files:
        path = Path(f)
        if not path.is_absolute():
            path = ROOT / f
        turns = load_turns(path)
        print(f"\n{'='*78}\n{path.name}：{len(turns)} 话轮\n{'='*78}")
        all_segs = []
        step = max(1, args.window - args.overlap)
        for lo in range(0, len(turns), step):
            hi = min(lo + args.window, len(turns))
            segs = await segment_window(client, turns, lo, hi)
            # 去掉与前窗重叠区里的重复（同 label 且区间高度重合）
            fresh = []
            for s in segs:
                dup = any(s["label"] == p["label"] and abs(s["start"] - p["start"]) <= args.overlap
                          for p in all_segs)
                if not dup:
                    fresh.append(s)
            all_segs.extend(fresh)
            print(f"  窗口 {lo:4d}-{hi:4d} → {len(segs)} 段（新增 {len(fresh)}）")
            if hi >= len(turns):
                break

        print(f"\n共 {len(all_segs)} 个情景片段：")
        # 后处理：同一 label 且区间重叠的，只留最长的那个（防"一个处境拆成好几段"）
        merged = []
        for s in sorted(all_segs, key=lambda x: -(x["end"] - x["start"])):
            dup = False
            for m in merged:
                if s["label"] == m["label"] and not (s["end"] < m["start"] or s["start"] > m["end"]):
                    dup = True
                    break
            if not dup:
                merged.append(s)
        if len(merged) != len(all_segs):
            print(f"  （合并重叠同名的 {len(all_segs) - len(merged)} 段）")
        all_segs = sorted(merged, key=lambda x: x["start"])

        for s in all_segs:
            print(f"  [{s['start']:4d}-{s['end']:4d}] {s['confidence']:.2f} 【{s['label']}】")
            if s.get("events"):
                print(f"        事件：{s['events']}")
            print(f"        对方：{s['other_party']}")
            print(f"        她：  {s['her_move']}")
            print(f"        判据：{s['why']}")

        if not args.dry:
            out = OUT_DIR / f"{path.stem}_scenes.json"
            out.write_text(json.dumps(
                {"source": str(path), "n_turns": len(turns), "segments": all_segs},
                ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"\n已保存 → {out}")


if __name__ == "__main__":
    asyncio.run(main())
