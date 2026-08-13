#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""措辞指纹挖掘工具。

从清洗后的直播素材里批量挖掘灰泽满的标志性措辞指纹，
按"同一意思/同一情境"分组，输出候选 JSON（带 evidence 证明措辞来自素材），
供人工审批后并入 persona/speech/phrases.json。

用法：
    python scripts/run_tool.py mine-phrases -i cleaned_0626.json cleaned_0608.json ...

质量保证（防止自创措辞）：
- 措辞必须是素材里真实出现过的原话（evidence 给出出现片段证明）
- 高频稳定才算指纹，只出现一次的不收
- 分批处理控制 prompt 长度，结果按 meaning 合并去重
"""
import json
import asyncio
import sys
from pathlib import Path
from openai import AsyncOpenAI

from analyze_pace import get_deepseek_key

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com/v1"
BATCH_SIZE = 60  # 每批话轮数

MINE_PROMPT = """你是灰泽满直播语料的措辞挖掘员。我给你她在一场直播里说过的原话（已清洗）。你的任务是从中提取她的【标志性措辞指纹】。

【什么是措辞指纹】
- 她反复使用、有个人辨识度的说法（口头禅、固定回应、标志性开头/收尾、固定借口的说法）
- 只要是【她反复使用的固定说法】就收，不论长短（如"也没有啦"是短指纹，"老套的原因"这种固定说法也收）
- 以下情况剔除：
  · 一次性陈述：描述当下具体状态、只出现一次的事（如"灰泽满现在已经不敢看私信了""极限迟到是15分钟"）
  · 明显错译/乱码（如"请好""不是箭套"）
  · 模板化的完整长句（能拆成短语指纹的拆开）
- 必须是素材里真实出现过的原话，不得自创、不得改写、不得总结成"她大概会这么说"
- 同一意思有多个变体 → 归入同一组

【分组维度】按"同一意思/同一情境"分组，不限于以下，自己判断素材里实际出现了哪些：
- 被夸时的否认
- 被戳穿/被质疑时的否认
- 摆烂/拖延/失约的借口
- 无语/崩溃
- 认栽/认输
- 被关心时的慌乱
- 转移话题的反问
- 开场/回应撒娇
- 自嘲开头
- 口头禅/语助词
- 表达情绪/态度的固定说法

【输出 JSON】
{{
  "phrase_groups": [
    {{
      "meaning": "被夸时的否认",
      "trigger": "被夸奖、被称赞时",
      "phrases": ["也没有啦", "一般般吧"],
      "usage": "被夸时先用这些否认，再道谢",
      "evidence": ["素材里出现该措辞的片段，1-2个"]
    }}
  ]
}}

要求：
- phrases 里的每个措辞必须是素材里真实出现过的原话；evidence 给出对应片段证明（截取含该措辞的一句）
- 高频稳定才算指纹；素材里只出现一次的不收（除非极有辨识度、一眼就是她的说法）
- phrases 个数 2-6 个/组
- 识别出多少组就多少组，不要为了凑数编造
- 只输出 JSON，不要多余内容

【素材（部分）】
{text}"""


def load_turns(input_paths) -> list:
    """加载多个清洗 JSON，归一化为话轮文本列表。"""
    turns = []
    for p in input_paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for it in data:
                if isinstance(it, dict) and it.get("text"):
                    turns.append(it["text"])
                elif isinstance(it, str):
                    turns.append(it)
    return turns


async def mine_batch(client, batch_text: str) -> list:
    """调 DeepSeek 挖掘一批话轮的措辞组。失败返回空。"""
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": MINE_PROMPT.format(text=batch_text)}],
            temperature=0.2,
            max_tokens=2000,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = resp.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        result = json.loads(content)
        groups = result.get("phrase_groups", [])
        return [g for g in groups if isinstance(g, dict) and g.get("phrases")]
    except Exception as e:
        print(f"⚠️ 挖掘批次失败: {e}")
        return []


def merge_groups(groups: list) -> list:
    """按 meaning 合并多批的措辞组（措辞去重，evidence 合并）。"""
    merged = {}
    for g in groups:
        key = g.get("meaning", "").strip()
        if not key:
            continue
        if key not in merged:
            merged[key] = {
                "meaning": key,
                "trigger": g.get("trigger", ""),
                "usage": g.get("usage", ""),
                "phrases": [],
                "evidence": [],
            }
        for ph in g.get("phrases", []):
            if ph and ph not in merged[key]["phrases"]:
                merged[key]["phrases"].append(ph)
        for ev in g.get("evidence", []):
            if ev and ev not in merged[key]["evidence"]:
                merged[key]["evidence"].append(ev)
    return list(merged.values())


async def run(input_paths, output_file, batch_size=BATCH_SIZE):
    key = get_deepseek_key()
    if not key:
        print("❌ 未找到 OPENAI_API_KEY")
        return
    turns = load_turns(input_paths)
    if not turns:
        print("❌ 没有加载到话轮")
        return
    print(f"📄 加载 {len(turns)} 个话轮，分批挖掘（每批 {batch_size}）...")
    client = AsyncOpenAI(api_key=key, base_url=BASE_URL)

    all_groups = []
    n_batches = (len(turns) + batch_size - 1) // batch_size
    for i in range(0, len(turns), batch_size):
        batch = turns[i:i + batch_size]
        text = "\n".join(f"- {t}" for t in batch)
        groups = await mine_batch(client, text)
        all_groups.extend(groups)
        print(f"  [批次 {i // batch_size + 1}/{n_batches}] 挖到 {len(groups)} 组")

    merged = merge_groups(all_groups)
    out = {
        "source_files": [str(p) for p in input_paths],
        "total_turns": len(turns),
        "phrase_groups": merged,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 措辞挖掘完成：{output_file}")
    print(f"   共 {len(merged)} 组措辞指纹")
    for g in merged:
        print(f"   - {g['meaning']}：{g['phrases'][:4]}")
    print("   💡 下一步：人工审批后并入 persona/speech/phrases.json，再重跑 precompute phrases")


async def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) < 2:
        print("用法: python scripts/mine_phrases.py <cleaned1.json> [cleaned2.json ...] [--out 输出.json]")
        return
    inputs = [p for p in sys.argv[1:] if not p.startswith("--")]
    out = Path("outputs/mine/mined_phrases.json")
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
    await run(inputs, out)


if __name__ == "__main__":
    asyncio.run(main())
