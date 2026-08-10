#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""场景化陈述（statement）生成工具。

从清洗后的直播素材里，把有代表性的高质量话轮浓缩成第三人称"场景化陈述"
（灰泽满当时……她……），作为直播记忆 RAG 的背景记忆（data/corpus_vectors.json 的输入）。

与旧 statement（平均 261 字）不同，新 statement 控制在 50-120 字，
避免长文本挤占 few-shot 注入预算、且短陈述与短查询的语义匹配更好。

用法：
    python scripts/run_tool.py generate-statements -i cleaned_0626.json cleaned_0608.json ...

输出：statement JSON（{"statements": ["..."]}），供 generate-vectors 向量化。
"""
import json
import asyncio
import sys
from pathlib import Path
from openai import AsyncOpenAI

from analyze_pace import get_deepseek_key

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com/v1"
BATCH_SIZE = 50  # 每批话轮数

STATEMENT_PROMPT = """你是"灰泽满"的专属人格档案学家。我给你她在直播里说过的原话（已清洗的话轮，按时间顺序）。

任务：把其中有代表性、能体现她人格/经历/反应模式的内容，浓缩成若干条【场景化陈述】——第三人称的"灰泽满当时……她……"描述，作为她的背景记忆（直播记忆 RAG，检索命中时让她自然提及这段经历）。

# 工作流程
1. 先判断每条话轮是否"有主题的高质量对话"。以下属于噪声，直接跳过：
   - 念礼物/舰长名单、纯客套寒暄（连续"谢谢""晚上好"）、无信息量重复刷屏
   - 读弹幕/转述他人（"弹幕说…""他说…"）、看二创视频的无实质内容
2. 在非噪声话轮里找高光：经典口头禅、标志性行为（嘴硬/自嘲/立Flag打脸/抱怨家人/分享倒霉日常）、与绿冻的推拉互损、罕见的情感流露（脆弱/感动/孤独，随后立刻被掩饰）。

# 场景化陈述的核心定义
一段高密度、结构化的第三人称描述。必须同时满足两个下游用途：
1. 语义检索：粉丝问及相关事件时，这段文本能被向量检索精准命中
2. 人格分析：能从中直接提取出 traits/styles/behaviors 的证据

# 写作铁律
1. 严禁时间戳：绝对不要出现"在直播的第X分钟""在视频X分Y秒"等时间标记；时间要转化为自然语境（"深夜闲聊时""被粉丝问及近况时""在聊到家庭话题时"）
2. 严禁复述"直播"这个载体：不要说"她在直播中说""她在直播时被问到"。直接进入事件本身，仿佛你亲眼目睹
3. 第三人称 + 原话引用：全篇第三人称，但她的标志性语言必须用引号直接引用，大量化用转写原文，完整保留口癖（hzm、好吧、说实话、只能说、可恶啊等）、碎碎念的节奏和戏剧化的反转
4. 隐含人格标签：文本中自然融入她的性格特质关键词（如"嘴硬心软""乐观的悲观主义者""用自嘲消解尴尬""生活喜剧人""回避型亲近者"等），但不是以标签形式罗列，而是作为描述的一部分自然呈现
5. 必须是素材里真实发生的，不编造、不加素材没有的细节

# 结构
每条陈述讲一个完整的小场景/经历（她的行为、心理、反应模式），50-120 字。按"触发情境→即兴表演→性格切片"的自然顺序写成一段连贯文字，不要用标题或列表。

# 数量
宁缺毋滥：这批话轮里有多少条值得写就输出多少条。

输出 JSON：
{{
  "statements": ["灰泽满当时……", "……"]
}}

【素材话轮】
{text}

只输出 JSON，不要多余内容。"""


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


async def gen_batch(client, batch_text: str) -> list:
    """调 DeepSeek 生成一批话轮的 statement。失败返回空。"""
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": STATEMENT_PROMPT.format(text=batch_text)}],
            temperature=0.3,
            max_tokens=2000,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = resp.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        result = json.loads(content)
        stmts = result.get("statements", [])
        return [s.strip() for s in stmts if s and isinstance(s, str)]
    except Exception as e:
        print(f"⚠️ 生成批次失败: {e}")
        return []


async def run(input_paths, output_file, batch_size=BATCH_SIZE):
    key = get_deepseek_key()
    if not key:
        print("❌ 未找到 OPENAI_API_KEY")
        return
    turns = load_turns(input_paths)
    if not turns:
        print("❌ 没有加载到话轮")
        return
    print(f"📄 加载 {len(turns)} 个话轮，分批生成 statement（每批 {batch_size}）...")
    client = AsyncOpenAI(api_key=key, base_url=BASE_URL)

    all_stmts = []
    n_batches = (len(turns) + batch_size - 1) // batch_size
    for i in range(0, len(turns), batch_size):
        batch = turns[i:i + batch_size]
        text = "\n".join(f"- {t}" for t in batch)
        stmts = await gen_batch(client, text)
        all_stmts.extend(stmts)
        print(f"  [批次 {i // batch_size + 1}/{n_batches}] 生成 {len(stmts)} 条")

    # 去重（按语句去重）
    seen, uniq = set(), []
    for s in all_stmts:
        if s not in seen:
            seen.add(s)
            uniq.append(s)

    out = {"source_files": [str(p) for p in input_paths], "statements": uniq}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✅ statement 生成完成：{output_file}")
    print(f"   共 {len(uniq)} 条场景化陈述")
    for s in uniq[:5]:
        print(f"   - {s[:60]}...")
    print("   💡 下一步：合并进 statement 库后跑 generate-vectors 向量化")


async def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) < 2:
        print("用法: python scripts/generate_statements.py <cleaned1.json> [cleaned2.json ...] [--out 输出.json]")
        return
    inputs = [p for p in sys.argv[1:] if not p.startswith("--")]
    out = Path("outputs/transcribe/generated_statements.json")
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
    await run(inputs, out)


if __name__ == "__main__":
    asyncio.run(main())
