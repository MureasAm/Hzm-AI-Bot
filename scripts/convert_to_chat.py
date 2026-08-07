"""直播原文 → QQ 聊天回复转化工具（试用版）。

把直播叙述体原文，转化为"灰泽满在 QQ 聊天里会怎么回复"，
并配一个触发情境（粉丝问了什么）。产出直接匹配 voice_samples.json 的 user/reply 结构。

用法：
    python scripts/run_tool.py convert-to-chat -i 原文列表.json [--out 输出.json]
"""
import json
import asyncio
import sys
from pathlib import Path
from openai import AsyncOpenAI

from analyze_pace import get_deepseek_key

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com/v1"

CONVERT_PROMPT = """你是灰泽满的"直播→聊天"翻译官。我给你一段她在直播里的原话（第三人称叙述、口语化）。

请把它转化为：她在 QQ 聊天里会怎么回复同一件事。要求：

【必须保留】
- 原文里已有的标志性措辞和口癖原样保留（原文有什么就保留什么，不要刻意添加新口癖）
- 她的语气节奏（省略号、断句、自嘲）和性格（嘴硬、心虚、自嘲化解）
- 原文的意思和细节，不增不减
- 【括号标注】原文里已有的括号情绪标注（（小声）（心虚）（警觉）（捂脸）（叹气）等）**必须原样保留**，这是她表达内心OS的方式
- 【最关键】她的自称：**必须用"灰泽满"或"hzm"自称，绝对不用"我"**。灰泽满说话永远自称"灰泽满/hzm"，这是她的人格标志（自我客体化）。原文里的"灰泽满"不许改成"我"，对话体里同样用"灰泽满/hzm"自称。

【必须改变】
- 形态：讲故事的旁观者视角 → 直接在回粉丝的对话体
- 去掉直播现场感（"弹幕说""在直播间""你看到我们聊天了吗"这类），适配 QQ 单轮聊天
- 第三人称叙述里描述别人的部分（"女同学跟灰泽满都没带伞"）可保持第三人称，但她的自称必须用"灰泽满/hzm"

【绝对禁止】
- 不要润色成标准、流畅、有文采的书面语——保留她口语的粗糙感和真实感
- 不要加原文没有的新梗或内容
- 绝对不要把"灰泽满"改成"我"
- 绝对不要为了"显得像灰泽满"而每句都加"好吧"或任何口癖——口癖只在原文确实有才保留
- 只针对当前这一条原文转化，不要使用或模仿任何其他条目的措辞
- 不要给回复加不必要的开头词；除非原文本就有"好吧"，否则不要凭空加"好吧"

输出 JSON：
{{
  "situation": "粉丝问/说了什么会触发这个回复，一句简洁的触发情境，如'被问到新家怎么样时'",
  "reply": "转化后的聊天回复"
}}

【直播原文】
{text}

只输出 JSON，不要多余内容。"""


async def convert_one(client, text: str) -> dict:
    """转化一条原文为聊天形态。失败返回降级结果。"""
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": CONVERT_PROMPT.format(text=text)}],
            temperature=0.3,
            max_tokens=200,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = resp.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)
    except Exception as e:
        print(f"⚠️ 转化失败: {e}")
        return {"situation": "", "reply": text}


async def run(input_paths, output_file):
    """参数化入口。"""
    key = get_deepseek_key()
    if not key:
        print("❌ 未找到 OPENAI_API_KEY")
        return

    # 加载原文：支持 [{text}] 或 [{"text":..}] 或 [statement] 数组
    items = []
    for p in input_paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            items.extend(data)

    # 归一化为文本列表：优先用 reply（已有对话），否则 text / statement
    texts = []
    for it in items:
        if isinstance(it, str):
            texts.append(it)
        elif isinstance(it, dict) and it.get("reply"):
            texts.append(it["reply"])
        elif isinstance(it, dict) and it.get("text"):
            texts.append(it["text"])
        elif isinstance(it, dict) and it.get("statement"):
            texts.append(it["statement"])

    client = AsyncOpenAI(api_key=key, base_url=BASE_URL)
    print(f"🧠 正在转化 {len(texts)} 条...")
    results = []
    for i, text in enumerate(texts):
        conv = await convert_one(client, text)
        results.append({"original": text, **conv})
        print(f"  [{i+1}/{len(texts)}] ✓")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 转化完成：{output_file}")


async def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) < 2:
        print("用法: python scripts/convert_to_chat.py <原文.json> [输出.json]")
        return
    inputs = [p for p in sys.argv[1:] if not p.startswith("--")]
    out = Path("outputs/transcribe/converted_chat.json")
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
    await run(inputs, out)


if __name__ == "__main__":
    asyncio.run(main())
