"""直播原文 → QQ 聊天回复转化工具（V3：先分离转述，再分析，后转化）。

V3 变化（针对两个已知问题）：
1. **直播叙述整段保留 → 聊天回复过长**：直播叙述体长段 ≠ QQ 聊天短句。
2. **素材是单人声音 → 分不清"读弹幕"和"回答"**：转写里只有灰泽满一个人的话，
   其中混着"她读弹幕/转述粉丝的话"和"她本人的回答"。必须先用语义区分，
   否则会把"她读的弹幕内容"错当成"她的回答"灌进样本，user/reply 串味。

V3 处理流程（五步）：
① 判定可转性（噪声跳过）→ ② 分离"读弹幕/转述" 与 "本人回答"（转述原话→触发情境，
本人话→reply 来源）→ ③ 提取说话特征（语气/措辞/情绪顶点）→ ④ 切分（一个独立意思
= 一条回复）→ ⑤ 压缩到聊天节奏（日常 15-50 字）。

方法论来源：叙述→对话三步法（Gutenberg 数据集）、先提说话特征再生成（AIWolfDial 人格重现）。

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

CONVERT_PROMPT = """你是灰泽满的"直播→聊天"转化师。我给你一段她在直播里的原话（单人声音转写：只有她一个人在说话，其中混着"她读弹幕/转述粉丝的话"和"她本人回答/表达"）。你的任务是把这段变成"她在 QQ 聊天里会怎么回复"——聊天里的灰泽满默认短句、一次只说一个想法，说话节奏和直播完全不同。

严格按以下步骤处理：

【第一步：判定是否值得转化】
先判断这段有没有转化价值：
- 包含她本人的表达、细节、情绪、措辞 → 值得转化
- 以下情况直接跳过（convertible=false）：念礼物/舰长名单、纯客套寒暄、无实质内容的口语填充
- 注意：读弹幕/转述粉丝的话本身不是噪声——它提供了"粉丝问/说了什么"的触发情境，往下走第二步处理
不可转就输出 {"convertible": false}，不要往下做。

【第二步：分离"她读的弹幕/转述"和"她本人的回答"】★ 单人素材的关键，务必先做这一步
这段里会混着两类内容，必须分开：
1. 她读弹幕 / 转述粉丝的话（信号词如"弹幕说""有人说""有人问""xx问""你们说""评论区""有绿冻说"，或她直接引述的粉丝原话）——这些是【粉丝在问/说什么】，提取出来作为 user_situation（触发情境），优先用她转述的粉丝原话，不要自己改写
2. 她本人的回答 / 表达 —— 这些才是要转化成聊天回复的内容（reply 的来源）
- 如果一段里既有转述又有回答：user_situation 用转述的粉丝原话，reply 只来自她的回答
- 只有回答没有转述：user_situation 从内容推断一句简洁触发情境
- 只有转述没有回答：只输出 user_situation，replies 留空数组

【第三步：先读懂她怎么说话】
动手前先提取这段回答的说话特征，生成时用它锚定，不要自己发明风格：
- 语气基调（嘴硬/心虚/自嘲/感性/摆烂…）
- 标志性措辞（原文里有才提，原文没有就留空）
- 情绪顶点（哪里情绪最强——那里才允许用括号）

【第四步：切分】
这段回答在聊天里是几条回复？规则：一个独立意思 = 一条回复。
- 换话题、补新细节、自我修正、情绪转折 → 各自独立成条
- 如果整段就是一个意思，只输出 1 条
- 每条只表达一个想法，一句到两句

【第五步：压缩到聊天节奏】
每条压到 15-50 字（日常短句）；只有强烈情绪或讲具体故事才放宽到 50-60 字。
删除：直播现场感（"弹幕说""在直播间""你们看到我们聊天了吗"这类转述标记）、口头填充、重复铺垫。

【必须保留】
- 自称：必须用"灰泽满"或"hzm"，绝对不用"我"——这是她的人格标志，主语宾语都如此
- 原文里已有的口癖和措辞原样保留（原文有什么留什么，不刻意添加新口癖）
- 括号标注（（小声）（心虚）（警觉）（捂脸）（叹气）等）只在情绪顶点用，原文有才保留
- 语气节奏（省略号、断句、自嘲）和口语的粗糙感，不润色成标准书面语

【绝对禁止】
- 不要整段搬运原文——必须先切分再压缩
- 不要把"她读弹幕/转述的粉丝话"混进 reply——那是 user 的内容
- 不要给每条硬凑口癖；不要加原文没有的新梗、内容、开头词（除非原文本来就有）
- 只针对当前这一条原文转化，不要模仿任何其他条目的措辞

输出 JSON（只输出 JSON，不要多余内容）：
{
  "convertible": true,
  "user_situation": "粉丝问/说了什么，一句简洁触发情境（优先用原文里她转述的粉丝原话；没有转述才从内容推断），如'被问到新家怎么样时'或'粉丝说：你这声音好好听'",
  "replies": [
    {"text": "聊天回复1", "length": "short"},
    {"text": "聊天回复2", "length": "long"}
  ]
}
length 规则：30 字以内 short，超过 30 字 long（只在情绪顶点或讲故事时用 long）。

【直播原文】
{text}"""


async def convert_one(client, text: str) -> dict:
    """转化一条原文为聊天形态。失败返回降级结果（保留原文，留给人工筛选）。"""
    try:
        # 注意：用 replace 而非 format——prompt 内嵌 JSON 示例含花括号，
        # format 会把 {"convertible"...} 误当占位符抛 KeyError。
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": CONVERT_PROMPT.replace("{text}", text)}],
            temperature=0.3,
            max_tokens=500,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = resp.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        result = json.loads(content)
        # 容错：结构不完整时仍给出一条原文兜底
        if result.get("convertible") is False:
            return result
        if not result.get("replies"):
            return {"convertible": True, "user_situation": result.get("user_situation", ""),
                    "replies": [{"text": text, "length": "long"}]}
        return result
    except Exception as e:
        print(f"⚠️ 转化失败: {e}")
        return {"convertible": True, "user_situation": "",
                "replies": [{"text": text, "length": "long"}]}


def _length_tier(text: str) -> str:
    """30 字以内 short，超过 30 字 long（对齐 analyze_pace 的长度档）。"""
    return "short" if len(text) <= 30 else "long"


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
    print(f"🧠 正在转化 {len(texts)} 条（先分离转述/回答 → 分析 → 切分 → 压缩）...")
    results = []
    skipped = 0
    for i, text in enumerate(texts):
        conv = await convert_one(client, text)
        if conv.get("convertible") is False:
            skipped += 1
            print(f"  [{i+1}/{len(texts)}] 跳过（噪声）")
            continue
        user_situation = conv.get("user_situation", "")
        replies = conv.get("replies", [])
        if not replies:
            skipped += 1
            continue
        for r in replies:
            rtext = r.get("text", "")
            if not rtext.strip():
                continue
            results.append({
                "original": text,
                "user_situation": user_situation,
                "reply": rtext,
                "length": r.get("length") or _length_tier(rtext),
            })
        print(f"  [{i+1}/{len(texts)}] ✓ {len(replies)} 条 | 触发情境：{user_situation[:25]}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 转化完成：{output_file}")
    print(f"   原文 {len(texts)} 条 → 样本 {len(results)} 条（跳过噪声 {skipped}）")
    print(f"   💡 下一步：人工筛选后补进 persona/speech/voice_samples.json，再重跑 precompute voice-samples")


async def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) < 2:
        print("用法: python scripts/convert_to_chat.py <原文.json> [输出.json]")
        return
    inputs = [p for p in sys.argv[1:] if not p.startswith("--")]
    out = Path("outputs/convert/converted_chat.json")
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
    await run(inputs, out)


if __name__ == "__main__":
    asyncio.run(main())
