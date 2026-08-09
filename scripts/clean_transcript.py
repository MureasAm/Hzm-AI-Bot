"""转写清洗工具：把 faster-whisper 原始转写清洗成语义完整的文本。

原始转写的典型问题：碎片化（一句话切成多段）、口语杂质（呃/那个）、
错字（陶婉安→桃晚安）、无标点连写。本工具用 DeepSeek 逐话轮清洗：
合并碎片 → 删口语杂质 → 修错字 → 加标点。

用法：
    python scripts/run_tool.py clean-transcript -i 原始转写.json
    python scripts/run_tool.py clean-transcript -i a.json -i b.json -o 输出.json

输出格式：[{"start","end","text(已清洗)"}, ...]
"""
import json
import asyncio
import sys
from pathlib import Path
from openai import AsyncOpenAI

from analyze_pace import aggregate_turns, load_transcript, has_timestamps, get_deepseek_key

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com/v1"

CLEAN_PROMPT = """你是直播转写的清洗员。我会给你一段直播转写的原始片段（可能含口语杂质、错字、无标点、碎片化的内容）。

请把它清洗成：一句话（或几句完整的话），修正明显错字，去掉口语杂质（"呃""那个""然后"等无意义填充），加上标点。

要求：
- 统一转为简体中文：如果原文有繁体字，一律转成简体（whisper 常把中文输出成繁体）
- 保留说话人真实的意思和语气，不要改写内容、不要加戏
- 保留她对粉丝的昵称（如"灰子们""绿冻"）和标志性自称（"hzm""灰泽满"）
- 已知的固定错字修正（务必应用）：
  · "灰子板""灰子满""灰计板""会计版""慧心满" → 灰泽满
  · "矿课" → 旷课
  · "联脉""联麦" → 连麦
  · "住宵""销户" → 注销
  · "陶婉安" → 桃晚安
  · "满巴" → 满爸（灰泽满的爸爸）
  · "三季嘉宾" → 3D嘉宾（虚拟主播平时 LIVE2D 直播，只有特殊场次开 3D 回）
- 如果一段里包含多个独立意思，用句号分开
- 直接输出清洗后的文本，不要任何解释

【原始片段】
{text}

【清洗后】"""


def load_raw_transcripts(input_paths) -> list:
    """加载原始转写。每条转为 {"start","end","text"} 或纯文本。"""
    items = []
    for p in input_paths:
        items.extend(load_transcript(p))
    return items


def build_clean_turns(raw_items, gap=2.0) -> list:
    """按时间间隔聚合话轮，返回 [{start,end,text}]。"""
    if not has_timestamps(raw_items):
        # 无时间戳：每条独立
        return [{"start": None, "end": None, "text": t}
                for t in raw_items if isinstance(t, dict) and t.get("text", "").strip()]

    # 聚合：记录每段的 start/end
    turns = []
    cur_texts, cur_start, cur_end = [], None, None
    for it in raw_items:
        text = it.get("text", "").strip()
        if not text:
            continue
        if cur_texts and cur_end is not None and it["start"] - cur_end > gap:
            turns.append({"start": cur_start, "end": cur_end, "text": "".join(cur_texts)})
            cur_texts, cur_start, cur_end = [], None, None
        if cur_start is None:
            cur_start = it["start"]
        cur_texts.append(text)
        cur_end = it["end"]
    if cur_texts:
        turns.append({"start": cur_start, "end": cur_end, "text": "".join(cur_texts)})
    return turns


async def clean_turn(client, text: str) -> str:
    """调 DeepSeek 清洗一个话轮。失败返回原文本。"""
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": CLEAN_PROMPT.format(text=text)}],
            temperature=0.1,
            max_tokens=200,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ 清洗失败: {e}")
        return text


async def run(input_paths, output_file, turn_gap=2.0):
    """参数化入口（供 run_tool 调用）。"""
    key = get_deepseek_key()
    if not key:
        print("❌ 未找到 OPENAI_API_KEY")
        return

    raw_items = load_raw_transcripts(input_paths)
    turns = build_clean_turns(raw_items, gap=turn_gap)
    print(f"📄 原始片段 {len(raw_items)} → 聚合出 {len(turns)} 个话轮")

    client = AsyncOpenAI(api_key=key, base_url=BASE_URL)
    print("🧠 正在清洗话轮...")
    out_turns = []
    for i, t in enumerate(turns):
        cleaned = await clean_turn(client, t["text"])
        out_turns.append({"start": t["start"], "end": t["end"], "text": cleaned})
        print(f"  [{i+1}/{len(turns)}] {t['text'][:25]}... → {cleaned[:25]}...")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out_turns, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 清洗完成，输出：{output_file}（{len(out_turns)} 个话轮）")


async def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) < 2:
        print("用法: python scripts/clean_transcript.py <转写.json> [输出.json]")
        return
    input_paths = [p for p in sys.argv[1:] if not p.startswith("--")]
    output_file = Path("outputs/transcribe/cleaned.json")
    if "--out" in sys.argv:
        output_file = Path(sys.argv[sys.argv.index("--out") + 1])
    await run(input_paths, output_file)


if __name__ == "__main__":
    asyncio.run(main())
