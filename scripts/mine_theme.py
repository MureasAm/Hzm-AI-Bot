#!/usr/bin/env python3
"""主题素材挖掘：调用模型专门找某类主题的素材瞬间（语义挖掘，非关键词硬搜）。

用法：
    python scripts/run_tool.py mine-theme -i <转写或清洗json> --theme 立Flag秒打脸 [-o 输出.json]

把转写片段按批（带重叠）喂给 DeepSeek，让它找出命中主题的完整瞬间（含原话+时间），
供人工审批后补进 behaviors/voice_samples。主题用一句话描述，可扩展。
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

from openai import AsyncOpenAI

BATCH = 35      # 每批段落数
OVERLAP = 8     # 批间重叠（跨批的主题瞬间不丢）
MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com/v1"

# 主题 → 详细定义（喂给模型的"什么算命中"）
THEMES = {
    "立Flag秒打脸": (
        "立下一个承诺/目标/Flag（如'明天一定早睡''这周不鸽''每天早起打卡''下次一定准时'），"
        "然后立刻（同场或紧接着）自我怀疑、讨价还价、找借口、承认做不到，或被自己的话推翻"
    ),
}


def load_segments(input_path: str) -> list:
    data = json.load(open(input_path, encoding="utf-8"))
    segs = []
    if isinstance(data, list):
        for it in data:
            if isinstance(it, dict) and it.get("text"):
                segs.append(it)
            elif isinstance(it, str):
                segs.append({"text": it})
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str):
                segs.append({"text": v})
    return segs


def batched(segs, size=BATCH, overlap=OVERLAP):
    step = max(1, size - overlap)
    for i in range(0, len(segs), step):
        yield segs[i:i + size]


async def mine_batch(client, batch: list, theme_meta: str, batch_no: int) -> list:
    lines = []
    for i, s in enumerate(batch):
        start = s.get("start", ""); end = s.get("end", "")
        ts = f"[{start}-{end}]" if start != "" else f"#{i}"
        lines.append(f"{ts} {s['text']}")
    prompt = f"""你是灰泽满直播素材的主题挖掘师。下面是直播转写片段（每段带编号/时间），共 {len(batch)} 段。

任务：找出【{theme_meta}】的完整瞬间。

要求：
- 只找"立Flag 和 秒打脸两个动作都能在转写里对上"的瞬间（立了Flag，然后同场/紧接着被自己怀疑/讨价还价/找借口/推翻）
- 对每个命中输出 JSON 对象：{{"start": "段落编号或时间", "quote": "最能代表这段的原话(尽量完整)", "reason": "一句话说明为什么命中"}}
- **宁缺毋滥**：拿不准就不报。这是找素材，绝不编造原话。

转写片段：
{chr(10).join(lines)}

只输出 JSON 数组（没有命中就 []），不要多余内容。"""
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = (resp.choices[0].message.content or "").strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)
    except Exception as e:
        print(f"⚠️ 批 {batch_no} 挖掘失败: {e}")
        return []


async def run(input_path: str, theme: str, output_file: str | None = None):
    segs = load_segments(input_path)
    if not segs:
        print("❌ 无有效片段")
        return
    theme_meta = THEMES.get(theme, theme)
    key = _common.get_api_key("OPENAI_API_KEY")
    if not key:
        print("❌ 未找到 OPENAI_API_KEY")
        return
    client = AsyncOpenAI(api_key=key, base_url=BASE_URL)
    print(f"🧠 用模型挖掘【{theme}】... 共 {len(segs)} 段，分 {len(list(batched(segs)))} 批")
    results = []
    for no, batch in enumerate(batched(segs), 1):
        hits = await mine_batch(client, batch, theme_meta, no)
        if hits:
            results.extend(hits)
            print(f"  ✓ 批 {no}: {len(hits)} 个命中")
    out = Path(output_file) if output_file else Path("outputs/transcribe/mined_theme.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n✅ 共命中 {len(results)} 条 → {out}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.get('start')} | {r.get('reason')}")
        print(f"    {r.get('quote', '')[:100]}")


def main():
    _common.ensure_utf8_stdout()
    p = argparse.ArgumentParser(description="主题素材挖掘（模型语义找素材）")
    p.add_argument("-i", "--input", required=True, help="转写或清洗后的 JSON")
    p.add_argument("--theme", required=True, choices=list(THEMES), help="要挖的主题")
    p.add_argument("-o", "--output", default=None, help="输出路径")
    args = p.parse_args()
    asyncio.run(run(args.input, args.theme, args.output))


if __name__ == "__main__":
    main()
