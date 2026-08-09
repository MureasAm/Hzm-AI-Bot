"""灰泽满"节奏地图"分析工具。

把直播转写按"话轮"聚合，调 DeepSeek 给每个话轮标注：
- 场景（被夸/被催播/被越界/日常/感性流露/摆烂…）
- 情绪强度（low/medium/high）
- 长度档（short≤30 / medium≤60 / long）
- 代表性措辞（她在这句话里的标志性说法）

最后汇总成"灰泽满的节奏地图"——哪类场景她短句、哪类她长句，
以及她的高频措辞清单。这份地图是之后补声音样本 / 措辞指纹库的依据。

用法：
    python scripts/analyze_pace.py <转写.json> [--out outputs/pace_map]

转写格式支持两种：
    1. 带时间戳：[{"start":0.0,"end":1.4,"text":"..."}]  → 自动按间隔聚合话轮
    2. 纯文本列表：[{"text":"..."}] 或 ["..."]            → 每条视为一个话轮
"""
import json
import sys
import asyncio
from pathlib import Path
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env.prod"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "pace_map"
MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com/v1"
TURN_GAP_SEC = 2.0  # 间隔 ≤2s 视为同一句话

# 场景分类（对齐 persona_behaviors + 直播常见场景）
SCENARIOS = [
    "被夸时", "被戳穿/被质疑", "被越界/被调戏", "被催播/催更",
    "日常闲聊", "感性流露/孤独", "摆烂/拖延", "分享倒霉事",
    "立Flag/承诺", "主动抛梗", "其他",
]


def get_deepseek_key():
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY"):
                    return line.split("=")[1].replace('"', '').strip()
    return None


def load_transcript(path) -> list:
    """加载转写，统一成 [{"start","end","text"}] 或 [{"text"}]。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and data and isinstance(data[0], str):
        return [{"text": t} for t in data]
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    raise ValueError(f"无法识别的转写结构: {path}")


def has_timestamps(items) -> bool:
    return bool(items) and "start" in items[0] and "end" in items[0]


def aggregate_turns(items, gap: float = TURN_GAP_SEC) -> list:
    """把带时间戳的连续片段聚合成话轮。间隔 ≤gap 视为同一句。"""
    if not has_timestamps(items):
        return [i["text"].strip() for i in items if i.get("text", "").strip()]
    turns, cur_parts, cur_end = [], [], None
    for it in items:
        text = it.get("text", "").strip()
        if not text:
            continue
        if cur_parts and cur_end is not None and it["start"] - cur_end > gap:
            turns.append("".join(cur_parts))
            cur_parts = []
        cur_parts.append(text)
        cur_end = it["end"]
    if cur_parts:
        turns.append("".join(cur_parts))
    return turns


ANALYZE_PROMPT = """你是灰泽满直播语料的节奏分析师。我会给你一段她的原话。
{focus_instruction}
【第一步：判定是否值得分析】
先判断这段是不是"有主题的高质量对话"。高质量 = 有明确话题、有画面感、能体现她的人格/口癖/反应模式。
以下情况属于【噪声】，直接判定 noise=true：
- 念礼物/舰长名单（"谢谢某某的xxx"）
- 纯客套寒暄（连续"谢谢""晚上好""中午好"）
- 无信息量的重复刷屏
- 读弹幕/转述他人（"弹幕说…""他说…"）
- 明显口语填充无实质内容

【第二步：若值得分析，再标注】
只有当 noise=false 时，才分析场景和措辞。场景必须是这段**她本人真实的表达**。

输出 JSON：
{{
  "noise": true 或 false,
  "scenario": "被夸时|被戳穿/被质疑|被越界/被调戏|被催播/催更|日常闲聊|感性流露/孤独|摆烂/拖延|分享倒霉事|立Flag/承诺|主动抛梗|其他",
  "emotion_intensity": "low|medium|high",
  "length_tier": "short|medium|long",
  "representative_phrases": ["她这段话里的标志性措辞，1-3个，如'好吧'、'神经吧！'、'被作业封印了'"]
}}

noise=true 时，scenario 填"其他"，其余字段给默认值。
【原话】
{text}

只输出 JSON，不要多余内容。"""


# 场景别名 → 正式标签（供 --focus 用，用户可传简称）
FOCUS_ALIASES = {
    "被调戏": "被越界/被调戏", "被越界": "被越界/被调戏", "越界": "被越界/被调戏",
    "被夸": "被夸时", "夸": "被夸时",
    "被戳穿": "被戳穿/被质疑", "被质疑": "被戳穿/被质疑", "戳穿": "被戳穿/被质疑",
    "立flag": "立Flag/承诺", "flag": "立Flag/承诺", "承诺": "立Flag/承诺",
    "失约": "被催播/催更", "被催播": "被催播/催更", "催播": "被催播/催更",
    "感性": "感性流露/孤独", "孤独": "感性流露/孤独",
    "倒霉": "分享倒霉事", "分享倒霉事": "分享倒霉事",
    "摆烂": "摆烂/拖延", "拖延": "摆烂/拖延",
    "抛梗": "主动抛梗",
    "日常": "日常闲聊",
}


def normalize_focus(focus) -> list:
    """把用户传的 focus 简称映射为正式场景标签；未知值原样保留。"""
    if not focus:
        return []
    out = []
    for f in focus:
        key = str(f).strip().lower()
        out.append(FOCUS_ALIASES.get(key, str(f).strip()))
    return out


def build_focus_instruction(focus: list) -> str:
    """聚焦模式下生成提示词片段：只精标 2-3 个场景，其余归"其他"，并要求给判定依据。"""
    if not focus:
        return ""
    focus_str = "、".join(focus)
    extra = ""
    if "被催播/催更" in focus:
        extra = (
            f"\n【失约片段的范围】\"被催播/催更\"不仅指弹幕直接催播，还包括：\n"
            f"- 迟到/鸽了/该播没播的开场解释（\"为什么迟到\"\"今天怎么又没播\"\"多久没直播了\"）\n"
            f"- 被弹幕戳穿作息/状态后，解释直播安排的回应\n"
            f"- 提到\"下周一定\"\"明天准时\"\"这周表全准时\"等补救承诺的片段\n"
            f"以上都属于失约片段，归入\"被催播/催更\"。\n"
        )
    return (
        f"【本场聚焦目标】\n"
        f"本场只关注以下 {len(focus)} 个场景：{focus_str}。\n"
        f"- 话轮属于其中任一场景 → scenario 填该具体场景，并新增字段 \"reasoning\"，用一句话写判定依据（你从原文看到什么信号，如'她在推开对方''她先否认再承认'）\n"
        f"- 不属于任何一个 → scenario 填\"其他\"\n"
        f"- 判定必须基于这段里她本人的表达，不是转述他人\n"
        f"{extra}"
    )


async def analyze_turn(client, text: str, focus: list = None) -> dict:
    """调 DeepSeek 分析一个话轮。失败返回降级标注。"""
    focus_instruction = build_focus_instruction(focus or [])
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": ANALYZE_PROMPT.format(text=text, focus_instruction=focus_instruction)}],
            temperature=0.1,
            max_tokens=120,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = resp.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        result = json.loads(content)
        result.setdefault("noise", False)
        return result
    except Exception as e:
        print(f"⚠️ 分析失败: {e}")
        return {"noise": True, "scenario": "其他", "emotion_intensity": "low",
                "length_tier": "short", "representative_phrases": []}


def length_tier(text: str) -> str:
    n = len(text)
    return "short" if n <= 30 else ("medium" if n <= 60 else "long")


def summarize(turns, annotations) -> dict:
    """汇总节奏地图。"""
    from collections import Counter
    scene_counts = Counter()
    scene_len = {}       # 场景 → 长度档统计
    scene_emotion = {}   # 场景 → 情绪强度统计
    phrase_counter = Counter()

    for text, ann in zip(turns, annotations):
        if ann.get("noise", False):
            continue  # 噪声话轮不计入节奏地图
        scene = ann.get("scenario", "其他")
        scene_counts[scene] += 1
        scene_len.setdefault(scene, Counter())[ann.get("length_tier", length_tier(text))] += 1
        scene_emotion.setdefault(scene, Counter())[ann.get("emotion_intensity", "low")] += 1
        for p in ann.get("representative_phrases", []):
            phrase_counter[p] += 1

    # 场景节奏总览：主要长度档
    scene_overview = {}
    for scene in scene_counts:
        lens = scene_len.get(scene, {})
        total = sum(lens.values())
        main_tier = max(lens, key=lens.get) if lens else "unknown"
        scene_overview[scene] = {
            "count": scene_counts[scene],
            "main_length_tier": main_tier,
            "tier_distribution": dict(lens),
            "main_emotion": max(scene_emotion.get(scene, {}), key=lambda k: scene_emotion.get(scene, {}).get(k, 0)) if scene_emotion.get(scene) else "unknown",
        }

    valid_turns = [t for t, a in zip(turns, annotations) if not a.get("noise", False)]
    return {
        "total_turns": len(valid_turns),
        "raw_turns": len(turns),
        "noise_turns": len(turns) - len(valid_turns),
        "avg_len": round(sum(len(t) for t in valid_turns) / len(valid_turns)) if valid_turns else 0,
        "scene_overview": scene_overview,
        "top_phrases": [{"phrase": p, "count": c} for p, c in phrase_counter.most_common(30)],
    }


def render_markdown(summary) -> str:
    lines = ["# 灰泽满节奏地图", ""]
    noise_note = f"，剔除噪声 {summary.get('noise_turns', 0)} 条" if summary.get("noise_turns") else ""
    lines.append(f"有效话轮数：**{summary['total_turns']}**{noise_note}，平均长度：**{summary['avg_len']}字**")
    lines.append("")
    lines.append("## 场景节奏总览")
    lines.append("")
    lines.append("| 场景 | 数量 | 主要长度档 | 长度分布 | 主要情绪 |")
    lines.append("|---|---|---|---|---|")
    for scene, info in sorted(summary["scene_overview"].items(), key=lambda x: -x[1]["count"]):
        dist = " / ".join(f"{k}:{v}" for k, v in sorted(info["tier_distribution"].items()))
        lines.append(f"| {scene} | {info['count']} | {info['main_length_tier']} | {dist} | {info['main_emotion']} |")
    lines.append("")
    lines.append("## 高频措辞（Top 30）")
    lines.append("")
    for item in summary["top_phrases"]:
        lines.append(f"- {item['phrase']}（{item['count']}次）")
    return "\n".join(lines)


def merge_pace(prev: dict | None, new_turns: list[dict]) -> dict:
    """把新一批带 session 标注的话轮并入既有节奏地图，重新汇总。"""
    prev_turns, prev_sessions = [], []
    if prev:
        prev_turns = prev.get("turns", [])
        prev_sessions = prev.get("sessions", [])
    sessions = prev_sessions + [
        s for s in {t.get("session") for t in new_turns}
        if s and s not in prev_sessions
    ]
    turns = prev_turns + new_turns
    summary = summarize([t["text"] for t in turns], turns)
    return {"sessions": sessions, "summary": summary, "turns": turns}


async def run(input_paths, out_prefix, sessions=None, merge=False, turn_gap=TURN_GAP_SEC,
              focus=None):
    """参数化入口（供 run_tool 调用）。支持多场合并、增量累积。

    focus: 每场聚焦的场景简称列表（如 ["被调戏", "被夸"]），会归一化为正式标签。
    """
    key = get_deepseek_key()
    if not key:
        print("❌ 未找到 OPENAI_API_KEY")
        return

    focus_labels = normalize_focus(focus) if focus else None
    if focus_labels:
        print(f"🎯 聚焦场景：{'、'.join(focus_labels)}（其余话轮归'其他'）")

    if sessions is None:
        sessions = [f"第{i+1}场" for i in range(len(input_paths))]
    if len(sessions) != len(input_paths):
        print("❌ --session 个数与 --input 个数不一致")
        return

    # 1. 加载既有结果（若 merge）
    prev = None
    out_json = Path(out_prefix).with_suffix(".json")
    if merge and out_json.exists():
        with open(out_json, "r", encoding="utf-8") as f:
            prev = json.load(f)

    # 2. 逐个分析本批次
    client = AsyncOpenAI(api_key=key, base_url=BASE_URL)
    all_detail = []
    for input_path, session in zip(input_paths, sessions):
        items = load_transcript(input_path)
        turns = aggregate_turns(items, gap=turn_gap)
        print(f"📄 {Path(input_path).name}: {len(items)} 片段 → {len(turns)} 话轮")
        print(f"🧠 正在分析 {session} ...")
        for i, text in enumerate(turns):
            ann = await analyze_turn(client, text, focus=focus_labels)
            ann["length_tier"] = length_tier(text)  # 长度档用真实字数，不用模型猜
            all_detail.append({"text": text, "len": len(text), "session": session, **ann})
            print(f"  [{i+1}/{len(turns)}] ({ann['scenario']}/{ann['length_tier']}) {text[:30]}...")

    # 3. 合并或覆盖
    merged = merge_pace(prev, all_detail) if merge else {
        "sessions": list({t["session"] for t in all_detail}),
        "summary": summarize([t["text"] for t in all_detail], all_detail),
        "turns": all_detail,
    }

    # 4. 输出
    out_md = Path(out_prefix).with_suffix(".md")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(render_markdown(merged["summary"]))

    print(f"\n✅ 完成")
    print(f"   JSON 明细: {out_json}")
    print(f"   Markdown 总览: {out_md}")


async def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if len(sys.argv) < 2:
        print("用法: python scripts/analyze_pace.py <转写.json> [--out 输出前缀] [--session 标签] [--merge]")
        return
    input_paths = [p for p in sys.argv[1:] if not p.startswith("--")]
    out_prefix = DEFAULT_OUT
    sessions = None
    merge = False
    if "--out" in sys.argv:
        out_prefix = Path(sys.argv[sys.argv.index("--out") + 1])
    if "--session" in sys.argv:
        sessions = [sys.argv[sys.argv.index("--session") + 1]]
    if "--merge" in sys.argv:
        merge = True
    await run(input_paths, out_prefix, sessions=sessions, merge=merge)


if __name__ == "__main__":
    asyncio.run(main())
