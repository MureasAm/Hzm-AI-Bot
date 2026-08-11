#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人格一致性评测（项目外测试，不影响线上运行代码）。

参考 InCharacter（复旦/人大）方法：
1. 用大五人格（NEO-FFI 精简版）开放题，让灰泽满以角色身份回答
2. 支持匿名化：把 "灰泽满/hzm" 替换为假名，防止模型靠名字记忆作弊
3. 用另一个 LLM 判断回答是否体现目标特质，输出"特质还原率"

用法：
    python scripts/persona_eval.py                 # 正常评测
    python scripts/persona_eval.py --anonymous     # 匿名版（替换自称）
    python scripts/persona_eval.py --traits 嘴硬,自嘲   # 只测指定特质
    python scripts/persona_eval.py --full          # 输出每题的完整回答

输出：outputs/persona_eval/eval_<时间戳>.json + .md

注意：这是项目之外的测试脚本，不 import 任何运行代码，不读写线上记忆文件。
"""
import json
import sys
import asyncio
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env.prod"
PERSONA_DIR = PROJECT_ROOT / "persona"
OUT_DIR = PROJECT_ROOT / "outputs" / "persona_eval"

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com/v1"
THINKING_DISABLED = {"extra_body": {"thinking": {"type": "disabled"}}}

# ==================== 人格数据加载 ====================

def get_deepseek_key():
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY"):
                    return line.split("=", 1)[1].replace('"', '').strip()
    return None


def load_persona(anonymous=False):
    """加载人设（system_prompt + traits/styles）。anonymous=True 时替换自称。"""
    parts = []
    sys_prompt = PERSONA_DIR / "system_prompt.txt"
    if sys_prompt.exists():
        parts.append(sys_prompt.read_text(encoding="utf-8"))

    for fname in ["persona_traits.json", "persona_styles.json"]:
        f = PERSONA_DIR / fname
        if not f.exists():
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        lines = []
        for item in data:
            name = item.get("name", "")
            desc = item.get("description", "")
            if name and desc:
                lines.append(f"- {name}：{desc}")
            elif desc:
                lines.append(f"- {desc}")
        if lines:
            label = "【性格基底】" if fname == "persona_traits.json" else "【语言风格】"
            parts.append(label + "\n" + "\n".join(lines))

    persona = "\n\n".join(parts)
    if anonymous:
        # 匿名化：把所有自称替换成中性假名，防模型靠名字记忆作弊
        persona = persona.replace("灰泽满", "小满").replace("hzm", "小满")
    return persona


# ==================== 大五人格开放题 ====================

# 每个特质：人格维度 + 一道"她会怎么回应"的开放面试题（对齐 InCharacter 改写法）
# 维度：O 开放性 / C 尽责性 / E 外向性 / A 宜人性 / N 神经质
BIG_FIVE_QUESTIONS = [
    # --- 开放性 Openness ---
    ("O", "开放性", "她说自己会主动找新番、新游戏、新香水，但也说'看多了会觉得还是老的好'。被问'你平时会主动尝试完全陌生的领域吗'，她会怎么回答？"),
    ("O", "开放性", "一个绿冻给她安利了一个她完全没听过的冷门作品，她第一反应会是什么？会去试吗？"),
    # --- 尽责性 Conscientiousness ---
    ("C", "尽责性", "她说好晚上八点直播，但白天玩累了很想睡。快七点半了，她会怎么想、怎么做？"),
    ("C", "尽责性", "被问'你答应别人的事都能做到吗'，她怎么回答？她怎么看待自己立过的Flag？"),
    ("C", "尽责性", "作业/工作截止日期快到了但她还在拖延，她心里是怎么想的？会怎么描述自己？"),
    # --- 外向性 Extraversion ---
    ("E", "外向性", "被问'你更喜欢一个人待着还是和大家一起玩'，她会怎么答？"),
    ("E", "外向性", "线下见到很多绿冻围着她，她是兴奋还是紧张？会怎么说？"),
    ("E", "外向性", "直播时弹幕突然安静下来没人说话，她会怎么反应？"),
    # --- 宜人性 Agreeableness ---
    ("A", "宜人性", "被绿冻很冒犯地开了个玩笑，她第一反应是怼回去还是忍下来？会怎么说？"),
    ("A", "宜人性", "有人当面说她不好/挑她毛病，她会怎么回应？"),
    ("A", "宜人性", "朋友心情不好来找她倾诉，她会怎么回应？"),
    # --- 神经质 Neuroticism ---
    ("N", "神经质", "被问'你最近压力大吗、会焦虑吗'，她会怎么描述自己的状态？"),
    ("N", "神经质", "直播前她会不会紧张/担心没人看？会怎么表达这种情绪？"),
    ("N", "神经质", "深夜一个人的时候，她心里一般是什么状态？"),
]


# 特质还原判定 prompt：让裁判 LLM 判断回答是否体现该人格维度及极性
JUDGE_PROMPT = """你是一个角色人格一致性裁判。下面是"灰泽满"（虚拟主播）的一段作答，以及一个待判定的人格维度。

请判断这段作答是否**稳定体现了**该维度（不看是否嘴硬，看回答背后的真实倾向）：

维度：{dimension}（{dim_label}）
说明：
- 开放性 O：是否好奇、愿尝试新事物 / 还是保守、怀旧、拒绝陌生
- 尽责性 C：是否自律、守约、有规划 / 还是拖延、随性、Flag必倒
- 外向性 E：是否从社交中获能、主动连接 / 还是独处回血、社交消耗
- 宜人性 A：是否友善、体谅、好说话 / 还是毒舌、挑剔、有棱角
- 神经质 N：是否易焦虑、敏感、内耗 / 还是情绪稳定、不内耗

【作答】
{answer}

输出 JSON：
{{
  "dimension": "O|C|E|A|N",
  "evident": true 或 false,
  "polarity": "high|low",
  "evidence": "从作答中哪句话看出来的，一句话",
  "confidence": 0.0-1.0
}}
只输出 JSON。"""


def build_answer_prompt(persona: str, question: str) -> str:
    """构造让灰泽满以角色回答开放题的 prompt。"""
    return (
        f"请以灰泽满的身份回答下面这个问题。\n"
        f"要求：用她的语气和风格回答（短句、自嘲、省略号、自称灰泽满/hzm），"
        f"控制在 2-4 句话。不要解释，直接回答。\n\n"
        f"【她的设定】\n{persona}\n\n"
        f"【问题】{question}\n\n"
        f"【她的回答】"
    )


# ==================== 评测逻辑 ====================

async def ask(client, prompt: str, max_tokens=300, temperature=0.7) -> str:
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        **THINKING_DISABLED,
    )
    return (resp.choices[0].message.content or "").strip()


async def judge(client, answer: str, dimension: str, label: str) -> dict:
    resp = await ask(
        client,
        JUDGE_PROMPT.format(dimension=dimension, dim_label=label, answer=answer),
        max_tokens=200, temperature=0,
    )
    content = resp
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"dimension": dimension, "evident": False, "polarity": "low",
                "evidence": "（裁判输出无法解析）", "confidence": 0.0}


async def run(anonymous=False, traits_filter=None, full=False):
    key = get_deepseek_key()
    if not key:
        print("❌ 未找到 OPENAI_API_KEY")
        return

    persona = load_persona(anonymous=anonymous)
    if not persona.strip():
        print("❌ 未加载到人格数据")
        return

    # 组装题目
    questions = BIG_FIVE_QUESTIONS
    if traits_filter:
        dim_map = {"O": "开放性", "C": "尽责性", "E": "外向性", "A": "宜人性", "N": "神经质"}
        allowed = []
        for t in traits_filter:
            t = t.strip().upper()
            if t in dim_map:
                allowed.append(t)
        if allowed:
            questions = [q for q in questions if q[0] in allowed]
        if not questions:
            print(f"❌ 没有匹配的维度，可选: O,C,E,A,N 或 {list(dim_map.values())}")
            return

    mode = "匿名" if anonymous else "实名"
    print(f"🔍 人格一致性评测（{mode}）| {len(questions)} 题 | 维度: {sorted(set(q[0] for q in questions))}")

    client = AsyncOpenAI(api_key=key, base_url=BASE_URL)
    results = []
    for dim, label, question in questions:
        answer = await ask(client, build_answer_prompt(persona, question))
        verdict = await judge(client, answer, dim, label)
        verdict["question"] = question
        verdict["answer"] = answer
        results.append(verdict)
        flag = "✓" if verdict.get("evident") else "✗"
        print(f"  [{flag}] {dim}/{label}: {verdict.get('evidence', '')[:40]}")

    # 汇总
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_stem = f"eval_{'anon_' if anonymous else ''}{ts}"
    summary = summarize(results, mode, ts)

    out_json = OUT_DIR / f"{out_stem}.json"
    out_json.write_text(json.dumps({
        "mode": mode,
        "timestamp": ts,
        "results": results,
        "summary": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# 灰泽满人格一致性评测（{mode}）", ""]
    lines.append(f"时间：{ts}")
    lines.append("")
    lines.append("## 汇总")
    for k, v in summary.items():
        if k == "总还原率":
            lines.append(f"- **{k}：{v}**")
        else:
            lines.append(f"- {k}：{v}")
    lines.append("")
    lines.append("## 逐题结果")
    for r in results:
        flag = "✓" if r.get("evident") else "✗"
        lines.append(f"\n### [{flag}] {r['dimension']}/{r['dim_label'] if 'dim_label' in r else r['question'][:20]}")
        lines.append(f"**问题**：{r['question']}")
        lines.append(f"**她的回答**：{r['answer']}")
        lines.append(f"**判定**：{'体现' if r.get('evident') else '未体现'}（极性 {r.get('polarity')}，置信 {r.get('confidence')}）— {r.get('evidence', '')}")
    out_md = OUT_DIR / f"{out_stem}.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n✅ 评测完成")
    print(f"   总还原率: {summary['总还原率']}")
    print(f"   明细: {out_json}")
    print(f"   报告: {out_md}")


def summarize(results: list, mode: str, ts: str) -> dict:
    from collections import Counter
    by_dim = {}
    for r in results:
        d = r.get("dimension", "?")
        by_dim.setdefault(d, []).append(bool(r.get("evident")))
    total_evident = sum(1 for r in results if r.get("evident"))
    total = len(results)
    summary = {"总还原率": f"{total_evident}/{total}（{total_evident / total * 100:.0f}%）" if total else "0"}
    for d, flags in sorted(by_dim.items()):
        e = sum(flags)
        summary[f"维度 {d}"] = f"{e}/{len(flags)}（{e / len(flags) * 100:.0f}%）" if flags else "无题"
    return summary


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    anonymous = "--anonymous" in sys.argv
    full = "--full" in sys.argv
    traits = None
    if "--traits" in sys.argv:
        traits = sys.argv[sys.argv.index("--traits") + 1].split(",")
    asyncio.run(run(anonymous=anonymous, traits_filter=traits, full=full))


if __name__ == "__main__":
    main()
