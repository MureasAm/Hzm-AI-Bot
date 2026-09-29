#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风格读数 v3：**成对比较**（绝对打分不稳，噪声 ≥ 信号）。

## 为什么换（这条有实测依据）

指纹距离那版量到的东西，**两次运行自己就波动 0.03**（生成温度 0.85），
而效应量是 0.01~0.06 —— **噪声和信号同量级**。两版安慰剂给出**相反**的判定，就是证据。

成对比较消掉了最大的方差源（"这段文字本身离她多远"），只留"**两句里谁更像她**"，
再用**位置随机 + 多次重复 + 二项检验**把裁判噪声也压掉。

## 三条纪律（缺一条结论就不成立）

1. **先用已知答案校准**：拿「她的真实回复 vs 客服腔」当对照 ——
   裁判必须几乎每次都选她的。**校准不过关 → 这个读数废，别用它下任何结论。**
2. **位置随机**：A/B 每次重排（本仓实测过位置会显著影响结果）
3. **报显著性**：只报"选中率 + 二项检验 p 值"，**不报一个孤零零的百分数**

## 用法
    python scripts/pairwise_style.py --calibrate            # ① 先跑这个
    python scripts/pairwise_style.py                        # ② 再判已有的一堆对子
    python scripts/pairwise_style.py --section 她的固定说法   # 只看某段
"""
import argparse
import asyncio
import json
import random
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402
nonebot.init()

from openai import AsyncOpenAI  # noqa: E402
from src.plugins.chatbot.constants import DEEPSEEK_BASE_URL, THINKING_DISABLED  # noqa: E402
from src.plugins.chatbot.config import _get_model_name  # noqa: E402

EVALDIR = ROOT / "outputs" / "eval"
CHAT_LOG = ROOT / "data" / "chat_log" / "chat.jsonl"
ENV_FILE = ROOT / ".env.prod"

NOT_HER = [
    "您好，很高兴为您服务，请问有什么可以帮到您的吗？",
    "我理解您的心情，建议您先深呼吸，一切都会好起来的，加油哦！！！",
    "这是一个很好的问题！让我来为您详细解答一下：首先我们需要明确目标；其次制定计划。",
    "感谢您的关注与支持，我们会继续努力，为您带来更好的体验！",
    "当然可以，我很乐意帮助您，请随时告诉我您的需求，我会尽力满足。",
    "作为一个人工智能助手，我无法回答这个问题，但我可以帮您查找相关信息。",
]

PROMPT = """下面是虚拟主播「灰泽满」本人真实说过的一些话，请先读一遍，感受她的说话方式：

{ref}

---

现在有两条候选回复，**其中一条是她本人写的，另一条不是**。
请判断：**哪一条更像她？**

A：{a}

B：{b}

只输出一个字母（A 或 B），不要解释。"""


def _key(name: str) -> str:
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(name):
            return line.split("=", 1)[1].replace('"', "").strip()
    return ""


def _her_lines(k: int = 20) -> list:
    out = []
    for line in CHAT_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        t = (r.get("reply") or "").strip()
        if 8 <= len(t) <= 45 and str(r.get("session", "")).isdigit():
            out.append(t)
    step = max(1, len(out) // k)
    return out[::step][:k]


async def ask(ds, ref: str, x: str, y: str, flip: bool):
    """判一次：flip=True 时把 y 放 A 位（控位置偏差）。返回 'x' 或 'y' 或 None。"""
    a, b = (y, x) if flip else (x, y)
    try:
        r = await ds.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": PROMPT.format(ref=ref, a=a, b=b)}],
            temperature=0, max_tokens=4, **THINKING_DISABLED)
        c = (r.choices[0].message.content or "").strip().upper()
        if c.startswith("A"):
            return "y" if flip else "x"
        if c.startswith("B"):
            return "x" if flip else "y"
    except Exception as e:
        print(f"  ⚠️ 判定失败：{type(e).__name__}")
    return None


def binom_two_sided(k: int, n: int) -> float:
    """二项检验（p=0.5）双尾 p 值——用来回答"这个偏离是不是运气"。

    ⚠️ **第一版写错了，而且错得很隐蔽**：只算了**下尾**再乘 2（`2*P(X≤k)`）。
    当 k > n/2（选中率超过一半）时下尾≈1 → p 恒等于 1.0 →
    **把"100% 选中、本该极显著"的结果判成"无差别"**。
    抓到它的方式：`100% 却 p=1.000` **自相矛盾** —— 结论必须自洽，矛盾就是 bug。
    正确做法：取**两侧尾部里较小的那个**再乘 2。
    """
    if n == 0:
        return 1.0
    lo = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    hi = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * min(lo, hi))


REWRITE_PROMPT = """下面这句话是虚拟主播「灰泽满」说的。请**改写**它：

要求
- **保留意思**（说的是同一件事）
- 但**把她的说话方式去掉**：
  · 不要用"灰泽满 / hzm / 小满"这类第三人称自称，改用"我"
  · 句子拉长、更书面、更客气
  · 可以加括号补充说明、加感叹号
- 读起来要像"**一个礼貌的普通助手**"，而不像她

只输出改写后的那一句，不要任何解释。

原句：{x}"""


async def calibrate_hard(ds, ref: str, n: int, reps: int) -> bool:
    """**难度匹配的校准**：她的真话 vs「意思一样但被她改写掉的版本」。

    ⚠️ 为什么需要它：拿"她 vs 客服腔"当校准**太容易**——过了只说明裁判"不完全瞎"，
    **不说明它够灵敏**。而我要判的（traits/styles 有没有用）是**细微**差别。
    这个对照才是难度匹配的：两句话**说的是同一件事**，差的**只有风格**。
    """
    lines = _her_lines(n)
    win = tot = 0
    for i, real in enumerate(lines):
        try:
            r = await ds.chat.completions.create(
                model=_get_model_name(),
                messages=[{"role": "user", "content": REWRITE_PROMPT.format(x=real)}],
                temperature=0.7, max_tokens=120, **THINKING_DISABLED)
            fake = (r.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  ⚠️ 改写失败：{type(e).__name__}")
            continue
        if len(fake) < 4:
            continue
        for k in range(reps):
            got = await ask(ds, ref, real, fake, flip=bool((i + k) % 2))
            if got is None:
                continue
            tot += 1
            win += 1 if got == "x" else 0
        if (i + 1) % 10 == 0:
            print(f"  …{i+1}/{len(lines)}（当前选中她 {win}/{tot}）")
    if tot == 0:
        print("❌ 一句都没改成功")
        return False
    rate = win / tot
    p = binom_two_sided(win, tot)
    print(f"\n【难度匹配校准】她  vs  「意思一样、风格被砍掉」的版本")
    print(f"  选中她：{win}/{tot} = {rate*100:.0f}%　p={p:.2g}")
    ok = rate >= 0.75 and p < 0.05
    print("  判定：" + ("✅ 尺子够灵敏 → 可以用它判『某段有没有用』"
                       if ok else
                       "❌ **尺子不够灵敏** → 它给出的『测不出』一律不能当证据，"
                       "得换更强的判据（或加大样本）"))
    return ok


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="校准（容易版）：她 vs 客服腔")
    ap.add_argument("--calibrate-hard", action="store_true",
                    help="★校准（难度匹配）：她 vs 意思一样但风格被砍掉的版本")
    ap.add_argument("--section", default=None, help="只看某段")
    ap.add_argument("--pairs", type=int, default=14, help="每个条件取几对")
    ap.add_argument("--reps", type=int, default=3, help="每对判几次（位置会重排）")
    ap.add_argument("--file", default="section_ablation.json")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ds = AsyncOpenAI(api_key=_key("OPENAI_API_KEY"), base_url=DEEPSEEK_BASE_URL)
    ref = "\n".join(f"· {t}" for t in _her_lines(20))
    print(f"📏 成对比较（裁判参考 = 她本人 {len(_her_lines(20))} 条真实发言）\n")

    # ---------- ① 校准：裁判能不能分出"她"和"客服腔" ----------
    if args.calibrate_hard:
        return 0 if await calibrate_hard(ds, ref, args.pairs, args.reps) else 1

    if args.calibrate:
        picks = _her_lines(args.pairs)
        win = tot = 0
        for i, real in enumerate(picks):
            for k in range(args.reps):
                got = await ask(ds, ref, real, NOT_HER[i % len(NOT_HER)], flip=bool((i + k) % 2))
                if got is None:
                    continue
                tot += 1
                win += 1 if got == "x" else 0
        p = binom_two_sided(win, tot)
        print(f"【校准】她 vs 客服腔：选中她 {win}/{tot} = {win/max(tot,1)*100:.0f}%　p={p:.2g}")
        print("  判定：" + ("✅ 校准通过（裁判能分出像不像）→ 这个读数可用"
                          if win / max(tot, 1) > 0.8 else
                          "❌ **校准不过关 → 读数废**，别用它下任何结论"))
        return 0

    # ---------- ② 判已有的一堆对子 ----------
    rows_all = []
    for f in (args.file, "section_ablation.json", "persona_ablation.json"):
        p = EVALDIR / f
        if p.exists():
            rows_all += json.loads(p.read_text(encoding="utf-8")).get("rows", [])
    if not rows_all:
        print(f"❌ 没有可判的对子（先跑 persona_ablation.py --sections）")
        return 1

    from collections import defaultdict
    buckets = defaultdict(list)          # 段 -> [(baseline, ablated), ...]
    for r in rows_all:
        for s, abl in (r.get("ablation") or {}).items():
            if args.section and args.section not in s:
                continue
            if isinstance(abl, str) and abl.strip():
                buckets[s].append((r["reply"], abl))
        # 兼容 persona_ablation.json 的形态：`with` = 现状，`without` = 拿掉了 traits+styles
        if r.get("with") and r.get("without") and not (args.section and "traits" not in args.section):
            buckets["【性格基底】+【语言风格】（traits+styles）"].append((r["with"], r["without"]))
    if not buckets:
        print("❌ 没找到符合条件的对子")
        return 1

    print(f"{'段':<26}{'判次':>6}{'她(有那段)被选中':>16}{'p':>10}")
    print("-" * 62)
    for s, pairs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        pairs = pairs[: args.pairs]
        win = tot = 0
        for i, (base, abl) in enumerate(pairs):
            for k in range(args.reps):
                got = await ask(ds, ref, base, abl, flip=bool((i + k) % 2))
                if got is None:
                    continue
                tot += 1
                win += 1 if got == "x" else 0
        if tot == 0:
            continue
        p = binom_two_sided(win, tot)
        rate = win / tot
        tag = "⭐ 有那段更像我" if (rate > 0.6 and p < 0.05) else \
              "⭐ 没那段更像我" if (rate < 0.4 and p < 0.05) else "无差别（在噪声里）"
        print(f"{s[:24]:<26}{tot:>6}{rate*100:>15.0f}%{p:>10.3f}   {tag}")

    print("\n⚠️ 读法：**只有 p<0.05 才算有效应**；50% 附近一律读作『这个读数分不出』。")
    print("   它回答『有没有那段更像她』——**这是价值层面的问题**（比指纹距离强：")
    print("   指纹只能说『变了』，成对比较能说『变好还是变坏』）。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
