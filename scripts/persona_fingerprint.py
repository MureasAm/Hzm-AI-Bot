#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人格层的尺子：**"像不像她自己在私聊里的样子"**（而不是"像不像她"——那个没有答案）。

## 为什么需要新尺子

人格层（traits/styles/骨架/behaviors）没有"标准答案"，不能用硬判据。
"像不像她"也无解（大五人格评测给她 14/14 = 100%，零区分度）。
所以换一个**可算**的问法：

    基准 = data/chat_log 里**她自己那 517 条真实私聊回复**的风格指纹
    读数 = 一段文本的指纹 **与基准的距离**（越小越像她自己）

⚠️ 为什么用 chat_log 而不是 voice_samples 当基准：那些是**直播语境**的产出，
   而我们要测的是**私聊**；chat_log 是同场景、真实产出的。

## 铁律：先验证尺子有分辨力，再用它下结论

`--validate` 做这件事：拿三类文本算指纹，看距离能不能把它们分开——

    ① 基准自身（她的 517 条真实回复）
    ② 她（真实链路生成，取现成产物）
    ③ 明显不像她的（客服腔 / 长句 / 不用自称——内置反例，零成本）

**如果 ③ 都被推到很近的距离 → 这把尺子废**（别再用它去测 traits/styles）。

## 用法
    python scripts/persona_fingerprint.py --validate        # 先跑这个
    python scripts/persona_fingerprint.py --baseline-stats  # 看看基准各维度的方差
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAT_LOG = ROOT / "data" / "chat_log" / "chat.jsonl"
EVALDIR = ROOT / "outputs" / "eval"

SELF = re.compile(r"(灰泽满|hzm|小满|满姐)")
# 她的形式特征（**全部对应文件里写死的东西**，不是随手挑的）
#  · 自称（第三人称保护） ← system_prompt【自我称呼】
#  · 长度短句 ← 【说话节奏】
#  · 括号「例外不是常态」← 骨架【括号与省略号】
#  · 省略号/问号/叹号 ← 【说话节奏】+ styles「碎碎念」「反问式推进」
def fingerprint(texts: list) -> dict:
    xs = [t or "" for t in texts if (t or "").strip()]
    if not xs:
        return {}
    n = len(xs)
    return {
        "自称率": sum(1 for t in xs if SELF.search(t)) / n,
        "「我」率": sum(1 for t in xs if "我" in t) / n,
        "括号率": sum(1 for t in xs if ("（" in t or "(" in t)) / n,
        "省略号率": sum(1 for t in xs if ("……" in t or "..." in t)) / n,
        "问号率": sum(1 for t in xs if re.search(r"[?？]", t)) / n,
        "叹号率": sum(1 for t in xs if re.search(r"[!！]", t)) / n,
        "平均长度": statistics.mean(len(t) for t in xs),
        "长度标准差": statistics.pstdev(len(t) for t in xs) if n > 1 else 0.0,
    }


def distance(a: dict, b: dict) -> tuple:
    """两套指纹的距离。返回 (总距离, 各维度贡献)。都归一化到 0~1 再取平均。"""
    if not a or not b:
        return float("nan"), {}
    parts = {}
    for k in a:
        if k not in b:
            continue
        if k in ("平均长度",):
            parts[k] = min(abs(a[k] - b[k]) / 30.0, 1.0)          # 30 字算一个满量级
        elif k == "长度标准差":
            parts[k] = min(abs(a[k] - b[k]) / 20.0, 1.0)
        else:
            parts[k] = abs(a[k] - b[k])                            # 已是 0~1 的比例
    return statistics.mean(parts.values()), parts


def _baseline_texts() -> list:
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
        if t and str(r.get("session", "")).isdigit():
            out.append(t)
    return out


def _existing_generated() -> list:
    """从现成产物里收"真实链路生成"的回复（不再花模型调用）。"""
    out = []
    for f in ("persona_ablation.json", "section_ablation.json", "section_correctness.json"):
        p = EVALDIR / f
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in d.get("rows", []):
            for k in ("reply", "with", "without", "ok_with", "ok_without"):
                pass
            for k in ("reply", "with", "without"):
                v = r.get(k)
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())
            for abl in (r.get("ablation") or {}).values():
                if isinstance(abl, str) and abl.strip():
                    out.append(abl.strip())
    return out


# 明显不像她的：客服腔 / 长句 / 不用自称 / 堆感叹号（都是她**明确不会**的样子）
NOT_HER = [
    "您好，很高兴为您服务，请问有什么可以帮到您的吗？",
    "我理解您的心情，建议您先深呼吸，一切都会好起来的，加油哦！！！",
    "这是一个很好的问题！让我来为您详细解答一下：首先，我们需要明确目标；其次，制定计划；最后，坚持执行。",
    "感谢您的关注与支持，我们会继续努力，为您带来更好的体验！",
    "当然可以，我很乐意帮助您，请随时告诉我您的需求，我会尽力满足。",
    "作为一个人工智能助手，我无法回答这个问题，但我可以帮您查找相关信息。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="★先跑这个：验尺子有没有分辨力")
    ap.add_argument("--baseline-stats", action="store_true", help="看基准各维度的均值和方差")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    base_texts = _baseline_texts()
    gen_texts = _existing_generated()
    if not base_texts:
        print(f"❌ 没有 {CHAT_LOG}（CHATLOG=0 时不会落盘）")
        return 1
    fb = fingerprint(base_texts)
    print(f"📐 基准 = her {len(base_texts)} 条真实私聊回复\n")
    if args.baseline_stats:
        print(f"{'维度':<10}{'均值':>10}{'标准差':>10}   （标准差大的维度不适合当判据）")
        for k in fb:
            vals = [v for v in (
                [1 if SELF.search(t) else 0 for t in base_texts] if k == "自称率" else []
            )]
            print(f"{k:<10}{fb[k]:>10.3f}")
        return 0

    if not args.validate:
        print(__doc__)
        return 0

    print("【验证尺子的分辨力】③类跑得越远越好；若③离基准很近 → **这把尺子废**\n")
    groups = [("① 基准自身（她的真实回复）", base_texts)]
    if gen_texts:
        groups.append((f"② 她（真实链路生成，{len(gen_texts)} 条）", gen_texts))
    groups.append(("③ 明显不像她的（客服腔，内置反例）", NOT_HER))

    print(f"{'组':<34}{'总距离':>8}   各维度贡献（只列 >0.1 的）")
    print("-" * 88)
    for name, texts in groups:
        f = fingerprint(texts)
        d, parts = distance(fb, f)
        big = "、".join(f"{k} {v:.2f}" for k, v in sorted(parts.items(), key=lambda x: -x[1]) if v > 0.1)
        print(f"{name:<34}{d:>8.3f}   {big}")

    print("\n【细看：各维度上的值】")
    print(f"{'维度':<10}{'基准':>10}{'②她':>10}{'③客服腔':>10}")
    for k in fb:
        f2 = fingerprint(gen_texts).get(k, float("nan")) if gen_texts else float("nan")
        f3 = fingerprint(NOT_HER).get(k, float("nan"))
        print(f"{k:<10}{fb[k]:>10.3f}{f2:>10.3f}{f3:>10.3f}")
    print("\n⚠️ 判断标准：**③ 的总距离要明显大于 ②**（即尺子能把『不像她』和『像她』分开）。")
    print("   如果 ③ ≈ ②，说明这些维度测不出『像不像』 → **得换维度**，")
    print("   别急着拿它去做 traits/styles 的消融（那就是拿废尺子量东西）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
