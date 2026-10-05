#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""内容普查（**零成本，不调任何模型**）：她的真实回复里，用上了哪些文件的内容。

## 为什么先做这个

用户的关键判断：**"我一直在没有真实回复数据的情况下去做优化。"**
——而 `data/chat_log/` 里有 1000+ 条**她真实说过的话**，不用花钱就能查：
**每个文件"要求她说的东西"，有多少真的出现在她的回复里？**

这一步回答的是「**出现过**」——**不等于**「**是因为注入了才出现**」（她自己本来也可能那么说话）。
所以：

    「出现过」是**上界**：连出现都没出现过 → 那个文件**几乎肯定没在起作用**（嫌疑最大）
    要确认「净作用」，必须做**消融**（`scripts/persona_ablation.py` 是第二级）

## 对每个文件用什么"探针"

| 文件 | 探针 | 匹配方式 |
|---|---|---|
| `legendary`（固定应答） | 那条固定应答的文本 | **字符 bigram 重合 ≥0.6**（她会照发或近似） |
| `behaviors`（16 条情景，内嵌她的原话 samples） | "她这么说过"那几句 | 同上 |
| `terms`（27 个词） | keyword + aliases | 子串 |
| `preferences`（23 条） | 该条的 `keywords` | 子串（⚠️ 那是"**用户会说的词**"，不是"她会说的词" → **弱探针**，只当参考） |

⚠️ **`corpus` / `core_stories` 故意不做**：它们是长陈述 / 成段原话，
词面探针测不出"有没有用上"（说同一件事可以一个词都不重）。**它们只能靠消融**。

## 用法

    python scripts/persona_usage.py              # 全部文件
    python scripts/persona_usage.py --file behaviors
    python scripts/persona_usage.py --list-unused   # 顺便列出"从没被用过"的条目
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAT_LOG = ROOT / "data" / "chat_log" / "chat.jsonl"
PERSONA = ROOT / "persona"


def _load(path: Path, *keys):
    d = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(d, list):
        return d
    for k in keys:
        if isinstance(d.get(k), list):
            return d[k]
    return [v for v in d.values() if isinstance(v, list)][-1]


def _bigrams(t: str) -> set:
    # ⚠️ 这里别写成 `r"[\s…""''…]"` —— 字符串里的 ASCII 双引号会把字面量**截断**，
    # 后半段变成非 raw 字符串、`\[` 就成了非法转义（`judge_experiment.py` 犯过同一个错）。
    # 标点类里只留全角引号，够用了。
    t = re.sub(r"[\s，。！？、；：（）【】()\[\]…~—\-]+", "", t or "")
    return {t[i:i + 2] for i in range(len(t) - 1)} or {t} if t else set()


def _overlap(a: str, b: str) -> float:
    """a 的 bigram 有多少出现在 b 里（0~1）。"""
    ba = _bigrams(a)
    if not ba:
        return 0.0
    return len(ba & _bigrams(b)) / len(ba)


# ==================== 各文件的探针 ====================

def probes_legendary():
    # ⚠️ 结构是 `{replies: {触发词: [应答...] 或 {replies: [...], pattern: ...}}}`（字典，不是列表）
    d = json.loads((PERSONA / "world" / "legendary.json").read_text(encoding="utf-8"))
    out = []
    for trigger, v in (d.get("replies") or {}).items():
        replies = v.get("replies") if isinstance(v, dict) else v
        for r in (replies or []):
            if isinstance(r, str) and len(r) >= 4:
                out.append((f"legendary:{trigger}", [r], "sim"))
    return out


def probes_behaviors():
    out = []
    for b in _load(PERSONA / "behavior" / "behaviors.json", "behaviors"):
        for s in (b.get("samples") or []):
            text = s.get("reply") if isinstance(s, dict) else s
            if isinstance(text, str) and len(text) >= 6:
                out.append((f"behavior:{(b.get('trigger') or '')[:12]}", [text], "sim"))
    return out


def probes_terms():
    # ⚠️ 只用 keyword（+aliases）：`meaning` 里的词不提（那是定义文字，拿去匹配回复会满地噪声）
    out = []
    for t in _load(PERSONA / "world" / "terms.json", "terms"):
        words = [t.get("keyword"), *(t.get("aliases") or [])]
        words = [str(w) for w in words
                 if w and len(str(w)) >= 2 and str(w) not in ("hzm", "灰泽满")]
        if words:
            out.append((f"term:{t.get('keyword')}", words, "sub"))
    return out


def probes_preferences():
    out = []
    for p in _load(PERSONA / "world" / "preferences.json", "entries"):
        words = [str(w) for w in (p.get("keywords") or []) if len(str(w)) >= 2]
        if words:
            out.append((f"pref:{p.get('id')}", words, "sub"))
    return out


BUILDERS = {
    "legendary": probes_legendary,
    "behaviors": probes_behaviors,
    "terms": probes_terms,
    "preferences": probes_preferences,
}


def main():
    ap = argparse.ArgumentParser(description="内容普查：她真实回复里用上了哪些文件的内容")
    ap.add_argument("--file", default=None, choices=list(BUILDERS))
    ap.add_argument("--list-unused", action="store_true", help="列出从没被用过的条目")
    ap.add_argument("--min-sim", type=float, default=0.6, help="sim 型探针的阈值（默认 0.6）")
    ap.add_argument("--show", type=int, default=3, help="每种文件举几个例子")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if not CHAT_LOG.exists():
        print(f"❌ 没有 {CHAT_LOG}（CHATLOG=0 时不会落盘）")
        return 1
    replies = []
    for line in CHAT_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        r = (rec.get("reply") or "").strip()
        if r and str(rec.get("session", "")).isdigit():     # 只看真实会话，排除测试会话
            replies.append(r)
    print(f"📊 扫了 {len(replies)} 条她的真实回复（零成本，没调模型）\n")

    names = [args.file] if args.file else list(BUILDERS)
    print(f"{'文件':<14}{'条目':>6}{'条目被用过':>12}{'回复命中率':>12}{'有反应':>8}")
    print("-" * 60)
    detail = {}
    for name in names:
        pr = BUILDERS[name]()
        used, hit_msgs = set(), 0
        for r in replies:
            hit = False
            for label, words, mode in pr:
                ok = (any(w in r for w in words) if mode == "sub"
                      else any(_overlap(w, r) >= args.min_sim for w in words))
                if ok:
                    used.add(label)
                    hit = True
            if hit:
                hit_msgs += 1
        detail[name] = (pr, used)
        print(f"{name:<14}{len(pr):>6}{len(used):>12}{hit_msgs / max(len(replies), 1) * 100:>11.0f}%"
              f"{'✅' if hit_msgs else '❌ 一次都没'}")
    print("\n⚠️ 「条目被用过」= 至少有一条回复出现过它的内容。")
    print("   这**不能证明**『是因为注入了才出现』（她自己本来也可能那么说）——")
    print("   它只说明『这个文件的内容**到得了**她的回复』。")
    print("   要确认**净作用**（拿掉它她会不会变），得做消融：`scripts/persona_ablation.py`。")

    if args.list_unused:
        print("\n" + "=" * 60)
        for name, (pr, used) in detail.items():
            miss = [lab for lab, _, _ in pr if lab not in used]
            if miss:
                print(f"\n【{name}】从没被用过的 {len(miss)} 条：")
                for lab in miss[:25]:
                    print(f"   · {lab}")
                if len(miss) > 25:
                    print(f"   …还有 {len(miss) - 25} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
