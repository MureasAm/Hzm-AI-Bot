#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""消融实验：**把【性格基底】+【语言风格】拿掉，她会变成什么样？**

## 为什么做

`docs/注入设计原理.md` §4.4 那张体检表里，**四个"每轮无条件注入"的静态块
（骨架 / traits / styles / 周表）一个都没做过消融**，而它们占了 base system 的绝大部分。
其中 **`traits` + `styles` = 610 字 ≈ base 的 17%** ——
**花了 17% 的预算，却连"有没有用"都不知道**。这就是本实验要填的洞。

⚠️ 注意区分"**在用**"和"**有用**"：前者是"代码读它了吗"（读了），
后者才是这个实验量的东西。

## 怎么做到"只变这一个变量"

一次检索，**生成两遍**：
  ① 现状：`messages[0]` = 骨架 + 【性格基底】+【语言风格】
  ② 消融：`messages[0]` = 骨架（把那两段**逐字删掉**），其余消息**完全相同**
所以两条回复的差别，只能来自那 610 字。（没走两遍链路——那样检索本身会抖，变量就不干净了）

## 怎么量"像不像她"（用她身上**可量化**的说话特征当代理）

| 指标 | 她的特征 | 判据 |
|---|---|---|
| **自称率** | 习惯用「灰泽满/hzm/小满」自称（第三人称自我保护） | ↑ 好（这是她最独特的一项） |
| 「我」率 | 少用「我」 | ↓ 好 |
| 长度 | 短句为主 | 别明显变长 |
| 括号率 | 「例外不是常态」，日常闲聊基本不用 | ↓ 好 |
| 叹号率 | 情绪克制 | ↓ 好 |

⚠️ **这些是代理指标，不是"像不像"的答案**——但它们是可复现、可对比的，
比"我读着觉得更像"可靠。三条硬话：**样本量小、单次有抖动、判据是我们定的**。

## 用法

    python scripts/persona_ablation.py              # 默认 16 条真实消息（每条约 1 次检索 + 2 次生成）
    python scripts/persona_ablation.py --n 32
    python scripts/persona_ablation.py --from-synthetic   # 不用真实聊天，用内置的几条（快，但会失真）

产物：`outputs/eval/persona_ablation.json` + 屏幕上的对照表。
⚠️ 用临时记忆目录，不碰真实 `user_memory/`。
"""
import argparse
import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402
from nonebot.adapters.onebot.v11 import Adapter as ONEBOT_V11Adapter  # noqa: E402

nonebot.init()
nonebot.get_driver().register_adapter(ONEBOT_V11Adapter)
nonebot.load_plugins("src/plugins")

import src.plugins.chatbot.core as core            # noqa: E402
import src.plugins.chatbot.memory as mem           # noqa: E402

OUT = ROOT / "outputs" / "eval" / "persona_ablation.json"
CHAT_LOG = ROOT / "data" / "chat_log" / "chat.jsonl"

# 自称：她最独特的特征（第三人称自我保护）
SELF = re.compile(r"(灰泽满|hzm|小满|满姐)")
FIRST_PERSON = re.compile(r"我")

SYNTHETIC = [
    "今天在干嘛呀", "你昨天直播怎么样", "好无聊啊，陪我聊会儿", "你吃饭了吗",
    "感觉你今天声音有点累", "你喜欢什么颜色", "明天播不播呀", "你最近在忙什么",
    "小满夸夸我", "你怎么这么可爱", "晚安", "你会想我们吗",
    "今天天气不错", "你在澳洲冷不冷", "你平时几点睡", "给我讲个你的事吧",
]

_captured = {}


_ORIG_REPLY = None      # 真正的 generate_reply（第一次跑前存下来）

PLACEHOLDER = "（占位）"


async def _capture(messages):
    """把真实链路的 messages 截下来（**不真生成**），后面两遍生成都用它。"""
    _captured["msgs"] = messages
    return PLACEHOLDER


# ⚠️ **指标必须对准被测对象自己声明的特征**（这一步我一开始就做错了，见文件头"踩过的坑"）：
#   `styles.json` 明写的是「反问与挑衅式推进」「重复与自我修正」「戏剧化叙事」，
#   所以要量的是**问号率 / 省略号率 / 自我修正标记**。
#   第一版量的是长度/括号/叹号 —— 那是**骨架里【说话节奏】管的**，量错了地方，
#   结果"差异恰好为 0"，差点得出"这 610 字没用"的假结论。
#   下面把两组分开：**该管的**（有差异才说明它起作用）vs **对照组**（该由骨架管，差异应≈0；
#   若对照组出现大差异，说明变量没控干净）。
def _overlap(a: str, b: str) -> float:
    """a 的字符 bigram 有多少出现在 b 里（0~1）。粗读用。"""
    import re as _re
    def bg(x):
        x = _re.sub(r"[\s，。！？、；：（）【】()\[\]…~—\-]+", "", x or "")
        return {x[i:i + 2] for i in range(len(x) - 1)} or ({x} if x else set())
    ba = bg(a)
    return 0.0 if not ba else len(ba & bg(b)) / len(ba)


def _features(reply: str) -> dict:
    r = reply or ""
    return {
        # —— traits/styles 自称会管的东西 ——
        "qmark": 1 if re.search(r"[?？]", r) else 0,                      # 反问式推进
        "ellipsis": 1 if ("……" in r or "..." in r) else 0,                # 碎碎念节奏
        "self_fix": 1 if re.search(r"(其实|不是，|或者说|应该是)", r) else 0,  # 自我推翻/修正
        # —— 通用 ——
        "len": len(r),
        "self": 1 if SELF.search(r) else 0,
        "wo": 1 if FIRST_PERSON.search(r) else 0,
        # —— 对照组：骨架【说话节奏】管的，差异应≈0 ——
        "paren": 1 if ("（" in r or "(" in r) else 0,
        "bang": r.count("！") + r.count("!"),
    }


def _load_messages(n: int, from_synthetic: bool, mix_synthetic: int = 0) -> list:
    """真实消息（等距取样）+ 可选混入 N 条**模拟消息**。

    为什么要混模拟：真实对话里有些场景**极少出现**（被夸、被质疑、被催…），
    只靠真实消息就永远测不到那些场景下各段的作用。模拟场景是合法的——
    只要**判据与场景的来源无关**（这里的判据是"哪句更像她"，与谁写的场景无关）。
    """
    if from_synthetic:
        return SYNTHETIC[:n]
    real = _load_real(n - mix_synthetic if mix_synthetic else n)
    if mix_synthetic > 0:
        # 模拟场景按"她会被怎么问"来写，覆盖真实对话里稀疏的那些
        extra = [s for s in SYNTHETIC if s not in real][:mix_synthetic]
        out = []
        for i in range(max(len(real), len(extra))):
            if i < len(real):
                out.append(real[i])
            if i < len(extra):
                out.append(extra[i])
        return out
    return real


def _load_real(n: int) -> list:
    if not CHAT_LOG.exists():
        return []
    rows = []
    for line in CHAT_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        u = (rec.get("user") or "").strip()
        # ⚠️ **只取私聊**（`kind == "private"`）。踩过的坑：群聊那栏存的是
        # `昵称：内容`（多个人拼起来的），我一开始只按"长度≤40 且无换行"筛，
        # 结果把 59 条群聊当**私聊消息**喂进了 handle_chat（没传 is_group=True）——
        # 模型看到一句以「真理X：」开头的话，却没有任何东西告诉它那是什么。
        # **内部对照仍然成立（两个条件吃同一批输入），但输入不真实 → 绝对水平不可信。**
        if (u and rec.get("kind") == "private" and len(u) <= 40 and "\n" not in u
                and str(rec.get("session", "")).isdigit()):
            rows.append(u)
    if not rows:
        return SYNTHETIC[:n]
    step = max(1, len(rows) // n)                    # 等距取样：确定性，不掷骰子
    picked, seen = [], set()
    for u in rows[::step]:
        if u not in seen:
            seen.add(u)
            picked.append(u)
        if len(picked) >= n:
            break
    return picked


def _section_of(msg: dict) -> str:
    """这条 system 消息是哪一段？（取开头的【…】当标题）

    ⚠️ **不硬编码段名清单**：真实 messages 里每段都以自己的【标题】开头，
    直接读出来最稳（标题是 `core.py` 里用变量/format 拼的，硬列清单一定会漏）。
    """
    c = (msg.get("content") or "").lstrip()
    if not c.startswith("【"):
        return "（无标题：骨架 / 性格基底 / 语言风格 那一段）"
    end = c.find("】")
    return c[: end + 1] if end > 0 else "（标题没闭合）"


SPECIAL = "（无标题：骨架 / 性格基底 / 语言风格 那一段）"

# ==================== 安慰剂对照 ====================
# ⚠️ **为什么必须有它**：删掉任何一段都会让上下文变短/变形，而模型对"上下文变了"本身就敏感。
# 所以"删掉它 → 输出变了"**不能**证明"那段的内容在起作用"。
# 正确对照：把那段换成**等长、真实、但与该段功能无关**的文本。
#   · 删掉 ≫ 安慰剂远  → 那段的内容确实在起作用 ✅
#   · 删掉 ≈ 安慰剂远  → 那个距离只是"上下文变了"，该段的影响是假的 ❌
_PLACEBO_POOL = None


_FILLER = "（本段内容暂时不可用）"


def _placebo_text(need: int) -> str:
    """**中性填充**：等长、但与她无关、不含任何信息。

    ⚠️ 第一版用的是 `statement_final` 里"关于她的真实陈述"——**那是错的**：
    那本身就跟她有关，等于给安慰剂组也塞了"她的内容"（平光镜也有度数），
    会让安慰剂组也变得"像她" → **低估各段真实作用**。改成纯占位文本。
    """
    if need <= 0:
        return ""
    out = ""
    while len(out) < need:
        out += _FILLER
    return out[:need]


def _placebo(msgs: list, section: str) -> list:
    """把该段的**内容**换成等长的无关文本（标题保留，长度保留）——只换内容。"""
    out = []
    skip_convo = False
    last = len(msgs) - 1
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role == "system" and _section_of(m) == section:
            c = m.get("content") or ""
            m2 = dict(m)
            m2["content"] = section + "\n" + _placebo_text(max(len(c) - len(section), 20))
            out.append(m2)
            if section.startswith("【灰泽满的说话方式参考"):
                skip_convo = True
            continue
        if role in ("user", "assistant") and skip_convo and i != last:
            continue                            # 风格样本那几对：安慰剂版本里也删掉
        out.append(m)
    return out


def _ablate(msgs: list, section: str) -> list:
    """把某一段从 messages 里删掉，其余**一字不动**。

    - 普通段（`【…】` 开头）：删掉所有属于该段的 system 消息
    - 特殊段 = 骨架那条：**只删性格基底/语言风格**（骨架本身不能整块删——
      会连带拿掉身份框架，得做"压缩版对照"，那是另一件事）
    - `【灰泽满的说话方式参考】`：连它**后面那些 user/assistant 对**一起删
      （那几对是她原话 few-shot），但**不能删最后那条真正的 user**
    """
    out = []
    skip_convo = False
    last = len(msgs) - 1
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role == "system":
            if _section_of(m) == section:
                if section == SPECIAL:
                    base = m.get("content") or ""
                    skel = core.SYSTEM_PROMPT
                    m2 = dict(m)
                    m2["content"] = skel if base.startswith(skel) else base
                    out.append(m2)          # 骨架留着，只掉性格基底/语言风格
                else:
                    skip_convo = section.startswith("【灰泽满的说话方式参考")
                continue
        if role in ("user", "assistant") and skip_convo and i != last:
            continue                          # 风格样本的那几对：一起删
        out.append(m)
    return out


_SKEL_SEC = re.compile(r"^## 【", re.M)


def _base_units(base: str):
    """把 `messages[0]` 拆成**可单独删的单元**：骨架的每个 `## 【小节】` + 性格基底/语言风格那一坨。

    ⚠️ 骨架**不能整块删**（会连带拿掉身份框架，她就不成形了）——**按小节删**才有意义，
    而且结论直接可行动：**哪几节能省**。
    """
    skel = core.SYSTEM_PROMPT
    tail = base[len(skel):] if base.startswith(skel) else ""
    idx = [m.start() for m in _SKEL_SEC.finditer(skel)]
    units = []
    if idx:
        units.append(("骨架·前言（【核心人格】之前那段）", skel[: idx[0]]))
    for i, st in enumerate(idx):
        en = idx[i + 1] if i + 1 < len(idx) else len(skel)
        title = skel[st: st + 16].split("\n")[0].strip()
        units.append((title, skel[st:en]))
    if tail.strip():
        units.append(("【性格基底】+【语言风格】", tail))
    return units


async def run_skeleton(n: int, from_synthetic: bool, min_chars: int, per_unit: int) -> int:
    """**骨架按小节消融**：一次检索 → 逐个删掉骨架的每一小节 → 看她的风格偏了多少。"""
    core.update_memory_task = lambda *a, **k: asyncio.sleep(0)
    tmp = tempfile.mkdtemp(prefix="skelabl_")
    mem.MEMORY_FILE = Path(tmp) / "short_term.json"
    global _ORIG_REPLY
    _ORIG_REPLY = core.generate_reply
    core.generate_reply = _capture

    msgs_list = _load_messages(n, from_synthetic)
    print(f"🦴 骨架按小节消融：{len(msgs_list)} 条消息\n")
    from collections import Counter
    hits = Counter()
    rows = []
    for i, msg in enumerate(msgs_list, 1):
        _captured.clear()
        try:
            await core.handle_chat(f"skel_{i}", msg)
        except Exception as e:
            print(f"  [{i}] 检索失败：{type(e).__name__}")
            continue
        msgs = _captured.get("msgs")
        if not msgs:
            continue
        base0 = msgs[0].get("content") or ""
        units = [(t, x) for t, x in _base_units(base0) if len(x) >= min_chars]
        if not units:
            print(f"  [{i}] 拆不出小节（base 结构变了？）"); continue
        base_reply = await _ORIG_REPLY([dict(m) for m in msgs])
        rows.append({"msg": msg, "reply": base_reply, "units": [t for t, _ in units], "ablation": {}})
        for title, text in units:
            if hits[title] >= per_unit:
                continue
            ab = [dict(m) for m in msgs]
            ab[0]["content"] = base0.replace(text, "", 1)
            if ab[0]["content"] == base0:
                continue
            rows[-1]["ablation"][title] = await _ORIG_REPLY(ab)
            hits[title] += 1
        print(f"  [{i}/{len(msgs_list)}] {msg[:18]}（{len(units)} 小节）")

    try:
        from persona_fingerprint import fingerprint, distance, _baseline_texts
        ref = fingerprint(_baseline_texts())
        d_base = distance(ref, fingerprint([r["reply"] for r in rows]))[0]
        print(f"\n【指纹读数】基准 = 她本人 {len(_baseline_texts())} 条真实回复"
              f"　baseline 距离 = {d_base:.3f}\n")
        print(f"{'骨架小节':<34}{'删掉':>8}{'比baseline':>10}   结论")
        print("-" * 74)
        for title, cnt in hits.most_common():
            texts = [r["ablation"][title] for r in rows if title in r["ablation"]]
            if len(texts) < 3:
                continue
            d = distance(ref, fingerprint(texts))[0]
            tag = "⭐ 影响大（这节有用）" if d - d_base > 0.03 else \
                  "有影响" if d - d_base > 0.012 else "≈ 无影响（可考虑省）"
            print(f"{title[:32]:<34}{d:>8.3f}{d - d_base:>+10.3f}   {tag}")
        print("\n  ⚠️ 同上：这只量『影响』不量『价值』；且骨架**还没有**安慰剂对照——")
        print("     删一小节 = 上下文变短，本身也会让输出微变。")
        print("     要下『可省』的结论，得再补安慰剂对照。")
    except Exception as e:
        print(f"（指纹读数跳过：{type(e).__name__}: {e}）")
    out = OUT.parent / "skeleton_ablation.json"
    out.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 明细已保存: {out}")
    return 0


async def run_compress(n: int, short_path: str, from_synthetic: bool) -> int:
    """**整块压缩对照**：现状（骨架+traits/styles，3758 字） vs 一个更短的等价版本。

    为什么做这个（2026-09-30）：
    - 逐文件消融测不出东西（23 道题 20 道答"差不多"）——**单文件的边际作用太小**。
    - 那就换个尺子能看见的问法：**整块值不值这份预算**。
    - ⚠️ 而且这一问的"分不出"**是可用的结论**：压缩版更便宜，分不出就换短的。
      （消融里"分不出"是废读数，因为删掉有风险；这里没有风险，只省钱。）

    ⚠️ 压缩的写法有讲究：**规则一条不删，只删解释/例子/重复表述**。
    否则比的就成了"两种写法"，不是"同一套规则的两种长度"。
    """
    short = Path(short_path)
    if not short.exists():
        print(f"❌ 找不到压缩版：{short}")
        return 1
    short_text = short.read_text(encoding="utf-8").strip()

    core.update_memory_task = lambda *a, **k: asyncio.sleep(0)
    tmp = tempfile.mkdtemp(prefix="compress_")
    mem.MEMORY_FILE = Path(tmp) / "short_term.json"
    global _ORIG_REPLY
    _ORIG_REPLY = core.generate_reply
    core.generate_reply = _capture

    msgs_list = _load_messages(n, from_synthetic)
    print(f"📦 整块压缩对照：{len(msgs_list)} 条消息 × 2 条件\n")
    rows, skipped, full_chars = [], 0, 0
    for i, msg in enumerate(msgs_list, 1):
        _captured.clear()
        try:
            await core.handle_chat(f"cmp_{i}", msg)
        except Exception as e:
            print(f"  [{i}] 检索失败：{type(e).__name__}")
            skipped += 1
            continue
        msgs = _captured.get("msgs")
        if not msgs:
            skipped += 1
            continue
        full_chars = len(msgs[0].get("content") or "")
        with_p = [dict(m) for m in msgs]
        short_p = [dict(m) for m in msgs]
        short_p[0]["content"] = short_text          # 只换这一整块，其余一字不动
        r_full = await _ORIG_REPLY(with_p)
        r_short = await _ORIG_REPLY(short_p)
        rows.append({"msg": msg, "with": r_full, "without": r_short,
                     "with_chars": full_chars, "without_chars": len(short_text)})
        print(f"  [{i}/{len(msgs_list)}] {msg[:22]}")
        print(f"       现状（{full_chars}字）：{r_full}")
        print(f"       压缩（{len(short_text)}字）：{r_short}")

    if not rows:
        print("\n❌ 一条都没跑成")
        return 1
    out = OUT.parent / "compress_ablation.json"
    out.write_text(json.dumps({
        "meta": {"full_chars": full_chars, "short_chars": len(short_text),
                 "short_file": str(short), "n": len(rows), "skipped": skipped},
        "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 明细已保存: {out}")
    return 0


async def run_sections(n: int, from_synthetic: bool, min_section_chars: int,
                       per_section: int = 20, only_section: str = None) -> int:
    """**按段消融**：一次检索 → 对每个注入段试"删掉它"，其余全部相同。

    ⚠️ 关键性质：这些段**已经在 messages 里**（检索已经跑完），
       所以消融 = **把那段删掉**，**不需要重跑检索** —— 一次检索能试所有段。

    `only_section`：只做名字里含这个子串的段（给"某个段还没测过、单独补一批"用）。不传 = 所有段。
    """
    core.update_memory_task = lambda *a, **k: asyncio.sleep(0)
    tmp = tempfile.mkdtemp(prefix="sectabl_")
    mem.MEMORY_FILE = Path(tmp) / "short_term.json"
    global _ORIG_REPLY
    _ORIG_REPLY = core.generate_reply
    core.generate_reply = _capture

    msgs_list = _load_messages(n, from_synthetic)
    print(f"🔬 按段消融：{len(msgs_list)} 条消息（每条 1 次检索 + N 次生成）\n")

    from collections import Counter, defaultdict
    tot = Counter()
    skipped_by_cap = Counter()
    hits = defaultdict(lambda: [0, 0, 0.0, 0.0])   # 段 -> [该段出现次数, 有它命中, 重合有, 重合无]
    rows = []

    for i, msg in enumerate(msgs_list, 1):
        _captured.clear()
        try:
            await core.handle_chat(f"sect_{i}", msg)
        except Exception as e:
            print(f"  [{i}] 检索失败：{type(e).__name__}: {e}")
            continue
        msgs = _captured.get("msgs")
        if not msgs:
            continue
        sections = []
        for m in msgs:
            if m.get("role") == "system":
                s = _section_of(m)
                if only_section and only_section not in s:
                    continue
                if s not in sections:
                    sections.append(s)
        base_reply = await _ORIG_REPLY([dict(m) for m in msgs])
        rows.append({"msg": msg, "reply": base_reply, "sections": sections})

        for s in sections:
            body = " ".join((m.get("content") or "") for m in msgs
                            if m.get("role") == "system" and _section_of(m) == s)
            if len(body) < min_section_chars:
                continue
            if hits[s][0] >= per_section:      # 预算帽：这一段已经测够了，别再为它花生成
                skipped_by_cap[s] += 1
                continue
            ab = _ablate(msgs, s)
            if len(ab) == len(msgs):
                continue
            r = await _ORIG_REPLY(ab)
            tot[s] += 1
            ov_with = _overlap(body, base_reply)
            ov_without = _overlap(body, r)
            hits[s][0] += 1
            hits[s][1] += 1 if ov_with >= 0.08 else 0     # 粗略：回复"用上了"这段的字面
            hits[s][2] += ov_with
            hits[s][3] += ov_without
            rows[-1].setdefault("ablation", {})[s] = r
            rp = await _ORIG_REPLY(_placebo(msgs, s))          # 安慰剂对照（等长无关文本）
            rows[-1].setdefault("placebo", {})[s] = rp
        print(f"  [{i}/{len(msgs_list)}] {msg[:20]}（{len(sections)} 段）")

    print(f"\n{'=' * 78}")
    # ==================== 指纹读数（尺子已在 persona_fingerprint.py --validate 验证过）====================
    # 拿"她本人的真实回复"当基准，看**每段消融后她的风格偏离了多少**。
    # ⚠️ 字面重合那两列测不了"设计上就要求别抄"的段；指纹距离才是通用读数。
    try:
        from persona_fingerprint import fingerprint, distance, _baseline_texts
        ref = fingerprint(_baseline_texts())
        base_fp = fingerprint([r["reply"] for r in rows])
        d_base = distance(ref, base_fp)[0]
        print(f"【指纹读数】基准 = 她本人 {len(_baseline_texts())} 条真实回复")
        print(f"  baseline（没消融）总距离 = {d_base:.3f}\n")
        print(f"{'段':<24}{'删掉':>8}{'比baseline':>9}{'安慰剂':>10}{'删-安':>9}   判定")
        print("-" * 78)
        sect_fp = {}
        for s in sorted(hits, key=lambda k: -hits[k][0]):
            texts = [r["ablation"][s] for r in rows if (r.get("ablation") or {}).get(s)]
            if len(texts) < 3:
                continue
            f = fingerprint(texts)
            d, parts = distance(ref, f)
            pt = [r["placebo"][s] for r in rows if (r.get("placebo") or {}).get(s)]
            dp = distance(ref, fingerprint(pt))[0] if len(pt) >= 3 else float("nan")
            sect_fp[s] = (d, len(texts))
            gap = d - dp
            verdict = ("✅ 内容在起作用" if gap > 0.015
                       else "❌ 只是上下文变了" if gap == gap else "（无安慰剂）")
            print(f"{s[:22]:<24}{d:>8.3f}{d - d_base:>+9.3f}{dp:>10.3f}{gap:>+9.3f}   {verdict}")
        print("\n  ⚠️ 读法（**关键是最后两列**）：")
        print("     「删-安」= 删掉它 与 换成等长无关文本，两者的距离差。")
        print("     >0.015 → 那段**的内容**在起作用；≈0 → 那个距离只是『上下文变了』造成的，")
        print("              说明该段的影响是假的（模型对上下文长度本身敏感）。")
        print("     接近 0 → 那段没在影响**风格**（可能它影响的是内容/事实——那要用任务正确性测）。")
    except Exception as e:
        print(f"（指纹读数跳过：{type(e).__name__}: {e}）")

    print(f"\n{'=' * 78}")
    print("【按段消融】拿掉这一段，她的回复变了吗？（字面重合，**粗读**）\n")
    print(f"{'段':<26}{'出现条数':>8}{'回复用过它':>10}{'重合(有)':>10}{'重合(无)':>10}{'差':>8}")
    print("-" * 78)
    for s, (cnt, hit, ow, oo) in sorted(hits.items(), key=lambda kv: -(kv[1][2] - kv[1][3])):
        d = (ow - oo) / max(cnt, 1)
        flag = "✅ 有净作用" if d > 0.02 else ("⚠️ 几乎无差异" if abs(d) <= 0.02 else "❌ 反常")
        print(f"{s[:24]:<26}{cnt:>8}{hit:>10}{ow / max(cnt,1):>10.3f}{oo / max(cnt,1):>10.3f}{d:>+8.3f}  {flag}")

    print("\n⚠️ **怎么读**：")
    print("   · 「重合(有) − 重合(无)」为正 = 那段的内容**真的进到了她的回复**（拿掉就少了）")
    print("   · 差异≈0 = 那段**没在影响回复的字面**（可能它影响的是别的东西，也可能它在白花）")
    print("   · ⚠️ 字面重合是**粗读**：像 `behaviors` 那种『照着学腔调、不照抄』的段，")
    print("     字面差异天然接近 0 —— **不能因此判它没用**（那正是它的设计目标）。")
    print("     这类要看输出里的 \"形式指标\"，或者换更细的判据。")
    out = OUT.parent / "section_ablation.json"
    out.write_text(json.dumps({"n": len(rows), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 明细（含每段消融后的完整回复）已保存: {out}")
    return 0


async def main():
    ap = argparse.ArgumentParser(description="消融：拿掉 traits+styles 会怎样")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--from-synthetic", action="store_true",
                    help="不用真实聊天，用内置的几句（快，但结论会失真）")
    ap.add_argument("--skeleton", action="store_true",
                    help="★骨架按小节消融（逐个删 ## 【小节】，看哪一节的缺失让她最不像她）")
    ap.add_argument("--mix-synthetic", type=int, default=0,
                    help="混入 N 条模拟消息（覆盖真实对话里稀疏的场景，如被夸/被质疑）")
    ap.add_argument("--sections", action="store_true",
                    help="★按注入段消融（一次检索试所有段：性格基底/行为/措辞/世界/记忆…）")
    ap.add_argument("--per-section", type=int, default=20,
                    help="★每一段最多消融几次（预算帽；跑很多消息时控成本）")
    ap.add_argument("--min-section-chars", type=int, default=60,
                    help="只看长于这个字数的段（太短的段测不出字面重合）")
    ap.add_argument("--only-section", default=None,
                    help="只消融名字里含这个子串的段（给某个段单独补一批用；不传=所有段）")
    ap.add_argument("--compress", default=None,
                    help="★整块压缩对照：把 messages[0] 整块换成这个短版文件（规则不删、只删水分）")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if args.compress:
        return await run_compress(args.n, args.compress, args.from_synthetic)

    if args.skeleton:
        return await run_skeleton(args.n, args.from_synthetic, args.min_section_chars,
                                  args.per_section)

    if args.sections:
        return await run_sections(args.n, args.from_synthetic, args.min_section_chars,
                                  per_section=args.per_section, only_section=args.only_section)

    # 临时记忆：不污染线上（长期提取关掉，只测回复本身）
    core.update_memory_task = lambda *a, **k: asyncio.sleep(0)
    tmp = tempfile.mkdtemp(prefix="ablation_")
    mem.MEMORY_FILE = Path(tmp) / "short_term.json"
    # ⚠️ 先把**真正的** generate_reply 存起来：下面要用它生成两遍。
    # 踩过的坑：光把 core.generate_reply 换成截获器就往下跑 → 两个条件调的都是截获器，
    # 拿回一堆"（占位）"，指标差异全是 0，**看起来像"traits/styles 完全没用"——那是假的**。
    global _ORIG_REPLY
    _ORIG_REPLY = core.generate_reply
    core.generate_reply = _capture

    msgs_list = _load_messages(args.n, args.from_synthetic, args.mix_synthetic)
    print(f"🔬 消融实验：{len(msgs_list)} 条消息 × 2 个条件"
          f"（每条 1 次真实检索 + 2 次生成）\n")

    rows, skipped = [], 0
    for i, msg in enumerate(msgs_list, 1):
        _captured.clear()
        try:
            await core.handle_chat(f"abl_{i}", msg)
        except Exception as e:
            print(f"  [{i}] 检索失败，跳过：{type(e).__name__}: {e}")
            skipped += 1
            continue
        msgs = _captured.get("msgs")
        if not msgs:
            skipped += 1
            continue
        base = msgs[0].get("content") or ""
        skeleton = core.SYSTEM_PROMPT
        if not base.startswith(skeleton):
            print(f"  [{i}] 认不出 base 结构，跳过")
            skipped += 1
            continue
        persona = base[len(skeleton):].strip()
        if not persona:
            # 本来就空（比如 traits/styles 文件被删）—— 那不是消融，是坏数据
            print(f"  [{i}] base 里没有【性格基底】/【语言风格】，跳过（这条没法消融）")
            skipped += 1
            continue

        with_p = [dict(m) for m in msgs]
        without = [dict(m) for m in msgs]
        without[0]["content"] = skeleton

        r_with = await _ORIG_REPLY(with_p)          # ← 用**真的**生成函数，不是截获器
        r_without = await _ORIG_REPLY(without)
        rows.append({"msg": msg, "with": r_with, "without": r_without,
                     "fw": _features(r_with), "fo": _features(r_without)})
        print(f"  [{i}/{len(msgs_list)}] {msg[:22]}")
        print(f"       有 traits/styles：{r_with}")
        print(f"       没有　　　　　　：{r_without}")

    if not rows:
        print("\n❌ 一条都没跑成（检查 .env.prod 的 key / 是否装了依赖）")
        return 1

    # ⚠️ 自检：如果拿回来的全是占位/全一个样，说明"真生成"那一步没跑起来。
    # 没有这个自检的话，指标会全 0、看起来像"这个东西完全没用"——**那正是我踩过的假结论**。
    replies = [r["with"] for r in rows] + [r["without"] for r in rows]
    if any(PLACEHOLDER in (x or "") for x in replies) or len(set(replies)) == 1:
        print(f"\n❌ 自检没过：拿回来的回复是占位符/全相同（{len(set(replies))} 种）。\n"
              f"   说明『真生成』那一步没跑（别再往下读指标——那些数字没有意义）。")
        return 1

    def avg(k, which):
        vals = [r[which][k] for r in rows]
        return sum(vals) / len(vals)

    print(f"\n{'=' * 66}\n【结果】{len(rows)} 条有配对（跳过 {skipped} 条）\n")
    print(f"{'指标':<12}{'有 traits/styles':>18}{'没有':>10}{'差异':>10}   怎么看")

    def row(label, k, pct, hint):
        a, b = avg(k, "fw"), avg(k, "fo")
        if pct:
            print(f"{label:<12}{a*100:>17.0f}%{b*100:>9.0f}%{(b-a)*100:>+9.0f}pp   {hint}")
        else:
            print(f"{label:<12}{a:>18.1f}{b:>10.1f}{b-a:>+10.1f}   {hint}")

    print("—— 这几项是 `styles.json` **自己写着**的特征，差异大 = 它确实在起作用 ——")
    row("反问率", "qmark", True, "styles：反问与挑衅式推进")
    row("省略号率", "ellipsis", True, "styles：碎碎念节奏")
    row("自我修正率", "self_fix", True, "styles：重复与自我修正")
    print("—— 通用 ——")
    row("自称率", "self", True, "↑ 好（她最独特的特征）")
    row("「我」率", "wo", True, "↓ 好")
    row("长度", "len", False, "短句为主，别明显变长")
    print("—— 对照组：这几项该由**骨架**的【说话节奏】管，差异应≈0（大了说明变量没控干净）——")
    row("括号率", "paren", True, "↓ 好（例外不是常态）")
    row("叹号/条", "bang", False, "↓ 好（情绪克制）")

    print("\n⚠️ 结论怎么读：**差异大 = 这 610 字在起作用；差异接近 0 = 它在被白花**。")
    print("   ⚠️ 但先问一句：**你量的指标，是它自己声明会管的东西吗？**")
    print("   但要记住三条限制：样本量小、单次有抖动、这些指标是**代理**不是'像不像'本身。")
    print("   要更硬的结论：加大 --n，或换一批消息看方向是否稳定。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n": len(rows), "skipped": skipped,
        "persona_chars": len(persona) if rows else 0,
        "avg_with": {k: avg(k, "fw") for k in ("len", "self", "wo", "paren", "bang")},
        "avg_without": {k: avg(k, "fo") for k in ("len", "self", "wo", "paren", "bang")},
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 明细已保存: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
