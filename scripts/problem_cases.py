#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题驱动：**看到的问题 → 回归用例**（把"标注"从"凭空判断"改成"记一笔"）。

## 为什么需要

原来要维护 `scripts/judge_labels.json` 那 41 条手工标注——**要凭空判断"这句话该不该命中"**，
难、烦、而且和 `retrieval_eval_cases.json` 是同源的（同一件事存两份，已经打架过。
详见 `scripts/retrieval_eval_cases.json` 的 `_change_log`）。

改成：**你不预先标，只在看到问题时记一笔**，用例由这条真实问题生成。

## 两条输入通道

① **抽查勾选**：`outputs/spot_check.md` 里把 `[ ] 有` 改成 `[x] 有：为什么不对`（带 `--from-spot`）
② **随手记**：根目录 `问题记录.md` 里写「用户说 / 她答 / 应该是」（带 `--from-note`）

## 核心机制：先把事实摆出来，再让你判

本脚本对每条问题**用线上真实的检索函数跑一遍**（`retrieval_eval.snapshot`，
与 `core.handle_chat` 同一条路），把"**这句话实际召回了什么**"打出来。
你在**眼前的事实**面前说"该中/不该中"——
这就是那句"判断难度从『凭空判断该不该命中』降到『这条是不是明显答错了』"。

## 落成哪一层（两种都管）

| 问题形态 | 落到哪 | 判据 |
|---|---|---|
| 该想起 X 却没想起 / 想起了不该想的 | `scripts/retrieval_eval_cases.json` | `should` / `contains` / `should_fire` / `should_not_hit` |
| 她话说错了（如拿旧事当现在） | `scripts/regression_cases.json` | `forbid` / `forbid_pattern` / 同质率 |

## ⭐ 先红后绿（别加白用例）

新用例**必须先 FAIL**——证明它复现了你看到的问题——**再改素材让它转绿**。
一加就绿的用例是白加，给的是假信心（本仓踩过：合成场景自测 8/8 却毫无意义）。
所以本脚本写完会提醒你**先跑一遍确认它是红的**。

## 用法

    python scripts/run_tool.py problems --query "你多高啊"   # 只看一条的检索实况（最常用）
    python scripts/run_tool.py problems --from-spot          # 读抽查勾选，打印实况+草案
    python scripts/run_tool.py problems --from-note          # 读 问题记录.md，同上
    python scripts/run_tool.py problems --from-note --write  # 确认后正式写进用例文件

## 问题记录.md 的字段（都在行首写「字段：值」，缺的脚本会问你）

    用户说：为什么今天不直播 我想你想的浑身难受     ← 必填（这条消息本身）
    她答：搬家的事还没弄完，今天真播不了            ← 生成层判据的来源
    应该是：不该拿两个月前的旧事当成现在的理由      ← 人话写就行
    层：生成                                        ← 检索 / 生成（不写就替你别猜并说明）
    禁词：搬家,搬东西                               ← 生成层：这些词一个都不能出现
    该中：preference:routine                         ← 检索层：该命中什么
    不该中：corpus,phrase                            ← 检索层：这些路必须为空
    跑次：3                                          ← 生成层跑几次（默认 3，越大越慢越贵）

**`该中` 的写法**：`<路>:<期望>`。期望写 `*` = "该有东西但不钉是哪条"（`should_fire`）；
写词 = 关键词包含（`contains`）；对 `preference`/`behavior`/`phrase`/`voice_sample`/`core_story`
这些 **id 稳定**的路，写词就是钉 id（`should`）。

⚠️ **`corpus` 只能写关键词，别钉 `#编号`**：corpus 的 item_id 是向量库下标，
**改一次语料就全体重排**（`docs/注入设计原理.md` §4.2 记过这个坑）——钉了就天天红。
"""
import argparse
import asyncio
import json
import re
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import _common

ROOT = _common.PROJECT_ROOT
sys.path.insert(0, str(ROOT))

RETRIEVAL_CASES = ROOT / "scripts" / "retrieval_eval_cases.json"
REGRESSION_CASES = ROOT / "scripts" / "regression_cases.json"
CHAT_LOG = ROOT / "data" / "chat_log" / "chat.jsonl"
SPOT_MD = ROOT / "outputs" / "spot_check.md"
NOTES_MD = ROOT / "问题记录.md"

# 这几路的 id 是**稳定名字**（preference/food、phrase/brag_deny…），可以钉；
# corpus 的 id 是向量库下标，改语料就重排，只能写关键词。
STABLE_ID_ROUTES = ("preference", "core_story", "behavior", "phrase")   # voice_sample 已删（2026-10-04）

FIELD_PREFIXES = {
    "用户说": "user", "用户": "user", "她说": "user",
    "她答": "reply", "回复": "reply", "答": "reply",
    "应该是": "should", "应该": "should",
    "层": "layer",
    "禁词": "forbid", "禁模式": "pattern", "跑次": "runs",
    "该中": "wants", "不该中": "nots",
    "备注": "note", "为什么": "note",
}


@dataclass
class Problem:
    """一条"看到的问题"。只装**人记下来的东西**，不装判断。"""
    user: str = ""            # 用户那句（必填）
    reply: str = ""           # 她答的那句
    should: str = ""          # 人话：应该是怎样
    layer: str = ""           # 检索 / 生成（空 = 未指定）
    forbid: list = field(default_factory=list)
    pattern: str = ""
    runs: int = 3
    wants: list = field(default_factory=list)   # [("preference:routine", ...)]
    nots: list = field(default_factory=list)    # ["corpus", "phrase"]
    note: str = ""
    # 抽查通道才有
    session: str = ""
    t: float = 0.0
    spot_why: str = ""


# ==================== 解析（纯函数，可测） ====================

def parse_notes(text: str) -> list:
    """解析 `问题记录.md`。**新问题以「用户说」那行为界**，一条一块。

    容错：认不出的行**追加到上一个字段**（人随手换行不该让解析崩）；
    块里没有「用户说」就整块忽略（那是标题、说明、或分隔符）。

    ⚠️ **``` 围栏里的内容一律不解析** —— 模板里的"示例"都是写在围栏里的，
    不加这条就会把示例当成真的问题（自己踩过）。
    """
    blocks, cur, in_fence = [], None, False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line or line.startswith("#") or line.startswith("---"):
            continue
        prefix, _, value = line.partition("：")
        prefix = prefix.strip()
        key = FIELD_PREFIXES.get(prefix)
        if key == "user":                      # 新问题开始
            cur = Problem()
            blocks.append(cur)
            cur.user = value.strip()
        elif cur is None:
            continue                            # 「用户说」之前的东西不是问题
        elif key:
            value = value.strip()
            if key == "forbid":
                cur.forbid = [w.strip() for w in re.split(r"[,，、\s]+", value) if w.strip()]
            elif key == "wants":
                cur.wants.append(value)
            elif key == "nots":
                cur.nots += [w.strip() for w in re.split(r"[,，、\s]+", value) if w.strip()]
            elif key == "runs":
                cur.runs = int(re.sub(r"\D", "", value) or 3)
            else:
                setattr(cur, key, value)
        elif cur is not None:
            # 续行：接到上一个非空字段（人随手换行）
            for f in ("should", "reply", "note"):
                if getattr(cur, f):
                    setattr(cur, f, getattr(cur, f) + " " + line)
                    break
    return blocks


def parse_spot_marks(text: str) -> list:
    """解析 `outputs/spot_check.md` 里**勾了「有」**的条目。

    只认 `[x] 有：...`（`[ ] 没有` 里也有 checkbox，所以必须钉 `有：` 前面那个）。
    条目的 user/reply 可能是截断的 —— 真正的全文靠 `(session, t)` 回 `chat.jsonl` 取
    （`t` 是毫秒级时间戳，实测全库零重复；**别用行号**，轮转会重排）。
    """
    text = text or ""
    out = []
    # 每条以 `### N. ` 开头；在每条内部找勾选与元数据
    parts = re.split(r"^###\s+\d+\.\s*", text, flags=re.M)
    for part in parts[1:]:
        m = re.search(r"\[[xX✓]\]\s*有：\s*(.*)", part)
        if not m:
            continue
        why = m.group(1).strip().strip("_ ").strip()
        meta = re.search(r"`session=(\S+)\s+kind=(\S+)\s+t=([\d.]+)`", part)
        p = Problem(spot_why=why, reply="")
        if meta:
            p.session, p.t = meta.group(1), float(meta.group(3))
            turn = lookup_turn(p.session, p.t)
            if turn:
                p.user = (turn.get("user") or "").strip()
                p.reply = (turn.get("reply") or "").strip()
        if not p.user:                          # 回查失败（日志轮转过）→ 退回 markdown 里那段
            head = re.match(r"「(.*?)」\s*\n(.*)", part, re.S)
            if head:
                p.user = head.group(1).strip()
                p.reply = head.group(2).strip().split("\n`")[0].strip()
        out.append(p)
    return out


def lookup_turn(session: str, t: float):
    """按 `(session, t)` 从聊天落盘里取那一轮原文。取不到返回 None。"""
    if not CHAT_LOG.exists() or not t:
        return None
    for line in CHAT_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if str(rec.get("session")) == str(session) and abs(float(rec.get("t") or 0) - t) < 1e-6:
            return rec
    return None


# ==================== 草案生成（纯函数，可测） ====================

def guess_layer(p: Problem) -> str:
    """猜这条是检索层还是生成层 —— **只是猜，会打印出来让你改**。

    生成层的特征：她说出来的那句话本身有问题（「她答」里有关键内容，
    或人明说了"不该说/别拿…当"）；否则按检索层（该不该想起某段素材）算。
    """
    if p.layer in ("检索", "生成"):
        return p.layer
    if p.forbid or p.pattern:
        return "生成"
    if re.search(r"(不该说|不能这么|别拿|别把|别再|编出|说错)", p.should):
        return "生成"
    return "检索"


def _expect_of(want: str) -> tuple:
    """`preference:routine` → ("preference", {"should": ["routine"]})。

    `*` → should_fire；corpus 或纯词 → contains；其余稳定 id 的路 → should。
    """
    route, _, spec = want.partition(":")
    route, spec = route.strip(), spec.strip()
    if not spec or spec == "*":
        return route, {"should_fire": True}
    if route == "corpus":                       # 下标会重排，只能钉关键词
        return route, {"contains": [spec]}
    if route in STABLE_ID_ROUTES:
        return route, {"should": [spec]}
    return route, {"contains": [spec]}


def draft_retrieval(p: Problem) -> dict:
    """把一条问题变成检索层用例草案。

    ⚠️ 同一路不能既"该中"又"不该中"（用例会自相矛盾、评测永久红）。
    真有这种需求（"该中室友那条，但别中女同学那条"）时**现有期望档表达不了**——
    所以这里**只保留更具体的那个（该中），并打印警告**，不静默拼出一个两头堵的用例。
    """
    expect = {}
    for w in p.wants:
        route, cond = _expect_of(w)
        expect.setdefault(route, {}).update(cond)
    for route in p.nots:
        route = route.strip()
        if route in expect:
            print(f"⚠️ 「{route}」既写了该中又写了不该中——**保留「该中」，丢掉「不该中」**。"
                  f"要表达'该中某条但不该中同类别的另一条'，现在的期望档做不到，"
                  f"请在 why 里写清楚，并考虑用 contains 钉关键词。")
            continue
        expect.setdefault(route, {}).update({"should_not_hit": True})
    if not expect:
        # 人没写「该中/不该中」——**不替人拍**，留成显式 TODO
        expect = {"corpus": {"should_fire": True}}   # 占位，写完必须人工改
    why = p.should or p.spot_why or "（待补：为什么这条算问题）"
    src = "抽查" if p.spot_why else "问题记录"
    return {
        "id": _slug(p.user),
        "query": p.user,
        "source": src,
        "why": why + (f"（提交人备注：{p.spot_why}）" if p.spot_why and p.should else ""),
        "expect": expect,
    }


def draft_generation(p: Problem) -> dict:
    """把一条问题变成生成层用例草案（`regression_cases.json` 的形状）。"""
    case = {
        "id": _slug(p.user),
        "desc": p.should or p.spot_why or "（待补：这条算什么错）",
        "user": p.user,
        "runs": p.runs,
        "source": (f"问题记录。她答：{p.reply}。" if p.reply else "问题记录。")
                  + (f"人判定：{p.should}" if p.should else ""),
    }
    if p.forbid:
        case["forbid"] = p.forbid
    if p.pattern:
        case["forbid_pattern"] = p.pattern
    if not p.forbid and not p.pattern:
        # 没有判据的用例跑不出结论 —— 明说，别假装成功
        case["_TODO"] = "还缺判据：补 forbid（不许出现的词）或 forbid_pattern（正则）"
    return case


def _slug(text: str) -> str:
    """从用户那句话造一个短 id。

    ⚠️ **不能用内置 `hash()`** —— 字符串 hash 每个进程都带随机盐，
    同一个问题两次跑会得到不同 id（id 就不可复现了）。用 crc32，确定性。
    """
    ascii_part = re.sub(r"[^0-9a-zA-Z]+", "_", text or "").strip("_").lower()
    stamp = zlib.crc32((text or "").encode("utf-8")) % 9973
    return (ascii_part[:24] or "case") + "_" + str(stamp)


# ==================== 写文件 ====================

def _write_cases_json(path: Path, doc: dict, compact_cases: bool) -> None:
    """写回标注集。`compact_cases=True` 时每条 case 压成一行（保人读性、diff 友好）。"""
    if not compact_cases:
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    lines = ["{"]
    for k, v in doc.items():
        if k == "cases":
            continue
        chunk = json.dumps(k, ensure_ascii=False) + ": " + json.dumps(v, ensure_ascii=False)
        lines.append("  " + chunk.replace("\n", "\n  ") + ",")
    lines.append('  "cases": [')
    for i, c in enumerate(doc.get("cases", [])):
        lines.append("    " + json.dumps(c, ensure_ascii=False) + ("," if i < len(doc["cases"]) - 1 else ""))
    lines.append("  ]")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_cases(path: Path, cases: list, compact_cases: bool) -> tuple:
    """把草案追加进用例文件。**已存在的 id 或同样的 query 一律跳过**（不覆盖人的东西）。"""
    doc = json.loads(path.read_text(encoding="utf-8"))
    existing_ids = {c.get("id") for c in doc.get("cases", [])}
    existing_qs = {c.get("query") for c in doc.get("cases", []) if c.get("query")}
    added, skipped = [], []
    for c in cases:
        if c["id"] in existing_ids or c.get("query") in existing_qs:
            skipped.append(c)
            continue
        doc.setdefault("cases", []).append(c)
        existing_ids.add(c["id"])
        if c.get("query"):
            existing_qs.add(c["query"])
        added.append(c)
    if added:
        _write_cases_json(path, doc, compact_cases)
    return added, skipped


# ==================== 复核已有用例（老用例是早期人手定的，可能有错） ====================

REPORT = ROOT / "outputs" / "eval" / "retrieval" / "eval_report.json"


def load_last_report() -> dict:
    """读最近一次检索评测报告（`--audit` 靠它拿"实测结果"）。

    ⚠️ **不现场重跑**：一次评测 ≈ 150 次模型调用。
    "现有期望 vs 实测"才是复核的主要依据，所以报告不存在时**让你先跑评测**，而不是悄悄跑一遍。
    """
    if not REPORT.exists():
        return {}
    try:
        return json.loads(REPORT.read_text(encoding="utf-8"))
    except Exception:
        return {}


def describe_expect(expect: dict) -> str:
    """把 expect 翻成人话。"""
    parts = []
    for route, cond in (expect or {}).items():
        if "should" in cond:
            parts.append(f"{route} 该命中 {cond['should']}")
        elif "contains" in cond:
            parts.append(f"{route} 该含关键词 {cond['contains']}")
        elif cond.get("should_fire"):
            parts.append(f"{route} 该有东西（不钉是哪条）")
        elif cond.get("should_not_hit"):
            parts.append(f"{route} 不该命中")
    return "；".join(parts) or "（没有期望 —— 这条本身写坏了）"


def audit_order(cases: list, report: dict) -> list:
    """按"最可能标错"排序：**当前失败的 → 边界案例 → 老标注（人手定的）→ 其余**。"""
    failed = {r.get("id") for r in report.get("results", []) if not r.get("passed")}

    def key(c):
        return (0 if c.get("id") in failed else 1,
                0 if c.get("borderline") else 1,
                0 if c.get("source") == "标注集" else 1)
    return sorted(cases, key=key)


def _audit(report: dict) -> int:
    """逐条复核现有用例，把"我觉得这条标错了"写进 `问题记录.md`（复用随手记通道）。"""
    doc = json.loads(RETRIEVAL_CASES.read_text(encoding="utf-8"))
    cases = doc.get("cases", [])
    results = {r.get("id"): r for r in report.get("results", [])}

    print(f"\n复核 {len(cases)} 条用例（顺序：**当前失败的 → 边界案例 → 老标注 → 其余**）")
    print("每条问一个问题：**这个期望你认可吗？**\n"
          "  [回车] 认可   [s] 这条标错了   [q] 退出\n")

    flagged = []
    for c in audit_order(cases, report):
        r = results.get(c["id"])
        mark = "（本次评测没跑到这条）" if r is None else ("✓ 当前是绿的" if r.get("passed") else "✗ 当前是红的")
        print("─" * 72)
        print(f"[{c['id']}]  {c['query']}")
        print(f"   现有期望：{describe_expect(c.get('expect'))}")
        print(f"   实测结果：{mark}")
        if r and not r.get("passed"):
            for chk in r.get("checks", []):
                if not chk.get("ok"):
                    print(f"             · {chk['source']}: {chk['detail']}")
        print(f"   为什么这么标：{c.get('why', '（没写）')[:120]}")
        print(f"   来源：{c.get('source', '?')}" + ("　⚠️ 边界案例（当初就是拿不准的）" if c.get("borderline") else ""))
        try:
            ans = input("   认可这条期望吗？ ").strip().lower()
        except EOFError:                     # 非交互环境（管道/CI）：当作退出，别死循环
            print("\n（读不到输入，退出复核）")
            break
        if ans.startswith("q"):
            break
        if ans.startswith("s"):
            flagged.append(c)

    if not flagged:
        print("\n✅ 复核完了，没有标为「有问题」的。")
        return 0

    with NOTES_MD.open("a", encoding="utf-8") as f:
        f.write("\n\n<!-- 以下由 problems --audit 生成：我认为这些**已有用例**标错了 -->\n")
        for c in flagged:
            f.write(f"\n用户说：{c['query']}\n")
            f.write(f"应该是：（这条已有用例的期望是「{describe_expect(c.get('expect'))}」，我认为不对）\n")
    print(f"\n✅ 标了 {len(flagged)} 条「我认为标错了」 → 已写进 {NOTES_MD.name}")
    print("   下一步：去那份文件把每条的『应该是』补完，然后按 `--from-note` 的流程走一遍")
    print("   （先红后绿：改完期望要能复现你看到的问题，否则那条判据没抓住东西）")
    return 0


# ==================== 跑 ====================

def _echo_problem(p: Problem, got, l3) -> None:
    """把一条问题的**事实**摆出来 —— 人就是对着这一段判的。"""
    from retrieval_eval import format_snapshot
    print(f"\n{'─' * 72}")
    print(f"用户说：{p.user}")
    if p.reply:
        print(f"她答　：{p.reply}")
    if p.should:
        print(f"应该是：{p.should}" + (f"（她备注：{p.spot_why}）" if p.spot_why else ""))
    elif p.spot_why:
        print(f"她勾了有问题：{p.spot_why}")
    print(f"—— 这句话实际召回了什么（线上同一条路）——")
    if got is None:
        print("   ⚠️ embedding 失败，取不到实况（检查 .env.prod 的 ZHIPU_API_KEY）")
    else:
        for line in format_snapshot(got, l3, full_text=True):
            print("   " + line)
    print(f"—— 我猜它该落到：【{guess_layer(p)}层】"
          f"{'（你写明了，就是它）' if p.layer in ('检索', '生成') else '（**猜的，你觉得不对就写「层：检索/生成」覆盖**）'}")


async def _clients():
    """构造与线上同款的两个客户端 + 行为表。"""
    from openai import AsyncOpenAI
    from src.plugins.chatbot.constants import ZHIPU_BASE_URL, DEEPSEEK_BASE_URL
    from src.plugins.chatbot.persona import load_persona_rules
    import retrieval_eval as RE
    zp, ds = RE._zhipu_key(), RE._deepseek_key()
    if not zp:
        print("❌ 未找到 ZHIPU_API_KEY（.env.prod）——查实况需要与线上同款 embedding")
        return None, None, None
    client = AsyncOpenAI(api_key=zp, base_url=ZHIPU_BASE_URL)
    ds_client = AsyncOpenAI(api_key=ds, base_url=DEEPSEEK_BASE_URL) if ds else None
    _, _, behaviors = load_persona_rules()
    return client, ds_client, behaviors


async def main(argv=None) -> int:
    """`argv=None` 时读 sys.argv（直接 `python scripts/problem_cases.py`）；
    `run_tool.py problems` 会把自己的参数转成 argv 传进来（两套入口同一份逻辑）。"""
    _common.ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="问题驱动：看到的问题 → 回归用例")
    ap.add_argument("--query", default="", help="只看这一条的检索实况（不写任何文件）")
    ap.add_argument("--from-spot", action="store_true", help=f"读 {SPOT_MD} 里勾了「有」的条目")
    ap.add_argument("--from-note", action="store_true", help=f"读 {NOTES_MD}")
    ap.add_argument("--write", action="store_true", help="确认草案没问题后，正式写进用例文件")
    ap.add_argument("--audit", action="store_true",
                    help="★复核已有用例（老用例是人手定的，可能有错）：逐条问「你认可吗」")
    args = ap.parse_args(argv)

    # ---- 模式⓪：复核已有用例（不调模型，只读最近一次评测报告） ----
    if args.audit:
        report = load_last_report()
        if not report:
            print(f"❌ 没有评测报告 {REPORT}\n"
                  f"   复核要靠「现有期望 vs 实测结果」对照着看，先跑一次：\n"
                  f"     python scripts/run_tool.py retrieval-eval")
            return 1
        return _audit(report)

    import retrieval_eval as RE
    client, ds_client, behaviors = await _clients()
    if client is None:
        return 1

    # ---- 模式①：看一条的实况 ----
    if args.query:
        got, l3 = await RE.snapshot(args.query, client, ds_client, behaviors)
        print(f"【{args.query}】")
        for line in RE.format_snapshot(got, l3, full_text=True) if got else ["（embedding 失败）"]:
            print("   " + line)
        return 0

    # ---- 模式②③：读问题 → 摆实况 → 生成草案 ----
    if args.from_spot:
        if not SPOT_MD.exists():
            print(f"❌ 没有 {SPOT_MD}——先跑 `python scripts/spot_check.py` 挑出可疑的几条")
            return 1
        problems = parse_spot_marks(SPOT_MD.read_text(encoding="utf-8"))
        print(f"📋 从 {SPOT_MD.name} 读到 {len(problems)} 条勾了「有」的")
    elif args.from_note:
        if not NOTES_MD.exists():
            NOTES_MD.write_text(
                "# 问题记录\n\n"
                "> 看到一条答得不对，就在这里记一笔。格式随意，一条一段，都以「用户说：」开头。\n"
                "> 详细字段说明见 `scripts/problem_cases.py` 的 docstring（或跑 `--help`）。\n\n"
                "用户说：\n她答：\n应该是：\n层：\n\n", encoding="utf-8")
            print(f"📝 {NOTES_MD.name} 不存在，已建空模板——写几条再跑一次")
            return 0
        problems = parse_notes(NOTES_MD.read_text(encoding="utf-8"))
        problems = [p for p in problems if p.user.strip()]
        print(f"📋 从 {NOTES_MD.name} 读到 {len(problems)} 条")
    else:
        print(__doc__)
        return 0

    if not problems:
        print("（没有可处理的条目）")
        return 0

    retro, gen = [], []
    for p in problems:
        got, l3 = await RE.snapshot(p.user, client, ds_client, behaviors)
        _echo_problem(p, got, l3)
        if guess_layer(p) == "生成":
            gen.append(draft_generation(p))
        else:
            retro.append(draft_retrieval(p))

    print(f"\n{'=' * 72}\n草案：检索层 {len(retro)} 条 → {RETRIEVAL_CASES.name}"
          f"｜生成层 {len(gen)} 条 → {REGRESSION_CASES.name}")
    for c in retro:
        print("  [检索] " + json.dumps(c, ensure_ascii=False))
    for c in gen:
        print("  [生成] " + json.dumps(c, ensure_ascii=False))

    if not args.write:
        print("\n（以上**只是草案**。核对/改好后加 --write 正式写入。）")
        return 0

    a1, s1 = add_cases(RETRIEVAL_CASES, retro, compact_cases=True)
    a2, s2 = add_cases(REGRESSION_CASES, gen, compact_cases=False)
    print(f"\n✅ 写入完成：检索层 +{len(a1)}（跳过 {len(s1)} 条已有的）"
          f"｜生成层 +{len(a2)}（跳过 {len(s2)}）")
    for c in a1 + a2:
        print(f"   - {c['id']}")
    if a1:
        print("\n⭐ **先红后绿**：现在就跑 `python scripts/run_tool.py retrieval-eval` —— "
              "新用例**应该 FAIL**（那才证明它复现了你看到的问题）。\n"
              "   若是绿的，说明这条判据没抓住问题，要改判据；改完素材再跑，才该转绿。")
    if a2:
        todo = [c["id"] for c in a2 if c.get("_TODO")]
        if todo:
            print(f"\n⚠️ 这几条还缺判据（跑不出结论）：{todo}——补 `forbid` 或 `forbid_pattern` 后再跑")
        print("   验证：`python scripts/run_tool.py regression --check --case <id>`"
              "（每条要跑 5~8 次生成，慢且花 API 钱）")
    print("\n下一步（改哪里）：\n"
          "   定位整条链 → python scripts/trace_chain.py \"<那句话>\"\n"
          "   查召回入口 → python scripts/entry_audit.py all\n"
          "   素材文件   → 检索层 persona/world/*.json ｜ 生成层 persona/behavior|speech/*.json")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
