#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读周报：最近 N 天聊天记录 + 条目级注入轨迹 → 确定性指标 + 问题轮次 + 候选。

为什么**只读、不调模型**（`docs/数据迭代.md` §8 第一项）：先把"该看哪几轮"用确定性
判据挑出来、落成人审报告；先跑一周看候选质量，确认这套挑法值得信，再决定要不要让
模型参与第 4 步（只读诊断、给 proposal）。

它**不改任何数据**：不写 persona / data / user_memory，只往 `outputs/review/`
（gitignore）落 `.md` + `.json`。
⚠️ 产物里带真实用户发言——**只留本机，别提交/外传**。

数据来源：
  · `data/chat_log/<会话>.jsonl`  用户说了什么 / 她回了什么（发送链里写）
  · `data/injection_trace.jsonl`  那一轮实际注入了哪些块/条目（core.handle_chat 写）

用法：
    python scripts/weekly_review.py                # 最近 7 天
    python scripts/weekly_review.py --days 14
    python scripts/weekly_review.py --session 2807992783
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

CHAT_DIR = _common.DATA_DIR / "chat_log"
TRACE_FILE = _common.DATA_DIR / "injection_trace.jsonl"
OUT_REVIEW = _common.OUTPUTS_DIR / "review"

# 她回复里出现这些信号 → 用户问的词她没接住（`docs/数据迭代.md` §5 反查入口）
CONFUSED_SIGNALS = (
    "听不懂", "不明白", "没懂", "啥意思", "不知道你在说什么", "没听说过", "这是什么",
)
# 用户下一轮出现这些说法 → 上一轮她答偏了（确定性、宁可窄）
CORRECTION_PATTERNS = (
    "不是这个", "不是这么", "不是那样", "不是让你", "没说让你", "没让你",
    "谁让你", "我说的是", "我是说", "我指的是", "我问的是", "问的是",
    "答非所问", "跑题", "理解错", "听错", "看错", "不要评价", "别评价",
)
# 检索类条目：一个都没有 = 这轮她几乎只能靠骨架答（terms 是常驻名词库，不算检索）
RETRIEVAL_KEYS = ("corpus", "behaviors", "preferences", "core_stories", "memory")

_MAX_EVIDENCE_QUOTE = 120      # 报告里单条引文上限（防止一整段转发刷屏）
_PAIR_SAME_TURN_SECONDS = 600  # 纠正判定：下一轮在 10 分钟内才算"在纠正上一轮"


# ==================== 读日志（只读） ====================

def _cutoff(days: float) -> float:
    return time.time() - days * 86400 if days else 0.0


def load_chat(days: float, session: str | None = None) -> list[dict]:
    """读 `data/chat_log/*.jsonl`，按时间排序。坏文件/坏行跳过。"""
    cutoff = _cutoff(days)
    out = []
    if not CHAT_DIR.exists():
        return out
    for f in sorted(CHAT_DIR.glob("*.jsonl")):
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (rec.get("t") or 0) < cutoff:
                continue
            if session and str(rec.get("session")) != str(session):
                continue
            out.append(rec)
    out.sort(key=lambda r: r.get("t") or 0)
    return out


def load_traces(days: float, session: str | None = None) -> list[dict]:
    """读 `data/injection_trace.jsonl`（缺文件返回空——轨迹是 2026-10-20 才加的）。"""
    cutoff = _cutoff(days)
    if not TRACE_FILE.exists():
        return []
    out = []
    for line in TRACE_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (rec.get("t") or 0) < cutoff:
            continue
        if session and str(rec.get("session")) != str(session):
            continue
        out.append(rec)
    return out


def pair_traces(turns: list[dict], traces: list[dict]) -> list[dict | None]:
    """按会话 + 时间把每轮聊天配上当时的注入轨迹。

    同一轮的轨迹**总是早于**聊天记录（前者在 `handle_chat` 里写、后者在发送链里写），
    所以按时间顺序走：每轮取"上一轮消费之后、本轮之前"的最后一份轨迹。
    配不到就返回 None——经典梗硬路由（不进 build_message_list）、群聊不接、
    轨迹上线前的旧记录都会这样，报告里单独计入"无轨迹轮次"。
    """
    by_session = defaultdict(list)
    for tr in traces:
        by_session[str(tr.get("session"))].append(tr)
    for lst in by_session.values():
        lst.sort(key=lambda r: r.get("t") or 0)

    cursor = defaultdict(int)
    paired = []
    for turn in turns:
        sess = str(turn.get("session"))
        lst = by_session.get(sess, [])
        i = cursor[sess]
        best = None
        while i < len(lst) and (lst[i].get("t") or 0) <= (turn.get("t") or 0):
            best = lst[i]
            i += 1
        cursor[sess] = i
        paired.append(best)
    return paired


# ==================== 确定性指标 ====================

def _quote(text: str, limit: int = _MAX_EVIDENCE_QUOTE) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= limit else t[:limit] + "…"


def _opener(reply: str) -> str:
    """回复的开头（到第一个标点为止，最多 8 字）——"同一 opener"按这个比。"""
    t = (reply or "").strip()
    for i, ch in enumerate(t):
        if ch in "。！？!?，,；;~～…、\n ":
            return t[:i][:8]
    return t[:8]


def _confused_reply(reply: str) -> str:
    for sig in CONFUSED_SIGNALS:
        if sig in (reply or ""):
            return sig
    return ""


def _correction_cue(text: str) -> str:
    for cue in CORRECTION_PATTERNS:
        if cue in (text or ""):
            return cue
    return ""


def _evidence(turn: dict, reply: str | None = None) -> dict:
    """一条证据（真实轮次）。用户原话只进本机报告。"""
    return {
        "session": str(turn.get("session")),
        "kind": turn.get("kind"),
        "t": round(float(turn.get("t") or 0), 3),
        "time": time.strftime("%m-%d %H:%M", time.localtime(turn.get("t") or 0)),
        "user": _quote(turn.get("user")),
        "reply": _quote(reply if reply is not None else turn.get("reply")),
    }


def analyse(turns: list[dict], traces: list[dict]) -> dict:
    """算指标 + 挑问题轮次。纯函数（不碰文件），方便测试。"""
    paired = pair_traces(turns, traces)
    n = len(turns)
    sessions = {str(t.get("session")) for t in turns}
    kinds = Counter(str(t.get("kind") or "?") for t in turns)
    with_trace = sum(1 for p in paired if p)

    candidates = []
    seen_candidate_keys = set()

    def add_candidate(kind: str, file_hint: str, turn: dict, problem: str,
                      expected_metric: str, extra_evidence: list | None = None) -> None:
        key = (kind, turn.get("session"), round(float(turn.get("t") or 0), 1))
        if key in seen_candidate_keys:
            return
        seen_candidate_keys.add(key)
        candidates.append({
            "file": file_hint,
            "item": "",
            "problem": problem,
            "evidence": [_evidence(turn)] + (extra_evidence or []),
            "proposal": None,
            "confidence": "待人工/模型诊断",
            "expected_metric": expected_metric,
            "kind": kind,
        })

    # ① 她回复里出现"听不懂"类信号 → 词条反查候选（§5）
    for turn in turns:
        sig = _confused_reply(turn.get("reply", ""))
        if sig:
            add_candidate("confused", "terms",
                          turn, f"回复里出现「{sig}」，可能用户问的词没接住",
                          "听不懂/不明白 回复数")

    # ② 用户下一轮明确纠正 → 上一轮作为问题轮
    by_session = defaultdict(list)
    for turn in turns:
        by_session[str(turn.get("session"))].append(turn)
    corrections = 0
    for lst in by_session.values():
        lst.sort(key=lambda r: r.get("t") or 0)
        for prev, nxt in zip(lst, lst[1:]):
            cue = _correction_cue(nxt.get("user", ""))
            if not cue:
                continue
            if (nxt.get("t") or 0) - (prev.get("t") or 0) > _PAIR_SAME_TURN_SECONDS:
                continue
            corrections += 1
            add_candidate("correction", "待定", prev,
                          f"用户下一轮用「{cue}」纠正了上一轮的回复",
                          "用户纠正/抱怨率",
                          extra_evidence=[_evidence(nxt)])

    # ③ 同一 opener 连续出现（复读/模板化）
    opener_repeats = 0
    openers = Counter()
    for lst in by_session.values():
        lst.sort(key=lambda r: r.get("t") or 0)
        last = ""
        for turn in lst:
            op = _opener(turn.get("reply", ""))
            if not op:
                last = ""
                continue
            openers[op] += 1
            if op == last:
                opener_repeats += 1
                add_candidate("repeat_opener", "behaviors/11_corpus", turn,
                              f"回复 opener『{op}…』与上一轮相同",
                              "同一 opener 重复率")
            last = op

    # ④ 检索空轮次（有轨迹的轮次里，检索类条目一个都没进提示词）
    empty_retrieval = 0
    for turn, tr in zip(turns, paired):
        if not tr:
            continue
        items = tr.get("items") or {}
        if not any(items.get(k) for k in RETRIEVAL_KEYS):
            empty_retrieval += 1
            add_candidate("empty_retrieval", "retrieval/terms", turn,
                          "这一轮检索类条目（corpus/behaviors/偏好/核心记忆/记忆）一条都没注入",
                          "检索空轮次")

    # ⑤ 注入健康度：块开火率 + 条目数
    block_counts = Counter()
    item_counts = defaultdict(list)
    nickname_turns = nickname_used = 0
    for tr in traces:
        blocks = set(tr.get("blocks") or [])
        block_counts.update(blocks)
        for k, v in (tr.get("items") or {}).items():
            item_counts[k].append(len(v))
        if "nickname" in blocks:
            nickname_turns += 1
            if tr.get("reply_used_nickname"):
                nickname_used += 1

    n_traced = len(traces)
    metrics = {
        "window_turns": n,
        "sessions": len(sessions),
        "private_turns": kinds.get("private", 0),
        "group_turns": kinds.get("group", 0),
        "turns_with_trace": with_trace,
        "trace_coverage": round(with_trace / n, 3) if n else 0.0,
        "confused_reply_count": sum(
            1 for turn in turns if _confused_reply(turn.get("reply", ""))),
        "correction_count": corrections,
        "correction_rate": round(corrections / n, 3) if n else 0.0,
        "repeated_opener_count": opener_repeats,
        "repeated_opener_rate": round(opener_repeats / n, 3) if n else 0.0,
        "empty_retrieval_turns": empty_retrieval,
        "top_openers": openers.most_common(8),
        "block_fire_rate": {
            k: round(v / n_traced, 3) for k, v in block_counts.most_common()
        } if n_traced else {},
        "avg_items_per_turn": {
            k: round(sum(v) / len(v), 2) for k, v in item_counts.items()
        },
        "nickname_hint_turns": nickname_turns,
        "nickname_used_turns": nickname_used,
    }
    return {"metrics": metrics, "candidates": candidates}


# ==================== 报告落盘 ====================

_NOT_COMPUTED = [
    ("偏好/行为冲突数", "要判断「她对 A 说过的偏好是否与 B 轮冲突」，确定性命中做不到——留给模型只读诊断"),
    ("答非所问", "也要语义判断；目前只能靠「用户下一轮纠正」间接发现（已算进候选）"),
    ("群聊该接没接", "需要回放 20 批真实群聊 + 人工判，不在本脚本里"),
    ("该命中没命中", "需要拿真实问法跑 `run_tool retrieval-eval` 对照，不在本脚本里"),
]


def render_markdown(report: dict, days: float, session: str | None) -> str:
    m = report["metrics"]
    lines = [
        "# 周报（确定性指标 + 问题轮次）",
        "",
        f"> 窗口：最近 {days:g} 天" + (f"；会话 {session}" if session else ""),
        f"> 生成：{time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. 概览",
        "",
        f"- 聊天轮次 **{m['window_turns']}**（私 {m['private_turns']} / 群 {m['group_turns']}），"
        f"会话 {m['sessions']} 个",
        f"- 有注入轨迹的轮次 **{m['turns_with_trace']}**（覆盖 {m['trace_coverage']:.0%}）"
        "；其余是经典梗硬路由 / 群聊不接 / 轨迹上线前的旧记录",
        "",
        "## 2. 确定性指标",
        "",
        "| 指标 | 读数 |",
        "|---|---|",
        f"| 听不懂/不明白 回复 | {m['confused_reply_count']} 轮 |",
        f"| 用户下一轮纠正 | {m['correction_count']} 轮（{m['correction_rate']:.1%}） |",
        f"| 同一 opener 连续 | {m['repeated_opener_count']} 轮（{m['repeated_opener_rate']:.1%}） |",
        f"| 检索空轮次 | {m['empty_retrieval_turns']} 轮 |",
        f"| 称呼提示 / 真用了 | {m['nickname_hint_turns']} / {m['nickname_used_turns']} |",
        "",
    ]
    if m["top_openers"]:
        lines += ["最高频 opener：" + "，".join(
            f"『{op}…』×{cnt}" for op, cnt in m["top_openers"]), ""]
    if m["block_fire_rate"]:
        lines += ["| 块 | 开火率 |", "|---|---|"]
        lines += [f"| {k} | {v:.0%} |" for k, v in m["block_fire_rate"].items()]
        lines.append("")
    if m["avg_items_per_turn"]:
        lines += ["平均每轮注入条目数：" + "，".join(
            f"{k} {v}" for k, v in m["avg_items_per_turn"].items()), ""]

    lines += ["## 3. 候选（问题轮次 + 文件提示）", ""]
    if not report["candidates"]:
        lines += ["本轮窗口没有挑出问题轮次。", ""]
    else:
        buckets = defaultdict(list)
        for c in report["candidates"]:
            buckets[c["kind"]].append(c)
        lines += ["`proposal` 留空——本脚本不调模型；下一步（§4 第 4 步）再用只读诊断补齐。", ""]
        for kind, items in buckets.items():
            lines += [f"### {kind}（{len(items)} 条）", ""]
            for c in items:
                ev = c["evidence"][0]
                lines += [
                    f"- **{c['problem']}**（文件提示：`{c['file']}`；预期改善：{c['expected_metric']}）",
                    f"  - 当时：`{ev['session']}` {ev['time']} 用户「{ev['user']}」→ 她「{ev['reply']}」",
                ]
                for extra in c["evidence"][1:]:
                    lines.append(
                        f"  - 下一轮：{extra['time']} 用户「{extra['user']}」")
            lines.append("")

    lines += ["## 4. 本脚本没算的（需人/模型）", ""]
    lines += [f"- **{name}**：{why}" for name, why in _NOT_COMPUTED]
    lines += [
        "",
        "## 5. 提醒",
        "",
        "- 报告里有真实用户发言：**只留本机**（`outputs/` 已 gitignore）。",
        "- 采纳任何修改前先跑验收：`pytest -q` + `run_tool regression --check`；"
        "动检索/语料/行为再加 `run_tool retrieval-eval`。",
        "",
    ]
    return "\n".join(lines)


def run(days: float = 7, session: str | None = None, out_dir: str | Path | None = None,
        date: str | None = None, max_per_kind: int = 30) -> dict:
    """跑一遍周报并落盘。返回 {turns, metrics, candidates, md, json}。

    拆出来给 `run_tool weekly-review` 复用（它不经过 argparse）。
    """
    turns = load_chat(days, session)
    traces = load_traces(days, session)
    if not turns:
        return {"turns": 0, "metrics": {}, "candidates": [], "md": None, "json": None}

    report = analyse(turns, traces)

    # 截断每类候选（指标在截断前算完，不受影响）
    kept, counts = [], Counter()
    for c in report["candidates"]:
        if counts[c["kind"]] >= max_per_kind:
            continue
        counts[c["kind"]] += 1
        kept.append(c)
    report["candidates"] = kept

    out = Path(out_dir) if out_dir else OUT_REVIEW
    out.mkdir(parents=True, exist_ok=True)
    stamp = date or time.strftime("%Y-%m-%d")
    md_path = out / f"{stamp}.md"
    json_path = out / f"{stamp}.json"
    md_path.write_text(render_markdown(report, days, session), encoding="utf-8")
    json_path.write_text(json.dumps(
        {"generated_at": time.time(), "days": days, "session": session,
         "metrics": report["metrics"], "candidates": report["candidates"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return {"turns": len(turns), "metrics": report["metrics"],
            "candidates": report["candidates"], "md": md_path, "json": json_path}


def main() -> int:
    ap = argparse.ArgumentParser(description="只读周报：确定性指标 + 问题轮次 + 候选")
    ap.add_argument("--days", type=float, default=7, help="窗口天数（默认 7）")
    ap.add_argument("--session", default=None, help="只看某个会话（QQ 号/群号）")
    ap.add_argument("--out-dir", default=str(OUT_REVIEW), help="输出目录（默认 outputs/review/）")
    ap.add_argument("--date", default=None, help="文件名日期（默认今天，YYYY-MM-DD）")
    ap.add_argument("--max-per-kind", type=int, default=30,
                    help="每类候选最多保留几条（默认 30；只截报告，不影响指标）")
    args = ap.parse_args()
    _common.ensure_utf8_stdout()

    result = run(days=args.days, session=args.session, out_dir=args.out_dir,
                 date=args.date, max_per_kind=args.max_per_kind)
    if not result["turns"]:
        print("窗口内没有聊天记录（data/chat_log/）。先让 bot 跑一阵再来。")
        return 0

    m = result["metrics"]
    print(f"窗口 {args.days:g} 天：{m['window_turns']} 轮 / {m['sessions']} 个会话"
          f"（轨迹覆盖 {m['trace_coverage']:.0%}）")
    print(f"  听不懂信号 {m['confused_reply_count']}｜用户纠正 {m['correction_count']}"
          f"｜opener 连续 {m['repeated_opener_count']}｜检索空 {m['empty_retrieval_turns']}")
    print(f"候选 {len(result['candidates'])} 条 → {result['md']}")
    print(f"                                    {result['json']}")
    print("⚠️ 含真实用户发言，只留本机。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
