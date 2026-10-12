#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文档数字体检：把"容易烂的数字"从各文档里扫出来，跟**实测**对一遍。

## 为什么需要它（2026-10-13）

同一组数字抄在 6 个文件里，各自漂移——光"测试数"一个概念就在 `docs/地图.md`
里同时出现过 **814 / 797 / 746** 三个值（实测是 877）。

**数字的权威在 `docs/地图.md §10`**：那里放数字 + 复核命令。
其它文件**允许**写数字（README 要对外、下一步.md 要交接），但写完跑一下这个脚本，
它会告诉你哪几处没跟上。

## 怎么标"这处是历史数字、别改我"

在那一行末尾加 `<!-- check-docs-ok -->`，脚本会跳过。
典型场景：README 的更新记录里"测试 807 → 746"这类**当时的快照**，
它们**不该**被改成今天的数字。

## 用法

    .venv/Scripts/python.exe scripts/check_docs.py          # 只报不一致
    .venv/Scripts/python.exe scripts/check_docs.py --list   # 连"扫到哪些、对上了"一起列
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# ⚠️ 本仓踩过的坑（见 docs/下一步.md「print 在 stdout 被重定向时会抛」）：
# 带 emoji 的 print 在 cp936 下写管道/文件会 UnicodeEncodeError。脚本自己要防这一下。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
VENDOR_PY = ROOT / ".venv" / "Scripts" / "python.exe"

SKIP_MARK = "check-docs-ok"


# ==================== 实测 ====================

def _count_tests() -> tuple[int, int]:
    """(collected, 测试函数数)。collected 由 pytest 收集，函数数按 `def test_` 数。"""
    out = subprocess.run(
        [str(VENDOR_PY), "-m", "pytest", "--collect-only", "-q"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
    ).stdout
    m = re.search(r"(\d+)\s+tests?\s+collected", out)
    collected = int(m.group(1)) if m else -1
    funcs = 0
    for p in (ROOT / "tests").glob("*.py"):
        funcs += len(re.findall(r"^\s*(?:async\s+)?def\s+test_", p.read_text(encoding="utf-8"), re.M))
    return collected, funcs


def _run_tool_subcommands() -> int:
    """run_tool 注册了几个子命令——直接从 --help 的 usage 里数。"""
    out = subprocess.run(
        [str(VENDOR_PY), str(ROOT / "scripts" / "run_tool.py"), "--help"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
    ).stdout
    m = re.search(r"\{([^}]+)\}", out)
    return len(m.group(1).split(",")) if m else -1


def measure() -> dict:
    def jlen(path, key):
        return len(json.loads(path.read_text(encoding="utf-8"))[key])

    collected, funcs = _count_tests()
    return {
        "测试用例数": collected,
        "测试函数数": funcs,
        "scripts .py": len(list((ROOT / "scripts").glob("*.py"))),
        "run_tool 子命令数": _run_tool_subcommands(),
        "运行时文件数": len(list((ROOT / "src/plugins/chatbot").glob("*.py"))),
        "tests 文件数": len(list((ROOT / "tests").glob("*.py"))),
        "behaviors 条数": jlen(ROOT / "persona/behavior/behaviors.json", "behaviors"),
        "骨架字数": len((ROOT / "persona/core/system_prompt.txt").read_text(encoding="utf-8")),
        "入库文件数": len(subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, cwd=str(ROOT)
        ).stdout.splitlines()),
    }


# ==================== 抓数字的正则 ====================
# 每条：(指标名, [能捕获该数字的正则...])。宁可窄一点——漏报比误报好。

PATTERNS = [
    ("测试用例数", [
        # 数字在词前：「**896** 个单元测试」
        r"(\d{3,4})\*{0,2}\s*个单元测试",
        # 数字在词后：「单元测试（896 个）」「测试 `tests/`（896 个）」「测试 **896** 全绿」
        # ⚠️ 这里必须**同时锚住前词和后缀**（个 / passed / 全绿 / tests）。
        #    只锚前词会被"下一处加粗数字"骗到——实测把「｜ **421** 条直播记忆」读成了测试数；
        #    完全不锚又会漏掉「（896 个）」这种把数字放进括号的写法。
        r"(?:测试|用例)[^\n]{0,12}?(\d{3,4})\*{0,2}\s*(?:个|passed|全绿|tests)",
        r"(\d{3,4})\s*passed",
        # ⚠️ 别加裸的 `应为\s*(\d{3,4})`——「应为 223」那种是**入库文件数**行，
        #    会被误当成测试数（实测踩过）。
        r"测试基线[^\n]{0,20}?(\d{3,4})",
    ]),
    ("tests 文件数", [
        r"tests[^\n]{0,8}（(\d{2,3})）",
        r"测试文件数[^\n]{0,10}?(\d{2,3})",
    ]),
    ("测试函数数", [
        r"(\d{3,4})\s*个测试函数",
    ]),
    ("scripts .py", [
        r"(\d{2,3})\s*个\s*\.py",
        r"(\d{2,3})\s*个脚本",
        r"脚本数（应为\s*(\d{2,3})）",
    ]),
    ("run_tool 子命令数", [
        r"(\d{1,3})\s*个子命令",
        r"（(\d{1,3})\s*子命令）",
    ]),
    ("运行时文件数", [
        r"chatbot/[^\n]{0,24}?（(\d{2,3})\s*个",
        r"运行时核心[^\n]{0,24}?（(\d{2,3})\s*个",
    ]),
    ("behaviors 条数", [
        r"behaviors[^\n]{0,24}?(\d{2,3})\s*条",
        r"(\d{2,3})\s*条情景",
    ]),
    ("骨架字数", [
        r"骨架[^\n]{0,16}?(\d{3,4})\s*字",
        r"(\d{3,4})\s*字[^\n]{0,10}骨架",
    ]),
    ("入库文件数", [
        r"(\d{2,3})\s*个入库文件",
    ]),
]

SCAN = ["CLAUDE.md", "README.md"] + [
    str(p.relative_to(ROOT)) for p in sorted((ROOT / "docs").glob("*.md"))
]


def main():
    ap = argparse.ArgumentParser(description="文档数字体检")
    ap.add_argument("--list", action="store_true", help="连对上的也列出来")
    args = ap.parse_args()

    got = measure()
    print("实测：" + " ｜ ".join(f"{k} {v}" for k, v in got.items()))
    print()

    bad, ok, skipped = [], [], 0
    for rel in SCAN:
        p = ROOT / rel
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if SKIP_MARK in line:
                skipped += 1
                continue
            for metric, pats in PATTERNS:
                # 「N 个 .py」既可能指 scripts 也可能指运行时核心——按上下文分派，别互相误判
                if metric == "scripts .py" and ("运行时核心" in line or "chatbot/" in line):
                    continue
                if metric == "运行时文件数" and "scripts" in line:
                    continue
                for pat in pats:
                    for m in re.finditer(pat, line):
                        n = int(m.group(1))
                        rec = (rel, i, metric, n, line.strip()[:90])
                        (ok if n == got[metric] else bad).append(rec)

    if args.list:
        print("—— 扫到且对上的 ——")
        for rel, i, metric, n, txt in ok:
            print(f"  ✅ {rel}:{i}  {metric}={n}  ｜ {txt}")
        print()

    if bad:
        print(f"❌ 有 {len(bad)} 处对不上（实测已列在上面）：")
        for rel, i, metric, n, txt in bad:
            print(f"  {rel}:{i}  {metric} 写了 {n}，实测 {got[metric]}")
            print(f"      {txt}")
        print()
        print(f"（另有 {len(ok)} 处对上、{skipped} 处按 `{SKIP_MARK}` 跳过）")
        return 1

    print(f"✅ 全部对上（{len(ok)} 处；{skipped} 处按 `{SKIP_MARK}` 跳过）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
