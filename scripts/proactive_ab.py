#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主动发言 A/B：拿真实动态跑 `compose_proactive`，把两版提示词的产出摆在一起看。

## 为什么有它（2026-10-14）

用户报了三类怪句——"顶着个万圣节南瓜早餐来上课" / 每次通知开播都带"来玩呗" /
每晚都是"睡了，你也早点睡"。**读 prompt 猜不出原因**，真跑一遍才发现：后两句是
prompt 里那两条"正例"被**逐字抄**了（详见 `proactive.py` 里 `PROACTIVE_PROMPT` 上方的注释）。
这脚本就是把那次复现固化成工具——以后改主动发言的提示词，先跑它，别凭感觉改。

## 用法

    .venv/Scripts/python.exe scripts/run_tool.py proactive-ab              # 线上那版
    .venv/Scripts/python.exe scripts/run_tool.py proactive-ab --n 2        # 每条跑 2 次看稳定性
    .venv/Scripts/python.exe scripts/run_tool.py proactive-ab --case "晚安🌙"
    .venv/Scripts/python.exe scripts/run_tool.py proactive-ab --prompt-file outputs/p_new.txt

⚠️ **会真的调 LLM**（每条两次：先判"说不说得清"，再生成）。10 条 × 2 次 ≈ 20 次调用。
"""
import argparse
import asyncio
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import _common  # noqa: E402

# 真实动态。前三条是用户 2026-10-14 报的那三条；其余从 `materials/灰泽满动态合集.txt` 里
# 挑同类型做对照——**只留报的那三条，会分不清"修好了"还是"这条恰好没犯"**。
DEFAULT_CASES = [
    ("日常(报的)", "早上好困顿上课中 一个有点🎃万圣节感的早餐"),
    ("直播通知(报的)", "通知晚了 21:30播一会儿"),
    ("晚安(报的)", "晚安🌙 （又熬夜了…"),
    ("直播通知", "明天18:00播吧"),
    ("直播通知", "突击一会儿 最后一场打电话 快润了"),
    ("直播通知", "明天不直播"),
    ("晚安", "好了提前晚安大家，hzm只适合当一条永远躺在被子里的东西😭"),
    ("状态", "怎么感觉要病了，突然浑身难受"),
    ("状态", "我就知道昨晚只睡了四个小时开播准没好事"),
    ("日常", "回家第一天：和睦 第二天：平静 第三天：吵架（还好马上要走了）"),
]


def _run(args) -> int:
    _common.ensure_utf8_stdout()

    # 插件包一导入就会拉起 NoneBot（compose_proactive 要它取 config / 检索素材）
    import nonebot
    from nonebot.adapters.onebot.v11 import Adapter as ONEBOT_V11Adapter

    nonebot.init()
    nonebot.get_driver().register_adapter(ONEBOT_V11Adapter)
    nonebot.load_plugins("src/plugins")

    from src.plugins.chatbot import proactive

    if args.prompt_file:
        proactive.PROACTIVE_PROMPT = Path(args.prompt_file).read_text(encoding="utf-8")
        print(f"（用 {args.prompt_file} 当 PROACTIVE_PROMPT）")
    else:
        print("（用线上 PROACTIVE_PROMPT）")

    cases = [("(命令行)", c) for c in (args.case or [])] or DEFAULT_CASES
    if args.case:
        print("（只跑 --case 给的那几条）")
    print("=" * 70)

    async def run_all():
        for label, text in cases:
            outs = []
            for _ in range(args.n):
                said = await proactive.compose_proactive(text, "B站")
                outs.append(said if said else "（NULL/未生成）")
            print(f"\n【{label}】{text}")
            for o in outs:
                print(f"   → {o}")

    asyncio.run(run_all())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="proactive_ab", description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1, help="每条生成几次（默认 1；温度 0.9，看稳定性就调 2）")
    ap.add_argument("--case", action="append", default=None,
                    help="只跑这一条动态（可重复传）；缺省跑内置的 10 条真实动态")
    ap.add_argument("--prompt-file", default=None,
                    help="用这个文件当 PROACTIVE_PROMPT（缺省=线上那版）")
    return _run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
