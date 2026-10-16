#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探针 / 张力最小对：拿同担交接包里的两组固定探针，量**我们自己的** bot。

## 这两组探针是什么（来自 assets/handoff/handoff-code-methods/）

- `probe_inputs.txt`：30 条**真实直播弹幕**（从 A 级语料里抽、长度 4~30 字、一条固定 seed
  抽一次就冻结，跨模型跨次运行可比）。用途是"给她一组固定刺激，看输出分布"。
- `probe_pairs.json`：7 对**张力最小对**（minimal pair，借自音位学：ship/sheep 只差一个音位）。
  每对两句**只翻转一个结构特征**（语用意图 / 评价方向 / 人称 / 在场预设…），用来测
  **"她会不会因为这一处不同而走不同姿态"**——测的是**选择逻辑**，不是句子分布。
  其中 6 对是"分化对"（预期 A/B 走向不同），1 对是"守恒对"（P03，预期 A/B **都**保持
  第三人称自称，测的是"招牌是内化的还是被弹幕带跑的"）。

## 为什么可以做（对我们）

它们的判读标准是现成的、写在 `probe_pairs.json` 的 `expect_pass` / `expect_fail` 里——
不需要 LLM 当裁判，人眼比对两版回复走向即可（这正是本项目"不用模型当裁判"的同款纪律）。

## 口径（重要）

**每次生成都是全新上下文**（不复用检索、不复用历史、不复用会话）——对齐他们的协议：
"A/B 必须独立会话上下文采样（防串扰：她刚对 A 说过的话会影响 B 的回复，分化判读失效）"。
所以这里的输出＝**纯人设**（骨架 + 词表 + 感知），不含检索命中的 behaviors/corpus。
想看"带检索"的版本另跑一轮即可（同一个脚本加 `--with-retrieval`）。

用法：
    python scripts/probe_run.py                    # 30 条探针 ×1 + 14 句张力对 ×2
    python scripts/probe_run.py --pair-runs 3
    python scripts/probe_run.py --with-retrieval   # 额外跑一轮"带检索"作对照
"""
import argparse
import asyncio
import json
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

import src.plugins.chatbot.core as core              # noqa: E402
import src.plugins.chatbot.memory as mem             # noqa: E402
from src.plugins.chatbot.config import _get_clients  # noqa: E402

FIX = ROOT / "assets" / "handoff" / "handoff-code-methods" / "tests" / "acceptance" / "fixtures"
PROBES = FIX / "probe_inputs.txt"
PAIRS = FIX / "probe_pairs.json"


async def _gen(text: str, client, zhipu, with_retrieval: bool) -> str:
    """一次生成。上下文全新（空 fused / 空记忆 / 空历史）。"""
    fused = []
    if with_retrieval:
        ctx = await core.gather_retrieval(
            query_text=text, retrieval_query=text, history_text="",
            deepseek_client=client, zhipu_client=zhipu, is_user_msg=True)
        fused = ctx["fused_items"]
    msgs = core.build_message_list(text, fused, [])
    return await core.generate_reply(msgs)


async def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-runs", type=int, default=2, help="每句张力对跑几次（默认 2，用来看分化稳不稳）")
    ap.add_argument("--with-retrieval", action="store_true", help="额外跑一轮带检索的对照")
    ap.add_argument("--tag", default="", help="输出文件名后缀")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import nonebot.log as _log
    _log.logger.remove()

    client, zhipu = _get_clients()
    core.update_memory_v2_shadow_task = lambda *a, **k: asyncio.sleep(0)
    mem.use_storage(Path(tempfile.mkdtemp(prefix="probe_")))

    probes = [ln.strip() for ln in PROBES.read_text(encoding="utf-8").splitlines() if ln.strip()]
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))["pairs"]
    print(f"探针 {len(probes)} 条；张力对 {len(pairs)} 对（{len(pairs) * 2} 句）\n")

    result = {"probes": [], "pairs": [], "with_retrieval": args.with_retrieval}

    print("=" * 70)
    print("【一】30 条固定探针 —— 同一组刺激，看她的输出")
    print("=" * 70)
    for i, q in enumerate(probes, 1):
        try:
            r = await _gen(q, client, zhipu, False)
        except Exception as e:
            r = f"<失败：{e}>"
        print(f"{i:2}. 弹幕：{q}")
        print(f"    她：{r}")
        result["probes"].append({"i": i, "q": q, "reply": r})

    print()
    print("=" * 70)
    print(f"【二】7 对张力最小对 —— 每句独立采样 ×{args.pair_runs}")
    print("=" * 70)
    for p in pairs:
        rec = {"id": p["id"], "kind": p["kind"], "posture": p["posture"],
               "flip": p["flip"], "a": p["a"], "b": p["b"],
               "expect_pass": p.get("expect_pass"), "expect_fail": p.get("expect_fail"),
               "A": [], "B": []}
        print(f"\n{p['id']} · {p['posture']} · {'分化对' if p['kind'] == '分化' else '守恒对'}")
        print(f"    只差一处：{p['flip']}")
        for side in ("a", "b"):
            tag = side.upper()
            for _ in range(args.pair_runs):
                try:
                    r = await _gen(p[side], client, zhipu, False)
                except Exception as e:
                    r = f"<失败：{e}>"
                rec[tag].append(r)
                print(f"    {tag}｜弹幕：{p[side]}")
                print(f"      → {r}")
        result["pairs"].append(rec)

    if args.with_retrieval:
        print()
        print("=" * 70)
        print("【三】同样的张力对，但走**带检索**的链路（对照）")
        print("=" * 70)
        for rec in result["pairs"]:
            rec["A_r"] = []
            rec["B_r"] = []
            for side, tag in (("a", "A"), ("b", "B")):
                try:
                    r = await _gen(rec[side], client, zhipu, True)
                except Exception as e:
                    r = f"<失败：{e}>"
                rec[f"{tag}_r"].append(r)
                print(f"    {rec['id']} {tag}(带检索)：{r}")

    out = ROOT / "outputs" / "eval" / f"probe_run{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 明细已存：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))