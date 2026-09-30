#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**判据自检**：喂假回复，看每条判据能不能被触发。

## 解决什么问题

杀毒测试（`criterion_mutation_test.py`）是**端到端**的：注入病因 → 真跑生成 → 看挂不挂。
但它分不开两种"没挂"：
```
(a) 病因注入得不对/不够       → 我这边的问题
(b) 那个病在现在的链路下不会再发生 → 判据守着的是个"已经灭绝的问题"
```

这个脚本用**纯单元测试**把机制层单独拎出来：
```
给每条判据喂一组"明显该被它抓住"的假回复 → 看 _judge 会不会报 FAIL
能触发 → 判据的机制是好的（端到端的"没挂"就归因于 (b) 或注入不够）
不能触发 → **判据本身写歪了**（该修）
```
零成本、不调模型。

用法：
    python scripts/criterion_selfcheck.py
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import nonebot                        # noqa: E402
try:
    nonebot.init()
except Exception:
    pass

import _common                        # noqa: E402
import regression_test as RT          # noqa: E402

# 每条用例 → 一组"明显该被它抓住"的假回复
SYNTH = {
    "no_old_fact_as_current_reason": [
        "搬家的事还没弄完，今天真播不了。", "今天不播，搬东西搬到一半。",
        "搬家太累了。", "有点事。", "困了。",
    ],
    "emoji_pitiful_not_read_as_speechless": [
        "……无语，你发这个干嘛。", "他对我无语了吧。", "这是什么意思。",
    ],
    "no_fixed_opener_when_deflecting": [
        "呃，这个嘛…灰泽满也没办法。"] * 5 + ["行吧行吧。", "你说得对。", "……知道了。"],
    "no_verbatim_repeat_after_late": [
        "呃，这个嘛…灰泽满也没办法。"] * 5 + ["行吧行吧。", "你说得对。", "……知道了。"],
    "no_third_person_self_reference": [
        "灰泽满是风纪委员，她不知道你在说什么。", "灰泽满就是个普通学生，她很普通。",
        "灰泽满在澳洲，她挺冷的。",
    ],
    "no_assistant_tone": [
        "您好，很高兴为您服务，请问有什么可以帮到您的吗？",
        "作为人工智能，我会尽力帮您。",
    ],
    "no_therapist_tone_when_sad": [
        "别难过，一切都会好起来的，相信自己。",
        "我理解你的感受，你一定能撑过去。",
    ],
    "no_new_specific_excuse_when_pressed": [
        "房东那边的事，说搬就得搬。", "搬家公司今天来不了，中介在等。",
        "就这两天的事。",
    ],
}


def _mk_res(replies: list) -> dict:
    openers = Counter(RT._opener(r) for r in replies)
    top, k = openers.most_common(1)[0]
    return {"replies": replies, "opener": top, "homogeneity": k / len(replies),
            "verbatim": max(Counter(replies).values())}


def main() -> int:
    _common.ensure_utf8_stdout()
    cases = {c["id"]: c for c in RT._load_cases()}
    print("判据自检：喂假回复，看每条判据能不能被触发（零成本，不调模型）\n")
    print(f"{'用例':<40}{'能触发':<8}报的什么")
    print("-" * 104)
    ok = 0
    for cid, replies in SYNTH.items():
        case = cases.get(cid)
        if not case:
            print(f"{cid:<40}{'? 缺用例':<8}")
            continue
        fails = RT._judge(case, _mk_res(replies))
        mark = "✅" if fails else "❌ 触发不了"
        if fails:
            ok += 1
        print(f"{cid:<40}{mark:<8}{(fails[0] if fails else '')[:62]}")
        if len(fails) > 1:
            for f in fails[1:]:
                print(f"{'':<48}{f[:62]}")

    print(f"\n【汇总】{ok}/{len(SYNTH)} 条判据的**机制**是好的")
    if ok < len(SYNTH):
        print("  ⚠️ 触发不了的那些：**判据本身写歪了**（不是「病没了」，是这个尺子量不到）→ 该修")
    else:
        print("  → 全部机制正常。那端到端杀毒测试里「没挂」的那些，原因只能是：")
        print("     ① 我注入的病因不够/不对，或")
        print("     ② 那个病在现在的链路下**真的不会再发生**了（守着的是已灭绝的问题）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
