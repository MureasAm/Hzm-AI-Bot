#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给每个数据文件的头部补一个统一的「角色」头（`docs/评测与实验.md` §七-6）。

## 为什么

> 现在角色散在包装语和文档里，**改数据的人在改的那一刻不会去翻文档**。
> corpus 那 49% 引语就是**这么漂移进去的**。

`persona.py::_persona_items` 的 docstring 已经说了这个道理、`_readme` 这个字段也早就有了，
但**没有一份文件开头写着"这份是干嘛的 + 什么东西该进这儿"**——
那条跨文件的归属判据只在 `terms.json` 里写过一次。

这个脚本给每份文件头部补两样（**幂等**，已经有了就跳过）：

1. **【角色】** 一句话：模型拿它干什么
2. **【一条东西该进哪个文件】** 同一份判据写进每一份——**在任何文件里都看得到该进哪儿**

⚠️ **只动 `_readme`（运行时根本不读）**，不碰任何数据键。
⚠️ **`system_prompt.txt` 不动**——它是**被注入的提示词本体**，加注释会进模型上下文。

用法：
    python scripts/persona_role_headers.py            # 预览
    python scripts/persona_role_headers.py --write    # 落盘
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

# 跨文件的归属判据（§1.2）——**同一份，写进每一份文件**
ROUTE = ("【一条东西该进哪个文件】—— 改数据前先对一眼（单一真值，别在两处维护）\n"
         "  这件事「发生过」        → `world/statement_final.json`（corpus）\n"
         "  这是她「一贯如此」       → `world/preferences.json`\n"
         "  这是她「会怎么开口」     → `speech/voice_samples.json`\n"
         "  这是「某情境下的反应」   → `behavior/behaviors.json`\n"
         "  这是「一个词的意思」     → `world/terms.json`\n"
         "  这是「固定该怎么答」     → `world/legendary.json`\n"
         "  ⚠️ 同一件事**只在一个文件维护**（历史上 terms/behaviors/提示词三处打架过）")

# 每份文件的角色（模型拿它干什么）
ROLE = {
    "core/traits.json": "**是这样的人**（底色）—— 让她「是这种人」，不是「知道这件事」。每条 = 倾向 + 证据。",
    "core/styles.json": "**怎么说**（概括层）—— 语言层面的稳定特征。⚠️ 只放**概括**，具体措辞→phrases、成句示范→voice_samples。",
    "behavior/behaviors.json": "**怎么做** —— 「什么情境 → 做什么」。⚠️ response 描述**做什么**，不写「用哪个词」（台词归 samples）。",
    "behavior/behavior_keywords.json": "**怎么做** 的确定性兜底 —— 用户含这些词就直接命中行为、跳过 L3。⚠️ 只是兜底，主判据是 L3。",
    "speech/voice_samples.json": "**怎么说** —— 她本人真实说过的话（问答对）。⚠️ 唯一「既传得进又不被抄」的形式；带 user 字段会撞车。",
    "speech/phrases.json": "**怎么说** —— 同意思 → 她的原话**碎片**（2~5 字，不自足 → 不会被整句抄）。",
    "world/terms.json": "**懂词** —— 某个词/人物/梗的意思。命中才注入。",
    "world/preferences.json": "**是这样的人** —— 去掉「某次直播」这个场合仍然成立的事。类别 + 事实。",
    "world/core_stories.json": "**记得**（最深处）—— 印象最深的经历，低阈值浮现。事件陈述。",
    "world/legendary.json": "**固定该怎么答** —— 命中触发词就**直接发出去**（硬路由，不进提示词）。",
    "world/schedule.json": "**地面真值** —— 她哪天几点播。每轮注入（被问时间以它为准）。",
    "world/statement_final.json": "**记得** —— 她「发生过什么」。⚠️ **只该有事实**：不要把「她当时怎么说的」写进去（引语会被整句搬走）。",
    "world/corpus_asks.json": "**入口工具**（不进提示词）—— corpus 每条「用户可能怎么说」，生成时逐条自验证过。",
    "world/corpus_keywords.json": "**入口工具**（不进提示词）—— corpus 每条的「钩子」：用户提到这些词就把该条递过去让 LLM 判。",
    "media/stickers.json": "**配图** —— 表情包标注（desc/tags/use_when），判「这句话配哪张」。",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    _common.ensure_utf8_stdout()

    done, skipped, missing = [], [], []
    for rel, role in ROLE.items():
        p = _common.PERSONA_DIR / rel
        if not p.exists():
            missing.append(rel)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            missing.append(rel + "（不是 dict 形状）")
            continue
        old = d.get("_readme") or ""
        if "【角色】" in old:
            skipped.append(rel)
            continue
        d["_readme"] = f"【角色】{role}\n\n{ROUTE}\n\n———\n{old}"
        # 保持 _readme 在最前（json 保序）
        d = {"_readme": d.pop("_readme"), **d}
        if args.write:
            p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        done.append(rel)

    print(f"{'已补' if args.write else '待补'}：{len(done)} 份")
    for r in done:
        print(f"  ✓ {r}")
    if skipped:
        print(f"已有角色头、跳过：{len(skipped)} 份")
    if missing:
        print(f"⚠️ 找不到/形状不对：{missing}")
    if not args.write and done:
        print(f"\n（预览模式，没落盘。加 --write 才写。）")
    print(f"\n⚠️ `core/system_prompt.txt` **不在名单里**——它是被注入的提示词本体，"
          f"加注释会进模型上下文。它的「角色」写在 `docs/评测与实验.md` §1.1。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
