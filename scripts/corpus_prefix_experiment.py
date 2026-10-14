#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""corpus 前缀实验：**每条自己带【这是过去发生过的事】，到底管不管用？**

## 为什么单独量它（2026-10-15）

用户问的原话是"文件开头有标注这是过去发生的事，我不知道这个包装语有没有起效果"。
项目自己对"包装语"已经有一条结论（`docs/架构.md` §4.5）：

    同样内容，只换外面那句说明（"用这些原话" vs "不整段复述"）→ **逐字率一模一样**。
    **包装语对"怎么用这段内容"零作用。**

但 §4.5 自己留了个口子：**"这不等于包装语全无用处——它可能仍在管别的事（如时间语义）"**。
这次要加的【这是过去发生过的事】正是**时间语义**，落在那条口子里面——所以**值得单独量一次**，
不能拿 §4.5 直接判它死刑。

## 设计（照抄 §4.5 的形状：只变外面那层）

注入**同一条** corpus（一条会被人当成"现在"的过去事），只换它的外包装：

    A  段头说明在、条目裸着     ← 加前缀**之前**的线上形态
    B  段头说明在 + 每条带前缀   ← 加前缀**之后**的线上形态（要验证的）
    C  段头说明和前缀都没有     ← 真正的"零时间定性"对照

然后问一句**会诱导她把它当现在**的话，判据是**回复里有没有出现那件过去事的特征词**
——出现 = 被她当成了"正在发生的事"。

用法：python scripts/corpus_prefix_experiment.py --runs 8
"""
import argparse
import asyncio
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

import src.plugins.chatbot.core as core              # noqa: E402
import src.plugins.chatbot.memory as mem             # noqa: E402
from src.plugins.chatbot.retrieval import RetrievalItem  # noqa: E402

# 一条**会被当成现在**的过去事（带"第二天"这种相对时间、带具体动作）
FACT = "她第二天要去拿搬家的钥匙，顺便看看还缺什么家具。"
LEAK = ("搬家", "搬东西", "钥匙", "家具")      # 出现 = 她把过去事当成"现在"了
QUESTION = "你最近在忙什么呀"

HEADER_RE = re.compile(r"^【她经历过的相关背景】")


def _msgs(kind: str, query: str) -> list:
    """按条件造 messages：只改 corpus 那一段的**外层包装**，条目内容一个字不动。"""
    it = RetrievalItem(source="corpus", item_id="0", score=0.9, text=FACT)
    msgs = core.build_message_list(query, [it], "", [])
    out = []
    for m in msgs:
        c = m.get("content") or ""
        if HEADER_RE.match(c):
            line = c[c.index("\n- "):] if "\n- " in c else ""
            body = line.replace("- 【这是过去发生过的事】", "- ").strip("- ").strip()
            if kind == "A":       # 段头说明在、条目裸着
                c = f"【她经历过的相关背景】以下是她过去直播里经历过的事（背景记忆，都是曾经发生的，不是现在）。\n- {body}"
            elif kind == "B":     # 线上现形态（段头 + 每条前缀）——原样不动
                pass
            else:                 # C：连段头都没有
                c = f"【相关背景】\n- {body}"
        out.append({**m, "content": c})
    return out


async def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=8)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import nonebot.log as _log
    _log.logger.remove()

    core.update_memory_v2_shadow_task = lambda *a, **k: asyncio.sleep(0)
    mem.use_storage(Path(tempfile.mkdtemp(prefix="corpusprefix_")))

    # 自检：那条 corpus 确实进了 messages（否则这个实验测的是空气）
    _probe = "\n".join(m["content"] for m in _msgs("B", QUESTION))
    assert FACT in _probe, "corpus 没注入，实验无效"
    print(f"自检 ✅ 注入的句子在 messages 里（共 {len(_probe)} 字）\n")

    conds = ["A", "B", "C"]
    res = {k: [] for k in conds}
    for _ in range(args.runs):
        for k in conds:
            r = await core.generate_reply(_msgs(k, QUESTION))
            leak = [w for w in LEAK if w in r]
            res[k].append((r, leak))
            print(f"[{k}] {'⚠️泄漏' if leak else '　干净'} {r!r}")

    print("\n" + "=" * 60)
    print(f"问的是：{QUESTION!r}　注入的过去事：{FACT}")
    label = {"A": "段头说明在、条目裸着", "B": "段头 + 每条带前缀（线上）", "C": "一点时间定性都没有"}
    for k in conds:
        n = sum(1 for _, leak in res[k] if leak)
        words = sorted({w for _, leak in res[k] for w in leak})
        print(f"{label[k]:<24}{n:>3}/{args.runs} 把过去事当现在   {'（泄漏词：' + '/'.join(words) + '）' if words else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))