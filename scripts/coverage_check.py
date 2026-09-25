# -*- coding: utf-8 -*-
"""覆盖度体检：拿真实对话跑检索，看有多少条"什么素材都搜不到"，以及各路命中率。

用法：
    python scripts/coverage_check.py

2026-09-25 首次跑的结果（167 条真实用户发言）：
    六路全空只有 1 条 = 1%  → **"完全没素材"几乎不存在**
    但各路命中率悬殊：phrase 99% / voice_sample 32% / core_story 31% / corpus 26% / preference 20%
    phrase 99% 是**假信号**（见 constants.py 里 PHRASE_THRESHOLD 的注释：短文本余弦分不开）。
    **结论：痛点不是"读不到"，是"读到了不该读的"。**

回答的问题不是"排序好不好"，是**"该有的有没有"**——
用户说的"很多读不到 / 需要时没有"多半是这个（覆盖问题），而它不该靠改排序解决。

判据：六路语义检索（corpus / voice_sample / phrase / preference / core_story）
有没有任何一路返回非空。**全空 = 她这一轮没有任何素材可依**，
只能靠 system_prompt 的底色硬答——很可能也是答得最飘的时刻。

只读 + 调 embedding，不改任何文件。
"""
import asyncio
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402
nonebot.init()

from openai import AsyncOpenAI  # noqa: E402
from src.plugins.chatbot.rag import embed_query  # noqa: E402
from src.plugins.chatbot.retrieval import (  # noqa: E402
    retrieve_corpus, retrieve_voice_samples, retrieve_phrases,
    retrieve_preferences, retrieve_core_stories,
)
from src.plugins.chatbot.constants import ZHIPU_BASE_URL  # noqa: E402

SHORT_TERM = ROOT / "user_memory" / "short_term.json"


def _key(name: str) -> str:
    for line in io.open(ROOT / ".env.prod", encoding="utf-8"):
        if line.strip().startswith(name):
            return line.split("=", 1)[1].strip().strip('"')
    return ""


def user_messages() -> list:
    """所有用户侧发言（去掉"用户："前缀）。群聊的批量文本也算了，它们是真实的输入。"""
    data = json.loads(SHORT_TERM.read_text(encoding="utf-8"))
    out = []
    for uid, turns in data.items():
        if not isinstance(turns, list):
            continue
        for t in turns:
            txt = (t.get("text") if isinstance(t, dict) else str(t)) or ""
            if txt.startswith("用户："):
                msg = txt[3:].strip()
                if msg:
                    out.append(msg)
    return out


async def main():
    zhipu = AsyncOpenAI(api_key=_key("ZHIPU_API_KEY"), base_url=ZHIPU_BASE_URL)
    msgs = user_messages()
    print(f"真实用户发言 {len(msgs)} 条\n")

    empty = []
    tally = Counter()
    for i, m in enumerate(msgs, 1):
        qv = await embed_query(zhipu, m)
        hits = {
            "corpus": retrieve_corpus(m, qv),
            "voice_sample": retrieve_voice_samples(m, qv),
            "phrase": retrieve_phrases(m, qv),
            "preference": retrieve_preferences(m, qv),
            "core_story": retrieve_core_stories(m, qv),
        }
        got = [k for k, v in hits.items() if v]
        for k in got:
            tally[k] += 1
        if not got:
            empty.append(m)
        if i % 50 == 0:
            print(f"  …{i}/{len(msgs)}")

    print("\n" + "=" * 56)
    print(f"六路全空：{len(empty)}/{len(msgs)} = {len(empty) / len(msgs):.0%}"
          f"   ← 这些轮次她没有任何素材可依")
    print("\n各路命中条数（一条消息可命中多路）：")
    for k, v in tally.most_common():
        print(f"  {k:<14} {v:>3} 条 = {v / len(msgs):>4.0%}")
    print("\n全空的例子（前 25 条）：")
    for m in empty[:25]:
        print(f"  · {m[:60]}")
    (ROOT / "outputs" / "_coverage_empty.txt").write_text(
        "\n".join(empty), encoding="utf-8")
    print(f"\n全部 {len(empty)} 条已存 outputs/_coverage_empty.txt")


asyncio.run(main())
