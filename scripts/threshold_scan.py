#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阈值体检：量出每条检索路的「噪声地板」，据此定阈值。

**为什么要这个**：阈值不能拍。实测（2026-09-25）——用五条**明确无关**的话去打三路，
   corpus       无关时最高 0.474 ｜ 阈值 0.48  ← 恰好卡在地板上，设计是对的
   voice_sample 无关时最高 0.610 ｜ 阈值 0.60  ← 地板压在阈值上，边缘
   phrase       无关时最高 0.575 ｜ 阈值 0.40  ← **地板远高于阈值，等于没有阈值**
   而且 phrase 的中位数就有 0.432 > 0.40 ⇒ **一半以上的措辞组在无关输入下都能过**。

**为什么会这样**：短文本的嵌入挤在向量空间中心——"毫不相关的两个短句"余弦也有
0.4~0.6；长文本（corpus 那些 50~120 字的陈述）分布散得开，无关的会掉到 0.2 以下。
所以**每路的地板不同，阈值必须按各自的地板定**。

用法：
    python scripts/threshold_scan.py            # 体检 + 给建议阈值
    python scripts/threshold_scan.py --margin 0.03   # 改安全余量

只读 + 调 embedding，不改任何文件。
"""
import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nonebot  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

nonebot.init()
from src.plugins.chatbot.rag import embed_query  # noqa: E402
from src.plugins.chatbot.retrieval import (  # noqa: E402
    load_vector_db, load_voice_sample_vectors, load_phrase_vectors,
    load_preference_vectors, load_core_story_vectors, cosine_similarity,
)
from src.plugins.chatbot.constants import (  # noqa: E402
    ZHIPU_BASE_URL, RAG_THRESHOLD, VOICE_SAMPLE_THRESHOLD, PHRASE_THRESHOLD,
    PREFERENCE_THRESHOLD, CORE_STORY_THRESHOLD,
)

ENV_FILE = Path(__file__).resolve().parent.parent / ".env.prod"

# 一批**明确和她的世界无关**的输入——用来量各路的噪声地板。
# 覆盖不同领域，避免只测到某一种"无关"。
NEG_QUERIES = [
    "1+1等于几", "帮我写个python脚本", "今天股市怎么样", "推荐一本物理书",
    "明天几点开会", "北京到上海高铁多久", "这道微积分题怎么解",
    "世界杯冠军是谁", "怎么给服务器装nginx", "糖尿病人能吃什么水果",
    "怎么申请护照", "Excel 怎么用 vlookup", "感冒吃什么药",
    "这道菜怎么做", "怎么给手机换电池",
]


def _key(name: str) -> str:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(name):
                return line.split("=", 1)[1].replace('"', "").strip()
    return ""


def _loaders():
    """每路：名字 → (取向量列表的函数, 现用阈值)。"""
    return [
        ("corpus", load_vector_db, RAG_THRESHOLD),
        ("voice_sample", load_voice_sample_vectors, VOICE_SAMPLE_THRESHOLD),
        ("phrase", load_phrase_vectors, PHRASE_THRESHOLD),
        ("preference", load_preference_vectors, PREFERENCE_THRESHOLD),
        ("core_story", load_core_story_vectors, CORE_STORY_THRESHOLD),
    ]


async def run(margin: float = 0.05):
    client = AsyncOpenAI(api_key=_key("ZHIPU_API_KEY"), base_url=ZHIPU_BASE_URL)
    print(f"负例输入 {len(NEG_QUERIES)} 条（明确和她的世界无关）\n")

    rows = []
    for name, loader, cur in _loaders():
        try:
            items = loader()
        except Exception as e:
            print(f"  ⚠️ {name} 向量加载失败: {e}")
            continue
        if not items:
            print(f"  ⚠️ {name} 没有向量，跳过")
            continue
        tops = []
        for q in NEG_QUERIES:
            v = await embed_query(client, q)
            tops.append(max(cosine_similarity(v, it.get("vector") or []) for it in items))
        tops.sort(reverse=True)
        n = len(tops)
        floor = tops[0]                              # 最保守：盖住最坏那条
        p90 = tops[max(0, int(n * 0.1))]
        med = tops[n // 2]
        rows.append({
            "path": name, "n": len(items), "floor": floor, "p90": p90, "median": med,
            "cur": cur,
            "suggest": round(floor + margin, 2),
            "ok": cur > floor,
        })

    print(f"{'路':<14}{'条数':>5}{'噪声地板':>10}{'p90':>8}{'中位':>8}{'现阈值':>8}  判定 / 建议")
    print("-" * 74)
    for r in rows:
        verdict = "✅ 阈值在地板之上" if r["ok"] else "❌ **地板 >= 阈值，会漏噪声**"
        sug = "" if r["ok"] else f"  → 建议 {r['suggest']}"
        print(f"{r['path']:<14}{r['n']:>5}{r['floor']:>10.3f}{r['p90']:>8.3f}"
              f"{r['median']:>8.3f}{r['cur']:>8.2f}  {verdict}{sug}")

    print(f"\n（地板 = {len(NEG_QUERIES)} 条无关输入下的最高分；建议值 = 地板 + {margin} 余量）")
    print("⚠️ 提阈值会连带挡掉弱的真信号——改完必须跑 retrieval_eval 看正例有没有被误杀。")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.05, help="地板之上的安全余量（默认 0.05）")
    args = ap.parse_args()
    asyncio.run(run(margin=args.margin))


if __name__ == "__main__":
    main()
