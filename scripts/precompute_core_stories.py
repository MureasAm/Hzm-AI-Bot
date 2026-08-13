#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线预计算 persona/world/core_stories.json 的核心记忆向量。

输出 persona/world/core_story_vectors.json（stories 数组：id/category/text/vector）。
运行时 retrieval.py 读缓存，按当前消息语义检索命中的核心记忆，注入【她的核心记忆】。

用法：
    python scripts/precompute_core_stories.py
    python scripts/run_tool.py precompute core-stories

注意：改过 persona/world/core_stories.json 后需重跑本脚本。
"""
import asyncio
import json
import sys
from pathlib import Path

from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env.prod"
STORIES_FILE = PROJECT_ROOT / "persona" / "persona/world/core_stories.json"
OUTPUT_FILE = PROJECT_ROOT / "persona" / "world" / "core_story_vectors.json"
EMBEDDING_MODEL = "embedding-3"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


def get_zhipu_key():
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("ZHIPU_API_KEY"):
                    return line.split("=")[1].replace('"', '').strip()
    return None


async def run(input_file: str | None = None, output_file: str | None = None):
    in_path = Path(input_file) if input_file else STORIES_FILE
    out_path = Path(output_file) if output_file else OUTPUT_FILE

    zhipu_key = get_zhipu_key()
    if not zhipu_key:
        print("❌ 未能在 .env.prod 中找到 ZHIPU_API_KEY，请检查文件！")
        return
    if not in_path.exists():
        print(f"❌ 未找到 {in_path}")
        return

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stories = data.get("stories", []) if isinstance(data, dict) else []
    stories = [s for s in stories if (s.get("text") or "").strip()]

    if not stories:
        print("⚠️ core_stories.json 里没有有内容的条目，跳过。")
        return

    print(f"🚀 正在向量化 {len(stories)} 条核心记忆...")
    client = AsyncOpenAI(api_key=zhipu_key, base_url=ZHIPU_BASE_URL)

    result = []
    for i, s in enumerate(stories):
        try:
            resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=s["text"])
            result.append({
                "id": s.get("id", str(i)),
                "category": s.get("category", ""),
                "text": s["text"],
                "vector": resp.data[0].embedding,
            })
            print(f"  [{len(result)}/{len(stories)}] {s.get('category', '')}: {s['text'][:20]}...")
        except Exception as e:
            print(f"❌ 向量化失败（{s.get('category', '')}）: {e}")
            return

    output = {"model": EMBEDDING_MODEL, "dim": len(result[0]["vector"]) if result else 0, "stories": result}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已生成 {out_path}（{len(result)} 条核心记忆）")


async def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    await run()


if __name__ == "__main__":
    asyncio.run(main())
