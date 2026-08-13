#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线预计算 persona/world/preferences.json 的偏好向量（第 5 路检索）。

输出 persona/world/preference_vectors.json（entries 数组：id/category/text/vector）。
运行时 retrieval.py 读缓存，按当前消息语义检索命中的偏好条目，注入【灰泽满的偏好】。

用法：
    python scripts/precompute_preference_vectors.py
    python scripts/run_tool.py precompute preferences

注意：改过 persona/world/preferences.json 后需重跑本脚本。
"""
import asyncio
import json
import sys
from pathlib import Path

from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env.prod"
PREF_FILE = PROJECT_ROOT / "persona" / "persona/world/preferences.json"
OUTPUT_FILE = PROJECT_ROOT / "persona" / "world" / "preference_vectors.json"
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
    """参数化入口（供 run_tool 调用）。空参数回落模块常量。"""
    in_path = Path(input_file) if input_file else PREF_FILE
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
    entries = data.get("entries", []) if isinstance(data, dict) else []
    entries = [e for e in entries if (e.get("text") or "").strip()]

    if not entries:
        print("⚠️ preferences.json 里没有有内容的条目，跳过。")
        return

    print(f"🚀 正在向量化 {len(entries)} 条偏好（只编码有 text 的条目）...")
    client = AsyncOpenAI(api_key=zhipu_key, base_url=ZHIPU_BASE_URL)

    result = []
    for i, e in enumerate(entries):
        try:
            resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=e["text"])
            result.append({
                "id": e.get("id", str(i)),
                "category": e.get("category", ""),
                "text": e["text"],
                "vector": resp.data[0].embedding,
            })
            print(f"  [{len(result)}/{len(entries)}] {e.get('category', '')}: {e['text'][:24]}...")
        except Exception as e:
            print(f"❌ 向量化失败（{e.get('category', '')}）: {e}")
            return

    output = {"model": EMBEDDING_MODEL, "dim": len(result[0]["vector"]) if result else 0, "entries": result}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已生成 {out_path}（{len(result)} 条偏好）")


async def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    await run()


if __name__ == "__main__":
    asyncio.run(main())
