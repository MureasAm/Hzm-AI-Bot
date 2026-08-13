"""离线预计算 persona/speech/phrases.json 中每个措辞组的 trigger 向量。

输出 data/phrase_vectors.json（自包含：连同 phrases 一起存），
运行时 retrieval.py 按当前用户消息情境检索相关措辞组。

用法：
    python scripts/precompute_phrase_vectors.py

注意：改过 persona/speech/phrases.json 后需重跑本脚本。
"""
import json
import asyncio
import sys
from pathlib import Path
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env.prod"
PHRASES_FILE = PROJECT_ROOT / "persona" / "persona/speech/phrases.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "phrase_vectors.json"
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
    in_path = Path(input_file) if input_file else PHRASES_FILE
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
    groups = data.get("phrase_groups", []) if isinstance(data, dict) else []

    if not groups:
        print(f"⚠️ {in_path.name} 中没有措辞组，无需预计算。")
        return

    print(f"🚀 正在向量化 {len(groups)} 个措辞组的 trigger ...")
    client = AsyncOpenAI(api_key=zhipu_key, base_url=ZHIPU_BASE_URL)

    result = []
    for i, g in enumerate(groups):
        trigger = g.get("trigger", "")
        if not trigger:
            continue
        try:
            resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=trigger)
            vector = resp.data[0].embedding
            result.append({
                "id": g.get("id", str(i)),
                "meaning": g.get("meaning", ""),
                "trigger": trigger,
                "phrases": g.get("phrases", []),
                "usage": g.get("usage", ""),
                "vector": vector,
            })
            print(f"  [{i+1}/{len(groups)}] 完成: {g.get('id','')}（{trigger[:25]}...）")
        except Exception as e:
            print(f"❌ 向量化失败（{trigger[:25]}...）: {e}")
            return

    output = {
        "model": EMBEDDING_MODEL,
        "dim": len(result[0]["vector"]) if result else 0,
        "phrase_groups": result,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已生成 {out_path}（{len(result)} 个措辞组）")


async def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    await run()


if __name__ == "__main__":
    asyncio.run(main())
