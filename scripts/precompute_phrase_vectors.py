"""⚠️ **已停用**（2026-09-26）——措辞不再走向量检索，本脚本不会被运行时消费。

为什么停：措辞组的 trigger 是 6~13 字的**类别标签**（"被夸奖、被称赞时"），不是句子，
拿它算余弦 = 用检索工具干分类的活。实测噪声地板 0.575 / 正例 0.573，**中位数 0.461 都过阈值
→ 100% 开火，等于没有判据**。现在改由 L3 的 LLM 分类给组 id（见 routing.classify_l3），
运行时读源文件 persona/speech/phrases.json（`retrieval.load_phrase_groups`）。

脚本与产出的 phrase_vectors.json 先留着（以防回退），但**改了 phrases.json 不需要再跑它**。

原说明：离线预计算 persona/speech/phrases.json 中每个措辞组的 trigger 向量，
输出 persona/speech/phrase_vectors.json，运行时按用户消息情境检索相关措辞组。
"""
import json
import asyncio
import sys
from pathlib import Path
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env.prod"
PHRASES_FILE = PROJECT_ROOT / "persona" / "persona/speech/phrases.json"
OUTPUT_FILE = PROJECT_ROOT / "persona" / "speech" / "phrase_vectors.json"
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
