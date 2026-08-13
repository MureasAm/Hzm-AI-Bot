"""离线预计算 persona/speech/voice_samples.json 中每条样本的 embedding 向量。

输出 persona/speech/voice_sample_vectors.json（自包含格式：连同 user/reply 一起存），
运行时 retrieval.py 直接读缓存，按当前用户消息情境检索相关的 few-shot 样本。

用法：
    python scripts/precompute_voice_sample_vectors.py

注意：改过 persona/speech/voice_samples.json 后需重跑本脚本。
"""
import json
import asyncio
import sys
from pathlib import Path
from openai import AsyncOpenAI

# 项目根目录（scripts/ 的上一级），所有路径基于它构建
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env.prod"
VOICE_FILE = PROJECT_ROOT / "persona" / "persona/speech/voice_samples.json"
OUTPUT_FILE = PROJECT_ROOT / "persona" / "speech" / "voice_sample_vectors.json"
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
    in_path = Path(input_file) if input_file else VOICE_FILE
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
    samples = data.get("samples", []) if isinstance(data, dict) else []

    if not samples:
        print(f"⚠️ {in_path.name} 中没有样本，无需预计算。")
        return

    print(f"🚀 正在向量化 {len(samples)} 条声音样本（只编码 user 情境字段）...")
    client = AsyncOpenAI(api_key=zhipu_key, base_url=ZHIPU_BASE_URL)

    result = []
    for i, s in enumerate(samples):
        # 只编码 user 字段（粉丝问：... 情境），与行为 trigger / corpus 编码策略一致
        text = s.get("user", "")
        if not text:
            continue
        try:
            resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=text)
            vector = resp.data[0].embedding
            result.append({
                "id": s.get("id", str(i)),
                "type": s.get("type", ""),
                "length": s.get("length", "short"),
                "user": s.get("user", ""),
                "reply": s.get("reply", ""),
                "vector": vector,
            })
            print(f"  [{i+1}/{len(samples)}] 完成: {text[:30]}...")
        except Exception as e:
            print(f"❌ 向量化失败（{text[:30]}...）: {e}")
            return

    output = {
        "model": EMBEDDING_MODEL,
        "dim": len(result[0]["vector"]) if result else 0,
        "samples": result,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已生成 {out_path}（{len(result)} 条样本）")


async def main():
    # Windows 控制台默认 GBK，强制 UTF-8 避免 emoji 打印报错
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    await run()


if __name__ == "__main__":
    asyncio.run(main())
