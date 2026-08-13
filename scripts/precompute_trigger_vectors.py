"""离线预计算 behavior/behaviors.json 中所有 trigger 的 embedding 向量。

输出 trigger_vectors.json（{trigger_text: vector}），运行时 persona.py 直接读缓存，
用户消息只对 query 调 1 次 embedding，避免每条消息对每个 trigger 都调 API。

用法：
    python scripts/precompute_trigger_vectors.py
"""
import json
import asyncio
import sys
from pathlib import Path
from openai import AsyncOpenAI

# 项目根目录（scripts/ 的上一级），所有路径基于它构建
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env.prod"
BEHAVIORS_FILE = PROJECT_ROOT / "persona" / "behavior/behaviors.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "trigger_vectors.json"
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
    in_path = Path(input_file) if input_file else BEHAVIORS_FILE
    out_path = Path(output_file) if output_file else OUTPUT_FILE

    zhipu_key = get_zhipu_key()
    if not zhipu_key:
        print("❌ 未能在 .env.prod 中找到 ZHIPU_API_KEY，请检查文件！")
        return

    if not in_path.exists():
        print(f"❌ 未找到 {in_path}")
        return

    with open(in_path, "r", encoding="utf-8") as f:
        behaviors = json.load(f)

    triggers = []
    for b in behaviors:
        t = b.get("trigger", "")
        if t:
            triggers.append(t)

    if not triggers:
        print("⚠️ behaviors 中没有 trigger，无需预计算。")
        return

    print(f"🚀 正在向量化 {len(triggers)} 个 trigger ...")
    client = AsyncOpenAI(api_key=zhipu_key, base_url=ZHIPU_BASE_URL)

    # 去重后批量计算
    unique_triggers = list(dict.fromkeys(triggers))
    result = {}
    for i, t in enumerate(unique_triggers):
        try:
            resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=t)
            result[t] = resp.data[0].embedding
            print(f"  [{i+1}/{len(unique_triggers)}] 完成: {t[:30]}...")
        except Exception as e:
            print(f"❌ 向量化失败（{t[:30]}...）: {e}")
            return

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已生成 {out_path}（{len(result)} 个 trigger）")


async def main():
    # Windows 控制台默认 GBK，强制 UTF-8 避免 emoji 打印报错
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    await run()


if __name__ == "__main__":
    asyncio.run(main())
