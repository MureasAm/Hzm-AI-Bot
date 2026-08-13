"""直播记忆 RAG：余弦相似度 + 语义检索。"""
import json

from .constants import VECTOR_FILE, EMBEDDING_MODEL


async def embed_query(zhipu_client, text: str):
    """对用户消息计算一次 embedding（供行为匹配与 RAG 共用）。失败/空输入返回 None。"""
    if not text or not str(text).strip():
        print("⚠️ 空 query 跳过 embedding")
        return None
    try:
        resp = await zhipu_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return resp.data[0].embedding
    except Exception as e:
        print(f"⚠️ 获取 query embedding 失败: {e}")
        return None


def cosine_similarity(v1, v2) -> float:
    """计算两个向量的余弦相似度。任一向量为零向量时返回 0.0。"""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = sum(a * a for a in v1) ** 0.5
    norm_v2 = sum(b * b for b in v2) ** 0.5
    if not norm_v1 or not norm_v2:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)


def load_vector_db():
    """加载直播记忆向量库；文件缺失或格式错误时返回空列表。"""
    if not VECTOR_FILE.exists():
        return []
    try:
        with open(VECTOR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []
