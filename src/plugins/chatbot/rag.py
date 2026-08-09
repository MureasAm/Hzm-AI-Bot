"""直播记忆 RAG：余弦相似度 + 语义检索。"""
import json

from .constants import VECTOR_FILE, RAG_THRESHOLD, RAG_TOP_K, EMBEDDING_MODEL


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


async def retrieve_semantic_contexts(user_query: str, query_vector, top_k: int = RAG_TOP_K) -> str:
    """基于 query 向量检索最相关的直播记忆片段。

    query_vector 由调用方传入（已在 persona 匹配中计算过，避免重复 embedding）。
    返回拼接好的文本，没有超过阈值的片段时返回空串。
    """
    vector_db = load_vector_db()
    if not vector_db or query_vector is None:
        return ""

    scored = []
    for item in vector_db:
        sim = cosine_similarity(query_vector, item["vector"])
        scored.append((sim, item["text"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    valid = [text for score, text in scored[:top_k] if score > RAG_THRESHOLD]
    if not valid:
        return ""
    return "\n".join([f"- {text}" for text in valid])
