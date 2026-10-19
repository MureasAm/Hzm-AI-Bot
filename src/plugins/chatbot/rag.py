"""直播记忆 RAG：余弦相似度 + 语义检索。"""
import json
import operator

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


def _norm(v) -> float:
    return sum(a * a for a in v) ** 0.5


def cosine_similarity(v1, v2) -> float:
    """计算两个向量的余弦相似度。任一向量为零向量时返回 0.0。"""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = _norm(v1)
    norm_v2 = _norm(v2)
    if not norm_v1 or not norm_v2:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)


def cosine_scores(query_vector, db: list) -> list:
    """query 对整库逐条打分，结果与逐条 `cosine_similarity(query_vector, it["vector"])` **逐位相同**。

    省掉的是重复劳动：query 的范数只算一次；db 是 `load_vector_db()` 缓存的那份时，
    库向量的范数直接用加载时算好的（测试 monkeypatch 进来的假 db 就现算）。
    """
    norm_q = _norm(query_vector)
    if _vector_db_cache is not None and db is _vector_db_cache[1]:
        norms = _vector_db_cache[2]
    else:
        norms = [_norm(it["vector"]) for it in db]
    scores = []
    for it, norm_v in zip(db, norms):
        if not norm_q or not norm_v:
            scores.append(0.0)
            continue
        scores.append(sum(map(operator.mul, query_vector, it["vector"])) / (norm_q * norm_v))
    return scores


# (文件签名, 向量库, 每条的范数)。16MB 的 JSON 每轮重读要 ~0.1s，而且是同步卡事件循环，
# 所以按 (mtime, size) 缓存——文件一改签名就变，下次调用自动重读，"改完不重启也生效"的行为不变。
_vector_db_cache = None


def load_vector_db():
    """加载直播记忆向量库；文件缺失或格式错误时返回空列表。"""
    global _vector_db_cache
    try:
        st = VECTOR_FILE.stat()
    except OSError:
        return []
    sig = (st.st_mtime_ns, st.st_size)
    if _vector_db_cache is not None and _vector_db_cache[0] == sig:
        return _vector_db_cache[1]
    try:
        with open(VECTOR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    _vector_db_cache = (sig, data, [_norm(it["vector"]) for it in data])
    return data
