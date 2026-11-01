"""wiki_lookup 单元测试：摘要清理、缓存/负缓存、超时、实体过滤。"""
import pytest

from src.plugins.chatbot import wiki_lookup as w


@pytest.fixture(autouse=True)
def _tmp_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(w, "WIKI_CACHE_FILE", tmp_path / "wiki_cache.json")
    w.reset_cache_for_tests()
    yield
    w.reset_cache_for_tests()


def test_clean_extract_strips_headings_and_truncates():
    text = ("== 简介 ==\n\n小松绿（Viridis）是一位隶属于VirtuaReal（二十八期）的虚拟UP主。"
            + "很长的描述。" * 40)
    out = w.clean_extract(text)
    assert "==" not in out
    assert out.startswith("小松绿")
    assert len(out) <= w.WIKI_MAX_CHARS + 1
    assert out.endswith(("。", "…", "！", "？", "；"))


async def test_lookup_caches_hit_and_missing(monkeypatch):
    calls = []

    async def fake(api, term):
        calls.append(api)
        if term == "小松绿":
            return {"title": "小松绿", "extract": "二十八期虚拟UP主", "resolved": "小松绿", "api": api}
        return {"missing": True}

    monkeypatch.setattr(w, "_query_one", fake)

    d = await w.lookup("小松绿")
    assert d["title"] == "小松绿"
    n = len(calls)
    d2 = await w.lookup("小松绿")
    assert d2["title"] == "小松绿"
    assert len(calls) == n  # 命中缓存，不再请求

    assert await w.lookup("不存在的词") is None
    n = len(calls)
    assert await w.lookup("不存在的词") is None
    assert len(calls) == n  # 三个镜像都 missing → 负缓存


async def test_lookup_timeout_not_cached(monkeypatch):
    async def boom(api, term):
        raise TimeoutError("timeout")

    monkeypatch.setattr(w, "_query_one", boom)
    assert await w.lookup("超时词") is None
    assert w._cache_get("超时词") is None  # 超时不能写负缓存，下一轮还要能重试


async def test_context_skips_known_or_absent_entity(monkeypatch):
    async def fake_lookup(term, force=False):
        return {"title": "小松绿", "extract": "二十八期虚拟UP主", "resolved": "小松绿", "api": "x"}

    monkeypatch.setattr(w, "lookup", fake_lookup)

    out = await w.wiki_context_for("小松绿是谁", ["灰泽满", "小松绿"], {"灰泽满"})
    assert out.startswith("小松绿：")
    assert await w.wiki_context_for("小松绿是谁", ["小松绿"], {"小松绿"}) == ""
    assert await w.wiki_context_for("今天天气", ["小松绿"], set()) == ""


async def test_context_keeps_redirect_mapping(monkeypatch):
    async def fake_lookup(term, force=False):
        return {"title": "鹿目圆", "extract": "《魔法少女小圆》的主角", "resolved": "鹿目圆", "api": "x"}

    monkeypatch.setattr(w, "lookup", fake_lookup)
    out = await w.wiki_context_for("小圆是谁", ["小圆"], set())
    assert out.startswith("小圆（百科词条：鹿目圆）：")


async def test_force_bypasses_cache(monkeypatch):
    calls = []

    async def fake(api, term):
        calls.append(term)
        return {"title": term, "extract": "资料", "resolved": term, "api": api}

    monkeypatch.setattr(w, "_query_one", fake)
    await w.lookup("小松绿")
    n = len(calls)
    await w.lookup("小松绿", force=True)
    assert len(calls) > n


async def test_fresh_intent_passes_force(monkeypatch):
    seen = {}

    async def fake_lookup(term, force=False):
        seen["force"] = force
        return None

    monkeypatch.setattr(w, "lookup", fake_lookup)
    await w.wiki_context_for("小松绿最近怎么样", ["小松绿"], set())
    assert seen["force"] is True