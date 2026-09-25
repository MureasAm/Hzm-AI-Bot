# -*- coding: utf-8 -*-
"""表情包挑选的测试。

这条路和群聊接话门**失败时的方向相反**：
- 接话门判不出来 → 放行（该说时不说更难发现）
- 表情包判不出来 → **不发**（发错比不发难看得多）
所以两边各有自己的兜底测试。
"""
import json
from types import SimpleNamespace

import pytest

from src.plugins.chatbot import stickers
from src.plugins.chatbot.stickers import load_stickers, pick_sticker, sticker_enabled


class _StubClient:
    def __init__(self, content=None, exc=None):
        self._content, self._exc = content, exc
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        if self._exc:
            raise self._exc
        return SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=self._content), finish_reason="stop")
        ])


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("STICKER", raising=False)
    monkeypatch.setattr(stickers, "_stickers_cache", None)


@pytest.fixture
def two_stickers(tmp_path, monkeypatch):
    """两张假表情（文件真的建出来，因为 load_stickers 会检查存在性）。"""
    d = tmp_path / "assets" / "stickers"
    d.mkdir(parents=True)
    items = []
    for sid in ("st_aaa", "st_bbb"):
        f = d / f"{sid}.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n")
        items.append({"id": sid, "file": f"assets/stickers/{sid}.png",
                      "desc": f"描述{sid}", "tags": ["无语"], "use_when": "无语时"})
    f = tmp_path / "stickers.json"
    f.write_text(json.dumps({"stickers": items}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(stickers, "STICKER_FILE", f)
    monkeypatch.setattr(stickers, "PROJECT_ROOT", tmp_path)
    return items


class TestToggle:
    def test_enabled_by_default(self):
        assert sticker_enabled() is True

    def test_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("STICKER", "0")
        assert sticker_enabled() is False


class TestLoad:
    def test_real_catalog_loads(self):
        # 仓库里那批是打好标的，应该能读出来且每条的文件都在
        items = load_stickers()
        assert len(items) >= 20
        for s in items:
            assert s["id"] and s["file"] and s.get("desc")

    def test_missing_file_entry_is_dropped(self, tmp_path, monkeypatch):
        # 标注是离线生成的，图可能被手动删过——文件没了的条目要剔掉，否则发送必失败
        f = tmp_path / "s.json"
        f.write_text(json.dumps({"stickers": [
            {"id": "gone", "file": "assets/stickers/nope.png", "desc": "x", "tags": [], "use_when": ""},
        ]}), encoding="utf-8")
        monkeypatch.setattr(stickers, "STICKER_FILE", f)
        monkeypatch.setattr(stickers, "PROJECT_ROOT", tmp_path)
        assert load_stickers() == []

    def test_broken_json_is_empty(self, tmp_path, monkeypatch):
        f = tmp_path / "s.json"
        f.write_text("{ 不是 json", encoding="utf-8")
        monkeypatch.setattr(stickers, "STICKER_FILE", f)
        assert load_stickers() == []


class TestPick:
    @pytest.mark.asyncio
    async def test_picks_a_real_id(self, two_stickers):
        c = _StubClient(content='{"id": "st_bbb", "why": "无语"}')
        hit = await pick_sticker(c, "灰泽满无语了")
        assert hit and hit["id"] == "st_bbb"

    @pytest.mark.asyncio
    async def test_null_means_no_sticker(self, two_stickers):
        c = _StubClient(content='{"id": null, "why": "不太对应"}')
        assert await pick_sticker(c, "今天天气不错") is None

    @pytest.mark.asyncio
    async def test_invented_id_is_rejected(self, two_stickers):
        # 模型编一个不存在的 id —— 必须拦住，否则发送时路径不存在
        c = _StubClient(content='{"id": "st_zzz", "why": "编的"}')
        assert await pick_sticker(c, "随便") is None

    @pytest.mark.asyncio
    async def test_code_fence_tolerated(self, two_stickers):
        c = _StubClient(content='```json\n{"id": "st_aaa"}\n```')
        hit = await pick_sticker(c, "无语")
        assert hit and hit["id"] == "st_aaa"

    @pytest.mark.asyncio
    async def test_error_means_no_sticker(self, two_stickers):
        # 与接话门相反：判不出来就不发
        c = _StubClient(exc=RuntimeError("boom"))
        assert await pick_sticker(c, "随便") is None

    @pytest.mark.asyncio
    async def test_disabled_never_picks(self, two_stickers, monkeypatch):
        monkeypatch.setenv("STICKER", "0")
        c = _StubClient(content='{"id": "st_aaa"}')
        assert await pick_sticker(c, "无语") is None

    @pytest.mark.asyncio
    async def test_empty_reply_never_picks(self, two_stickers):
        c = _StubClient(content='{"id": "st_aaa"}')
        assert await pick_sticker(c, "   ") is None

    @pytest.mark.asyncio
    async def test_avoid_ids_are_enforced_in_code(self, two_stickers):
        """提示词里说了"别再选"，但**模型不一定听**（实测 2 次里 1 次照选）。

        所以还要代码拦一道：宁可这轮不发，也不连着甩同一张。
        """
        c = _StubClient(content='{"id": "st_aaa", "why": "又选它"}')
        assert await pick_sticker(c, "无语", avoid_ids=["st_aaa"]) is None

    @pytest.mark.asyncio
    async def test_avoid_ids_lets_other_stickers_through(self, two_stickers):
        c = _StubClient(content='{"id": "st_bbb"}')
        hit = await pick_sticker(c, "无语", avoid_ids=["st_aaa"])
        assert hit and hit["id"] == "st_bbb"

    @pytest.mark.asyncio
    async def test_avoid_ids_reaches_prompt(self, two_stickers):
        seen = {}

        class _Capture(_StubClient):
            async def create(self, **kw):
                seen["prompt"] = kw["messages"][0]["content"]
                return await super().create(**kw)

        await pick_sticker(_Capture(content='{"id": null}'), "无语", avoid_ids=["st_aaa"])
        assert "st_aaa" in seen["prompt"]
        assert "别再选" in seen["prompt"]
