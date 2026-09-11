"""terms.json（名词库 / lorebook）的结构校验。

为什么值得单独测：`load_terms()`（persona.py:17-30）是**模块级永久缓存**，而且
JSON 写坏会**静默返回 []**——整个名词库在进程生命周期内消失，不报错、不崩溃，
表现只是"她突然什么都不懂了"。所以"能加载 + 字段合法"必须由测试守住。

另外守住"一个词只属于一个条目"：HIKAMI/你好神、久远澪/910 曾经各占两条，
注入哪条看用户打哪个词。同义的要并成一个条目、另一个进 aliases。
"""
import json
import re

import pytest

from src.plugins.chatbot import persona
from src.plugins.chatbot.constants import TERMS_FILE

CATEGORIES = {"world", "slang", "relation", "person", "family", "meme", "identity"}
PRIORITIES = {"always", "on-demand"}


def _matches(term: dict, text: str) -> bool:
    """复刻 core.build_terms_note 的命中判定：keyword/aliases 裸 substring + pattern 正则。"""
    keys = [term["keyword"]] + [str(a) for a in term.get("aliases", []) if a]
    if any(k in text for k in keys):
        return True
    pat = term.get("pattern")
    return bool(pat and re.search(pat, text))


@pytest.fixture(scope="module")
def terms():
    data = json.loads(TERMS_FILE.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and "terms" in data, "terms.json 顶层要是 {_readme, terms}"
    return data["terms"]


class TestTermsSchema:
    def test_file_loads_and_has_entries(self, terms):
        assert terms, "terms.json 不能为空——空库会让【灰泽满的世界】整段消失"

    def test_load_terms_matches_file(self, terms):
        # 守住"写坏了静默变 []"这个坑：文件里有几条，load_terms 就必须给几条
        assert len(persona.load_terms()) == len(terms)

    def test_required_fields_present(self, terms):
        for t in terms:
            for f in ("keyword", "category", "priority", "meaning"):
                assert t.get(f), f"{t.get('keyword', t)} 缺字段 {f}"

    def test_category_in_enum(self, terms):
        for t in terms:
            assert t["category"] in CATEGORIES, \
                f"{t['keyword']} 的 category 未知: {t['category']}（新增分类要同步改本测试）"

    def test_priority_in_enum(self, terms):
        for t in terms:
            assert t["priority"] in PRIORITIES, f"{t['keyword']} 的 priority 非法: {t['priority']}"

    def test_aliases_shape(self, terms):
        for t in terms:
            if "aliases" in t:
                assert isinstance(t["aliases"], list), f"{t['keyword']} 的 aliases 要是 list"
                for a in t["aliases"]:
                    assert isinstance(a, str) and a.strip(), f"{t['keyword']} 的 aliases 有空项"

    def test_pattern_compiles(self, terms):
        for t in terms:
            if t.get("pattern"):
                re.compile(t["pattern"])  # 坏正则不该进运行时

    def test_usage_triggers_shape(self, terms):
        for t in terms:
            if "usage_triggers" in t:
                assert isinstance(t["usage_triggers"], list), f"{t['keyword']} 的 usage_triggers 要是 list"


class TestCohortPatterns:
    """期数写法（`27期` / `VR27` / `277` / `二十七期`）要落到正确的条目上。

    这里**故意不用 aliases 而用 pattern**：aliases 是裸 substring，`27期` 会命中 `127期`、
    `277` 会命中 `12777`。pattern 带 `(?<!\\d)...(?!\\d)` 边界才安全。
    期数简写规律：**`<期数> + 7`**（277=27期、287=28期、207=20期），通用于所有期数。
    """

    def _hit(self, text):
        return {t["keyword"] for t in persona.load_terms() if _matches(t, text)}

    @pytest.mark.parametrize("text", ["27期", "VR27期", "VR27", "277", "二十七期", "27期生", "vr27期"])
    def test_tongqi_forms(self, text):
        assert "同期" in self._hit(text), f"{text!r} 该命中同期"

    @pytest.mark.parametrize("text", ["28期", "287", "VR28", "二十八期", "29期", "297"])
    def test_houbei_forms(self, text):
        assert "后辈" in self._hit(text), f"{text!r} 该命中后辈"

    @pytest.mark.parametrize("text", ["20期", "207", "24期", "247", "26期", "267", "VR25", "二十期", "二十六期"])
    def test_qianbei_forms(self, text):
        assert "前辈" in self._hit(text), f"{text!r} 该命中前辈"

    @pytest.mark.parametrize("text", ["127期", "12777", "2777", "127", "2807"])
    def test_no_false_positive_on_longer_numbers(self, text):
        # 数字边界：`127期` 不该被当成 27 期；`12777` 不该被当成 277
        assert not (self._hit(text) & {"同期", "前辈", "后辈"}), f"{text!r} 误命中期数词条"

    def test_three_cohorts_do_not_overlap(self):
        assert self._hit("277") == {"同期"}
        assert self._hit("287") == {"后辈"}
        assert self._hit("207") == {"前辈"}


class TestNoRomajiInInjectedText:
    """罗马音只能待在 `aliases`（匹配用户输入），**不能出现在任何注入给模型看的文本里**。

    踩坑：meaning 曾写成"四时小路Komichi、小松绿Viridis、羽啾chu2u、枝堇Sumire"，
    结果被问"后辈都有谁"时她照抄了罗马音 `Komichi`——可平时该说"四时小路/小路/路神"。

    为什么放 aliases 就安全：`build_terms_note` 只注入 meaning/reaction/usage，
    aliases 只参与"用户消息里有没有这个词"的匹配，从不进提示词。
    """

    # 只列这几位已知会踩坑的罗马音，不做通用"禁英文"检查（VR/VirtuaReal/wqndyd 都是合法用词）
    ROMAJI = ("Komichi", "Viridis", "chu2u", "Sumire", "Liko", "Kloa", "Ameki", "Harei")

    def test_no_romaji_in_term_injected_fields(self, terms):
        for t in terms:
            for field in ("meaning", "reaction", "usage"):
                text = t.get(field) or ""
                for r in self.ROMAJI:
                    assert r not in text, \
                        f"{t['keyword']} 的 {field} 里出现罗马音 {r!r}——她会照抄（罗马音请放 aliases）"

    def test_no_romaji_in_behavior_injected_fields(self):
        _, _, behaviors = persona.load_persona_rules()
        for b in behaviors:
            fields = [b.get("trigger") or "", b.get("response") or ""]
            for s in b.get("samples", []):
                fields += [s.get("user") or "", s.get("reply") or ""]
            for text in fields:
                for r in self.ROMAJI:
                    assert r not in text, \
                        f"行为「{b.get('name')}」的注入文本里有罗马音 {r!r}——她会照抄（罗马音请放 terms 的 aliases）"

    def test_romaji_still_kept_in_aliases(self, terms):
        # 反向确认：aliases 本就是给"用户打罗马音"匹配用的，别一起删了
        aliases = [a for t in terms for a in t.get("aliases", [])]
        assert "Komichi" in aliases, "用户可能打 Komichi，这个别名要留在 aliases 里"


class TestTermsNoAmbiguity:
    """一个词只能属于一个条目，否则注入哪条看运气。"""

    def test_no_duplicate_keyword(self, terms):
        kws = [t["keyword"] for t in terms]
        dupes = sorted({k for k in kws if kws.count(k) > 1})
        assert not dupes, f"keyword 重复: {dupes}——同义的要并成一个条目、另一个进 aliases"

    def test_alias_not_another_keyword(self, terms):
        kws = {t["keyword"] for t in terms}
        for t in terms:
            for a in t.get("aliases", []):
                assert a not in kws, \
                    f"{t['keyword']} 的别名 {a!r} 同时是另一个条目的 keyword（同一条会双重注入）"
