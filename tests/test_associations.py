# -*- coding: utf-8 -*-
"""关联背景（作品 → 作者 → 喜好）的测试。

数据在 `persona/injection/07_terms/associations.json`：**只存实体别名 + 边**，
事实在 terms.json、态度在 preferences。这里守三件事：
1. 别名能命中、普通词（《真相》）不误触；
2. 沿边走到作者/代表作，并把 `has_preference` 指到的偏好 id 带出来；
3. 反向：天气/闲聊不产生任何关联内容。
"""
from src.plugins.chatbot import retrieval


class TestMatchEntities:
    def test_work_alias_matches(self):
        assert retrieval.match_entities("你看过空中杀人事件吗") == ["空中杀人现场"]

    def test_author_alias_matches(self):
        assert "东野圭吾" in retrieval.match_entities("东野最近咋样")

    def test_generic_word_is_not_an_entity(self):
        # 《真相》平时是普通词（"我想知道真相"），不能拿来子串匹配
        assert retrieval.match_entities("我想知道真相") == []

    def test_weather_chat_matches_nothing(self):
        assert retrieval.match_entities("今天好冷，还下雨") == []

    def test_probe_entity_is_canonicalized(self):
        # 探测给的是简称/全名都认；不认识的丢弃
        assert retrieval.match_entities("那本书", ["东野"]) == ["东野圭吾"]
        assert retrieval.match_entities("那本书", ["某本没收录的书"]) == []

    def test_seed_count_is_capped(self):
        found = retrieval.match_entities("东野 村上 周杰伦 袋鼠")
        assert len(found) <= 3


class TestResolveAssociations:
    def test_work_walks_to_author_works_and_preference(self):
        r = retrieval.resolve_associations("你看过空中杀人事件吗")
        assert "《空中杀人现场》是东野圭吾的作品。" in r["note"]
        assert "东野圭吾的代表作有" in r["note"] and "《白夜行》" in r["note"]
        assert r["preference_ids"] == ["author"]
        assert r["entities"][:2] == ["空中杀人现场", "东野圭吾"]

    def test_author_alias_gets_representative_works(self):
        r = retrieval.resolve_associations("东野")
        assert "代表作" in r["note"]
        assert r["preference_ids"] == ["author"]

    def test_unknown_work_uses_probe_author(self):
        """用户只说了"那本小说"——探测补出作者，仍然走作者的边。"""
        r = retrieval.resolve_associations("那本小说怎么样", ["某本没收录的书", "东野圭吾"])
        assert r["preference_ids"] == ["author"]
        assert "代表作" in r["note"]

    def test_animal_walks_to_place(self):
        r = retrieval.resolve_associations("你在澳洲见过袋鼠吗")
        assert "袋鼠生活在澳洲。" in r["note"]
        assert "袋鼠" in r["entities"]

    def test_no_entity_returns_empty(self):
        assert retrieval.resolve_associations("今天好冷") == {
            "entities": [], "note": "", "preference_ids": []}

    def test_note_respects_budget(self):
        r = retrieval.resolve_associations("东野 村上 周杰伦 袋鼠")
        assert len(r["note"]) <= 300


class TestPreferenceExtraIds:
    def test_extra_id_comes_first(self):
        items = retrieval.retrieve_preferences("东野", extra_ids=["author"])
        assert items and items[0]["id"] == "author"

    def test_without_edge_no_preference(self):
        # "东野"不是 preference 的关键词（关键词是"东野圭吾"）——关联边才是桥梁
        assert retrieval.retrieve_preferences("东野") == []

    def test_dedupes_keyword_and_edge(self):
        items = retrieval.retrieve_preferences("东野圭吾", extra_ids=["author"])
        assert [i["id"] for i in items].count("author") == 1

    def test_top_n_still_respected(self):
        items = retrieval.retrieve_preferences("西柚", extra_ids=["author", "singer"], top_n=1)
        assert [i["id"] for i in items] == ["author"]
