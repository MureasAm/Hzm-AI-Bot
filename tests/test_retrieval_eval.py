"""检索评测复用共同入口，并显式报告不可用检查；全程不联网。"""
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import retrieval_eval as evaluation
from src.plugins.chatbot.retrieval import RetrievalItem


def _got(**changes):
    got = {name: [] for name in ("corpus", "behavior", "preference", "core_story", "terms", "associations", "facts")}
    got.update(changes)
    return got


@pytest.mark.parametrize("ids,expected", [([], True), (["food"], True),
                                           (["routine"], False), (["food", "routine"], False)])
def test_allowed_ids_can_be_empty_but_cannot_expand_to_unrelated_material(ids, expected):
    case = {"id": "optional-food", "query": "吃月饼了吗", "expect": {
        "preference": {"allowed": ["food"]}}}
    got = _got(preference=[{"id": item_id} for item_id in ids])
    assert evaluation.check_case(case, got)["passed"] is expected


@pytest.mark.parametrize("condition,ids,expected", [
    ({"allowed": ["food"], "must_not": ["food"]}, ["food"], False),
    ({"allowed": ["food"], "should": ["food"]}, [], False),
    ({"allowed": ["food"], "should": ["food"]}, ["food"], True),
    ({"allowed": [], "should_not_hit": True}, [], True),
])
def test_allowed_and_other_conditions_all_apply(condition, ids, expected):
    case = {"id": "combined", "query": "饮食背景", "expect": {"preference": condition}}
    assert evaluation.check_case(case, _got(preference=[{"id": x} for x in ids]))["passed"] is expected


def test_required_id_does_not_bypass_a_text_condition():
    case = {"id": "both", "query": "记录内容", "expect": {
        "corpus": {"should": ["77"], "contains": ["袋鼠蛋"]}}}
    wrong = RetrievalItem("corpus", "77", 1, text="与目标事件无关")
    assert not evaluation.check_case(case, _got(corpus=[wrong]))["passed"]


@pytest.mark.parametrize("facts,expected", [
    ([{"id": "terms:袋鼠", "text": "袋鼠是动物"}], False),
    ([{"id": "terms:袋鼠", "text": "袋鼠是动物"},
      {"id": "association:澳洲", "text": "它生活在澳洲"}], True),
    ([], False),
])
def test_facts_contains_requires_every_piece_across_approved_materials(facts, expected):
    case = {"id": "facts", "query": "你在澳洲见过袋鼠吗", "expect": {
        "facts": {"contains": ["澳洲", "袋鼠"]}}}
    assert evaluation.check_case(case, _got(facts=facts))["passed"] is expected


@pytest.mark.parametrize("condition", [
    {"allowed": "food"}, {"allowed": [None]}, {"allowed": [""]},
    {"should": []}, {"contains": []}, {"should_fire": False},
])
def test_invalid_optional_or_empty_positive_conditions_do_not_count_as_checks(condition):
    case = {"id": "invalid", "query": "未有效标注", "expect": {"preference": condition}}
    result = evaluation.check_case(case, _got())
    assert not result["passed"] and result["checks"][0]["missing"]


def test_missing_facts_are_not_treated_as_a_valid_empty_result():
    case = {"id": "missing-facts", "query": "作者是谁", "expect": {
        "facts": {"contains": ["东野圭吾"]}}}
    result = evaluation.check_case(case, {})
    assert not result["passed"] and result["checks"][0]["missing"]


async def test_snapshot_facts_use_only_final_approved_information_and_group_context(monkeypatch):
    seen = {}
    selected = RetrievalItem("corpus", "selected", 1, text="已进入预算的事件")
    dropped = RetrievalItem("corpus", "dropped", .5, text="预算丢掉的事件")

    async def gather(*args, **kwargs):
        seen["context"] = kwargs["context"]
        return {"fused_items": [selected], "corpus_items": [selected, dropped],
                "preference_items": [{"id": "author", "text": "她以前喜欢过东野圭吾"}],
                "core_stories": [], "approved_terms": {"袋鼠"}, "denied_terms": set(),
                "association_entities": ["澳洲"], "association_context": "澳洲的已批准背景",
                "embedding_available": True, "l3": {"classification_ok": True},
                "selection_trace": {"available": True, "candidates": {"secret": ["未批准秘密资料"]}}}

    def terms(*args, **kwargs):
        kwargs["hit_out"].append("袋鼠")
        return "袋鼠：澳洲常见动物"

    monkeypatch.setattr(evaluation, "gather_retrieval", gather)
    monkeypatch.setattr(evaluation, "build_terms_note", terms)
    got, l3 = await evaluation.snapshot("群游戏动作", object(), object(), [],
                                        history="群友在玩游戏", is_group=True, addressed=False)
    text = "\n".join(item["text"] for item in got["facts"])
    assert all(value in text for value in ("已进入预算的事件", "东野圭吾", "袋鼠", "澳洲的已批准背景"))
    assert "预算丢掉的事件" not in text and "未批准秘密资料" not in text
    assert seen["context"].is_group and not seen["context"].addressed
    assert any("facts" in line for line in evaluation.format_snapshot(got, l3))
    assert "facts" in evaluation._summarize_got(got)


async def test_behavior_samples_do_not_become_factual_evidence(monkeypatch):
    async def gather(*args, **kwargs):
        return {"fused_items": [RetrievalItem("behavior", "style", 1, text="风格例句：她见过活袋鼠")],
                "preference_items": [], "core_stories": [], "approved_terms": set(),
                "embedding_available": True, "l3": {}, "selection_trace": {"available": True}}

    monkeypatch.setattr(evaluation, "gather_retrieval", gather)
    monkeypatch.setattr(evaluation, "build_terms_note", lambda *args, **kwargs: "")
    got, _ = await evaluation.snapshot("见过袋鼠吗", object(), object(), [])
    assert got["behavior"]
    assert got["facts"] == []


async def test_missing_term_approval_cannot_supply_facts_from_keyword_fallback(monkeypatch):
    from src.plugins.chatbot import core

    async def gather(*args, **kwargs):
        return {"fused_items": [], "preference_items": [], "core_stories": [],
                "embedding_available": True, "l3": {}, "selection_trace": {"available": False}}

    monkeypatch.setattr(evaluation, "gather_retrieval", gather)
    monkeypatch.setattr(core, "load_terms", lambda: [{"keyword": "棉花糖", "meaning": "匿名提问"}])
    got, l3 = await evaluation.snapshot("棉花糖", object(), object(), [])
    assert got["terms"] == [] and got["facts"] == []
    assert l3["_degraded"]


async def test_optional_source_empty_does_not_pass_a_degraded_run(offline, monkeypatch):
    async def snapshot(*args, **kwargs):
        return _got(), {"_degraded": True, "_degraded_reasons": ["selection_unavailable"]}

    monkeypatch.setattr(evaluation, "snapshot", snapshot)
    case = {"id": "optional-degraded", "query": "没有确认结果", "expect": {
        "preference": {"allowed": ["food"]}}}
    assert await evaluation.run([case], verbose=False) == 1
    report = json.loads((offline / "eval_report.json").read_text(encoding="utf-8"))
    assert report["case_degraded"] == 1 and report["case_passed"] == 0
    assert report["results"][0]["retrieval_passed"] is True


@pytest.mark.parametrize("case_id,allowed", [
    ("pref4_mooncake", "food"), ("pref5_mooncake2", "food"),
    ("pref8_jellyfish", "jellyfish"), ("pref9_singer", "singer"),
    ("n7_early_sleep", "routine"),
])
def test_calibrated_boundaries_allow_relevant_background_without_requiring_it(case_id, allowed):
    case = next(case for case in evaluation.load_cases() if case["id"] == case_id)
    assert case["expect"]["preference"] == {"allowed": [allowed]}
    if case_id == "pref8_jellyfish":
        assert case["query"] == "测试群友：/抓水母"
        assert case["is_group"] and case["addressed"] is False and case["history"]
    if case_id == "n7_early_sleep":
        assert case["expect"]["corpus"] == {"should_not_hit": True}


def test_canonical_information_is_required_without_mandating_duplicate_associations():
    cases = {case["id"]: case for case in evaluation.load_cases()}
    author = cases["as2_author_alias"]["expect"]
    assert author["preference"]["should"] == ["author"]
    assert author["facts"]["contains"] == ["东野圭吾"]
    assert "should" not in author["associations"]
    kangaroo = cases["as3_kangaroo"]["expect"]
    assert "facts" not in kangaroo
    assert "77" in kangaroo["corpus"]["must_not"]


@pytest.mark.parametrize("terms,facts", [
    ([], []),
    ([{"id": "澳洲"}], [{"id": "terms", "text": "澳洲：澳大利亚"}]),
])
def test_personal_kangaroo_experience_does_not_require_unrelated_background(terms, facts):
    case = next(case for case in evaluation.load_cases() if case["id"] == "as3_kangaroo")
    assert evaluation.check_case(case, _got(terms=terms, facts=facts))["passed"]


@pytest.mark.parametrize("facts,corpus,expected", [
    ([], [], False),
    ([{"id": "terms", "text": "澳洲：澳大利亚"}], [], False),
    ([{"id": "terms", "text": "袋鼠是澳洲常见动物"}], [], True),
    ([{"id": "terms", "text": "袋鼠是澳洲常见动物"}],
     [RetrievalItem("corpus", "77", 1, text="袋鼠蛋很臭")], False),
])
def test_kangaroo_definition_requires_real_information_without_duplicate_route_requirement(facts, corpus, expected):
    case = next(case for case in evaluation.load_cases() if case["id"] == "cal_kangaroo_definition")
    assert evaluation.check_case(case, _got(facts=facts, corpus=corpus))["passed"] is expected


@pytest.mark.parametrize("case_id,source,item_id", [
    ("cal_jellyfish_preference", "preference", "jellyfish"),
    ("cal_personal_routine", "preference", "routine"),
    ("cal_personal_singer_recommendation", "preference", "singer"),
    ("cal_question_box_name", "terms", "棉花糖"),
])
def test_new_paired_positives_preserve_strong_required_material(case_id, source, item_id):
    case = next(case for case in evaluation.load_cases() if case["id"] == case_id)
    assert item_id in case["expect"][source]["should"]


def test_anonymous_read_status_and_kangaroo_egg_have_their_own_controls():
    cases = {case["id"]: case for case in evaluation.load_cases()}
    read_status = cases["cx08_anaphoric_question_box"]
    assert read_status["query"] == "那这个你看到了吗？"
    assert read_status["expect"]["terms"] == {"allowed": ["棉花糖"]}
    assert read_status["expect"]["core_story"] == {"should_not_hit": True}
    assert cases["cal_kangaroo_egg_memory"]["expect"]["corpus"]["contains"] == ["袋鼠蛋"]


def test_same_egg_record_fails_live_kangaroo_question_and_passes_its_actual_memory():
    cases = {case["id"]: case for case in evaluation.load_cases()}
    egg = RetrievalItem("corpus", "77", 1, text="朋友在纪念品店买了袋鼠蛋，闻起来巨臭")
    got = _got(corpus=[egg], facts=[{"id": "corpus:77", "text": egg.text},
                                  {"id": "terms", "text": "袋鼠是澳洲常见动物"}])
    assert not evaluation.check_case(cases["as3_kangaroo"], got)["passed"]
    assert evaluation.check_case(cases["cal_kangaroo_egg_memory"], got)["passed"]


def test_legacy_corpus_contains_keeps_alternative_wording_semantics():
    case = {"id": "legacy-alias", "query": "女同学的事", "expect": {
        "corpus": {"contains": ["女室友", "女同学"]}}}
    item = RetrievalItem("corpus", "story", 1, text="她和女同学的一件往事")
    assert evaluation.check_case(case, _got(corpus=[item]))["passed"]


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "_zhipu_key", lambda: "test-key")
    monkeypatch.setattr(evaluation, "AsyncOpenAI", lambda **kwargs: object())
    monkeypatch.setattr(evaluation, "_get_clients", lambda: (object(), object()))
    monkeypatch.setattr(evaluation, "load_persona_rules", lambda: [])
    monkeypatch.setattr(evaluation, "OUT_DIR", tmp_path)
    return tmp_path


async def test_snapshot_uses_shared_fused_results_and_context(monkeypatch):
    seen = {}
    corpus = RetrievalItem("corpus", "chosen", .9, text="选中并进入预算的内容")
    behavior = RetrievalItem("behavior", "reaction", 1, text="反应样本")

    async def gather(raw, resolved, history, chat, embed, **kwargs):
        seen.update(raw=raw, resolved=resolved, history=history, **kwargs)
        return {"fused_items": [corpus, behavior], "preference_items": [{"id": "p"}],
                "core_stories": [{"id": "s"}], "l3": {"behavior": "reaction"},
                "embedding_available": True, "selection_trace": {"available": True},
                "denied_terms": {"否认的词"}, "approved_terms": {"已批准词"},
                "association_entities": ["已批准实体"]}

    async def no_embedding(*args, **kwargs):
        return None

    def terms(query, denied_terms=None, hit_out=None, selected_terms=None):
        seen["denied_terms"] = denied_terms
        seen["selected_terms"] = selected_terms
        hit_out.append("已批准词")
        return "词义"

    monkeypatch.setattr(evaluation, "gather_retrieval", gather, raising=False)
    monkeypatch.setattr(evaluation, "embed_query", no_embedding, raising=False)
    monkeypatch.setattr(evaluation, "build_terms_note", terms)
    got, l3 = await evaluation.snapshot(
        "那本书呢", object(), object(), [], history="用户：昨天聊到一本书",
        resolved_query="之前那本书叫什么", probe_meta={"target": "书", "speech_act": "question"})
    assert got["corpus"] == [corpus] and got["behavior"] == [behavior]
    assert got["preference"] == [{"id": "p"}] and got["core_story"] == [{"id": "s"}]
    assert got["terms"] == [{"id": "已批准词", "score": 1.0}]
    assert got["associations"] == [{"id": "已批准实体", "score": 1.0}]
    assert seen["denied_terms"] == {"否认的词"}
    assert seen["selected_terms"] == {"已批准词"}
    assert seen["context"].history == seen["history"] == "用户：昨天聊到一本书"
    assert seen["context"].target == "书" and seen["resolved"] == "之前那本书叫什么"
    assert l3["_embedding_available"] is True and not l3["_degraded"]


async def test_snapshot_preserves_other_sources_when_embedding_unavailable(monkeypatch):
    async def gather(*args, **kwargs):
        return {"fused_items": [], "preference_items": [{"id": "rescued"}], "core_stories": [],
                "embedding_available": False, "selection_trace": {"available": True},
                "l3": {"behavior": ""}, "denied_terms": set(), "association_entities": []}

    async def no_embedding(*args, **kwargs):
        return None

    monkeypatch.setattr(evaluation, "gather_retrieval", gather, raising=False)
    monkeypatch.setattr(evaluation, "embed_query", no_embedding, raising=False)
    monkeypatch.setattr(evaluation, "build_terms_note", lambda *args, **kwargs: "")
    got, l3 = await evaluation.snapshot("你喜欢猫吗", object(), object(), [])
    assert got is not None and got["preference"] == [{"id": "rescued"}]
    assert l3["_embedding_available"] is False
    assert l3["_degraded"] and "embedding_unavailable" in l3["_degraded_reasons"]


@pytest.mark.parametrize("status", [False, True, None])
async def test_classification_failure_cannot_false_pass_negative_behavior(offline, monkeypatch, status):
    async def gather(*args, **kwargs):
        l3 = {"behavior": ""}
        if status is not None:
            l3["classification_ok"] = status
        return {"fused_items": [], "preference_items": [], "core_stories": [], "l3": l3,
                "embedding_available": True, "selection_trace": {"available": True},
                "denied_terms": set(), "approved_terms": set(), "association_entities": []}

    monkeypatch.setattr(evaluation, "gather_retrieval", gather)
    monkeypatch.setattr(evaluation, "build_terms_note", lambda *args, **kwargs: "")
    _, l3 = await evaluation.snapshot("这句不是在催你", object(), object(), [])
    assert l3["_classification_available"] is status
    assert l3["_degraded"] is (status is False)
    assert ("l3_unavailable" in l3["_degraded_reasons"]) is (status is False)

    case = {"id": "negative-behavior", "query": "这句不是在催你",
            "expect": {"behavior": {"should_not_hit": True}}}
    assert await evaluation.run([case], verbose=False) == (1 if status is False else 0)
    report = json.loads((offline / "eval_report.json").read_text(encoding="utf-8"))
    assert report["case_total"] == 1
    assert report["case_degraded"] == (1 if status is False else 0)
    assert report["case_passed"] == (0 if status is False else 1)
    assert report["results"][0]["diagnostics"]["_classification_available"] is status


@pytest.mark.parametrize("status", [False, True, None])
async def test_material_needs_failure_is_degraded_even_if_other_steps_available(offline, monkeypatch, status):
    async def gather(*args, **kwargs):
        l3 = {"behavior": "", "classification_ok": True, "needs": []}
        if status is not None:
            l3["needs_ok"] = status
        return {"fused_items": [], "preference_items": [], "core_stories": [], "l3": l3,
                "embedding_available": True, "selection_trace": {"available": True},
                "denied_terms": set(), "approved_terms": set(), "association_entities": []}

    monkeypatch.setattr(evaluation, "gather_retrieval", gather)
    monkeypatch.setattr(evaluation, "build_terms_note", lambda *args, **kwargs: "")
    _, l3 = await evaluation.snapshot("这只是普通闲聊", object(), object(), [])
    assert l3["_needs_available"] is status
    assert l3["_degraded"] is (status is False)
    assert ("needs_unavailable" in l3["_degraded_reasons"]) is (status is False)

    case = {"id": "no-material-needs", "query": "这只是普通闲聊",
            "expect": {"core_story": {"should_not_hit": True}}}
    assert await evaluation.run([case], verbose=False) == (1 if status is False else 0)
    report = json.loads((offline / "eval_report.json").read_text(encoding="utf-8"))
    assert report["case_total"] == 1
    assert report["case_degraded"] == (1 if status is False else 0)
    assert report["case_passed"] == (0 if status is False else 1)
    assert report["results"][0]["diagnostics"]["_needs_available"] is status


def test_absent_source_cannot_pass_negative_expectation():
    result = evaluation.check_case(
        {"id": "missing", "query": "原始标注", "expect": {"corpus": {"should_not_hit": True}}}, {})
    assert not result["passed"]
    assert result["checks"][0]["missing"] is True


def test_case_without_expectations_is_reported_as_missing_check():
    result = evaluation.check_case({"id": "unlabeled", "query": "尚未标注"}, _got())
    assert not result["passed"]
    assert len(result["checks"]) == 1 and result["checks"][0]["missing"] is True


@pytest.mark.parametrize("ids,passed", [(["game_animal"], True), (["game_animal", "animals"], False),
                                         (["animals"], False)])
def test_required_and_forbidden_ids_are_both_enforced(ids, passed):
    case = {"id": "specific", "query": "动森岛建", "expect": {
        "preference": {"should": ["game_animal"], "must_not": ["animals"]}}}
    result = evaluation.check_case(case, _got(preference=[{"id": item_id} for item_id in ids]))
    assert result["passed"] is passed
    assert not result["checks"][0]["missing"]


@pytest.mark.parametrize("ids,passed", [([], True), (["别的反应"], True), (["失约被催时认栽滑跪"], False)])
def test_standalone_forbidden_ids_allow_other_behavior_modes(ids, passed):
    case = {"id": "quotation", "query": "同事又鸽了", "expect": {
        "behavior": {"must_not": ["失约被催时认栽滑跪"]}}}
    items = [RetrievalItem("behavior", item_id, 1, text="样本") for item_id in ids]
    assert evaluation.check_case(case, _got(behavior=items))["passed"] is passed


@pytest.mark.parametrize("condition", [
    {"should": ["game_animal"], "must_not": "animals"},
    {"should": ["game_animal"], "must_nott": ["animals"]},
    {"must_not": []},
])
def test_invalid_or_empty_standalone_conditions_are_missing_checks(condition):
    case = {"id": "invalid", "query": "动森", "expect": {"preference": condition}}
    result = evaluation.check_case(case, _got(preference=[{"id": "game_animal"}]))
    assert not result["passed"] and result["checks"][0]["missing"]


async def test_missing_embedding_key_still_evaluates_all_cases(offline, monkeypatch):
    monkeypatch.setattr(evaluation, "_zhipu_key", lambda: "")

    async def snapshot(query, client, *args, **kwargs):
        assert client is None
        return _got(), {"_embedding_available": False, "_degraded": True,
                        "_degraded_reasons": ["embedding_unavailable"]}

    monkeypatch.setattr(evaluation, "snapshot", snapshot)
    cases = [{"id": str(i), "query": "原始问题", "expect": {"corpus": {"should_not_hit": True}}}
             for i in range(2)]
    assert await evaluation.run(cases, verbose=False) == 1
    report = json.loads((offline / "eval_report.json").read_text(encoding="utf-8"))
    assert report["case_total"] == report["case_degraded"] == 2
    assert report["source_stats"]["corpus"]["total"] == 2


async def test_run_keeps_legacy_unavailable_case_in_denominator(offline, monkeypatch):
    async def snapshot(*args, **kwargs):
        return None, {"behavior": ""}

    monkeypatch.setattr(evaluation, "snapshot", snapshot)
    case = {"id": "kept-id", "query": "原始标注", "expect": {"corpus": {"should_fire": True}}}
    assert await evaluation.run([case], verbose=False) == 1
    report = json.loads((offline / "eval_report.json").read_text(encoding="utf-8"))
    assert report["case_total"] == 1 and report["case_passed"] == 0
    assert report["case_degraded"] == 1 and report["check_missing"] == 1
    assert report["source_stats"]["corpus"]["total"] == 1
    assert report["results"][0]["id"] == case["id"]
    assert report["results"][0]["query"] == case["query"]


@pytest.mark.parametrize("diagnostics", [
    {"_embedding_available": False, "_degraded": True, "_degraded_reasons": ["embedding_unavailable"]},
    {"_selection_trace": {"available": False}, "_degraded": True, "_degraded_reasons": ["selection_unavailable"]},
])
async def test_degraded_run_is_nonzero_even_if_negative_checks_pass(offline, monkeypatch, diagnostics):
    async def snapshot(*args, **kwargs):
        return _got(), {"behavior": "", **diagnostics}

    monkeypatch.setattr(evaluation, "snapshot", snapshot)
    case = {"id": "negative", "query": "空结果", "expect": {"corpus": {"should_not_hit": True}}}
    assert await evaluation.run([case], verbose=False) == 1
    report = json.loads((offline / "eval_report.json").read_text(encoding="utf-8"))
    assert report["case_total"] == 1 and report["case_degraded"] == 1
    assert report["case_passed"] == 0
    assert report["source_stats"]["corpus"]["total"] == 1
    assert report["source_stats"]["corpus"]["pass"] == 0
    assert report["source_stats"]["corpus"]["degraded"] == 1


async def test_source_stats_keep_healthy_and_degraded_checks_in_same_denominator(offline, monkeypatch):
    async def snapshot(query, *args, **kwargs):
        failed = query == "unavailable"
        return _got(), {"behavior": "", "_degraded": failed,
                        "_degraded_reasons": ["l3_unavailable"] if failed else []}

    monkeypatch.setattr(evaluation, "snapshot", snapshot)
    cases = [{"id": query, "query": query,
              "expect": {"behavior": {"should_not_hit": True}}}
             for query in ("healthy", "unavailable")]
    assert await evaluation.run(cases, verbose=False) == 1
    report = json.loads((offline / "eval_report.json").read_text(encoding="utf-8"))
    assert report["case_total"] == 2 and report["case_passed"] == 1
    assert report["source_stats"]["behavior"] == {"pass": 1, "total": 2, "degraded": 1}


async def test_functional_failure_returns_nonzero(offline, monkeypatch):
    async def snapshot(*args, **kwargs):
        return _got(), {"behavior": "", "_degraded": False}

    monkeypatch.setattr(evaluation, "snapshot", snapshot)
    case = {"id": "no-hit", "query": "问题", "expect": {"corpus": {"should_fire": True}}}
    assert await evaluation.run([case], verbose=False) == 1


async def test_healthy_run_preserves_annotations_and_explicit_output(offline, monkeypatch):
    async def snapshot(*args, **kwargs):
        return _got(), {"behavior": "", "_degraded": False}

    monkeypatch.setattr(evaluation, "snapshot", snapshot)
    case = {"id": "healthy", "query": "原句", "expect": {"corpus": {"should_not_hit": True}}}
    output = offline / "nested" / "report.json"
    assert await evaluation.run([case], verbose=False, output=output) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["case_total"] == report["case_passed"] == 1
    assert report["case_degraded"] == report["check_missing"] == 0


async def test_snapshot_error_is_counted_and_later_cases_still_run(offline, monkeypatch):
    async def snapshot(query, *args, **kwargs):
        if query == "broken":
            raise RuntimeError("offline failure")
        return _got(), {"behavior": "", "_degraded": False}

    monkeypatch.setattr(evaluation, "snapshot", snapshot)
    cases = [{"id": q, "query": q, "expect": {"corpus": {"should_not_hit": True}}}
             for q in ("broken", "healthy")]
    assert await evaluation.run(cases, verbose=False) == 1
    report = json.loads((offline / "eval_report.json").read_text(encoding="utf-8"))
    assert report["case_total"] == 2 and report["case_passed"] == 1
    assert report["check_missing"] == report["case_degraded"] == 1
