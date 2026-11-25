"""按原有方式召回候选，再用同一份本轮语境选择各路真正适用的材料。"""
import asyncio

from .corpus_judge import CORPUS_JUDGE_PROMPT, judge_enabled
from .persona import load_terms
from .retrieval import (
    load_preferences, match_entities, resolve_associations, retrieve_corpus_candidates,
    retrieve_core_stories, retrieve_preferences,
)
from .retrieval_context import (
    CONTEXT_GUIDANCE, MaterialCandidate, RetrievalContext, Selection, select_materials,
)
from .term_hints import term_candidates


_CORPUS_POLICY = (
    # 保留事实判据，输出合同由共享选择器按证据模式统一添加。
    CORPUS_JUDGE_PROMPT.split("只输出 JSON：", 1)[0].format(
        query="下方本轮语境的原话、最近对话和独立任务理解", candidates="下方候选材料")
    + "\n\n" + CONTEXT_GUIDANCE
    + "\n\n【任务优先】先根据共用语境确定本轮实际任务；候选经历不能反过来改写用户意图。"
      "独立任务理解只作参考，仍以原话和最近对话为证据。"
)


def _joined_relations(groups, budget=300):
    parts = []
    for group in groups:
        for sentence in group.get("note", "").split("；"):
            if sentence and sentence not in parts and len("；".join(parts + [sentence])) <= budget:
                parts.append(sentence)
    return "；".join(parts)


async def retrieve_contextual_materials(context: RetrievalContext, query_vector, client, *,
                                        extra_preference_ids=None, behavior_candidate=None) -> dict:
    """共用一份语境；corpus 专用事实判据与其余材料判定分批并行。"""
    # 向量看指代补全；词面同时保留原话，防辅助解释漏掉实体或圈内用词。
    lexical_query = "\n".join(dict.fromkeys(x for x in (context.raw_text, context.query) if x))
    corpus = (retrieve_corpus_candidates(lexical_query, query_vector)
              if context.is_user_msg and judge_enabled() else [])
    stories = retrieve_core_stories(context.query, query_vector)
    groups = []
    preference_ids = list(extra_preference_ids or [])
    for seed in match_entities(context.raw_text, context.entities):
        group = dict(resolve_associations("", [seed]))
        group["id"] = seed
        preference_ids.extend(group.get("preference_ids") or [])
        if group.get("note"):
            groups.append(group)
    preferences = retrieve_preferences(lexical_query, top_n=max(1, len(load_preferences())),
                                       extra_ids=list(dict.fromkeys(preference_ids)))
    terms = term_candidates(lexical_query)
    known = {"灰泽满", "Hazel", "灰泽满Hazel"} | {
        str(t.get("keyword")) for t in load_terms() if t.get("keyword")}
    known.update(entity for group in groups for entity in group.get("entities") or [])
    wiki = list(dict.fromkeys(entity for entity in context.entities
                             if 2 <= len(entity) <= 20 and entity not in known))
    candidates = [MaterialCandidate("corpus", item.item_id, item.text) for item in corpus]
    candidates += [MaterialCandidate("core_story", str(item["id"]), item["text"],
                                    {"category": item.get("category", "")}) for item in stories]
    candidates += [MaterialCandidate("preference", str(item["id"]), item["text"],
                                    {"category": item.get("category", "")}) for item in preferences]
    candidates += [MaterialCandidate("term", term["keyword"], "；".join(str(term.get(key) or "")
                   for key in ("meaning", "reaction", "usage")),
                   {"category": term.get("category", ""),
                    "usage_triggers": term.get("usage_triggers", [])}) for term in terms]
    candidates += [MaterialCandidate("association", group["id"], group["note"]) for group in groups]
    candidates += [MaterialCandidate("wiki", entity, entity) for entity in wiki]
    if behavior_candidate:
        candidates.append(MaterialCandidate("behavior", behavior_candidate.item_id, behavior_candidate.text))
    corpus_selection, material_selection = await asyncio.gather(
        select_materials(client, context, [item for item in candidates if item.source == "corpus"],
                         policy=_CORPUS_POLICY, require_evidence=True),
        select_materials(client, context, [item for item in candidates if item.source != "corpus"]),
    )
    approved_corpus = corpus_selection if corpus_selection.available else Selection()
    approved_materials = material_selection if material_selection.available else Selection()
    selection = Selection(
        selected=approved_corpus.selected | approved_materials.selected,
        disputed=approved_corpus.disputed | approved_materials.disputed,
        entity_mentions={**approved_corpus.entity_mentions, **approved_materials.entity_mentions},
        available=corpus_selection.available and material_selection.available,
        evidence=approved_corpus.evidence,
    )

    def allowed(source, item_id):
        return (source, str(item_id)) in selection.selected

    accepted_groups = [group for group in groups if allowed("association", group["id"])]
    approved_terms = {term["keyword"] for term in terms if allowed("term", term["keyword"])}
    all_terms = {str(term.get("keyword")) for term in load_terms() if term.get("keyword")}
    trace = {"available": selection.available, "needs": context.needs,
             "batches": {"corpus": corpus_selection.available, "materials": material_selection.available},
             "attempts": {"corpus": corpus_selection.attempts, "materials": material_selection.attempts},
             "verification": {item_id: check for (source, item_id), check in corpus_selection.verification.items()
                              if source == "corpus"},
             "evidence": {item_id: {"quote": proof["quote"], "relation": proof["relation"]}
                          for (source, item_id), proof in selection.evidence.items() if source == "corpus"},
             "candidates": {}, "approved": {}}
    for candidate in candidates:
        trace["candidates"].setdefault(candidate.source, []).append(candidate.item_id)
        if allowed(candidate.source, candidate.item_id):
            trace["approved"].setdefault(candidate.source, []).append(candidate.item_id)
    return {
        "corpus_items": [item for item in corpus if allowed("corpus", item.item_id)],
        "preference_items": [item for item in preferences if allowed("preference", item["id"])],
        "core_stories": [item for item in stories if allowed("core_story", item["id"])],
        "behavior_items": ([behavior_candidate] if behavior_candidate and allowed("behavior", behavior_candidate.item_id) else []),
        "denied_terms": all_terms - approved_terms,
        "approved_terms": approved_terms,
        "association_context": _joined_relations(accepted_groups),
        "association_entities": list(dict.fromkeys(entity for group in accepted_groups for entity in group.get("entities") or [])),
        "wiki_entities": [entity for entity in wiki if allowed("wiki", entity)],
        "entity_mentions": selection.entity_mentions,
        "selection_trace": trace,
    }
