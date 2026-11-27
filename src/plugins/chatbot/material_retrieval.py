"""按原有方式召回候选，再用同一份本轮语境选择各路真正适用的材料。"""
import asyncio

from .corpus_judge import judge_enabled
from .persona import load_terms
from .material_usage import MaterialUse, render_material_use
from .retrieval import (
    load_preferences, match_entities, resolve_associations, retrieve_corpus_candidates,
    retrieve_core_stories, retrieve_preferences,
)
from .retrieval_context import (
    CONTEXT_GUIDANCE, MaterialCandidate, RetrievalContext, Selection, select_materials,
)
from .term_hints import term_candidates


_CORPUS_POLICY = (
    "你为角色对话选择她本人的既有事实或经历，不生成回答。\n" + CONTEXT_GUIDANCE
    + "\n本轮语境与独立任务理解先于候选；不得因候选同话题就改写用户的任务。"
      "根据下方唯一用途合同选择，没有适用材料可以为空。"
)


def _joined_relations(groups, budget=300, *, hit_ids=None):
    parts = []
    for group in groups:
        used = False
        for sentence in group.get("note", "").split("；"):
            if sentence and sentence in parts:
                used = True
            elif sentence and len("；".join(parts + [sentence])) <= budget:
                parts.append(sentence)
                used = True
        if used and hit_ids is not None and group["id"] not in hit_ids:
            hit_ids.append(group["id"])
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
    association_uses = {group["id"]: MaterialUse("association", group["id"], "background")
                        for group in accepted_groups}
    # 与生成出口相同地去重短注，并计入每条短注到正文之间的换行。
    relation_notes = list(dict.fromkeys(render_material_use(use) for use in association_uses.values()))
    relation_body_budget = max(0, 300 - sum(len(note) + 1 for note in relation_notes))
    association_use_ids = []
    association_context = _joined_relations(accepted_groups, budget=relation_body_budget,
                                             hit_ids=association_use_ids)
    material_uses = []
    for candidate in candidates:
        if not allowed(candidate.source, candidate.item_id):
            continue
        if candidate.source == "association":
            if candidate.item_id in association_use_ids:
                material_uses.append(association_uses[candidate.item_id])
            continue
        proof = selection.evidence.get((candidate.source, candidate.item_id), {})
        purpose = (proof.get("purpose", "past_example") if candidate.source == "corpus" else
                   {"core_story": "past_example", "term": "definition", "behavior": "interaction"}
                   .get(candidate.source, "background"))
        material_uses.append(MaterialUse(candidate.source, candidate.item_id, purpose,
                                         fact=proof.get("fact", ""), quote=proof.get("quote", "")))
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
        "association_context": association_context,
        "association_use_ids": association_use_ids,
        "association_entities": list(dict.fromkeys(entity for group in accepted_groups
            if group["id"] in association_use_ids for entity in group.get("entities") or [])),
        "wiki_entities": [entity for entity in wiki if allowed("wiki", entity)],
        "entity_mentions": selection.entity_mentions,
        "selection_trace": trace,
        "material_uses": material_uses,
    }
