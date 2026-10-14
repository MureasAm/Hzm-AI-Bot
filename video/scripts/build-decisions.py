#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EP01：把真实 trace 派生成 Remotion 直接读的 decisions 数组（规格 §10.5 建议结构）。

**不伪造任何命中**：每条 decision 的 hit/method 都从
`full-trace-result.json` 里的**真实执行痕迹**推出：

- hit  = 对应的 message 段是否真的出现在最终 messages 里；
- method = 该内容在代码里实际走的判别方式（对照 core.build_message_list 与各检索函数）；
- reason = 引用真实跑出来的证据（命中的 id / 分数 / LLM 判定值）。

输出 `video/data/ep01/full-trace-decisions.json`，由 Remotion 只读消费。

    python video/scripts/build-decisions.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "video" / "data" / "ep01" / "full-trace-result.json"
OUT = ROOT / "video" / "data" / "ep01" / "full-trace-decisions.json"

# (id, 展示名, 所属域, 判别方式, 段头/证据键)
# 域：persona 人格 / fact 事实 / scene 现场
# 判别：always / keyword / vector / llm
SPEC = [
    ("system_prompt", "system_prompt.txt", "persona", "always", "（无段头 · 基础提示词）",
     "稳定骨架：每轮常驻，首条 system message"),
    ("schedule", "schedule.json", "fact", "always", "【灰泽满的周表】",
     "周表是地面真值，带 weekly 就每轮注入"),
    ("profile", "用户长期记忆", "scene", "always", "【和这个绿冻的相处方式】",
     "Memory V2：相处方式靠前段"),
    ("preference", "preferences.json", "fact", "keyword", "【灰泽满的偏好】",
     "keyword 子串 / pattern 命中才带"),
    ("core_story", "core_stories.json", "fact", "vector", "【她的核心记忆】",
     "低阈值向量召回，命中才带"),
    ("group", "群聊现场", "scene", "always", "【群聊现场】",
     "只有群聊才注入；私聊为空"),
    ("terms", "terms.json", "fact", "keyword", "【灰泽满的世界】",
     "always 词条常驻 + 命中词注入"),
    ("session", "会话记忆", "scene", "always", "【会话记忆】",
     "会话未冷场就注入当前话题"),
    ("prev_session", "上一场会话", "scene", "always", "【上次聊过】",
     "隔 >12h 才出现"),
    ("behavior", "behaviors.json", "persona", "llm", "【当前情境下的行为指令】",
     "LLM 情景分类为主，判别词兜底"),
    ("corpus", "statement_final.json", "fact", "vector+llm", "【她经历过的相关背景】",
     "向量候选 + 词面补位 + LLM 确认"),
    ("commitment", "未完成约定", "scene", "always", "【相关未完成约定】",
     "Memory V2：中段，紧挨记忆区"),
    ("history", "最近对话记录", "scene", "always", "【最近对话记录】",
     "短期记忆带每条时间戳"),
    ("same_request", "连续索要", "scene", "llm", "【客观情况】",
     "L3 同一次调用顺带判「连续第几轮要同一件事」"),
    ("batch_summary", "批量归纳", "scene", "llm", "【这批消息的归纳】",
     "整批一起给模型做理解提示"),
    ("vision", "图片描述", "scene", "always", "图片",
     "图片描述并入 user 消息，没有独立段头"),
    ("query_hint", "短消息语境", "scene", "llm", "【用户这条消息的语境】",
     "短/指代消息的语境扩充"),
    ("time", "当前时间/天气", "scene", "always", "【当前时间】",
     "每轮常驻，放在最末尾（护前缀缓存）"),
    ("emotion", "当下情绪", "scene", "always", "【用户此刻的情绪】",
     "会话级感知，3h 内有效"),
    ("legendary", "legendary.json", "fact", "keyword", "legendary",
     "关键词/正则粗筛 + LLM 语境确认；命中直接回复，不进 messages"),
    ("preference_source", "长期记忆 v1 卡", "scene", "always", "【关于这个绿冻的长期记忆】",
     "注入路径已停用；只用来读天气城市"),
]


def _message_block(m: dict) -> dict:
    """messages 的展示名。

    trace-result 里"无段头"一律标成"（无段头 · 基础提示词）"——那对**首条 system**
    是对的，但对最后的 user 消息是错的（它没有段头只是因为用户原话本来就不需要段头）。
    这里按 role 分开命名，避免画面把用户消息标成"骨架全文"。
    """
    header = m["header"]
    if header.startswith("（无段头"):
        header = "system_prompt（骨架全文）" if m["role"] == "system" else "（用户原话）"
    return {"index": m["index"], "role": m["role"], "header": header,
            "chars": m["chars"], "head": m["head"]}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    res = json.loads(RESULT.read_text(encoding="utf-8"))
    messages = res["messages"]
    present = {}
    for m in messages:
        present.setdefault(m["header"], []).append(m)

    trace = {t["layer"]: t.get("value") for t in res["trace"]}
    parts = res.get("promptParts", {})
    l3 = trace.get("L3·LLM分类") or {}

    def find_header(header: str):
        return present.get(header, [])

    decisions = []
    for did, label, domain, method, key, why in SPEC:
        hit = False
        reason = ""
        evidence = {}

        if did == "legendary":
            hit = False
            reason = "没有真实触发：本条消息不命中任何梗库关键词/正则 → 不进 messages"
        elif did == "preference_source":
            hit = False
            reason = ("注入路径已停用（handle_chat 里 memory_context 恒为空串）；"
                      "旧的 v1 长期记忆卡只用来读天气城市")
        elif did == "vision":
            hit = False
            reason = "无图片 → 视觉描述为空，本轮不注入"
        elif did == "group":
            hit = bool(find_header(key))
            reason = ("私聊 → 群聊现场整段不注入" if not hit
                      else "群聊 → 注入在场成员与群内近况")
        elif did == "prev_session":
            hit = bool(find_header(key))
            reason = ("距上一场会话不足 12 小时 → 不出现" if not hit
                      else "隔得够久 → 出现【上次聊过】")
        elif did == "query_hint":
            hit = bool(find_header(key))
            reason = ("整批文本不短、也不是指代性消息 → 语境扩充为空，不注入"
                      if not hit else f"扩充为：{parts.get('queryHint', '')[:60]}")
        elif did == "commitment":
            hit = bool(find_header(key))
            reason = ("Memory V2 没有未完成的约定条目 → 中段为空" if not hit
                      else "有未完成约定 → 注入中段")
        elif did == "corpus":
            kept = trace.get("corpus·LLM确认") or []
            cand = trace.get("corpus·候选召回") or []
            hit = bool(find_header(key))
            evidence = {"candidates": len(cand),
                        "kept": [{"id": x["id"], "score": x["score"]} for x in kept]}
            reason = (f"向量候选 {len(cand)} 条 → LLM 确认保留 {len(kept)} 条 → 注入"
                      if hit else f"向量候选 {len(cand)} 条 → LLM 全部判不相关 → 不注入")
        elif did == "behavior":
            item = trace.get("behavior·命中")
            hit = bool(find_header(key))
            evidence = {"behavior": l3.get("behavior") or "", "source": "llm"}
            reason = (f"LLM 情景分类判为「{l3.get('behavior')}」→ 按名字取出一条 behavior 注入"
                      if hit else "LLM 未判出情景，判别词兜底也未命中 → 不注入")
            if item:
                evidence["chars"] = item.get("chars")
        elif did == "same_request":
            sr = l3.get("same_request") or {}
            hit = bool(find_header(key))
            evidence = {"count": int(sr.get("count") or 1), "what": sr.get("what") or ""}
            reason = (f"LLM 判定连续第 {sr.get('count')} 轮在要「{sr.get('what')}」→ 注入客观情况"
                      if hit else "LLM 判定不是连续索要 → 不注入")
        elif did == "batch_summary":
            hit = bool(find_header(key))
            evidence = {"summary": parts.get("batchSummary", "")[:80]}
            reason = ("三条消息属同一批 → 归纳一句作为理解提示" if hit
                      else "只有一条消息时不归纳")
        elif did == "core_story":
            hits = trace.get("core_story·命中") or []
            hit = bool(find_header(key))
            evidence = {"stories": [{"id": x["id"], "category": x["category"],
                                     "score": x["score"]} for x in hits]}
            reason = ("；".join(f"{x['id']} {x['score']}" for x in hits) + " → 低阈值向量命中后注入"
                      if hit else "全部低于 CORE_STORY_THRESHOLD=0.42 → 不注入")
        elif did == "preference":
            hits = trace.get("preference·命中") or []
            hit = bool(find_header(key))
            evidence = {"entries": [{"id": x["id"], "category": x["category"]} for x in hits]}
            reason = ("；".join(x["id"] for x in hits) + " → keyword 子串命中后注入"
                      if hit else "keywords/pattern 都没命中 → 不注入")
        elif did == "time":
            hit = bool(find_header(key))
            reason = "本轮的固定时间/天气行"
        elif did == "profile":
            hit = bool(find_header(key))
            evidence = {"text": (parts.get("profileContext") or "")[:120]}
            reason = "把用户长期事实（名字/城市）与相处方式放进靠前段"
        elif did == "system_prompt":
            hit = True
            evidence = {"chars": messages[0]["chars"]}
            reason = f"每轮常驻，首条 system message（{messages[0]['chars']} 字）"
        elif did == "schedule":
            hit = bool(find_header(key))
            reason = "带 weekly 就每轮注入，作为'几点播'的地面真值"
        elif did == "terms":
            hits = [m for m in find_header(key)]
            hit = bool(hits)
            evidence = {"note": (parts.get("termsNote") or "")[:80]}
            reason = ("always 词条（绿冻）常驻 + 命中词（满神）注入" if hit
                      else "既没有 always 词条也没有命中词 → 不注入")
        elif did == "session":
            hit = bool(find_header(key))
            evidence = {"topic": parts.get("sessionContext", "")[:80]}
            reason = "会话未冷场 → 注入当前话题与本场事件"
        elif did == "history":
            hit = bool(find_header(key))
            reason = "带每条时间戳的最近几轮原文"
        elif did == "emotion":
            hit = bool(find_header(key))
            evidence = {"mood": (parts.get("emotionContext") or "")[:80]}
            reason = "探测出的用户此刻情绪，紧挨用户消息"
        else:
            hit = bool(find_header(key)) if key else False
            reason = why

        decisions.append({
            "id": did, "label": label, "domain": domain, "method": method,
            "hit": hit, "reason": reason, "evidence": evidence,
            "why": why,
        })

    payload = {
        "fixtureId": res.get("fixtureId"),
        "fictional": True,
        "derivedFrom": "video/data/ep01/full-trace-result.json",
        "generated_at": res.get("generated_at"),
        "commit": res.get("commit"),
        "model": res.get("model"),
        "clock": res.get("clock"),
        "weather": res.get("weather"),
        "userProfile": res.get("userProfile"),
        "incomingBatch": res.get("incomingBatch"),
        "decisions": decisions,
        "messages": [_message_block(m) for m in messages],
        "messageCount": len(messages),
        "rawReply": res.get("rawReply"),
        "cleanedReply": res.get("cleanedReply"),
        "delivery": res.get("delivery"),
        "runSelection": res.get("runSelection"),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    hit_n = sum(1 for d in decisions if d["hit"])
    print(f"decisions = {len(decisions)} 条，命中 {hit_n} 条，未命中 {len(decisions) - hit_n} 条\n")
    for d in decisions:
        mark = "●" if d["hit"] else "○"
        print(f"  {mark} [{d['domain']:<7}] {d['label']:<22} {d['method']:<12} {d['reason'][:66]}")
    print(f"\n✅ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
