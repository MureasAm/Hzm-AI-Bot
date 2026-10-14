#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EP01 Phase 0-B：用真实代码跑出完整链路 trace（规格 §10）。

## 铁律（规格 §10.5）

画面必须来自**真实项目逻辑**产生的 trace，不允许手工伪造"命中"。

所以做法与 `scripts/trace_chain.py` 一致：**包装真实函数、不重新实现**。
这里调的是真的 `core.build_message_list` / `core.classify_l3` / `retrieve_corpus_candidates` /
`judge_corpus` / `select_behavior_item` / `retrieve_preferences` / `retrieve_core_stories` /
`fuse_and_truncate` / `generate_reply`，输入是固定的虚构 fixture。

## 隔离（规格 §10.1、验收标准「隐私与可复现性」）

**绝不读写真 user_memory/ 或 data/chat_log/**：

- `memory.use_storage(tmp)` + `session_memory.use_storage(tmp)` 把短期/会话记忆整体重定向；
- `core._MEMORY_V2_STORE` 换成指向临时文件的 store 实例（长期记忆隔离）；
- `context_probe.warm_weather` 打桩 → 不联网取天气；`get_now_context` 打桩 → 固定时钟；
- `chatlog.log_turn` 打桩 → 不写聊天落盘；
- `memory_v2` 的提取任务不触发（我们直接调 build 路径，不调 handle_chat 的异步尾巴）。

## 用法

    python video/scripts/capture-full-trace.py
    python video/scripts/capture-full-trace.py --generate     # 连真实模型生成回复
"""
import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402

nonebot.init()

from scripts import _common  # noqa: E402
from src.plugins.chatbot import constants as C  # noqa: E402
from src.plugins.chatbot import context_probe, memory, session_memory  # noqa: E402
import src.plugins.chatbot.core as core  # noqa: E402

FIXTURE = ROOT / "video" / "data" / "ep01" / "full-trace-fixture.json"
OUT = ROOT / "video" / "data" / "ep01" / "full-trace-result.json"

CST = timezone(timedelta(hours=8))

# ---- 与 fixture 对齐的本地参考判据（只用于交叉核对，不冒充线上判定）----
import re  # noqa: E402


def _local_bigram_count(sequence: list[str], threshold: float = 0.6) -> int:
    """memory.count_consecutive_requests 的判据**原样复刻**，仅作参考值。

    ⚠️ 该函数在生产聊天路径里**没有被调用**（死代码）；线上是 classify_l3 的 LLM 判定。
    这里复刻它只为给结果文件留一个"本地确定性参考"，帮助人工核对（规格 §15 Phase 0 第 5 步）。
    """
    def bg(s: str) -> set:
        s = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", s)
        return {s[i:i + 2] for i in range(len(s) - 1)}

    cur = bg(sequence[-1])
    n = 1
    for line in reversed(sequence[:-1]):
        other = bg(line)
        if not other or len(cur & other) / min(len(cur), len(other)) < threshold:
            break
        n += 1
    return n


def _items(items) -> list:
    """把一路检索结果压成可展示结构。

    ⚠️ 两路返回值形状不同：`retrieve_corpus_candidates` / `judge_corpus` 返回
    `RetrievalItem`（属性风格），`retrieve_core_stories` 返回**普通 dict**
    （retrieval.py:391-396）。统一在这里兼容，别假设都是对象。
    """
    out = []
    for it in items or []:
        def get(name, default=""):
            if isinstance(it, dict):
                return it.get(name, default)
            return getattr(it, name, default)
        out.append({
            "source": get("source", "?"),
            "id": str(get("item_id", "") or get("id", "")),
            "category": get("category", ""),
            "score": round(float(get("score", 0.0) or 0.0), 4),
            "chars": len(get("text", "") or ""),
            "text": (get("text", "") or "")[:120],
        })
    return out


class Recorder:
    """把每个真实函数的输出记下来。**只包不实现**。"""

    def __init__(self):
        self.log: list[dict] = []

    def wrap(self, owner, name: str, label: str, extract=None):
        orig = getattr(owner, name, None)
        if orig is None:
            self.log.append({"layer": label, "error": f"{name} 不存在，未包装"})
            return
        rec = self

        async def aw(*a, **k):
            out = await orig(*a, **k)
            rec.log.append({"layer": label, "value": extract(out) if extract else out})
            return out

        def sw(*a, **k):
            out = orig(*a, **k)
            rec.log.append({"layer": label, "value": extract(out) if extract else out})
            return out

        setattr(owner, name, aw if asyncio.iscoroutinefunction(orig) else sw)

    def install(self):
        self.wrap(core, "confirm_ambiguous_terms", "terms·LLM语境确认", lambda v: sorted(v or []))
        self.wrap(core, "retrieve_corpus_candidates", "corpus·候选召回", _items)
        self.wrap(core, "judge_corpus", "corpus·LLM确认", _items)
        self.wrap(core, "retrieve_preferences", "preference·命中", lambda v: [
            {"id": x.get("id"), "category": x.get("category"),
             "text": (x.get("text") or "")[:80]} for x in (v or [])])
        self.wrap(core, "retrieve_core_stories", "core_story·命中", _items)
        self.wrap(core, "select_behavior_item", "behavior·命中", lambda b: None if not b else {
            "id": b.item_id, "score": b.score, "chars": len(b.text or ""),
            "text": (b.text or "")[:160]})
        self.wrap(core, "classify_l3", "L3·LLM分类")
        self.wrap(core, "fuse_and_truncate", "RRF·融合后", _items)
        self.wrap(core, "generate_reply", "生成·原始回复")


def build_now_context(fixture: dict) -> str:
    """按 fixture 的固定时钟/天气拼【当前时间】。

    格式与 `context_probe.get_now_context`（context_probe.py:187-205）一致，
    但时间/天气来自 fixture 而不是系统时钟与网络。
    """
    dt = datetime.fromisoformat(fixture["clock"])
    wd = "周" + "一二三四五六日"[dt.weekday()]
    mel = dt.astimezone(timezone(timedelta(hours=10)))  # 澳洲墨尔本（10 月为夏令时 +10）
    w = fixture["weather"]
    return (
        f"【当前时间】{dt.month}月{dt.day}日 {wd} {dt.hour:02d}:{dt.minute:02d} 北京时间"
        f"（澳洲当地时间 {mel.hour:02d}:{mel.minute:02d}），灰泽满所在地：澳洲，"
        f"灰泽满现在没有在直播，天气：{w['text']} {w['temperatureC']}℃"
    )


async def capture_once(fx: dict, generate: bool) -> dict:
    """跑**一次**真实链路，返回这一次的结果 dict（所有中间状态都在里面）。

    每次调用都用**全新的临时目录**，所以多次 run 之间互不污染（会话记忆每次从 fixture 重建）。
    """
    seed = fx["seed"]
    uid = seed["traceUserId"]
    batch = fx["incomingBatch"]
    combined = "\n".join(batch)

    rec = Recorder()
    rec.install()

    tmp_root = Path(tempfile.mkdtemp(prefix="ep01-trace-"))
    v2_store = core.memory_v2.ShadowMemoryStore(
        state_file=tmp_root / "memory_v2_shadow.json",
        audit_file=tmp_root / "memory_v2_audit.jsonl")

    result: dict = {"warnings": []}
    try:
        # ---------- 隔离：把三套存储都指向临时目录 ----------
        memory.use_storage(tmp_root)
        session_memory.use_storage(tmp_root)
        core._MEMORY_V2_STORE = v2_store

        # 长期记忆：走真实 ingest API 写入（不手改内部状态文件）
        observed = fx["clock"]
        v2_store.ingest(uid, seed["memoryV2Items"], source={
            "session_id": "ep01-fixture-session", "observed_at": observed,
            "user_text": "（EP01 固定 fixture 的长期记忆种子）"},
            now=datetime.fromisoformat(observed))
        result["memoryV2Seeded"] = [
            {"kind": m["kind"], "key": m["key"], "value": m["value"], "status": m["status"]}
            for m in v2_store.snapshot(uid).get("memories", [])
        ]

        # 会话记忆：写入 topic/events/last_active（围绕固定时钟）
        dt = datetime.fromisoformat(observed)
        (tmp_root / "session").mkdir(parents=True, exist_ok=True)
        (tmp_root / "session" / f"{uid}.json").write_text(json.dumps({
            "topic": seed["sessionState"]["topic"],
            "events": seed["sessionState"]["events"],
            "last_active": (dt - timedelta(seconds=240)).isoformat(),
        }, ensure_ascii=False, indent=1), encoding="utf-8")

        # 短期记忆：写入前两轮（含两次索要）
        (tmp_root / "short_term").mkdir(parents=True, exist_ok=True)
        (tmp_root / "short_term" / f"{uid}.json").write_text(
            json.dumps(seed["shortTerm"], ensure_ascii=False, indent=1), encoding="utf-8")

        fixed_dt = dt.replace(tzinfo=None)  # session_memory / memory 用 naive datetime

        # ---------- 打桩：时钟、天气、落盘 ----------
        # 只换 `datetime` 这个名字（提供固定的 now()），fromisoformat 仍是真货——
        # 否则 `datetime.fromisoformat(...)` 会返回 Mock，算术与比较全炸。
        with patch.object(session_memory, "datetime", _FrozenDatetime(fixed_dt)), \
             patch.object(memory.time, "time", return_value=dt.timestamp()), \
             patch.object(context_probe, "warm_weather", new=_noop_async), \
             patch.object(context_probe, "get_now_context", new=lambda city="": build_now_context(fx)), \
             patch.object(core.context_probe, "get_now_context", new=lambda city="": build_now_context(fx)), \
             patch.object(core.context_probe, "warm_weather", new=_noop_async), \
             patch.object(core, "get_user_memory", new=lambda _uid: {"weather_city": fx["weather"]["city"],
                                                                     "user_name": fx["userProfile"]["name"]}):

            # ---------- 真实调用链 ----------
            from src.plugins.chatbot.config import _get_clients
            deepseek_client, zhipu_client = _get_clients()

            history_timed = memory.get_user_history_timed(uid)
            history_text = "\n".join(history_timed[-8:]) if history_timed else ""

            retrieval_query = combined
            try:
                retrieval_query = await session_memory.probe_session(
                    uid, combined, history_text, deepseek_client)
            except Exception as e:  # noqa: BLE001
                result["warnings"].append(f"probe_session 失败: {e}")

            session_context = session_memory.build_session_context(uid)
            prev_note = session_memory.previous_session_note(uid)
            emotion_context = session_memory.mood_note(uid)

            ctx = await core.gather_retrieval(combined, retrieval_query, history_text,
                                             deepseek_client, zhipu_client)

            # Memory V2 的两段（靠前「相处方式+相关事实」/ 中段「未完成约定」）。
            # 位置与 core.handle_chat:920-925 一致：同一个开关、同两个方法。
            profile_context = commitment_context = ""
            if core.memory_v2.injection_enabled():
                profile_context = core._MEMORY_V2_STORE.build_preference_context(uid, retrieval_query)
                commitment_context = core._MEMORY_V2_STORE.build_commitment_context(uid, retrieval_query)

            batch_summary = ""
            try:
                batch_summary = await core.summarize_batch([(t, "") for t in batch])
            except Exception as e:  # noqa: BLE001
                result["warnings"].append(f"summarize_batch 失败: {e}")

            denied_terms = await core.confirm_ambiguous_terms(combined, deepseek_client)
            query_hint = retrieval_query if (retrieval_query and retrieval_query != combined) else ""
            terms_note = core.build_terms_note(combined, denied_terms=denied_terms)

            messages = core.build_message_list(
                combined, ctx["fused_items"], "", history_timed,
                vision_desc="", weather_city=fx["weather"]["city"],
                batch_summary=batch_summary,
                preference_items=ctx["preference_items"], core_stories=ctx["core_stories"],
                session_context=session_context, query_hint=query_hint,
                denied_terms=denied_terms, group_context="",
                history_gap_note="", prev_session_note=prev_note,
                same_request=ctx.get("same_request"),
                profile_context=profile_context,
                commitment_context=commitment_context,
                emotion_context=emotion_context,
            )

            result["promptParts"] = {
                "retrievalQuery": retrieval_query,
                "batchSummary": batch_summary,
                "termsNote": terms_note,
                "sessionContext": session_context,
                "prevSessionNote": prev_note,
                "emotionContext": emotion_context,
                "queryHint": query_hint,
                "deniedTerms": sorted(denied_terms or []),
                "profileContext": profile_context,
                "commitmentContext": commitment_context,
            }

            raw_reply = ""
            if generate:
                raw_reply = await core.generate_reply(messages)
            else:
                raw_reply = "（未生成：加 --generate 才会真的调模型）"

        # ---------- 组装 ----------
        from src.plugins.chatbot.reply_style import clean_reply, split_reply
        from src.plugins.chatbot.voice import should_voice

        cleaned = clean_reply(raw_reply)
        parts = split_reply(cleaned) if C.SPLIT_REPLY_ENABLED else [cleaned]
        parts = [clean_reply(p) for p in parts]
        voice = should_voice(cleaned)

        result["messages"] = [
            {"index": i, "role": m["role"], "chars": len(m["content"]),
             "header": _block_header(m["content"]),
             "head": m["content"][:80], "content": m["content"]}
            for i, m in enumerate(messages)
        ]
        result["rawReply"] = raw_reply
        result["cleanedReply"] = cleaned
        result["delivery"] = {
            "mode": "voice" if voice else "text",
            "shouldVoice": voice,
            "parts": [] if voice else parts,
            "note": ("语音成功时不再发拆分文字（互斥不双发）" if voice
                     else "未走语音 → split_reply 拆成多个文字气泡"),
        }
        result["trace"] = rec.log
        result["localReference"] = {
            "consecutiveRequestCount": _local_bigram_count(
                seed["sameRequestReference"]["sequence"],
                seed["sameRequestReference"]["threshold"]),
            "method": "local-reference（复刻 memory.count_consecutive_requests 的判据）",
            "caveat": ("生产聊天路径**不调用**它（该函数目前是死代码）；"
                       "线上是 classify_l3 的 LLM 判定，见 trace 里的 L3 项。"),
        }
        return result
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        # 把模块级存储指回去，避免影响同进程里后续的 run
        memory.use_storage(ROOT / "user_memory")
        session_memory.use_storage(ROOT / "user_memory")
        core._MEMORY_V2_STORE = core.memory_v2.ShadowMemoryStore()


def _l3_of(run: dict):
    for t in run.get("trace", []):
        if t.get("layer") == "L3·LLM分类" and isinstance(t.get("value"), dict):
            return t["value"]
    return {}


def _summarize_run(run: dict) -> dict:
    l3 = _l3_of(run)
    sr = l3.get("same_request") or {}
    layers = {}
    for t in run.get("trace", []):
        if t.get("layer"):
            layers[t["layer"]] = t.get("value")
    return {
        "behavior": l3.get("behavior") or "",
        "sameRequestCount": int(sr.get("count") or 1),
        "sameRequestWhat": sr.get("what") or "",
        "corpusKept": len(layers.get("corpus·LLM确认") or []),
        "preferenceHits": [x.get("id") for x in (layers.get("preference·命中") or [])],
        "coreStoryHits": [x.get("id") for x in (layers.get("core_story·命中") or [])],
        "blockCount": len(run.get("messages") or []),
        "headers": [m["header"] for m in (run.get("messages") or [])],
        "cleanedReply": run.get("cleanedReply", ""),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true",
                    help="真的调用模型生成回复（否则只组装 messages）")
    ap.add_argument("--runs", type=int, default=3,
                    help="最多跑几次真实链路（L3 是 LLM，有随机性；命中行为就提前停）")
    args = ap.parse_args()
    _common.ensure_utf8_stdout()

    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))

    runs: list[dict] = []
    for i in range(args.runs):
        print(f"\n===== 真实链路 run {i + 1}/{args.runs} =====")
        run = await capture_once(fx, args.generate)
        runs.append(run)
        s = _summarize_run(run)
        print(f"  L3 behavior = {s['behavior'] or '∅'}")
        print(f"  连续索要     = 第 {s['sameRequestCount']} 轮（{s['sameRequestWhat']}）")
        print(f"  messages     = {s['blockCount']} 段")
        if s["behavior"]:
            print("  → 行为已命中，提前停止（规格 §10.3 预期 behavior 亮起）")
            break

    # 选择规则：规格 §10.3 明确要求 behavior 亮起。
    # 规则 = **取第一个 behavior 命中的真实执行**（不挑回复好坏）；若都没命中，
    # 取第一次并在 result 里如实记录"本轮 L3 未判定行为"。
    chosen_idx = next((i for i, r in enumerate(runs) if _l3_of(r).get("behavior")), 0)
    chosen = runs[chosen_idx]

    out = {
        "fixtureId": fx["fixtureId"],
        "fictional": True,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "commit": fx["frozenAt"]["commit"],
        "model": _common.get_model_name(),
        "clock": fx["clock"], "weather": fx["weather"],
        "userProfile": fx["userProfile"], "session": fx["session"],
        "incomingBatch": fx["incomingBatch"],
        "expectedPaths": fx["expectedPaths"],
        "isolation": {
            "storageRoot": "临时目录（每次 run 新建，进程内用完全部删除）",
            "note": "未读写真实 user_memory/ 与 data/chat_log/；时钟/天气/直播状态全部来自 fixture",
        },
        "runSelection": {
            "capturedRuns": len(runs),
            "selectedRun": chosen_idx,
            "rule": ("取第一个 L3 命中 behavior 的真实执行（规格 §10.3 要求该路径亮起）。"
                     "没有人为改写过任何一步的输出；若多次都未命中则取第 0 次并如实标注。"),
            "reason": (f"run {chosen_idx} 的 behavior = "
                       f"{_l3_of(chosen).get('behavior') or '∅'}"),
            "nondeterminismNote": (
                "classify_l3 / judge_corpus / summarize_batch / probe_session 都是**真实 LLM 调用**，"
                "同一输入多次运行结果可能不同（实测同一批消息 behavior 一次为空、一次为"
                "「被推着要求时打太极」）。这是链路真实的随机性，不是采集错误。"),
            "allRuns": [_summarize_run(r) for r in runs],
        },
    }
    out.update({k: v for k, v in chosen.items() if k != "warnings"})
    out["warnings"] = chosen.get("warnings", [])

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n选定 run {chosen_idx}（共采集 {len(runs)} 次）")
    print(f"messages = {len(out['messages'])} 段")
    for m in out["messages"]:
        print(f"  [{m['index']:>2}] {m['role']:<9} {m['chars']:>5}字  {m['header'] or m['head'][:40]}")
    print(f"\n原始回复 = {out['rawReply']}")
    print(f"清洗后   = {out['cleanedReply']}")
    n_parts = len(out["delivery"]["parts"])
    print(f"投递     = {out['delivery']['mode']}"
          + ("（语音，不发文字）" if out["delivery"]["shouldVoice"] else f"（{n_parts} 条文字）"))
    print(f"\n✅ {OUT}")
    return 0


async def _noop_async(*_a, **_k):
    return None


class _FrozenDatetime:
    """假的 `datetime` 名字：`now()` 固定，其余（fromisoformat 等）转发真货。

    只把"现在几点"钉死；真 datetime 对象仍从 fromisoformat 出来，
    否则下游 `datetime.fromisoformat(...)` 拿到 Mock，时间算术立刻炸。
    """

    def __init__(self, fixed: datetime):
        self._fixed = fixed

    def now(self, tz=None):
        if tz is not None:
            return self._fixed.replace(tzinfo=tz)
        return self._fixed

    def __getattr__(self, name):
        return getattr(datetime, name)


_HEADERS = [
    "【灰泽满的周表】", "【灰泽满的偏好】", "【她的核心记忆】", "【群聊现场】",
    "【灰泽满的世界】", "【会话记忆】", "【上次聊过】", "【当前情境下的行为指令】",
    "【她经历过的相关背景】", "【关于这个绿冻的长期记忆】", "【最近对话记录】",
    "【关于这个绿冻的上一轮记忆】", "【客观情况】", "【这批消息的归纳】",
    "【用户发了表情】", "【用户这条消息的语境】", "【当前时间】",
    "【和这个绿冻的相处方式】", "【相关用户事实】", "【相关未完成约定】",
    "【用户此刻的情绪】",
]


def _block_header(content: str) -> str:
    for h in _HEADERS:
        if content.startswith(h):
            return h
    if content.startswith("【别自相矛盾】") or "【别自相矛盾】" in content:
        return "【最近对话记录】"
    return "（无段头 · 基础提示词）"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
