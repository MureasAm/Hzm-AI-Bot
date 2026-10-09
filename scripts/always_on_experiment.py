"""可行性实验：常驻注入（性格 / 痛处）会不会让回复更像老朋友、或者更像"背资料/哪壶不开提哪壶"。

隔离法：**绕开检索**，只调 `core.build_message_list(...)` + `core.generate_reply`，
把三档注入塞进 `profile_context`：
  A = 现状（话题相关的事实）
  B = A + 常驻"TA 是什么样的人"（性格，转成相处提示）
  C = B + 常驻"TA 的痛处 / 雷区"（明确说"别主动提"）
同一轮的同一句话、三条回复，交给裁判模型盲评（A/B/C 顺序每轮打乱）。

只写 `outputs/always_on_ab/`，不碰任何线上数据。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

try:  # direct script via run_tool.py
    import _common
except ModuleNotFoundError:  # imported as scripts.always_on_experiment
    from . import _common

if str(_common.PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_common.PROJECT_ROOT))

try:
    import memory_v2_tool as _v2tool
    import memory_ab as _ab
except ModuleNotFoundError:
    from . import memory_v2_tool as _v2tool
    from . import memory_ab as _ab

DEFAULT_STATE = _common.PROJECT_ROOT / "outputs" / "memory_ab" / "ab_v2_shadow.json"
DEFAULT_OUT_DIR = _common.OUTPUTS_DIR / "always_on_ab"

# 一条 past 事实算不算"痛处"：命中这些词就算（保守，宁可少算）。
_PAIN_MARKERS = (
    "霸凌", "车祸", "去世", "丧", "自杀", "抑郁", "躁郁", "落榜", "骗", "被PUA",
    "欺负", "童年", "家暴", "离婚", "癌", "病", "分手", "失恋",
)

_JUDGE_PROMPT = """下面是同一个用户说的同一句话（{user_msg}），以及三种记忆注入配置下机器人的回复。

{blocks}

请判断（只输出 JSON）：
{{
  "most_natural": "A|B|C  哪条最像在跟认识很久的老朋友自然聊天",
  "reciting": "A|B|C|none  哪条最像在'背资料/硬提背景'",
  "inappropriate": "A|B|C|none  哪条提到了不该提的（尤其没头没尾地翻旧账 / 哪壶不开提哪壶）",
  "reason": "一句话"
}}
只输出 JSON，不要多余内容。"""


def _text(response) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


def load_memories(path: Path) -> dict[str, list[dict]]:
    state = _v2tool.load_state(path)
    out: dict[str, list[dict]] = {}
    for uid, user in (state.get("users") or {}).items():
        mems = [m for m in (user.get("memories") or []) if m.get("status") in {"confirmed", "active"}]
        if mems:
            out[str(uid)] = mems
    return out


def load_v1_memories(root: Path) -> dict[str, list[dict]]:
    """从 v1 记忆卡取素材（印象=性格、含痛处的 user_fact=痛处）。

    可行性实验测的是"把这条内容常驻注入，模型反应如何"——素材由谁抽出来不影响这个反应。
    v1 卡覆盖 ~150 个真人，样本比刚开跑、只有 3 个人有记忆的 v2 影子库大得多。
    """
    out: dict[str, list[dict]] = {}
    for path in sorted(Path(root).glob("*.json")):
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(card, dict):
            continue
        mems: list[dict] = []
        for imp in card.get("impressions") or []:
            tag = imp.get("tag") if isinstance(imp, dict) else imp
            if tag and str(tag).strip():
                mems.append({"kind": "fact", "key": "personality",
                             "value": str(tag).strip(), "temporal_scope": "timeless"})
        for fact in card.get("user_facts") or []:
            text = fact.get("fact") if isinstance(fact, dict) else fact
            if text and str(text).strip():
                text = str(text).strip()
                mems.append({"kind": "fact", "key": "fact", "value": text,
                             "temporal_scope": "past" if _is_pain(text) else "current"})
        if mems:
            out[path.stem] = mems
    return out


def _is_pain(text: str) -> bool:
    return any(marker in text for marker in _PAIN_MARKERS)


def split_memories(memories: list[dict]) -> dict[str, list[dict]]:
    """把记忆分成三堆：话题相关（事实，含近况）/ 性格（常驻）/ 痛处（常驻，避雷）。"""
    from src.plugins.chatbot.memory_v2 import _ALWAYS_ON_FACT_KEYS

    personality, pain, rest = [], [], []
    for memory in memories:
        if memory.get("kind") != "fact":
            rest.append(memory)
            continue
        key = str(memory.get("key") or "")
        value = str(memory.get("value") or "")
        scope = memory.get("temporal_scope")
        if scope == "past" and _is_pain(value):
            pain.append(memory)
        elif scope == "timeless" and key not in _ALWAYS_ON_FACT_KEYS:
            personality.append(memory)
        else:
            rest.append(memory)
    return {"personality": personality, "pain": pain, "rest": rest}


def render_personality(items: list[dict]) -> str:
    if not items:
        return ""
    lines = ["【关于TA这个人】相处时把下面这些放在心上（别直接念出来，只是提醒你怎么接话）："]
    lines += [f"- {m.get('value')}" for m in items]
    return "\n".join(lines)


def render_pain(items: list[dict]) -> str:
    if not items:
        return ""
    lines = ["【TA的痛处——别主动提、别拿来开玩笑】下面是TA经历过的不好的事，"
             "**除非TA自己先提起，否则不要主动碰**；TA提起时轻轻接住，别追问细节："]
    lines += [f"- {m.get('value')}" for m in items]
    return "\n".join(lines)


def build_variants(base: str, personality: str, pain: str) -> dict[str, str]:
    b = "\n".join(x for x in (base, personality) if x)
    c = "\n".join(x for x in (base, personality, pain) if x)
    return {"A": base, "B": b, "C": c}


async def run(*, source: str, state_path: Path, turns: int, per_user: int, out_dir: Path, judge: bool) -> dict:
    from openai import AsyncOpenAI

    import nonebot

    try:
        nonebot.get_driver()
    except ValueError:
        nonebot.init()

    from src.plugins.chatbot import core
    from src.plugins.chatbot import memory_v2
    from src.plugins.chatbot.constants import THINKING_DISABLED

    api_key = _common.get_api_key("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("未检测到 OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key, base_url=_common.get_openai_base_url())
    model = _common.get_model_name()
    store = memory_v2.ShadowMemoryStore(
        state_file=state_path, audit_file=state_path.with_suffix(".audit.jsonl"),
    )

    memories_by_user = load_memories(state_path) if source == "v2" else load_v1_memories(state_path)
    split = {uid: split_memories(m) for uid, m in memories_by_user.items()}

    out_dir = _v2tool._safe_replay_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 取这些用户真实说过的话（只取有记忆的人），每人最多 per_user 条。
    # 迭代本身不花钱（花钱的只有后面的生成），所以给一个大上限，扫到够为止。
    paths = _ab.order_paths_by_activity(_v2tool._resolve_inputs(None))
    want = set(memories_by_user)
    records: list[dict] = []
    for record in _ab.iter_turns_by_user(paths, limit=5000, per_user=per_user):
        if str(record["session"]) in want:
            records.append(record)
        if len(records) >= turns:
            break

    rows = []
    for record in records:
        uid = str(record["session"])
        msg = str(record["user"])
        groups = split.get(uid, {"personality": [], "pain": [], "rest": []})
        base = store.build_preference_context(uid, msg) if source == "v2" else ""
        variants = build_variants(base, render_personality(groups["personality"]), render_pain(groups["pain"]))

        gen = {}
        for tag, block in variants.items():
            messages = core.build_message_list(msg, [], "", [], profile_context=block)
            gen[tag] = await core.generate_reply(messages)

        row = {"user": _v2tool.redact_user_id(uid), "msg": msg[:60],
               "A": gen["A"], "B": gen["B"], "C": gen["C"]}
        if judge:
            # 盲评：把 A/B/C 三个字母**随机**分给三种策略，判卷按字母答，再映射回策略。
            # ⚠️ 标签必须用 A/B/C（和裁判 schema 的 "A|B|C" 一致）——用「甲乙丙」会让
            #    裁判按 schema 答 A/B/C，而我们按甲乙丙映射 → 结果全错位（踩过一次）。
            order = ["A", "B", "C"]
            random.shuffle(order)           # order[i] = 展示在第 i 个字母下的策略
            blocks = "\n\n".join(
                f"{label}：{gen[tag]}" for label, tag in zip(["A", "B", "C"], order)
            )
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _JUDGE_PROMPT.format(user_msg=msg[:80], blocks=blocks)}],
                temperature=0.2, max_tokens=200,
                **THINKING_DISABLED,
            )
            try:
                verdict = json.loads(_text(resp).strip().strip("`").replace("json\n", "", 1))
            except Exception:
                verdict = {}
            mapping = dict(zip(["A", "B", "C"], order))
            row["verdict"] = {
                field: mapping.get(verdict.get(field), verdict.get(field))
                for field in ("most_natural", "reciting", "inappropriate")
            }
            row["verdict"]["reason"] = verdict.get("reason", "")
        rows.append(row)

    await client.close()

    tally = Counter()
    for row in rows:
        v = row.get("verdict") or {}
        if v.get("most_natural"):
            tally[f"natural:{v['most_natural']}"] += 1
        if v.get("reciting") not in (None, "none"):
            tally[f"reciting:{v['reciting']}"] += 1
        if v.get("inappropriate") not in (None, "none"):
            tally[f"inappropriate:{v['inappropriate']}"] += 1

    report = {
        "users": len(memories_by_user),
        "turns": len(rows),
        "tally": dict(sorted(tally.items())),
        "split": {u: {k: len(v) for k, v in g.items()} for u, g in split.items()},
        "rows": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="always_on_experiment")
    parser.add_argument("--source", choices=("v1", "v2"), default="v1",
                        help="素材来源：v1 记忆卡（覆盖广）或 v2 影子库")
    parser.add_argument("--state", default=None, help="v2 影子文件；--source v2 时用")
    parser.add_argument("--v1-dir", default=str(_common.PROJECT_ROOT / "user_memory" / "long_term"))
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--per-user", type=int, default=12)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args(argv)
    state_path = Path(args.state) if args.state else (
        DEFAULT_STATE if args.source == "v2" else Path(args.v1_dir)
    )
    report = asyncio.run(run(
        source=args.source, state_path=state_path, turns=args.turns,
        per_user=args.per_user, out_dir=Path(args.out_dir), judge=not args.no_judge,
    ))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
