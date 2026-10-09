"""实验：给模型注入「用户的性格」，到底改不改回复？标签 vs 行为句哪个好？

三档（同一句话、只差注入这一块）：
  A = 不注入
  B = 注入**性格标签**（v1 印象原样，如"讨好型人格"）
  C = 注入同一批印象**改写成的行为句**（如"她容易自责，回她时主动一点"）

**先用确定性指标，再谈别的**：
  ① 回复差异度（A vs B / A vs C 的文本相似度）——若≈1.0（几乎一样），
     说明注入根本没影响回复，"标签 vs 行为句"的争论就不用打了；
  ② 长度变化；③ 标签有没有被念出来。
LLM 判官不参与（仓库立场：风格类不能用模型当裁判——上一版实验就是死在判官是噪声上）。

只写 `outputs/profile_experiment/`。
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import re
import sys
from pathlib import Path

try:
    import _common
except ModuleNotFoundError:
    from . import _common

if str(_common.PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_common.PROJECT_ROOT))

try:
    import memory_ab as _ab
    import memory_v2_tool as _v2tool
except ModuleNotFoundError:
    from . import memory_ab as _ab
    from . import memory_v2_tool as _v2tool

DEFAULT_V1_DIR = _common.PROJECT_ROOT / "user_memory" / "long_term"
DEFAULT_OUT_DIR = _common.OUTPUTS_DIR / "profile_experiment"

REWRITE_PROMPT = """下面每一条都是模型给某个用户下的**性格标签**。请把它改写成**行为句**——
也就是"跟她说话时该怎么做"的具体做法，而不是换一个标签。

要求：
- 一句话，直接从**行为**写起（如"她自责时先接住情绪，别急着讲道理"）
- **不要复述标签词**（别写"她有讨好型人格"，要写她会怎么做/你该怎么接）
- 保持原意，**不要加戏**、不要编标签里没有的东西。想不出可操作的写法，就写"（无）"

标签：{tags}

只输出 JSON：{{"<原标签>": "<行为句>"}}，不要多余内容。"""


def _text(response) -> str:
    try:
        c = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    return c if isinstance(c, str) else ""


def load_impressions(root: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for path in sorted(Path(root).glob("*.json")):
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(card, dict):
            continue
        tags = []
        for imp in card.get("impressions") or []:
            tag = imp.get("tag") if isinstance(imp, dict) else imp
            if tag and str(tag).strip():
                tags.append(str(tag).strip())
        if tags:
            out[path.stem] = tags
    return out


def render_labels(tags: list[str]) -> str:
    return "【关于这个用户】TA 是这种人：" + "、".join(tags) + "。"


def render_behaviors(behaviors: list[str]) -> str:
    items = [b for b in behaviors if b and b != "（无）"]
    if not items:
        return ""
    return "【和这个用户相处】" + "；".join(items) + "。"


def _bigrams(text: str) -> set[str]:
    t = re.sub(r"[^一-鿿]", "", text or "")
    return {t[i:i + 2] for i in range(len(t) - 1)}


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


async def rewrite_behaviors(client, model: str, tags: list[str], thinking: dict) -> dict[str, str]:
    """把标签批量改写成行为句（一次调用处理一批，省调用）。"""
    out: dict[str, str] = {}
    for i in range(0, len(tags), 8):
        chunk = tags[i:i + 8]
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": REWRITE_PROMPT.format(tags="、".join(chunk))}],
                temperature=0.3, max_tokens=600, **thinking,
            )
            data = json.loads(_text(resp).strip().strip("`").replace("json\n", "", 1))
        except Exception:
            continue
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str) and v.strip():
                    out[str(k).strip()] = v.strip()
    return out


async def run(*, v1_dir: Path, turns: int, per_user: int, out_dir: Path) -> dict:
    from openai import AsyncOpenAI

    import nonebot

    try:
        nonebot.get_driver()
    except ValueError:
        nonebot.init()

    from src.plugins.chatbot import core
    from src.plugins.chatbot.constants import THINKING_DISABLED

    client = AsyncOpenAI(api_key=_common.get_api_key("OPENAI_API_KEY"),
                        base_url=_common.get_openai_base_url())
    model = _common.get_model_name()

    impressions = load_impressions(v1_dir)
    all_tags = [t for tags in impressions.values() for t in tags]
    print(f"[实验] {len(impressions)} 个用户有性格标签，共 {len(all_tags)} 条；开始改写行为句…", flush=True)
    tag2behavior = await rewrite_behaviors(client, model, sorted(set(all_tags)), THINKING_DISABLED)
    print(f"[实验] 改写完成：{len(tag2behavior)}/{len(set(all_tags))} 条", flush=True)

    out_dir = _v2tool._safe_replay_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = _ab.order_paths_by_activity(_v2tool._resolve_inputs(None))
    want = set(impressions)
    records = []
    for rec in _ab.iter_turns_by_user(paths, limit=5000, per_user=per_user):
        if str(rec["session"]) in want:
            records.append(rec)
        if len(records) >= turns:
            break
    print(f"[实验] 采到 {len(records)} 轮真实对话", flush=True)

    rows = []
    for rec in records:
        uid, msg = str(rec["session"]), str(rec["user"])
        tags = impressions.get(uid, [])
        arm_b = render_labels(tags)
        arm_c = render_behaviors([tag2behavior.get(t, "（无）") for t in tags])
        gen = {}
        # ⚠️ A2 是**对照组**：和 A 完全同一条件再生成一次。CHAT_TEMPERATURE=0.85 很高，
        #    同 prompt 两次也会差很多——没有 A2 就没法把"注入的影响"和"采样噪声"分开。
        for name, block in (("A", ""), ("A2", ""), ("B", arm_b), ("C", arm_c)):
            messages = core.build_message_list(msg, [], "", [], profile_context=block)
            gen[name] = await core.generate_reply(messages)
        rows.append({"user": _v2tool.redact_user_id(uid), "msg": msg, **gen,
                     "tags": tags, "behaviors": [tag2behavior.get(t, "") for t in tags]})
    await client.close()

    # 确定性指标
    n = len(rows)
    def mean(xs): return round(sum(xs) / len(xs), 3) if xs else 0.0
    ident_ab = sum(1 for r in rows if r["A"].strip() == r["B"].strip())
    ident_ac = sum(1 for r in rows if r["A"].strip() == r["C"].strip())
    # 标签泄漏：B/C 里出现了 B 组标签的 bigram（扣掉 A 组也有的公共词）
    base = set()
    for r in rows:
        base |= _bigrams(r["A"])
    def leaked(arm, r):
        return bool((_bigrams(r[arm]) - base) & _bigrams("、".join(r["tags"])))
    report = {
        "turns": n,
        "arms": {
            "A 无注入": {"mean_len": mean([len(r["A"]) for r in rows])},
            "B 标签": {"mean_len": mean([len(r["B"]) for r in rows])},
            "C 行为句": {"mean_len": mean([len(r["C"]) for r in rows])},
        },
        "similarity": {
            # ⚠️ 先看 A_vs_A2：这是**噪声地板**（同条件两次生成）。注入的真效果必须明显低于它。
            "A_vs_A2（噪声地板）": mean([similarity(r["A"], r["A2"]) for r in rows]),
            "A_vs_B": mean([similarity(r["A"], r["B"]) for r in rows]),
            "A_vs_C": mean([similarity(r["A"], r["C"]) for r in rows]),
            "B_vs_C": mean([similarity(r["B"], r["C"]) for r in rows]),
        },
        "identical_reply_rate": {
            "A==A2": round(sum(1 for r in rows if r["A"].strip() == r["A2"].strip()) / n, 3) if n else 0,
            "A==B": round(ident_ab / n, 3) if n else 0,
            "A==C": round(ident_ac / n, 3) if n else 0,
        },
        "tag_leak_rate": {"B": round(sum(leaked("B", r) for r in rows) / n, 3) if n else 0,
                          "C": round(sum(leaked("C", r) for r in rows) / n, 3) if n else 0},
        "rows": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="profile_experiment")
    ap.add_argument("--v1-dir", default=str(DEFAULT_V1_DIR))
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--per-user", type=int, default=6)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args(argv)
    rep = asyncio.run(run(v1_dir=Path(args.v1_dir), turns=args.turns,
                          per_user=args.per_user, out_dir=Path(args.out_dir)))
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
