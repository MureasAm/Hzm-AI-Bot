#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判别方法对照实验：**同一个任务，不同判据，谁最好用。**

## 为什么做

链路里每一路都在"判"，但判据是历史演化出来的，从没横向比过。
候选方法其实不止三种（见 `注入设计原理.md` 第二节），本脚本把常用的几种放在**同一份标注集**上比。

## ⚠️ 地基是标注，标注是人定的

比较方法的前提是"正确答案"。而正确答案**不能由 LLM 生成**——那是循环论证
（本仓踩过：`待办清单.md` 里"合成场景自测 = 8/8，但毫无意义"）。
所以这个脚本里的标注是**手工标注**，而且**只在 `scripts/retrieval_eval_cases.json` 一处维护**
（`want` 由该文件的 `expect` 推导，本文件不存标签——存两份必然打架）。
**边界案例标了 `borderline`**，它们的标签最该由人来裁决。

## 指标

- **误报率（FP）** 是主指标——本仓一贯取向"宁可漏不可错"（想起错的经历是主动伤害）。
- 召回率（TP）次之。
- 两个一起报，免得"把所有东西都判成不该命中"也能拿满分。

用法：
    python scripts/judge_experiment.py                  # 跑全部任务
    python scripts/judge_experiment.py --task preference
"""
import argparse
import asyncio
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nonebot  # noqa: E402
nonebot.init()

from openai import AsyncOpenAI  # noqa: E402
from src.plugins.chatbot import retrieval as R  # noqa: E402
from src.plugins.chatbot.rag import embed_query  # noqa: E402
from src.plugins.chatbot.routing import classify_l3  # noqa: E402
from src.plugins.chatbot.corpus_judge import judge_corpus  # noqa: E402
from src.plugins.chatbot.persona import load_persona_rules  # noqa: E402
from src.plugins.chatbot.constants import (  # noqa: E402
    ZHIPU_BASE_URL, DEEPSEEK_BASE_URL, THINKING_DISABLED, RAG_THRESHOLD,
)

ENV_FILE = ROOT / ".env.prod"


def _key(name: str) -> str:
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(name):
            return line.split("=", 1)[1].replace('"', "").strip()
    return ""


# ==================== 标注集 ====================
# ⚠️ **标注只在 `scripts/retrieval_eval_cases.json` 一处维护**（2026-09-29 起）。
#
# 历史教训：这里原来还有一份 `TASKS_FALLBACK` 内置字典，加上 `scripts/judge_labels.json`，
# 同一个东西存了三份 —— 于是同一条标注在三个地方有两个值
# （「小满 给我推荐一首歌」：文件里 true、兜底里 false、导出件写否）。
# **维护两份必然咬人**，所以现在 want 由评测集**推导**，本文件不再存任何标签。
CASES_FILE = ROOT / "scripts" / "retrieval_eval_cases.json"

TASK_DESC = {
    "preference": "用户这句话，该不该注入对应那条偏好",
    "corpus": "用户这句话，该不该唤起她某段直播经历",
}

# ==================== 判据实现 ====================

def _bigrams(t: str) -> list:
    # ⚠️ 原来这里写的是 `r'[...""''...'` —— 中间那对 `''` 把字符串**截断**了，
    # 后半段成了非 raw 字符串，`\[` 就成了非法转义（每次 import 都刷一条 SyntaxWarning）。
    # 现在写成单个 raw 字符串，**字符类与原来逐字符相同**（原来那对 `''` 其实被字面量边界吃掉了）。
    t = re.sub(r'[\s，。！？、；：""（）【】()\[\]]+', '', t)
    return [t[i:i + 2] for i in range(len(t) - 1)] or [t]


class BM25:
    """字符 bigram 的 BM25。不装依赖，够用。"""

    def __init__(self, docs: list, k1=1.5, b=0.75):
        self.k1, self.b, self.docs = k1, b, docs
        self.toks = [_bigrams(d) for d in docs]
        self.avg = sum(len(t) for t in self.toks) / max(len(self.toks), 1)
        self.df = Counter()
        for t in self.toks:
            self.df.update(set(t))
        self.N = len(docs)

    def score(self, i: int, q: str) -> float:
        q_t = set(_bigrams(q))
        tf = Counter(self.toks[i])
        s = 0.0
        for w in q_t:
            if w not in tf:
                continue
            idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
            s += idf * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * len(self.toks[i]) / self.avg))
        return s

    def top(self, q: str, n: int) -> list:
        return sorted(range(self.N), key=lambda i: -self.score(i, q))[:n]


def load_tasks() -> dict:
    """从评测集推导每个任务的 (query, want) —— **单一真值**。

    推导规则（看 `expect.<route>` 里写了什么）：
        should / contains / should_fire  → want=True  （该路该出东西）
        should_not_hit                   → want=False （该路该是空的）
    该 route 两边都没写的 query → 不参与这个任务。
    ⚠️ `contains` 也算"该命中"——它是"命中且 top 文本含关键词"，
      只把它当 should_not_hit 的反面会**漏掉走 contains 的正例**（踩过：co1/co2/co6 被判成不命中）。
    `borderline` 原样带过来 —— **边界案例的标签直接决定'哪个判据赢'**，最该由人裁决。
    """
    raw = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    tasks = {}
    for task, desc in TASK_DESC.items():
        cases = []
        for c in raw.get("cases", []):
            cond = c.get("expect", {}).get(task)
            if not cond:
                continue
            if "should_not_hit" in cond:
                want = False
            else:
                want = bool("should" in cond or "contains" in cond or cond.get("should_fire"))
            item = {"q": c["query"], "want": want, "why": c.get("why", "")}
            if c.get("borderline"):
                item["borderline"] = True
            cases.append(item)
        tasks[task] = {"desc": desc, "cases": cases}
    print(f"（标注来自 {CASES_FILE.name} 推导："
          + "、".join(f"{k} {len(v['cases'])} 条" for k, v in tasks.items()) + "）")
    return tasks


def _parse_json(c: str):
    if "```json" in c:
        c = c.split("```json")[1].split("```")[0]
    elif "```" in c:
        c = c.split("```")[1].split("```")[0]
    return json.loads(c.strip())


# ==================== 各路的方法 ====================

def methods_preference(env):
    """偏好：该不该注入。返回 {方法名: fn(query)->bool}"""
    prefs = R.load_preferences()

    def by_keyword(q):
        return bool(R.retrieve_preferences(q))

    def by_cosine(q):
        v = env["embed"](q)
        best = max((R.cosine_similarity(v, e["_vec"]) for e in env["pref_vecs"]), default=0)
        return best >= 0.55           # 原来的阈值

    def by_bm25(q):
        return bool(env["pref_bm25"].top(q, 1)) and env["pref_bm25"].score(
            env["pref_bm25"].top(q, 1)[0], q) > 3.0

    async def by_llm(q):
        defs = "\n".join(f"- {e['id']}（{e.get('category','')}）：{e.get('text','')[:60]}" for e in prefs)
        p = (f"用户对虚拟主播灰泽满说了一句话。她的偏好档案如下：\n{defs}\n\n"
             f"用户消息：{q}\n\n"
             "判断：**用户这句话是不是在聊上面某个话题、需要她的偏好信息？**\n"
             "⚠️ 只有冲着这个话题来的才算。用户说自己的事、求语音、撒娇、问技术 → 都不算。\n"
             "只输出 JSON：{\"hit\": true/false}")
        r = await env["ds"].chat.completions.create(
            model=env["model"], messages=[{"role": "user", "content": p}],
            temperature=0, max_tokens=20, **THINKING_DISABLED)
        try:
            return bool(_parse_json(r.choices[0].message.content or "").get("hit"))
        except Exception:
            return False

    return {"子串命中(现方案)": by_keyword, "余弦≥0.55(旧方案)": by_cosine,
            "BM25": by_bm25, "LLM 判": by_llm, }


def methods_corpus(env):
    """corpus：该不该唤起某段经历。"""
    db = R.load_vector_db()

    def by_cosine(q):
        v = env["embed"](q)
        return max((R.cosine_similarity(v, it["vector"]) for it in db), default=0) >= RAG_THRESHOLD

    def by_gate(q):
        v = env["embed"](q)
        return bool(R.retrieve_corpus(q, v))

    async def by_gate_llm(q):
        v = env["embed"](q)
        direct = R.retrieve_corpus(q, v)
        ids = {i.item_id for i in direct}
        cands = [c for c in R.retrieve_corpus_candidates(q, v) if c.item_id not in ids]
        return bool(direct or await judge_corpus(env["ds"], q, cands))

    async def by_bm25_llm(q):
        top = env["corpus_bm25"].top(q, 6)
        cands = [R.RetrievalItem(source="corpus", item_id=str(i), score=0.0, text=db[i]["text"])
                 for i in top]
        return bool(await judge_corpus(env["ds"], q, cands))

    async def by_hyde_llm(q):
        # HyDE：先让 LLM 写一段"假想的她的相关经历"，再拿它去检索 → 治"问句 vs 陈述不同构"
        p = ("灰泽满是一个虚拟主播。用户问了她一句话，请用【她的经历陈述口吻】"
             "写一句**假想的**与这个问题最相关的往事（不知道真假，只要形态像）——\n"
             "格式：'灰泽满曾经…'\n\n用户消息：" + q + "\n\n只输出那一句话。")
        r = await env["ds"].chat.completions.create(
            model=env["model"], messages=[{"role": "user", "content": p}],
            temperature=0, max_tokens=80, **THINKING_DISABLED)
        hypo = (r.choices[0].message.content or "").strip()
        v = await embed_query(env["zp"], hypo)          # 假想答案的向量（不在缓存里，现算）
        order = sorted(range(len(db)), key=lambda i: -R.cosine_similarity(v, db[i]["vector"]))
        cands = [R.RetrievalItem(source="corpus", item_id=str(i), score=0.0, text=db[i]["text"])
                 for i in order[:6]]
        return bool(await judge_corpus(env["ds"], q, cands))

    async def by_union_llm(q):
        """多路召回取**并集**再交 LLM 判：余弦 / BM25 / HyDE 各出一批候选。
        单路的短板是"候选池里没有正确的"，并集把三种互补的召回合并，判据仍只有一个（LLM）。"""
        v = env["embed"](q)
        idx = set()
        idx |= {i for i in sorted(range(len(db)), key=lambda i: -R.cosine_similarity(v, db[i]["vector"]))[:6]}
        idx |= set(env["corpus_bm25"].top(q, 6))
        p = ("灰泽满是一个虚拟主播。用户问了她一句话，请用【她的经历陈述口吻】"
             "写一句**假想的**与这个问题最相关的往事（不知道真假，只要形态像）——\n"
             "格式：'灰泽满曾经…'\n\n用户消息：" + q + "\n\n只输出那一句话。")
        r = await env["ds"].chat.completions.create(
            model=env["model"], messages=[{"role": "user", "content": p}],
            temperature=0, max_tokens=80, **THINKING_DISABLED)
        hv = await embed_query(env["zp"], (r.choices[0].message.content or "").strip())
        idx |= {i for i in sorted(range(len(db)), key=lambda i: -R.cosine_similarity(hv, db[i]["vector"]))[:6]}
        cands = [R.RetrievalItem(source="corpus", item_id=str(i), score=0.0, text=db[i]["text"])
                 for i in sorted(idx)]
        return bool(await judge_corpus(env["ds"], q, cands, ))

    return {"余弦≥0.48 + 关键词门": by_gate, "纯余弦≥0.48": by_cosine,
            "门 + LLM 判(现方案)": by_gate_llm, "BM25 + LLM 判": by_bm25_llm,
            "HyDE + LLM 判": by_hyde_llm, "余弦+BM25+HyDE 并集 + LLM": by_union_llm, }


# ==================== 跑 ====================

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all")
    args = ap.parse_args()

    ds = AsyncOpenAI(api_key=_key("OPENAI_API_KEY"), base_url=DEEPSEEK_BASE_URL)
    zp = AsyncOpenAI(api_key=_key("ZHIPU_API_KEY"), base_url=ZHIPU_BASE_URL)
    model = _key("OPENAI_MODEL") or "deepseek-flash"
    _, _, behaviors = load_persona_rules()

    # 预热：偏好向量（旧方案要）+ BM25 索引
    prefs = R.load_preferences()
    pref_vecs = []
    for e in prefs:
        v = await embed_query(zp, e.get("text", ""))
        pref_vecs.append({**e, "_vec": v})

    cache: dict = {}

    def embed(q):
        """同步取缓存向量。所有 case 的 query 在跑之前会统一预热，所以这里必定命中。"""
        if q not in cache:
            raise KeyError(f"query 没预热：{q[:30]}")
        return cache[q]

    env = {
        "ds": ds, "zp": zp, "model": model, "behaviors": behaviors,
        "pref_vecs": pref_vecs,
        "pref_bm25": BM25([e.get("text", "") for e in prefs]),
        "corpus_bm25": BM25([it["text"] for it in R.load_vector_db()]),
        "embed": embed,
    }

    for task, spec in load_tasks().items():
        if args.task not in ("all", task):
            continue
        cases = spec["cases"]
        # 预先把所有 query 的 embedding 算好（同步方法里要用）
        for c in cases:
            if c["q"] not in cache:
                cache[c["q"]] = await embed_query(zp, c["q"])

        ms = methods_preference(env) if task == "preference" else methods_corpus(env)
        print(f"\n{'='*74}\n【{task}】{spec['desc']}   标注 {len(cases)} 条"
              f"（其中边界案例 {sum(1 for c in cases if c.get('borderline'))} 条）\n")

        rows = []
        for name, fn in ms.items():
            tp = fp = tn = fn_ = 0
            errs = []
            for c in cases:
                try:
                    got = fn(c["q"])
                    if asyncio.iscoroutine(got):
                        got = await got
                except Exception as e:
                    got = False
                    errs.append(f"{c['q'][:16]}:{type(e).__name__}")
                w = bool(c["want"])
                if w and got:
                    tp += 1
                elif w and not got:
                    fn_ += 1
                    errs.append(f"漏(应中未中)「{c['q'][:20]}」")
                elif not w and got:
                    fp += 1
                    errs.append(f"**误报**「{c['q'][:20]}」")
                else:
                    tn += 1
            prec = tp / (tp + fp) if tp + fp else float("nan")
            rec = tp / (tp + fn_) if tp + fn_ else float("nan")
            rows.append((name, tp, fp, fn_, tn, prec, rec, errs))

        print(f"{'方法':<24}{'中/该中':>8}{'误报':>6}{'漏':>5}{'准确率':>9}{'召回率':>9}")
        print("-" * 66)
        for name, tp, fp, fn_, tn, prec, rec, errs in rows:
            print(f"{name:<24}{tp:>4}/{tp+fn_:<4}{fp:>6}{fn_:>5}{prec:>9.2f}{rec:>9.2f}")
        print("\n各方法的错在哪：")
        for name, *_r in rows:
            errs = _r[-1]
            if errs:
                print(f"  {name}: " + " ｜ ".join(errs[:4]))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    asyncio.run(main())
