#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判别方法对照实验：**同一个任务，不同判据，谁最好用。**

## 为什么做

链路里每一路都在"判"，但判据是历史演化出来的，从没横向比过。
候选方法其实不止三种（见 `注入设计原理.md` 第二节），本脚本把常用的几种放在**同一份标注集**上比。

## ⚠️ 地基是标注，标注是人定的

比较方法的前提是"正确答案"。而正确答案**不能由 LLM 生成**——那是循环论证
（本仓踩过：`待办清单.md` 里"合成场景自测 = 8/8，但毫无意义"）。
所以这个脚本里的标注是**手工标注**，来源：
  · `scripts/retrieval_eval_cases.json`（项目已有的精选回归）
  · `data/chat_log/chat.jsonl`（真实对话里**有明确依据**的）
  · `记录.txt`（用户手放的真实翻车）
**边界案例单独标出来**（`borderline`），它们的标签最该由人来裁决。

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
# ⚠️ **标注在 `scripts/judge_labels.json`，该文件优先**（人改那份，不用碰代码）。
# 下面这份是内置兜底（文件不存在时用）。
# want: True=该命中  False=不该命中。  borderline=True 的标签最该由人裁决。
LABELS_FILE = ROOT / "scripts" / "judge_labels.json"

TASKS_FALLBACK = {
    # ---------- 第 5 路：偏好该不该注入 ----------
    "preference": {
        "desc": "用户这句话，该不该注入对应那条偏好",
        "cases": [
            # —— 正例（有明确依据）——
            {"q": "中秋节了，hzm吃月饼了吗", "want": True, "why": "问吃的→食物偏好"},
            {"q": "好吧，hzm爱吃月饼吗", "want": True, "why": "问爱吃什么→食物偏好"},
            {"q": "你熬夜一般干啥", "want": True, "why": "直接问作息"},
            {"q": "你睡了吗", "want": True, "why": "作息话题"},
            {"q": "你喜欢吃什么呀", "want": True, "why": "评测集 pref1"},
            {"q": "你怕不怕猫", "want": True, "why": "评测集 pref2"},
            {"q": "你会玩宝可梦吗", "want": True, "why": "评测集 pref3"},
            {"q": "真理X：/抓水母", "want": True, "borderline": True, "why": "水母（群聊动作，算不算提话题？）"},
            {"q": "满神有没有那种说华流是最吊的，十张专辑封面全是欧洲的流行乐歌手推荐", "want": True,
             "borderline": True, "why": "歌手话题，但问的是'推荐别人'不是'你喜欢谁'"},
            {"q": "宝宝我明天要早起我先去睡了", "want": True, "borderline": True,
             "why": "用户在说自己，不是问她——但作息话题确实相关"},
            # —— 反例 ——
            {"q": "今天天气不错，出去走走吗", "want": False, "why": "评测集 co4"},
            {"q": "9点是你那边", "want": False, "why": "评测集 co3"},
            {"q": "灰泽满是不是很懒", "want": False, "why": "评测集 co5"},
            {"q": "明天几点开会", "want": False, "why": "评测集 co7"},
            {"q": "感冒吃什么药", "want": False, "why": "问药，不是问她"},
            {"q": "我要听你的声音", "want": False, "why": "求语音，与偏好无关"},
            {"q": "亲一个嘛，亲一个，不亲我就跳了", "want": False, "why": "撒娇/越界"},
            {"q": "能再发几个表情包吗满神", "want": False, "why": "求表情包"},
            {"q": "你这个声音是怎么调试的这么真实的啊", "want": False, "why": "问技术"},
            {"q": "睡不着好难过", "want": False, "why": "在倾诉情绪，不是问作息"},
            {"q": "小满 给我推荐一首歌", "want": False, "why": "求推荐，不是问她喜欢什么"},
            {"q": "你笑死我了", "want": False, "why": "纯情绪"},
        ],
    },
    # ---------- corpus：该不该唤起某段经历 ----------
    "corpus": {
        "desc": "用户这句话，该不该唤起她某段直播经历",
        "cases": [
            # —— 正例 ——
            {"q": "你不是以前有喜欢的男学霸吗，你直播里说的", "want": True, "why": "明确指向她的经历"},
            {"q": "你就说过最近直播里", "want": True, "why": "明确指向直播内容"},
            {"q": "你在哪里直播", "want": True, "why": "她的日常/所在地"},
            {"q": "什么平台的直播间", "want": True, "borderline": True, "why": "可能与经历相关，也可能是事务"},
            {"q": "你多高啊", "want": True, "why": "评测集 co6（身高那条）"},
            {"q": "你和女同学一起上学的事", "want": True, "why": "评测集 co1"},
            {"q": "你和室友关系怎么样", "want": True, "why": "评测集 co2"},
            # —— 反例 ——
            {"q": "今天天气不错，出去走走吗", "want": False, "why": "评测集 co4"},
            {"q": "9点是你那边", "want": False, "why": "评测集 co3"},
            {"q": "灰泽满是不是很懒", "want": False, "why": "评测集 co5"},
            {"q": "明天几点开会", "want": False, "why": "评测集 co7"},
            {"q": "感冒吃什么药", "want": False, "why": "评测集 co8"},
            {"q": "我要听你的声音", "want": False, "why": "求语音"},
            {"q": "亲一个嘛，亲一个，不亲我就跳了", "want": False, "why": "撒娇越界"},
            {"q": "能再发几个表情包吗满神", "want": False, "why": "求表情包"},
            {"q": "你这个声音是怎么调试的这么真实的啊", "want": False, "why": "问技术"},
            {"q": "宝宝我明天要早起我先去睡了", "want": False, "why": "日常告知"},
            {"q": "我是特别特别想你的意思不是想死", "want": False, "why": "澄清情绪"},
            {"q": "満神，我感觉你的直播间氛围太好了，就像在恬静的乡下和邻里聊家长里短", "want": False,
             "why": "夸奖直播间氛围，不是问她的经历"},
        ],
    },
}


# ==================== 判据实现 ====================

def _bigrams(t: str) -> list:
    t = re.sub(r'[\s，。！？、；：""''（）【】()\[\]]+', '', t)
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
    """优先读 judge_labels.json（人改那份）；没有就用内置兜底。"""
    if LABELS_FILE.exists():
        try:
            d = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
            tasks = d.get("tasks") or {}
            if tasks:
                print(f"（标注来自 {LABELS_FILE.name}）")
                return tasks
        except Exception as e:
            print(f"⚠️ {LABELS_FILE.name} 读不出来，用内置兜底：{e}")
    return TASKS_FALLBACK


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
