# -*- coding: utf-8 -*-
"""corpus 语义判：这段经历和用户刚说的话，是不是**一回事**。

**为什么需要它**：corpus 的召回原本只靠「余弦阈值 + 关键词门」。但用户是**口语问句**
（"你多高啊"），库里是**第三人称陈述**（"灰泽满被问到身高，说1米6"）——两种形态不同构，
余弦只有 0.443，低于噪声地板 0.474 → 被当"不相关"扔掉 → **她明明有的经历想不起来，只能现编**。

**为什么不能让向量/词重叠自己判**：实测（2026-09-25）——
  · 「你多高啊」对身高那条 0.443，而**无关**的「明天几点开会」对库里那条「明天几点下课」
    能打出 **0.822**（比部分正例还高）。语义、共享 bigram、IDF 加权重叠**三种都试过，全分不开**。
  · 差别不在文本里，在"她**到底有没有**这段经历"这个**事实**——这是判断，不是相似度。→ 只有 LLM 判得开。
  · LLM 判实测（top-6 候选）：反例 **10/10 全拒**（含「明天几点开会」），正例 2/5 保留。

**判据措辞是本模块最要紧的东西**（改词=改行为，实测过判据稍动结果就从 0% 跳到 74%）：
  · 判的是"**内容是不是同一件事**"，不是"句式像不像"——提示词里用真实反例把它钉死。
  · **同话题也算**（用户说"我室友好吵"，她那段是"室友作息健康"→ 该想起来）。

**兜底方向：失败/判不出来 → 一律不放行**（宁可漏不可错）。
⚠️ 与 `group_gate`（失败放行）**相反**，与 `stickers`（失败不发）一致。
理由：漏掉只是"没加分"，而注入错的经历是**主动伤害**——她会理直气壮地说一件没发生过的事
（"搬家"那个坑就是这么来的）。这是本仓一贯取向：宁可漏，不可错。
"""
import json
import os

from .constants import CORPUS_JUDGE_MAX_KEEP, MAX_RETRIEVAL_ITEM_CHARS, THINKING_DISABLED
from .config import log_cache_usage, _get_model_name, extract_chat_content

CORPUS_JUDGE_PROMPT = """灰泽满是一个虚拟主播。下面是**她本人真实经历过的片段**（不是编的，是她直播里发生过的事）。


\
判断：用户这句话，是不是在**问她、或者聊到**上面哪段经历里的事？

⚠️ 判的是「**用户是不是在问她这件事**」，**不是「有没有撞到同一个词」**：

   ✗ 用户说"感冒吃什么药" → 他在问药，不是在问她感冒过 → 不保留
   ✗ 用户说"9点是你那边" → 他在说时区，不是在问室友 → 不保留
   ✗ 用户说"明天几点开会" → 句式像"她说过明天几点下课"，但不是一回事 → 不保留
   ✗ 用户说"今天天气不错" → 他在聊天气，不是在问她 → 不保留

   ✓ 用户说"你多高啊"，候选是"她被问到身高，说自己1米6" → 措辞完全不同，但是在问她 → 保留
   ✓ 用户说"你和女同学的事"、"你室友平时都干嘛" → 在问她那边的情况 → 保留

⚠️ 只"提到同一个词/同一个话题"**不算**——必须是**冲着她来的**。
⚠️ **拿不准 → 不保留**。她想不起来只是没加分，**想起错的**是让她说一件没发生过的事
   （例：用户问"感冒吃什么药"，她想起自己感冒那段 → 她会说"我最近也感冒了" → 那是错的）。

【用户刚说的话】
{query}

【候选经历】
{candidates}

只输出 JSON：{{"keep": [编号, …]}}；一段都不相关就 {{"keep": []}}"""


def judge_enabled() -> bool:
    """环境变量 CORPUS_JUDGE=0 可整体关掉（快速回滚用），缺省开。

    关掉后 corpus 完全退回"只靠关键词门放行"的老行为（判定的候选一条都不带）。
    """
    return os.environ.get("CORPUS_JUDGE", "1") != "0"


async def judge_corpus(deepseek_client, query: str, candidates: list) -> list:
    """从候选里挑出**真的和用户这句话相关**的经历，返回保留下来的条目。

    candidates 是 retrieval.retrieve_corpus_candidates 给的 RetrievalItem 列表（已按分排序）。
    **一次调用判完所有候选**（不是每条一次）——省调用，且模型能横向比较谁更相关。

    **失败一律返回 []（宁可漏不可错）**：判不出来就不带经历，让她正常按人设回，
    也不要冒着"想起错的"的风险。这跟 group_gate 的失败放行是**相反**的取向。
    """
    items = [c for c in (candidates or []) if getattr(c, "text", "")]
    if not items or not (query or "").strip():
        return []
    if not judge_enabled():
        return []

    numbered = "\n".join(
        f"{k + 1}. {(c.text or '').strip()[:MAX_RETRIEVAL_ITEM_CHARS]}"
        for k, c in enumerate(items)
    )
    prompt = CORPUS_JUDGE_PROMPT.format(query=query.strip(), candidates=numbered)
    try:
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=60,
            **THINKING_DISABLED,
        )
        log_cache_usage(resp, "直播记忆判")
        content = extract_chat_content(resp)
        # 剥 ``` 围栏。**要连语言标签一起剥**（模型常返回 ```json），
        # 只取 ``` 之间那段的话，剩下的 `json\n{...}` 会让 json.loads 直接失败。
        if "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
        content = content.strip()
        if content[:4].lower() == "json":
            content = content[4:].strip()
        data = json.loads(content)
        keep = data.get("keep")
        if not isinstance(keep, list):
            print("⚠️ corpus 判定返回的 keep 不是列表（不带经历）")
            return []
        # 拦住模型编的编号——只认候选范围内、且不重复的
        picked = []
        for k in keep:
            if not isinstance(k, int) or isinstance(k, bool):
                continue
            if 1 <= k <= len(items) and items[k - 1] not in picked:
                picked.append(items[k - 1])
        if len(picked) > CORPUS_JUDGE_MAX_KEEP:      # 限爆破半径：判错也只是多说一句
            picked = picked[:CORPUS_JUDGE_MAX_KEEP]
        print(f"[corpus判定] 候选{len(items)}条 → 保留{len(picked)}条")
        return picked
    except Exception as e:
        print(f"⚠️ corpus 判定失败（不带经历）: {e}")
        return []
