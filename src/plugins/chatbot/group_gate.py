# -*- coding: utf-8 -*-
"""群聊接话判定：决定灰泽满在群里这一批要不要开口。

**为什么需要它**：群聊里"有话就接"不像群友——真人只挑着接。
实测（2026-09-25，拿真实群聊记录回放）：不判定的话她每条都接；
加了这个门之后她只在"有人表达情绪 / 有人分享经历 / 被点名"时开口，其余安静看着，
而且接话内容也从"一个合理的群友"变回"灰泽满本人"（因为走的是完整链路）。

**判定粒度是"批"，不是单条**：读秒窗口攒出来的那批就是她的一个观察单位。
真人也是看一群人聊了一阵、等安静下来，再决定要不要插一句。
（逐条判会"蹭余温"——情绪一出现，后面每条都靠它的余温被判成该接。实测过。）

**判据**（用户 2026-09-25 标定，2026-09-26 补第 ⑤ 档）：
  ① 有人明确问她 / 点她名 / 引用她的话 → **必接**
  ② 本批有情绪表达（"我服了""好惨"，**纯情绪也算，不需要有具体事件**）→ 可以接
  ③ 本批有人分享自己的经历或状态 → 偶尔接
  ④ 纯事务（发文件/要不要转/在不在）、纯附和（"哈哈哈""对""。。。"）、收尾语 → 不接
  ⑤ 纯事件（陈述一件跟她无关的事，如"明天要下雨"）→ **按概率接**（见 constants.GROUP_EVENT_REPLY_PROB）

⚠️ **只在群聊用**。私聊被人直接找来必回，不过这个门（见 chat_window._flush）。
"""
import json
import os
import random

from .constants import THINKING_DISABLED, AT_SELF_MARK, GROUP_EVENT_REPLY_PROB
from .config import _get_model_name, extract_chat_content

GROUP_GATE_PROMPT = """灰泽满是一个 QQ 群里的虚拟主播，正混在这个群里。她本人不在下面这些发言者里。

【群里刚才聊的】
{history}

【刚安静下来的一批】
{batch}

\
这一批消息发完了、群里安静下来了。判断：**她要开口说一句吗？**

判断对象是**整批**——看这段话聊完，她作为一个群友想不想插一句。
⚠️ 只看**这批本身**，不要因为前面几批有人感慨过，就觉得这批也该接。
⚠️ 情绪在群里很常见，真人只在"这批里确实有我想接的东西"时才开口。

判据（并给出这批判成了哪一档 kind）：
1. 有人明确问她、点她名、引用她的话 → **必须接**（kind=点名）
2. 这批里有**情绪表达**（"我服了""好惨""笑死了"，纯情绪也算，不需要有具体事件）→ **可以接**（kind=情绪）
3. 这批里有人在**分享自己的经历或状态** → **偶尔**接一句（kind=经历）
4. 这批里只是在**陈述一件跟自己无关的事**（天气、新闻、某个东西坏了/更新了）→ 少接（kind=事件）
5. 纯事务（发文件、要不要转、在不在）、纯附和（"哈哈哈""对""。。。"）、收尾语 → **不接**（kind=事务）

kind 填最贴近的那**一档**（一档就好，别自己造新词）。

只输出 JSON：{{"reply": true 或 false, "kind": "点名|情绪|经历|事件|事务", "why": "不超过25字"}}"""


def gate_enabled() -> bool:
    """环境变量 GROUP_GATE=0 可整体关掉（A/B 与快速回滚用），缺省开。"""
    return os.environ.get("GROUP_GATE", "1") != "0"


# 她自称/被叫的名字。消息里出现这些 → **一定是在跟她说话**，不用判。
# 别名表是**兜底**：@ 本身由 AT_SELF_MARK 覆盖（at 段在纯文本里看不见），
# 这里是"群友手打她名字"那条路。别往里塞太泛的字（"满"单独出现多半不是叫她）。
_SELF_NAMES = ("灰泽满", "hzm", "小满", "满姐", "满宝", "满哥",
               "灰灰", "小灰", "满区")


def _addressed_to_her(batch_text: str) -> bool:
    """这批消息里有没有直接点她的名 / @ 她——**确定性信号，不该交给 LLM 判**。

    踩坑（2026-09-25）：群里 @了她、还写了名字，却被门判成"不用接"。
    判据本身没问题（提示词第一条就是"被点名必须接"），但**能确定的事不该调 LLM**——
    多一次调用就多一次判错的机会，而这个信号是白纸黑字的。

    坑二（2026-09-26）：光有名字表不够——**@ 是独立消息段**，
    `extract_plain_text()` 看不见，「@她 + 正文」进来时正文里根本没有名字。
    现在由 `__init__._extract_at_self` 在消息前插 `AT_SELF_MARK`，这里连着一起认。
    """
    t = batch_text or ""
    return AT_SELF_MARK in t or any(n in t for n in _SELF_NAMES)


async def should_reply_in_group(deepseek_client, history_text: str, batch_text: str) -> bool:
    """群里这一批，灰泽满要开口吗？

    **失败一律返回 True（宁滥勿缺）**：判不出来就让她回，别因为一次上游抖动
    把她变成哑巴——"偶尔安静"是风格，"该说时不说"是故障，后者更难发现也更伤。
    """
    if not gate_enabled():
        return True
    if not (batch_text or "").strip():
        return False
    # 点名了她 → 直接接，不调 LLM（确定性信号优先）
    if _addressed_to_her(batch_text):
        print("[群聊接话] 接（点名了她——确定性放行，不判定）")
        return True

    prompt = GROUP_GATE_PROMPT.format(
        history=(history_text or "").strip()[-1500:] or "（这是群里最早的一批）",
        batch=batch_text.strip(),
    )
    try:
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
            **THINKING_DISABLED,
        )
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
        take = bool(data.get("reply"))
        kind = str(data.get("kind") or "").strip()
        why = str(data.get("why") or "")[:30]
        # 纯事件档：**判据自己定不了**——真人是"大多数时候不接、偶尔插一句"，
        # 那是个概率不是一个是非。所以 LLM 说"不接"时改成掷骰子。
        # ⚠️ 只**加**不**减**：LLM 明确说"接"就照接（和别档一样），
        # 不能让骰子把她已经判成该说的话又吞掉——"该说时不说"是这个门最怕的故障。
        if kind in ("事件", "event") and not take:
            take = random.random() < GROUP_EVENT_REPLY_PROB
            print(f"[群聊接话] {'接' if take else '不接'}（纯事件，"
                  f"{int(GROUP_EVENT_REPLY_PROB * 100)}% 概率放行）")
            return take
        print(f"[群聊接话] {'接' if take else '不接'}"
              f"{'（' + kind + '）' if kind else ''}{'（' + why + '）' if why else ''}")
        return take
    except Exception as e:
        print(f"⚠️ 群聊接话判定失败（放行=接）: {e}")
        return True
