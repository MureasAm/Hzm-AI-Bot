# -*- coding: utf-8 -*-
"""主动发言：把"她发了动态/微博"变成"她会主动跟你说的一句话"。

**为什么要这个**：现在抓到新动态，推的是一条**结构化通知**——
「灰泽满刚刚发了动态哦！\n\n动态内容：xxx\n\n<空间链接>」。
那一眼就是机器播报，不像她。人不会这样跟朋友说话，她会随口提一句。

**和被动回复的区别**：没有"用户消息"。所以不走 `build_message_list` 那套十层注入，
只用 system_prompt + 性格基底 + **按这条动态检索到的风格样本**（检索仍然有用：
动态讲熬夜，就该捞她讲熬夜时怎么说话）。

**失败/不适合 → 返回 None**，调用方据此决定退回原通知模板还是不发。
"""
import json
import os

from .constants import (
    SYSTEM_PROMPT_FILE, THINKING_DISABLED,
)
from .config import _get_clients, _get_model_name, extract_chat_content
from .persona import load_persona_rules, build_global_persona_context
from .rag import embed_query
from .retrieval import retrieve_voice_samples

# 主动发言的长度上限：私聊里蹦出一大段很出戏
PROACTIVE_REPLY_TRIM = 60

PROACTIVE_PROMPT = """你刚在{source}发了下面这条内容。

【你发的内容】
{content}

现在你想在私聊里跟绿冻**随口提一句**这件事。写出你会发的那句话。

要求：
- **就一句话，短**，像 QQ 私聊随手打的，不是发公告
- 用"灰泽满"自称（你的说话习惯）
- **绝对别像播报**：不要"我发了动态""快去看看"这种，就像跟朋友顺口提一句
- ⚠️ **但"随口"指的是口气，不是少说信息。** 这条内容里对绿冻**有用的东西**
  （什么时间、去哪儿、做什么、有什么看点）**必须说出来**——时间和那件事本身一个字都不能省。
  踩坑：曾把"明天来玩这个音游"缩成"明天音游有翻唱"，**粉丝该被邀请的那件事没了**。
- 不要贴链接、不要提"B站/微博/动态"这些平台词
- 可以有态度、可以自嘲、可以吐槽，别干巴巴复述内容
- 如果这条**不适合**拿来主动找人聊天（纯转发、抽奖、广告、只是打卡签到、
  转发别人的东西），只输出 NULL

只输出你要说的那句话（或 NULL），不要引号、不要任何前缀解释。"""


def proactive_enabled() -> bool:
    """环境变量 PROACTIVE=0 可整体关掉（退回原来的通知模板），缺省开。"""
    return os.environ.get("PROACTIVE", "1") != "0"


async def compose_proactive(content: str, source: str = "B站") -> str | None:
    """把一条动态/微博转成她会主动说的私聊消息；不适合主动说返回 None。

    失败（异常/空）也返回 None —— 由调用方决定退回通知模板还是不发。
    """
    text = (content or "").strip()
    if not proactive_enabled() or not text:
        return None

    try:
        deepseek_client, zhipu_client = _get_clients()
        system = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
        traits, styles, _ = load_persona_rules()
        persona = build_global_persona_context(traits, styles)
        if persona:
            system += "\n\n" + persona

        messages = [{"role": "system", "content": system}]

        # 按这条动态检索风格样本：动态讲熬夜，就捞她讲熬夜时怎么说话。
        # 检索失败不影响主流程（宁可不给样本，也别因为检索挂了就不发）。
        try:
            vector = await embed_query(zhipu_client, text)
            for s in retrieve_voice_samples(text, vector)[:2]:
                u, r = s.extra.get("user", ""), s.extra.get("reply", "")
                if u and r:
                    messages.append({"role": "user", "content": u})
                    messages.append({"role": "assistant",
                                     "content": r[:PROACTIVE_REPLY_TRIM]})
        except Exception as e:
            print(f"⚠️ 主动发言：风格样本检索失败（照常生成）: {e}")

        messages.append({"role": "user",
                         "content": PROACTIVE_PROMPT.format(source=source, content=text)})

        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=messages,
            temperature=0.9,
            max_tokens=120,
            **THINKING_DISABLED,
        )
        out = extract_chat_content(resp).strip().strip('"').strip("“”")
        if not out or out.upper().startswith("NULL"):
            print("[主动发言] 判定不适合主动说")
            return None
        # 只取第一行：模型偶尔会多写一句
        out = out.split("\n")[0].strip()
        return out or None
    except Exception as e:
        print(f"⚠️ 主动发言生成失败（返回 None）: {e}")
        return None
