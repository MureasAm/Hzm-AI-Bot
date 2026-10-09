# -*- coding: utf-8 -*-
"""主动发言：把"她发了动态/微博"变成"她会主动跟你说的一句话"。

**为什么要这个**：现在抓到新动态，推的是一条**结构化通知**——
「灰泽满刚刚发了动态哦！\n\n动态内容：xxx\n\n<空间链接>」。
那一眼就是机器播报，不像她。人不会这样跟朋友说话，她会随口提一句。

**素材走主链路**：跟聊天**共用** `core.gather_retrieval` + `core.build_message_list`——
人设骨架 / 周表 / 当前时间（含直播状态、天气）/ 名词库 / 偏好 / 核心记忆，全部按这条动态检索一遍。
（原来这里是自己拼的一套：system_prompt + 自己写的规则 + 几条样本——**素材层整层没接**，
输出干巴、没有生活感；用户 2026-09-29 指出"只在一个代码里做了提示词和调用"，就是这个。）

**跟聊天唯一的区别**：没有"用户消息"，这条动态是**她自己发的**。所以
`gather_retrieval(..., is_user_msg=False)`：跳过两个前提不成立的层
（L3 行为 = "用户这句想干嘛"；corpus 判 = "用户是不是在问她这段"），其余照旧。

**两道闸，都只拦"这一句"，原文通知照发**（信息不丢）：
  ① `content_is_clear()`（**开口前**先判断）：内容本身缺信息、单独看不知道在说什么 → 不发。
     典型（2026-09-27 用户报）：她微博写"写不完了"，而直播里说的是"作业写不完了"——
     缺了"写什么"这半边，硬说一句只会说歪或说空。
  ② 生成时的 NULL：模型写着发现**不适合**主动说（纯转发/抽奖/广告/打卡）→ 不发。

**配图也给它看**（`image_paths`）：她在内容里写"⬇️/这个/图片里那个"时指的是配图，
不给图它就只会拿正文里别的名字瞎填（2026-09-27 实测踩坑）。

**任何失败都返回 None**。调用方（bili_bridge / weibo_bridge）**无论成败都会先发原文通知**，
返回 None 只是"少她那句"，信息不会丢（两个桥现在都是「原文 + 她那句」两条）。
"""
import asyncio
import os
import re
from pathlib import Path

from .constants import THINKING_DISABLED
from .config import log_cache_usage, _get_clients, _get_model_name, extract_chat_content, parse_json_block
from .core import gather_retrieval, build_message_list
from .vision import describe_image_bytes

# 主动发言的长度上限：私聊里蹦出一大段很出戏
PROACTIVE_REPLY_TRIM = 60

# 有实义字符（字母/数字/汉字；标点、符号、emoji 都不算）——用来确定性地挡掉"..."这类空内容
_CONTENT_CHAR_RE = re.compile(r"\w")

# 原文里的 emoji（她发动态爱带：🌙💤📑🥺😭…）。**不含箭头 2B00-2BFF**——
# "到这个⬇️里面"那种箭头是**指代配图的标记**，不是表情，带进她的话里就成了乱码。
_EMOJI_PAT = re.compile(
    "[\U0001F000-\U0001FAFF☀-➿]"                  # 基础 emoji（1F000+ 表情 / 2600-27BF 杂项符）
    "(?:[️‍][\U0001F000-\U0001FAFF☀-➿]?)*"   # 跟在后面的变体选择符/零宽连接（如 🌡️）
)


def ensure_trailing_emoji(reply: str, source: str) -> str:
    """把她原文里的 emoji 放到这句话**结尾**（原文有才加）。

    **确定性做法，不走提示词**：她的聊天样本里一个 emoji 都没有（那批是直播口语
    转写，说话打不出 emoji），光靠提示词她基本不会主动用；而"照抄原文那一个"
    本来就是确定的，不需要模型判断。取原文**第一个** emoji（通常跟主句绑在一起；
    结尾那个往往是链接/标签的符号，如 "📦…🔗"）。已经出现过就不重复加。
    """
    emojis = _EMOJI_PAT.findall(source or "")
    if not emojis:
        return reply
    tail = emojis[0]
    if not reply or tail in reply:
        return reply
    return f"{reply}{tail}"

# 内容里**指向配图**的说法（"到这个⬇️里面"、"图里那个"）。
# 只有出现这些才去读配图：① 别的动态里图片跟正文没关系，描述它等于往上下文里塞噪声
# （实测会干扰：条件2 有 2/8 把点名的游戏名泛化成了"游戏"）；
# ② 读图要调一次视觉模型，没必要每次动态都花。
_REFERS_TO_IMAGE_RE = re.compile(r"⬇|⬆|👇|👆|↓|↑|图里|图片里|配图|下图|上图|这个图")

PROACTIVE_PROMPT = """你刚在网上发了一条内容，现在想在私聊里跟绿冻**随口提一句**这件事。写出你会发的那句话。

⚠️ **你是在跟绿冻说话，不是在向别人报告你刚才干了什么。**
反例（错的，读起来像第三人在转述）：
  "灰泽满刚说了句晚安，你也早点睡吧"   ← 这是旁白，不是她在说话
  "灰泽满刚发了条动态，说明天要直播"
正例（对的，就是本人顺口一句）：
  "睡了，你也早点睡"                  ← 内容"晚安，明天见"→ 她本人说的
  "明天来玩啊，别迟到"                ← 内容"明晚八点音游"→ 她本人在邀人

要求：
- **就一句话，短**，像 QQ 私聊随手打的，不是发公告
- **不要复述"灰泽满刚才发了什么/说了什么"**：没有"灰泽满刚……"这种句式——
  你就是正在说这句话的人，不是转述者
- **绝对别像播报**：不要"灰泽满发了动态""快去看看"这种，就像跟朋友顺口提一句
  （**自称怎么写由人设决定**：system_prompt 的【自我称呼】+ 上面的说话片段说了算，
   这里**不要额外规定自称**——曾在此写"自称用'我'最自然"，等于把人格文件覆盖了）
- ⚠️ **但"随口"指的是口气，不是少说信息。** 这条内容里对绿冻**有用的东西**
  （什么时间、去哪儿、做什么、有什么看点）**必须说出来**——时间和那件事本身一个字都不能省。
  踩坑：曾把"明天来玩这个音游"缩成"明天音游有翻唱"，**粉丝该被邀请的那件事没了**。
- ⚠️ **反过来，内容里没写的，一个字也别补。** 她写的是什么、为什么、结果怎样，
  内容没讲就**别猜**——宁可说得笼统，也不许编一个具体的东西出来。
  （原文通知已经把整条原样发给绿冻了，所以这句笼统一点完全没关系；编错才糟。）
  反例（错的）：内容「写完了…终于写完了…」→「**写稿**写得快不行了」
  （"稿"内容里根本没有；而且「终于写完了」是**已经写完**，不该说成"快不行了"）
  正例：「终于写完了，人快没了…晚上来打游戏回回血」
- ⚠️ **别把内容里两件事揉成一件**：「⬇️ / 这个 / 图片里那个」指的是**配图**，
  不是正文里另外那个名字——拿它去填就会把两件事说混。
  反例（错的）：内容「会到这个⬇️里面鲨会儿人」+ 正文另有"20:00多来玩BanG Dream"
  →「晚上八点多来玩BanG Dream! OurNotes嘛，让灰泽满也鲨会儿人」
  （把**两件事揉成一件**，而且鲨人的根本不是 BanG Dream）
  **配图看不清、或这条没配图，就只挑一件说**（优先时间/邀请——那件对绿冻最有用）。
- 不要贴链接、不要提"B站/微博/动态"这些平台词
- 可以有态度、可以自嘲、可以吐槽，别干巴巴复述内容
- 如果这条**不适合**拿来主动找人聊天（纯转发、抽奖、广告、只是打卡签到、
  转发别人的东西），只输出 NULL

只输出你要说的那句话（或 NULL），不要引号、不要任何前缀解释。

【你发的内容（来源：{source}）】
{content}
{image_block}"""


# 开口前的第一道闸：这条内容**说不说得清**？
# 用户 2026-09-27 提：她微博写"写不完了"（直播里说的是"作业写不完了"），
# 单独看这句连"写什么"都不知道——硬说一句只会说歪或说空，不如不说。
# 注意判据只拦"缺信息"，不拦"短"——"晚安，明天见""今天好累"这种短但完整的照样能开口。
PROACTIVE_JUDGE_PROMPT = """灰泽满刚在网上发了条内容，粉丝会在私聊里收到一条"她发了什么"的原文通知，
另外她还打算**顺口跟粉丝说一句**这件事。

判断：**这条内容够不够撑起她顺口说的那一句？**

只有一种情况要拦下来——**内容本身缺了关键信息，单独看不知道在说什么**：
- 缺了"做什么/什么东西/去哪儿"这半边（如"写不完了"——写什么？"到了"——到哪了？"那个真好用"——哪个？）
- 只有半句话、一个词、一个符号、纯表情

⚠️ 必须区分**"短"和"缺"**：
- "晚安，明天见"、"今天好累"、"直播推迟到九点" → **短但完整，不拦**
- "写不完了"、"到了"、"还是不行" → **缺关键信息，拦**

⚠️ 拿不准就**放行**（够=true）——她少说一句没人发现，但把每条都拦下来等于这个功能没了。

只输出 JSON：{{"enough": true 或 false, "why": "不超过20字"}}

【这条内容】
{content}"""


def proactive_enabled() -> bool:
    """环境变量 PROACTIVE=0 可整体关掉（退回原来的通知模板），缺省开。"""
    return os.environ.get("PROACTIVE", "1") != "0"


def judge_enabled() -> bool:
    """环境变量 PROACTIVE_JUDGE=0 可关掉开口前的"说不说得清"判断，缺省开。"""
    return os.environ.get("PROACTIVE_JUDGE", "1") != "0"


async def content_is_clear(deepseek_client, text: str) -> bool:
    """这条内容够不够撑起一句主动搭话？（"写不完了"这种说不清的拦下来）

    **失败一律放行（True）**：和接话门同一个兜底方向——"该说时不说"比"多说一句"
    难发现得多。而且判官挂了时，紧接着的生成调用多半也会挂（同一个客户端），
    实际结果是退回"只发原文通知"，不会出洋相。
    """
    # 纯标点/纯符号/纯表情（"..."、"。。。"、"😭"）没有实义内容——**确定性拦掉，不调 LLM**：
    # 能确定的事别多一次判错的机会（同 group_gate 里"点名了她"那条）。
    # 实测（2026-09-27）：交给 LLM 判时，"..." 会被判成"够"。
    # 注意这条在 judge_enabled() **之前**——它不是"判断"，是"这里没有内容"，
    # 所以 PROACTIVE_JUDGE=0 也拦。
    if not _CONTENT_CHAR_RE.search(text or ""):
        print("[主动发言] 内容没有实义（纯符号/纯表情），不发这句")
        return False
    if not judge_enabled():
        return True
    prompt = PROACTIVE_JUDGE_PROMPT.format(content=text)
    try:
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=80,
            **THINKING_DISABLED,
        )
        log_cache_usage(resp, "主动发言")
        data = parse_json_block(extract_chat_content(resp))
        enough = bool(data.get("enough"))
        why = str(data.get("why") or "")[:25]
        if not enough:
            print(f"[主动发言] 内容说不清，不发这句（{why}）")
        return enough
    except Exception as e:
        print(f"⚠️ 主动发言'说不说得清'判断失败（放行=发）: {e}")
        return True


async def _describe_first_image(zhipu_client, image_paths) -> str:
    """第一张配图的视觉描述，给"⬇️/这个/图片里那个"当**指代目标**；失败返回空串。

    **只描述、不判断**：这里不管内容对不对，只把图里有什么说清楚——
    没有它，模型面对一个箭头只能瞎填（实测：她把"到这个⬇️里面鲨会儿人"和正文里
    另一句"来玩 BanG Dream"揉成了一句，说成"来玩 BanG Dream 顺便鲨会儿人"）。

    不抛异常（与调用方并行跑，抛了会连累整条生成）。
    """
    if not image_paths:
        return ""
    try:
        data = Path(image_paths[0]).read_bytes()
        return await describe_image_bytes(zhipu_client, data)
    except Exception as e:
        print(f"⚠️ 主动发言：配图描述失败（照常生成）: {e}")
        return ""


async def compose_proactive(content: str, source: str = "B站",
                            image_paths: list | None = None) -> str | None:
    """把一条动态/微博转成她会主动说的私聊消息；不适合主动说返回 None。

    **素材走主链路**：`gather_retrieval`（人设 + 按这条内容检索的素材）+
    `build_message_list`（周表 / 感知 / 名词库 / 样本 / 偏好 / 核心记忆 一次到位），
    跟聊天共用同一份人格数据。

    失败（异常/空）也返回 None —— 由调用方决定退回通知模板还是不发。
    """
    text = (content or "").strip()
    if not proactive_enabled() or not text:
        return None

    try:
        deepseek_client, zhipu_client = _get_clients()
        # 配图描述与"够不够"判断**并行**跑：两者互不依赖，串行等于白等多一次调用。
        # 且**只在她指了配图时**才读（"到这个⬇️里面"），否则那条描述是纯噪声。
        want_image = bool(image_paths) and bool(_REFERS_TO_IMAGE_RE.search(text))
        desc_task = (asyncio.create_task(_describe_first_image(zhipu_client, image_paths))
                     if want_image else None)
        try:
            # 第一道闸：内容自己说不说得清。说不清 → 不发这一句（原文通知照发，信息不丢）。
            if not await content_is_clear(deepseek_client, text):
                return None
            image_desc = await desc_task if desc_task else ""
        finally:
            # 判官拦下时别把描述任务挂在那儿（它已经没用了）
            if desc_task and not desc_task.done():
                desc_task.cancel()
        image_block = ""
        if image_desc:
            image_block = (f"\n【这条的配图】{image_desc}"
                           f"\n（她内容里写「⬇️」「这个」「图片里那个」时，指的就是这张图）")
        # 素材：跟聊天同一个入口取。
        # `is_user_msg=False`——这条动态是**她自己发的**，不是"用户说的话"，
        # 所以跳过两个前提不成立的层（L3 行为、corpus 的"用户是不是在问她"判）。
        ctx = await gather_retrieval(text, text, "", deepseek_client, zhipu_client,
                                     is_user_msg=False)

        # 组装：跟聊天同一套（周表 / 当前时间+直播状态 / 名词库 / 偏好 / 核心记忆…）。
        # 记忆两路传空——主动发言是**广播给所有好友**的，没有"这个用户"这回事。
        messages = build_message_list(
            PROACTIVE_PROMPT.format(source=source, content=text, image_block=image_block),
            ctx["fused_items"], "", [],
            preference_items=ctx["preference_items"],
            core_stories=ctx["core_stories"],
        )

        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=messages,
            temperature=0.9,
            max_tokens=120,
            **THINKING_DISABLED,
        )
        log_cache_usage(resp, "主动发言")
        out = extract_chat_content(resp).strip().strip('"').strip("“”")
        if not out or out.upper().startswith("NULL"):
            print("[主动发言] 判定不适合主动说")
            return None
        # 只取第一行：模型偶尔会多写一句
        out = out.split("\n")[0].strip()
        if not out:
            return None
        # 原文带 emoji → 这句话结尾也带一个（照抄原文的，不是自造）
        return ensure_trailing_emoji(out, text)
    except Exception as e:
        print(f"⚠️ 主动发言生成失败（返回 None）: {e}")
        return None
