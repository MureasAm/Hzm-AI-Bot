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

**投递**（2026-10-23 用户定）：主动发言生成后只发她那句——语音优先，失败/不适合再发文字；
不再推动态/微博原文，配图随她说的话一起发。`broadcast_proactive` 负责广播：
**同一句话只合成一次 TTS**，再按并发上限并行发给所有目标（300 人时这是关键）。
"""
import asyncio
import os
import re
from pathlib import Path

from .constants import THINKING_DISABLED
from .config import (
    log_cache_usage, _get_clients, _get_model_name, extract_chat_content,
    parse_json_block, set_llm_route, get_vision_client,
    aux_clients_or, aux_model,   # 辅助链（2026-10-29）
)
from .core import gather_retrieval, build_message_list
from .memory import append_bot_message
from ._bridge_common import to_memory_line
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

# ⚠️ **这版删掉了所有"正例"**（2026-10-14，用户报了三类怪句后实测复现）：
# 原来 prompt 里写着两条当作范例的句子——"睡了，你也早点睡"（配"晚安"内容）和
# "明天来玩啊，别迟到"（配"几点播"内容）。**模型把它们当标准答案逐字抄了**：
#   动态"晚安🌙" → 她说"睡了，你也早点睡🌙"（每次晚安都一样）
#   动态"明天18:00播吧" → 她说"明天18:00来玩啊，别迟到"（每次通知开播都带"来玩X"）
# 拿真实动态真跑 10 条复现过（`scripts/proactive_ab.py`）。**而且加了更多正例会更糟**：
# 用户 2026-10-14 提的"不好的话就多加几条正例候选"那一版，模型直接把新例句原样吐了出来
# 两次（"不是，这早饭也太万圣节了"／"行，21:30，别又说灰泽满鸽"）——A/B/C 三版对照见
# `历史/交接文档.md`。结论：**范例一旦写进提示词，就是被复读的样本**（黄金律 1 的另一面：
# 要么放数据层，要么只写"要什么口气"，别写"该说哪句话"）。改这条 prompt 时**别再往里加例句**。
#
# ⚠️ **任务定义 = 把内容"换成私聊口气讲一遍"**（用户 2026-10-14 明确）：
# 发两条是有意的——① 原文通知（机器口吻，信息兜底）② 这句（同样的信息，她的私聊口气）。
# 所以这句**该复述内容**，**不该是对内容的评论/回应**。
# 我一度把它改成"不用复述、补一句反应"，被用户否掉了（"都九点半了才通知到位"那种就是错的）。
#
# ⚠️ 另一处别再犯的错：**"冒出来的时间"不一定是编的**。主动发言这条路会注入
# 【当前时间】（含澳洲当地时间，见 `context_probe`），所以"这个点还没睡""都两点了"
# 是**引用真事实**。我曾把"凌晨两点还没睡的"当成编造写进 prompt（2026-10-14 当天发现并撤回）。
#
# ⚠️ **"别补内容里没有的"被读成了"什么也别多说"**（2026-10-30 用户提）：
# 动态只写了道晚安那种一个状态词，那一句就被压成了同一个词——**等于把原文又发了一遍**，
# 既没告诉绿冻她接下来干嘛，也不像在跟人说话。根子是上面那条"内容没写的一个字别补"太绝对，
# 模型于是连"道晚安 = 要睡了"这种**原文本身就暗示的意思**都不敢说。
# 现在补了一条：状态类内容要**说成人话**（她接下来做什么、跟绿冻什么关系），
# 同时把"私聊用'你'"也写进去（之前冒出过"大家晚安"——**这是对一个人说话**）。
# ⚠️ 仍然**不许写例句**：试过把想要的句子写进规则，模型就逐字吐出来（见上面 2026-10-14 那条）。
# 改动前后的对照跑在 `outputs/_proactive_ab2.py`（5 条动态 × 2 次采样）。
#
# ⚠️ **"内容里夹了个看不懂的词"要单独规定**（2026-10-30 用户给的真实动态）：
# 那条动态是「晚安晚安 哇星露谷」（她那晚在播星露谷，所以发了这么一句道别），
# 生成成了「晚安，灰泽满去玩星露谷了」——**"哇星露谷"只是个感叹名词，模型却替它补了"去玩"**。
# 根因：上面"有用的东西必须说出来"那条在逼它把每个词都说成一个动作。
# 现在写成：**说不清的那个词只丢它自己，确认得了的部分照常说**（哪怕只剩一句道别）；
# 并且明说**含糊不构成 NULL 的理由**——第一版没写这句，模型把整条判成不适合、直接不发。
# 实测（gemini-2.5-lite，6 条 × 3 次）：那条原文 3/3 只说了道晚安、一个"星露谷"都没提。
PROACTIVE_PROMPT = """你刚在网上发了一条内容。现在要**把这条内容讲给绿冻听**——用你平时在私聊里
跟他说话的口气，把它说一遍。写出你会说的那一句。

⚠️ **你是在跟绿冻说话，不是在向别人报告你刚才干了什么。**
反例（错的，读起来像第三人在转述）：
  "灰泽满刚发了条动态，说明天要直播"

要求：
- **就一句话，短**，像 QQ 私聊随手打的，不是发公告
- ⚠️ **补的是"人"，不是"事"。** 内容里**有动作、只是没说是谁**（谁被私信、谁被求、谁在看）时，
  把"灰泽满"补出来；内容里**只有名词或感叹、压根没有动作**时，别替它造一个动作。
  不是她做的事，别写成她做的。
- **提到自己时带上"灰泽满"，把句子说完整**——主动找人说话，本来就该让对方知道是她在讲。
- **内容本身带着跟人搭话的口气（提问、惊叹），就照那个口气说**，别压成一句干巴巴的陈述。
- ⚠️ **它是那条内容的"私聊版"**：同一件事，换成你随口讲给一个人听的口气。
  **不是对它的评论、感想或回应**——你要把这件事**告诉**他，不是**冲着**他说别的话。
- **不要复述"灰泽满刚才发了什么/说了什么"**：没有"灰泽满刚……"这种句式——
  你就是正在说这句话的人，不是转述者
- **绝对别像播报**：不要"灰泽满发了动态""快去看看"这种，就像跟朋友顺口提一句
  （**自称怎么写由人设决定**：system_prompt 的【自我称呼】+ 说话片段说了算。
   上面"带上灰泽满"只是**照人设说话**，不是另立一套称呼规矩——曾在此写"自称用'我'最自然"，
   那才等于把人格文件覆盖了）
- ⚠️ **"随口"指的是口气，不是少说信息。** 这条内容里对绿冻**有用的东西**
  （什么时间、去哪儿、做什么、有什么看点）**必须说出来**——时间和那件事本身一个字都不能省。
  踩坑：曾把"明天来玩这个音游"缩成"明天音游有翻唱"，**粉丝该被邀请的那件事没了**。
- ⚠️ **内容只是一句状态时（道晚安、说累了、说要去睡、说在吃饭），别把那句话原样再说一遍。**
  这不是"补细节"，是**把状态说成一句人话**：说清她接下来要做什么、这件事跟绿冻有什么关系。
  内容里已经暗示的东西（道晚安→要睡了）可以直说；但时间、地点、和谁、做什么具体事
  这些内容没写的，一个字也不许补。
- ⚠️ **这是私聊，对一个人说话**：称呼对方用“你”；写成“大家”“各位”“朋友们”就成了播报。
- ⚠️ **反过来，内容里没写的，一个字也别补。** 她写的是什么、为什么、结果怎样，
  内容没讲就**别猜**——宁可说得笼统，也不许编一个具体的东西出来。
  （原文通知已经把整条原样发给绿冻了，所以这句笼统一点完全没关系；编错才糟。）
  反例（错的）：内容「写完了…终于写完了…」→「**写稿**写得快不行了」
  （"稿"内容里根本没有；而且「终于写完了」是**已经写完**，不该说成"快不行了"）
  ⚠️ **但"现在几点"不算补**：【当前时间】那块是注入给她的真事实，说"这个点还不睡"
  "都两点了"是**用真时间**，不是编造——别把这条规则误读成"不许提时间"。
- ⚠️ **内容里夹着一个说不清是什么的词的，只丢那个词，别丢整条。** 那个词只有一个名词或感叹、
  看不出她具体干了什么或要干什么时，**别替它补一个动作、也别给它加评价**，直接不提它；
  但内容里**确认得了的部分照常说**——哪怕只剩一句道别或一个状态，这一条也照常发。
  也别把它换个说法（谐音、比喻、近义词）带出来——看不懂的那个词就是不存在。
  **只有整条都是转发/广告/抽奖这类才输出 NULL**：说得含糊不构成不发这条的理由。
- ⚠️ **时间是"照原文的说法搬"，不是"换算成几点"。** 原文写"10分钟后""马上""一会儿"
  "还有半小时"就照这么说——**别自己改成"十点""晚上八点"这种钟点**。换算是另一件事：
  要挑时区（她人在澳洲、观众在国内），挑错就差几个小时，而原文的相对说法本来就准。
  踩坑（2026-10-11 真实发出）：动态写"好无聊 **10分钟**后来玩会儿游戏吧"，
  那句成了"**十点后**来玩会儿游戏嘛"——把 19:55 说成了 22:00，粉丝按错的时间等了俩小时。
- ⚠️ **别把内容里两件事揉成一件**：「⬇️ / 这个 / 图片里那个」指的是**配图**，
  不是正文里另外那个名字——拿它去填就会把两件事说混。
  反例（错的）：内容「会到这个⬇️里面鲨会儿人」+ 正文另有"20:00多来玩BanG Dream"
  →「晚上八点多来玩BanG Dream! OurNotes嘛，让灰泽满也鲨会儿人」
  （把**两件事揉成一件**，而且鲨人的根本不是 BanG Dream）
  **配图看不清、或这条没配图，就只挑一件说**（优先时间/邀请——那件对绿冻最有用）。
- 不要贴链接、不要提"B站/微博/动态"这些平台词
- **口气是你的，信息是那条内容的**：可以有态度、可以自嘲、可以吐槽，但别干巴巴照抄原文
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

    ⚠️ 2026-10-30 试过让它顺带拆出"看得懂的那部分"再喂给生成，**三版都失败**：
    让它输出 usable → 原样抄回整条；让它输出要删的字 → 删过头（把"好好看的剧求继续看"、"今天"都删了）；
    改问事后"这句有没有把碎片编成事" → gemini 判反、deepseek 全判 true。
    结论：这种"哪几个字凑不成句"的粒度，两个模型都给不出稳定答案，**别再往回加**。
    """
    text = (text or "").strip()
    # 纯标点/纯符号/纯表情（"..."、"。。。"、"😭"）没有实义内容——**确定性拦掉，不调 LLM**：
    # 能确定的事别多一次判错的机会（同 group_gate 里"点名了她"那条）。
    # 实测（2026-09-27）：交给 LLM 判时，"..." 会被判成"够"。
    # 注意这条在 judge_enabled() **之前**——它不是"判断"，是"这里没有内容"，
    # 所以 PROACTIVE_JUDGE=0 也拦。
    if not _CONTENT_CHAR_RE.search(text):
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


async def _describe_first_image(vision_client, image_paths) -> str:
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
        return await describe_image_bytes(vision_client, data)
    except Exception as e:
        print(f"⚠️ 主动发言：配图描述失败（照常生成）: {e}")
        return ""


async def deliver_proactive(bot, user_id, text: str,
                            image_paths: list | None = None, *,
                            voice: bool = False) -> str:
    """把一条主动发言送到一个用户；返回 "voice" / "text"。

    2026-10-23 用户定：触发主动发言时**直接走语音**，不再推动态/微博原文；
    配图随语音一起发。TTS 合成失败或内容不适合念（长英文/链接残留）→ 发文字版
    （仍然不推原文）。所有异常由调用方按“单个好友失败不中断”处理。
    """
    from nonebot.adapters.onebot.v11 import Message, MessageSegment

    from .voice import prepare_proactive_tts, send_voice

    if voice:
        prepared, why = prepare_proactive_tts(text)
        if prepared:
            try:
                ok = await send_voice(bot, str(user_id), True, prepared)
            except Exception as e:
                print(f"⚠️ 主动语音发送异常（改发文字）: {e}")
                ok = False
            if ok:
                if image_paths:
                    try:
                        msg = Message()
                        for p in image_paths:
                            msg += MessageSegment.image(file=str(p))
                        await bot.send_private_msg(user_id=user_id, message=msg)
                    except Exception as e:
                        print(f"⚠️ 主动语音配图发送失败（语音已发）: {e}")
                return "voice"
            print("[主动] 语音合成/发送失败 → 改发文字版")
        else:
            print(f"[主动] 内容不适合语音（{why}）→ 改发文字版")

    msg = Message(text)
    for p in (image_paths or []):
        msg += MessageSegment.image(file=str(p))
    await bot.send_private_msg(user_id=user_id, message=msg)
    return "text"


async def broadcast_proactive(bot, targets, text: str,
                              image_paths: list | None = None, *,
                              voice: bool = False, record: bool = True) -> dict:
    """把一条主动发言广播给多个用户：**TTS 只合成一次**，再按并发上限并行发送。

    为什么单独有这个（2026-10-24）：300 个好友逐个 `send_voice` 会串行跑——每个都要
    等 TTS 锁、复检缓存、再发消息，整体几十秒起步。这里把合成挪到循环外，发送用
    `asyncio.Semaphore` 限并发（默认 8，可用 `PROACTIVE_PUSH_CONCURRENCY` 调），
    单个好友失败不影响其他人。
    """
    from nonebot.adapters.onebot.v11 import Message, MessageSegment

    from .voice import prepare_proactive_tts, send_voice_file, synthesize_voice

    ids = list(dict.fromkeys(str(t) for t in (targets or []) if t))
    if not ids:
        return {"voice": 0, "text": 0, "failed": 0, "total": 0}
    try:
        concurrency = max(1, int(os.environ.get("PROACTIVE_PUSH_CONCURRENCY", "8")))
    except ValueError:
        concurrency = 8

    wav_path = None
    if voice:
        prepared, why = prepare_proactive_tts(text)
        if prepared:
            wav_path = await synthesize_voice(prepared)
            if wav_path is None:
                print("[主动] 语音合成失败 → 全量改发文字")
        else:
            print(f"[主动] 内容不适合语音（{why}）→ 全量改发文字")

    sem = asyncio.Semaphore(concurrency)
    stats = {"voice": 0, "text": 0, "failed": 0, "total": len(ids)}

    async def _send_text(uid: str) -> None:
        msg = Message(text)
        for p in (image_paths or []):
            msg += MessageSegment.image(file=str(p))
        await bot.send_private_msg(user_id=uid, message=msg)

    async def _send_images(uid: str) -> None:
        if not image_paths:
            return
        try:
            msg = Message()
            for p in image_paths:
                msg += MessageSegment.image(file=str(p))
            await bot.send_private_msg(user_id=uid, message=msg)
        except Exception as e:
            print(f"⚠️ 主动语音配图发送失败（语音已发）user={uid}: {e}")

    async def one(uid: str) -> None:
        async with sem:
            try:
                if wav_path:
                    try:
                        ok = await send_voice_file(bot, uid, True, wav_path)
                    except Exception as e:
                        print(f"⚠️ 主动语音发送异常（改发文字）user={uid}: {e}")
                        ok = False
                    if ok:
                        await _send_images(uid)
                        stats["voice"] += 1
                    else:
                        await _send_text(uid)
                        stats["text"] += 1
                else:
                    await _send_text(uid)
                    stats["text"] += 1
                if record:
                    append_bot_message(uid, to_memory_line(text))
            except Exception as e:
                stats["failed"] += 1
                print(f"⚠️ 主动推送失败 user={uid}: {e}")

    await asyncio.gather(*(one(uid) for uid in ids))
    print(f"[主动] 广播 {len(ids)} 人：语音 {stats['voice']} / 文字 {stats['text']} / "
          f"失败 {stats['failed']}（并发 {concurrency}）")
    return stats


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
    set_llm_route("主动")   # 用量分账：这条链路的调用单独算一档（见 config._ROUTE）

    try:
        deepseek_client, zhipu_client = aux_clients_or(_get_clients)
        # 配图描述与"够不够"判断**并行**跑：两者互不依赖，串行等于白等多一次调用。
        # 且**只在她指了配图时**才读（"到这个⬇️里面"），否则那条描述是纯噪声。
        want_image = bool(image_paths) and bool(_REFERS_TO_IMAGE_RE.search(text))
        desc_task = (asyncio.create_task(_describe_first_image(get_vision_client(), image_paths))
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
            ctx["fused_items"], [],
            preference_items=ctx["preference_items"],
            core_stories=ctx["core_stories"],
        )

        resp = await deepseek_client.chat.completions.create(
            model=aux_model(_get_model_name()),
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
