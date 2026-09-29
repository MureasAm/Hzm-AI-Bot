"""灰泽满 AI 聊天机器人插件入口。

模块结构：
- constants.py  路径 / API / 阈值常量
- config.py     NoneBot 配置读取助手
- context_probe.py  时间/农历/天气感知
- persona.py    人格加载与语义行为匹配（trigger 向量缓存）
- memory.py     短期记忆（带锁）+ 长期记忆封装
- rag.py        直播记忆检索（余弦相似度）
- vision.py     视觉理解（glm-4.6v）
- core.py       消息处理主循环
- chat_window.py  读秒窗口（方案B：消息攒批，回复前统一读图+归纳）
- bili_bridge.py  B站直播/动态监听 + 私聊广播（启动时注册后台任务）
"""
import asyncio
import ast
import re
import time
from pathlib import Path

from nonebot import get_driver, on_message, on_request, on_type
from nonebot.adapters.onebot.v11 import (
    Bot, Event, FriendRequestEvent, RequestEvent, NoticeEvent,
)

from .constants import AUTO_ACCEPT_FRIEND, PROJECT_ROOT, AT_SELF_MARK
from .qq_faces import face_name
from . import bili_bridge  # noqa: F401  导入即注册启动时的后台监听任务
from . import weibo_bridge  # noqa: F401  导入即注册启动时的微博后台监听任务
from . import chat_window

chat = on_message(priority=10, block=True)

# ==================== 💓 心跳 + 连接信号（watchdog 自愈用） ====================
# 双信号，帮 watchdog 区分两种"死法"：
# - data/heartbeat   bot 进程心跳：每 30s touch，进程活着就一直在 → bot 崩溃检测
# - data/qq_alive    QQ 会话在线标记：只在 OneBot WebSocket 已连接时才 touch；
#                    连接一断就停更 → 假死（显示在线但收不到消息）检测
# - data/qq_offline  断连标记：on_bot_disconnect 时写时间戳，让 watchdog 快速感知
# 全部纯文件操作，无副作用，不依赖网络。
HEARTBEAT_FILE = PROJECT_ROOT / "data" / "heartbeat"
QQ_ALIVE_FILE = PROJECT_ROOT / "data" / "qq_alive"
QQ_OFFLINE_FILE = PROJECT_ROOT / "data" / "qq_offline"

# OneBot WS 连接状态（SnowLuma 作为 WS 客户端连到本 bot 的反向 WS）
_connected = {"state": False}
# QQ 账号在线状态：bot_offline/bot_online 通知事件是权威信号（被踢时 WS 可能还连着，
# 仅看 WS 状态会把"假死"误判为正常，所以必须双条件）
_account_online = {"state": True}


@get_driver().on_bot_connect
async def _on_bot_connect(bot: Bot) -> None:
    _connected["state"] = True
    _account_online["state"] = True
    try:
        QQ_OFFLINE_FILE.unlink(missing_ok=True)  # 连上了就清掉断连标记
    except OSError:
        pass


@get_driver().on_bot_disconnect
async def _on_bot_disconnect(bot: Bot) -> None:
    _connected["state"] = False
    try:
        QQ_OFFLINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # 已有被踢标记则保留（踢出优先于普通 WS 断开，别覆盖掉）
        if not QQ_OFFLINE_FILE.exists():
            QQ_OFFLINE_FILE.write_text("disconnect", "utf-8")
    except OSError:
        pass  # 写失败不影响 bot 运行


# QQ 账号被踢下线 / 恢复上线的通知事件（比 WS 断开更准确、更早）。
# 适配器没有专门的 bot_offline/bot_online 类，统一走 NoticeEvent 按 notice_type 分流。
account_notice = on_type(NoticeEvent, priority=1, block=False)


@account_notice.handle()
async def _on_account_notice(bot: Bot, event: NoticeEvent) -> None:
    if event.notice_type == "bot_offline":
        _account_online["state"] = False
        try:
            QQ_OFFLINE_FILE.parent.mkdir(parents=True, exist_ok=True)
            QQ_OFFLINE_FILE.write_text("bot_offline", "utf-8")
        except OSError:
            pass
    elif event.notice_type == "bot_online":
        _account_online["state"] = True
        try:
            QQ_OFFLINE_FILE.unlink(missing_ok=True)
        except OSError:
            pass


async def _heartbeat_loop() -> None:
    while True:
        try:
            HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
            HEARTBEAT_FILE.touch()
            if _connected["state"] and _account_online["state"]:
                QQ_ALIVE_FILE.touch()  # 连接 + 账号在线双条件才更新在线标记
        except OSError:
            pass  # 心跳写失败不影响 bot 运行
        await asyncio.sleep(30)


@get_driver().on_startup
async def _start_heartbeat() -> None:
    # bot 重启时若残留 qq_offline（上次被踢还没恢复），先保持离线标记，
    # 等真正的连接/bot_online 事件再翻转，避免重启窗口期误判在线
    if QQ_OFFLINE_FILE.exists():
        _account_online["state"] = False
    asyncio.create_task(_heartbeat_loop())

# 自动通过好友申请：新加的绿冻立即变双向好友，B站推送才能到达（NapCat 无法给单向好友发消息）
friend_req = on_request(priority=1, block=False)


@friend_req.handle()
async def _auto_accept_friend(bot: Bot, event: RequestEvent):
    if not isinstance(event, FriendRequestEvent) or not AUTO_ACCEPT_FRIEND:
        return  # 群申请等不处理
    try:
        await bot.set_friend_add_request(flag=event.flag, approve=True, remark="")
        print(f"[好友申请] ✅ 已自动通过 user={event.user_id}")
    except Exception as e:
        print(f"[好友申请] ⚠️ 自动通过失败 user={event.user_id}: {e}")


def _extract_image_source(msg) -> tuple[str, str]:
    """取消息第一张图的 (url, file)。两者都返回：url 优先下载，file 留作 CDN 失败时读本地缓存兜底。

    只取源不解析——真正的视觉解析推迟到回复前（chat_window._flush），
    这样"读图"是读整批的一部分，而不是消息一到就秒解析。
    """
    for seg in msg:
        if seg.type == "image":
            url = seg.data.get("url") or ""
            file_ = seg.data.get("file") or ""
            return url, file_
    return "", ""


def _extract_at_self(msg, event, is_group: bool = True) -> bool:
    """这条消息有没有 @ 她本人。**两道信号都要看，缺一不可。**

    ① **`event.to_me`**（适配器算好的）：`_check_at_me` 在 @ 落在**开头或结尾**时，
       会把那个 at 段**整个删掉**（然后置 `to_me=True`）——所以光看消息段**永远看不到**
       这种。踩坑（2026-09-27）：先加了段检测，"@她"还是被判成没点名，根因就在这：
       `@灰泽满 在吗` 恰好是最常见的写法，而它轮到我们处理时 at 段已经没了。
       ⚠️ 私聊的 `to_me` 是适配器**无条件**置的 True，所以**只在群聊认它**。
    ② **消息段**：@ 落在**中间**（"你们看 @她 这个"）时适配器不动它，段还在消息里。

    "@全体成员"（qq=all）不算——那是 @ 所有人，不是点她。
    """
    if is_group and getattr(event, "to_me", False):
        return True
    self_id = str(getattr(event, "self_id", "") or "")
    if not self_id:
        return False
    for seg in msg:
        if seg.type == "at" and str(seg.data.get("qq") or "").strip() == self_id:
            return True
    return False


def _extract_face_text(msg) -> str:
    """提取 QQ 内置表情（face）的含义文字；无 face 返回空串。

    face 段的 raw 里有 faceText（如 '/比爱心'），解析成可读含义，纯表情消息也能回。
    """
    for seg in msg:
        if seg.type != "face":
            continue
        raw = seg.data.get("raw")
        text = ""
        if isinstance(raw, dict):
            text = raw.get("faceText", "") or ""
        elif isinstance(raw, str) and raw.strip():
            try:
                d = ast.literal_eval(raw)  # python dict repr
                text = d.get("faceText", "") or ""
            except Exception:
                m = re.search(r'["\']faceText["\']\s*[:=]\s*["\']([^"\']*)["\']', raw)
                text = m.group(1) if m else ""
        if text:
            return text.lstrip("/")
        fid = seg.data.get("id")
        if fid:
            # 兜底：NapCat 没给 faceText 时，用 QQ 客户端自带的 id→名字表还原。
            # 否则会退化成"QQ表情6"，模型完全不知道那是什么表情（实测踩坑）。
            return face_name(fid) or f"QQ表情{fid}"
    return ""


# 被引用消息注入时的文本上限：引用是"指路"，不是把整条消息搬进来
QUOTE_TEXT_MAX = 80

# 内层转发最多往里展开几层（防炸/防环：每层都要再调一次 get_forward_msg）
_NESTED_FORWARD_MAX_DEPTH = 2

# 引用的消息**没有文字**时，用它是什么类型来指路。
# 不加这个映射的话，语音/转发卡片/文件这类引用会被**静默丢掉**——
# 她看不到引的是什么，就会答非所问（实测 2026-09-29 群 901907410：
# 豆子引用一条转发问她"能不能把灰泽满调成这样"，她回了句"调成哪样啊"）。
_QUOTE_SEG_WHAT = {
    "image": "一张图", "face": "一个表情", "mface": "一张表情包",
    "record": "一条语音", "video": "一段视频", "file": "一个文件",
    "forward": "一条转发聊天记录", "json": "一张卡片", "music": "一首歌",
    "poke": "一个戳一戳", "dice": "一个骰子", "rps": "一次猜拳",
}

# 「只含这些段、又没有文字」的消息：给个占位提示，别让它**静默消失**
# （NapCat 对卡片类消息有时给 `json`/`file`/`video` 段——不在 _extract_forward_text 的
#  识别范围内，于是正文为空、又没有图，整条消息就一声不响地没了）
_UNKNOWN_SEG_HINT = dict(_QUOTE_SEG_WHAT, json="一张卡片（可能是分享/转发卡片）")


def _extract_quote_text(event, forward_detail: str = "") -> str:
    """把"用户在引用哪条消息"还原成可注入的文本；没有引用返回空串。

    `forward_detail`：引用的那条**本身是转发卡片**时，由 `_quoted_forward_detail`
    取回来的内容摘要（引用的正文是空的，不给它的话"这样/那个"就没有指向）。

    **不用自己调 get_msg**：适配器在 `Bot.handle_event` 里对每条消息都跑过
    `_check_reply`（nonebot/adapters/onebot/v11/bot.py:22）——发现 reply 段就自动
    `get_msg(message_id=int(seg.data["id"]))`，结果挂在 `event.reply` 上，
    同时把 reply 段从 `event.message` 删掉（所以 extract_plain_text 看不到，得从这取）。

    **为什么要带"是谁说的"**：引用她自己的话 vs 引用另一个群友的话，她的反应完全不同
    （前者是"解释/defend 自己说过的话"，后者是"接别人的话"）。群聊里这点尤其要紧
    ——群记忆早就叮嘱过"别把不同的人当成同一个你"。
    """
    reply = getattr(event, "reply", None)
    if reply is None:
        # 有 reply 段、却没解析出 `event.reply`：适配器 `_check_reply` 在 get_msg 失败时
        # **只记一条 WARNING 就返回**，于是引用**静默消失**。不能让它消失——
        # 她看不到引的是什么就会答非所问（见 _QUOTE_SEG_WHAT 上的实测记录）。
        if "[CQ:reply" in (getattr(event, "raw_message", "") or ""):
            print("[引用] ⚠️ 引用了消息但没取到内容（适配器 get_msg 失败？）")
            return "[有人引用了一条消息问你，但内容没取到——看不懂就直接反问对方引的是什么]"
        return ""

    text = ""
    try:
        text = (reply.message.extract_plain_text() or "").strip()
    except Exception:
        text = ""
    if not text:
        # 兜底用 raw_message（CQ 串），但**必须先把 CQ 码剥掉**——
        # 否则会把 `[CQ:record,file=91257a….amr]` 这种丑代码原样注进上下文（实测漏过）
        text = re.sub(r"\[CQ:[^\]]*\]", "",
                      getattr(reply, "raw_message", "") or "").strip()
    if len(text) > QUOTE_TEXT_MAX:
        text = text[:QUOTE_TEXT_MAX] + "…"

    try:
        sender_id = str(reply.sender.user_id)
    except Exception:
        sender_id = ""
    if sender_id and sender_id == str(getattr(event, "self_id", "")):
        who = "灰泽满自己"          # 引用的是她自己说过的话
    elif sender_id and sender_id == str(event.get_user_id()):
        who = "对方自己"
    else:
        nick = ""
        try:
            nick = (reply.sender.nickname or "").strip()
        except Exception:
            pass
        who = f"另一个绿冻{nick}" if nick else "另一个绿冻"

    # 引用的那条还带了什么（图片/表情/类型）——不然模型只知道文字部分
    has_image, qface, whats = False, "", []
    try:
        qface = _extract_face_text(reply.message)
        for seg in reply.message:
            if seg.type == "image":
                has_image = True
            what = _QUOTE_SEG_WHAT.get(seg.type)
            # 表情有名字就更精确（"表情「可怜」"比"一个表情"有用），别重复列两遍
            if seg.type == "face" and qface:
                what = f"表情「{qface}」"
            if what and what not in whats:
                whats.append(what)
    except Exception:
        pass

    if not text:
        # 引用的不是文字（图/语音/转发卡片…）：没有正文可引，但**不能让引用整个丢失**。
        # 用"引的是什么类型"指路，她才知道该怎么接（曾静默丢弃 → 她只能反问"调成哪样啊"）。
        if not whats:
            return ""   # 引用了一条空消息（没有任何段），没什么可指路的
        # 引的是**转发卡片**时，forward_detail 是取回来的内容摘要——
        # 不给内容的话"这样/那个"就没有指向（实测踩坑见 _quoted_forward_detail）
        detail = f"（{forward_detail}）" if forward_detail else ""
        return f"[引用{who}发的{'、'.join(whats[:2])}{detail}]"

    extra = ""
    if has_image:
        extra += "（带图）"
    if qface:
        extra += f"（表情：{qface}）"
    return f"[引用{who}说的：「{text}」{extra}]"


def _flatten_forward_content(content) -> str:
    """把转发里一条消息的 content 拍成纯文本。

    各实现给的形状不一样（字符串 / 段数组 / Message 对象），所以三种都认。
    图片/表情拍成占位符——她要的是"知道聊了啥"，不需要图片字节。
    """
    if isinstance(content, str):
        return content.strip()
    if hasattr(content, "extract_plain_text"):      # nonebot Message 对象
        try:
            return (content.extract_plain_text() or "").strip()
        except Exception:
            return ""
    if isinstance(content, list):
        out = []
        for s in content:
            if not isinstance(s, dict):
                continue
            t = s.get("type")
            if t == "text":
                out.append(str((s.get("data") or {}).get("text", "")))
            elif t == "image":
                out.append("[图片]")
            elif t == "face":
                out.append("[表情]")
            elif t == "forward":
                # ⚠️ 内层转发**必须留个痕迹**：旧实现什么都不加，于是"聊天记录里
                # 还套着一条聊天记录"时，内层内容**整段静默消失**——
                # 摘要只能写出"有人在群里发图并接话"这种空话（实测 2026-09-29）。
                # 正常情况下这段标记会被 `_expand_nested_forwards` 换成取回的内容。
                out.append("[转发的聊天记录]")
        return "".join(out).strip()
    return ""


def _nested_forward_ids(content) -> list:
    """挑出 content 里**内层转发**的 id（可多个）。段数组和 CQ 串两种形状都认。"""
    ids = []
    if isinstance(content, str):
        ids = re.findall(r"\[CQ:forward,[^\]]*id=([^,\]]+)", content)
    elif isinstance(content, list):
        for s in content:
            if isinstance(s, dict) and s.get("type") == "forward":
                fid = str((s.get("data") or {}).get("id") or "").strip()
                if fid:
                    ids.append(fid)
    return ids


async def _expand_nested_forwards(bot, content, depth: int = 0):
    """把 content 里的**内层转发段**取回来、递归展开成文字段；其余段原样保留。

    为什么要它（2026-09-29 用户实测）：她转发的那条记录里**还套着一条记录**
    （"蓝泽大肥鱼被拉进群后的暴力发言"就在里层）。旧实现遇到内层 forward 段直接跳过，
    摘要于是只剩"有人在群里发图并接话，提到自己这儿还有'攻击的'"——
    **真正的内容一个字都没进上下文**，粉丝接着问"能不能给灰泽满调成这样"，
    她手里没有任何可关联的东西。
    """
    if depth >= _NESTED_FORWARD_MAX_DEPTH or not _nested_forward_ids(content):
        return content
    if isinstance(content, str):
        out = content
        for fid in _nested_forward_ids(content):
            inner = await _fetch_nested(bot, fid, depth)
            out = out.replace(f"[CQ:forward,id={fid}]", inner, 1)
        return out
    if isinstance(content, list):
        out = []
        for s in content:
            if isinstance(s, dict) and s.get("type") == "forward":
                fid = str((s.get("data") or {}).get("id") or "").strip()
                out.append({"type": "text", "data": {"text": await _fetch_nested(bot, fid, depth)}})
            else:
                out.append(s)
        return out
    return content


async def _fetch_nested(bot, fid: str, depth: int) -> str:
    """取一层内层转发 → 文字（失败给一句说明，**绝不留空**）。"""
    try:
        sub = await bot.get_forward_msg(id=fid)
    except Exception as e:
        print(f"⚠️ 内层转发取回失败: {e}")
        return "［内层转发，但没取到内容］"
    inner = await _render_forward_deep(bot, sub, depth + 1)
    return f"［内层转发：{inner}］" if inner else "［内层转发，但里面没有文字内容］"


async def _render_forward_deep(bot, resp, depth: int = 0) -> str:
    """`_render_forward` 的**递归**版：内层转发也取回来展开。

    `_render_forward`（同步、不联网）保留给不需要展开的场景与测试；
    真正的取回走这里。
    """
    nodes = []
    if isinstance(resp, dict):
        nodes = resp.get("message") or resp.get("messages") or []
    elif isinstance(resp, list):
        nodes = resp
    lines = []
    for nd in nodes:
        if not isinstance(nd, dict):
            continue
        d = nd.get("data") if isinstance(nd.get("data"), dict) else nd
        content = d.get("content") or d.get("message") or ""
        content = await _expand_nested_forwards(bot, content, depth)
        text = _flatten_forward_content(content)
        if not text:
            continue
        nick = str(d.get("nickname") or d.get("name") or "").strip()
        lines.append(f"{nick}：{text}" if nick else text)
    return "\n".join(lines)


def _render_forward(resp) -> str:
    """把 get_forward_msg 的返回拍成「昵称：内容」逐行。认不出的形状返回空串。"""
    nodes = []
    if isinstance(resp, dict):
        nodes = resp.get("message") or resp.get("messages") or []
    elif isinstance(resp, list):
        nodes = resp
    lines = []
    for nd in nodes:
        if not isinstance(nd, dict):
            continue
        d = nd.get("data") if isinstance(nd.get("data"), dict) else nd
        text = _flatten_forward_content(d.get("content") or d.get("message") or "")
        if not text:
            continue
        nick = str(d.get("nickname") or d.get("name") or "").strip()
        lines.append(f"{nick}：{text}" if nick else text)
    return "\n".join(lines)


async def _fetch_forward_summary(bot, fid: str) -> tuple[bool, str]:
    """取回一条合并转发并摘要。返回 `(取到了吗, 摘要短语或失败说明)`。

    **直接转发 和 "引用一条转发" 共用这一份**——取回 / 渲染 / 摘要三步，
    两处各写一套必然漂（直接转发那条早就通了，引用那条一直漏着）。
    """
    try:
        resp = await bot.get_forward_msg(id=fid)
    except Exception as e:
        print(f"⚠️ 取转发内容失败: {e}")
        return False, "但没取到内容"
    # **用递归版**：里面还套着转发时把内层也取回来展开
    #（不展开的话内层内容会整段消失，摘要只剩空话——2026-09-29 实测踩坑）
    raw = await _render_forward_deep(bot, resp)
    if not raw:
        return False, "但里面没有文字内容"
    from .core import summarize_forward
    summary = await summarize_forward(raw)
    n = raw.count("\n") + 1
    print(f"[转发] 取回 {n} 条（含内层展开）→ 摘要 {len(summary)} 字")
    return True, (f"共 {n} 条：{summary}" if summary else f"共 {n} 条（内容较长没细看）")


async def _quoted_forward_detail(event, bot) -> str:
    """引用的那条消息**自己是一条转发聊天记录**时，把里面的内容取回来（摘要）。

    踩坑（2026-09-29 群 901907410）：豆子引用一条转发（里面是怎么把 AI 调成那种性格的
    对话）问"能不能给灰泽满调成这样"——转发卡片在 `extract_plain_text()` 里是**空的**，
    旧代码遇到这种引用`return ""`，于是**转发内容整段消失**：
    她只看到"能不能给灰泽满调成这样"，完全不知道"这样"指什么，
    只能回一句"调成哪样啊"。
    """
    reply = getattr(event, "reply", None)
    if reply is None:
        return ""
    try:
        segs = list(reply.message)
    except Exception:
        return ""
    for seg in segs:
        if seg.type != "forward":
            continue
        fid = str(seg.data.get("id") or "").strip()
        if not fid:
            continue
        ok, inner = await _fetch_forward_summary(bot, fid)
        print(f"[转发] 引用里的转发：{'已取回' if ok else inner}")
        return inner
    return ""


async def _extract_forward_text(event, bot) -> str:
    """把"用户转发的聊天记录"取回并**摘要**成可注入的一段；没有转发返回空串。

    取回走 `bot.get_forward_msg`（走 Bot.__getattr__ → call_api，OneBot 标准动作）。
    **只注入摘要、不注入原文**：转发几十条几千字，原文会挤爆上下文——
    详见 core.summarize_forward 的说明。
    """
    try:
        segs = list(event.get_message())
    except Exception:
        return ""
    for seg in segs:
        if seg.type != "forward":
            continue
        fid = str(seg.data.get("id") or "").strip()
        if not fid:
            continue
        ok, inner = await _fetch_forward_summary(bot, fid)
        return f"[转发的聊天记录（{inner}）]" if ok else f"[转发了一条聊天记录，{inner}]"
    return ""


@chat.handle()
async def _handle_chat(bot: Bot, event: Event):
    msg = event.get_message()
    user_msg = msg.extract_plain_text().strip()
    image_url, image_file = _extract_image_source(msg)
    face_text = _extract_face_text(msg)

    # 引用回复：把"引用了哪条"放到消息最前——它是这条消息的语境，不是新内容。
    # 引用的那条**本身是转发卡片**时，正文取不到 → 单独把里面的内容取回来（摘要）
    quote_text = _extract_quote_text(event, await _quoted_forward_detail(event, bot))
    if quote_text:
        user_msg = f"{quote_text} {user_msg}".strip() if user_msg else quote_text

    # 合并转发：取回聊天记录 → 摘要 → 注入（只注入摘要，原文不进上下文）
    forward_text = await _extract_forward_text(event, bot)
    if forward_text:
        user_msg = f"{forward_text} {user_msg}".strip() if user_msg else forward_text

    # QQ 内置表情（face）：把含义转成消息，纯表情也能回应
    if face_text:
        face_msg = f"[表情：{face_text}]"
        user_msg = f"{user_msg} {face_msg}".strip() if user_msg else face_msg

    is_private = getattr(event, "message_type", "private") == "private"

    # @ 她本人：放进文本最前面（同上，是这条消息的语境）。
    # 两个作用：① 群里"点名必接"能命中 ② **只 @ 不带字**的消息不再被当空消息丢掉。
    at_self = _extract_at_self(msg, event, is_group=not is_private)
    if at_self:
        user_msg = f"{AT_SELF_MARK} {user_msg}".strip() if user_msg else AT_SELF_MARK

    # **只含"我们没解析的段"的消息不能静默丢掉**（卡片/文件/视频/位置/残留的 reply 段…）：
    # 这类消息 `extract_plain_text()` 是空的、又没有图/表情，会掉进下面"空消息"那一行
    # **一声不响地消失**——连一条日志都没有。
    # 踩坑（2026-09-29 群 901907410）：豆老湿先发了一条内容（疑似卡片/转发），
    # 3 分钟后才问"能不能给灰泽满调成这样"。那条先发的**没进任何一批**，
    # 所以她永远不知道"这样"指什么，只能回一句"调成哪样啊"。
    # 现在给个占位提示：她至少知道"对方发了个我解析不出来的东西"，可以反问。
    if not user_msg and not image_url and not image_file:
        unknown = [seg.type for seg in msg
                   if seg.type not in ("text", "at", "reply", "image", "face")]
        if unknown:
            what = "、".join(dict.fromkeys(_UNKNOWN_SEG_HINT.get(t, f"{t} 段") for t in unknown))
            print(f"[收到消息] ⚠️ 只含未识别的段 {unknown} → 注入占位提示（否则这条会静默消失）")
            user_msg = f"[对方发来{what}，但内容没能解析出来——看不懂就问他发了什么]"

    if not user_msg and not image_url and not image_file:
        return  # 真正空消息（无文字无图片无表情），不回复

    user_id = event.get_user_id()
    target_id = str(user_id if is_private else getattr(event, "group_id", ""))
    print(f"[收到消息] user={user_id}, msg={user_msg[:40]!r}, "
          f"img={'有' if (image_url or image_file) else '无'}, "
          f"引用={'有' if quote_text else '无'}, "
          f"转发={'有' if forward_text else '无'}, "
          f"@={'有' if at_self else '无'}")

    # 读秒窗口（方案B）：攒批 + 静默后统一回复（含读图/归纳/分批发送）
    # target_id=会话标识（私聊=user_id，群聊=group_id），群聊按群攒批实现多人对话
    chat_window.enqueue(target_id, user_id, user_msg, image_url, image_file, bot, is_private)
