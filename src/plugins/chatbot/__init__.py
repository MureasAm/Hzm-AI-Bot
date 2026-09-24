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

from .constants import AUTO_ACCEPT_FRIEND, PROJECT_ROOT
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


def _extract_quote_text(event) -> str:
    """把"用户在引用哪条消息"还原成可注入的文本；没有引用返回空串。

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
        return ""

    text = ""
    try:
        text = (reply.message.extract_plain_text() or "").strip()
    except Exception:
        text = ""
    if not text:
        text = (getattr(reply, "raw_message", "") or "").strip()
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

    # 引用的那条还带了什么（图片/表情）——不然模型只知道文字部分
    has_image, qface = False, ""
    try:
        has_image = any(seg.type == "image" for seg in reply.message)
        qface = _extract_face_text(reply.message)
    except Exception:
        pass

    if not text:
        # 引用的是一条纯图片/纯表情：没有文字可指路，但**不能让引用整个丢失**
        if has_image and qface:
            what = "一张带表情的图"
        elif has_image:
            what = "一张图"
        elif qface:
            what = f"表情「{qface}」"
        else:
            return ""   # 引用了一条空消息，没什么可说的
        return f"[引用{who}发的{what}]"

    extra = ""
    if has_image:
        extra += "（带图）"
    if qface:
        extra += f"（表情：{qface}）"
    return f"[引用{who}说的：「{text}」{extra}]"


@chat.handle()
async def _handle_chat(bot: Bot, event: Event):
    msg = event.get_message()
    user_msg = msg.extract_plain_text().strip()
    image_url, image_file = _extract_image_source(msg)
    face_text = _extract_face_text(msg)

    # 引用回复：把"引用了哪条"放到消息最前——它是这条消息的语境，不是新内容
    quote_text = _extract_quote_text(event)
    if quote_text:
        user_msg = f"{quote_text} {user_msg}".strip() if user_msg else quote_text

    # QQ 内置表情（face）：把含义转成消息，纯表情也能回应
    if face_text:
        face_msg = f"[表情：{face_text}]"
        user_msg = f"{user_msg} {face_msg}".strip() if user_msg else face_msg

    if not user_msg and not image_url and not image_file:
        return  # 真正空消息（无文字无图片无表情），不回复

    user_id = event.get_user_id()
    is_private = getattr(event, "message_type", "private") == "private"
    target_id = str(user_id if is_private else getattr(event, "group_id", ""))
    print(f"[收到消息] user={user_id}, msg={user_msg[:40]!r}, "
          f"img={'有' if (image_url or image_file) else '无'}, "
          f"引用={'有' if quote_text else '无'}")

    # 读秒窗口（方案B）：攒批 + 静默后统一回复（含读图/归纳/分批发送）
    # target_id=会话标识（私聊=user_id，群聊=group_id），群聊按群攒批实现多人对话
    chat_window.enqueue(target_id, user_id, user_msg, image_url, image_file, bot, is_private)
