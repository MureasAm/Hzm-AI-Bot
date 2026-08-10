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
import ast
import re

from nonebot import on_message, on_request
from nonebot.adapters.onebot.v11 import Bot, Event, FriendRequestEvent, RequestEvent

from .constants import AUTO_ACCEPT_FRIEND
from . import bili_bridge  # noqa: F401  导入即注册启动时的后台监听任务
from . import chat_window

chat = on_message(priority=10, block=True)

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


def _extract_image_source(msg) -> str:
    """取消息第一张图的源：优先 url，否则 file（NapCat 缓存名），无图返回空串。

    只取源不解析——真正的视觉解析推迟到回复前（chat_window._flush），
    这样"读图"是读整批的一部分，而不是消息一到就秒解析。
    """
    for seg in msg:
        if seg.type == "image":
            url = seg.data.get("url") or ""
            if url:
                return url
            file_ = seg.data.get("file") or ""
            if file_:
                return file_
            print(f"⚠️ 图片段既无 url 也无 file: {seg.data}")
            return ""
    return ""


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
        if fid:  # 兜底：无 raw 时用 id
            return f"QQ表情{fid}"
    return ""


@chat.handle()
async def _handle_chat(bot: Bot, event: Event):
    msg = event.get_message()
    user_msg = msg.extract_plain_text().strip()
    image_source = _extract_image_source(msg)
    face_text = _extract_face_text(msg)

    # QQ 内置表情（face）：把含义转成消息，纯表情也能回应
    if face_text:
        face_msg = f"[表情：{face_text}]"
        user_msg = f"{user_msg} {face_msg}".strip() if user_msg else face_msg

    if not user_msg and not image_source:
        return  # 真正空消息（无文字无图片无表情），不回复

    user_id = event.get_user_id()
    is_private = getattr(event, "message_type", "private") == "private"
    target_id = str(user_id if is_private else getattr(event, "group_id", ""))
    print(f"[收到消息] user={user_id}, msg={user_msg[:40]!r}, img={'有' if image_source else '无'}")

    # 读秒窗口（方案B）：攒批 + 静默后统一回复（含读图/归纳/分批发送）
    chat_window.enqueue(user_id, user_msg, image_source, bot, is_private, target_id)
