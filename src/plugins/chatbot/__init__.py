"""灰泽满 AI 聊天机器人插件入口。

模块结构：
- constants.py  路径 / API / 阈值常量
- persona.py    人格加载与语义行为匹配（trigger 向量缓存）
- memory.py     短期记忆（带锁）+ 长期记忆封装
- rag.py        直播记忆检索（余弦相似度）
- core.py       消息处理主循环
"""
from nonebot import on_message
from nonebot.adapters.onebot.v11 import Bot, Event, Message
from nonebot.params import EventPlainText

from .core import handle_chat

chat = on_message(priority=10, block=True)


@chat.handle()
async def _handle_chat(bot: Bot, event: Event, user_msg: str = EventPlainText()):
    user_id = event.get_user_id()
    print(f"[收到消息] user={user_id}, msg={user_msg}")

    reply = await handle_chat(user_id, user_msg)
    await chat.finish(Message(reply))
