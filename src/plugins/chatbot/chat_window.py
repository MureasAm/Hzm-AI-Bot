"""读秒窗口（方案B）：把同一用户几秒内的多条消息攒批，静默后统一回复。

架构（参考 Koishi 三模块）：
- 采集：__init__ 的处理器把每条消息 (text, image_source) 交给 enqueue()
- 会话：enqueue 重置去抖定时器；用户停手超过随机读秒窗口（5~10s）才触发一次回复
- 回复：_flush 在回复前统一"读图 + 归纳"（作为读整批的一部分），再交给 handle_chat，
        复用分批发送（A）；发送中途用户插话 → 取消未发送的分段，优先回新消息

generation 计数用于区分代际：插话取消旧任务后，旧任务不会误清掉新任务正在用的窗口。
参数在 constants.py 配置（READ_WINDOW_MIN/MAX_SECONDS）。
"""
import asyncio
import random

from nonebot.adapters.onebot.v11 import Message

from .core import (
    handle_chat, split_reply, split_delay, summarize_batch, clean_reply, _get_clients,
)
from .vision import describe_image
from .constants import (
    READ_WINDOW_MIN_SECONDS, READ_WINDOW_MAX_SECONDS, SPLIT_REPLY_ENABLED,
)


class _UserWindow:
    """单用户的会话窗口：缓冲 + 去抖/发送任务 + 代际计数。"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.pending: list[tuple[str, str]] = []   # (text, image_source)
        self.task: asyncio.Task | None = None      # 当前"窗口等待/发送"任务
        self.generation = 0                        # 每次 enqueue 自增
        self.bot = None
        self.is_private = True
        self.target_id = ""


_windows: dict[str, _UserWindow] = {}


def _combine_text(msgs: list[tuple[str, str]]) -> str:
    """合并本批文本（图片源不参与文本合并）。"""
    return "\n".join(t for t, _ in msgs if t.strip()).strip()


async def _describe_image_src(bot, src: str) -> str:
    """把图片源（url 或 NapCat file）解析成视觉描述；失败返回空串。"""
    if not src:
        return ""
    url = src if src.startswith(("http://", "https://")) else ""
    if not url:
        try:
            info = await bot.get_image(file=src)
            url = info.get("url", "") or ""
        except Exception as e:
            print(f"⚠️ get_image 解析图片失败: {e}")
            return ""
    if not url:
        return ""
    try:
        _, zhipu_client = _get_clients()
        print(f"[视觉] 图片url: {url[:100]}")
        desc = await describe_image(zhipu_client, url)
        print(f"[视觉] 图片描述: {desc[:80]}")
        return desc
    except Exception as e:
        print(f"⚠️ 视觉接入失败（忽略）: {e}")
        return ""


def enqueue(user_id: str, text: str, image_source: str,
            bot, is_private: bool, target_id: str) -> None:
    """采集一条消息进缓冲，重置读秒窗口。

    若该用户正在等待回复/发送分段，先取消（插话优先：取消未发送的，先回新消息）。
    image_source 只存图片源（url/file），真正解析推迟到回复前统一做。
    """
    if not text.strip() and not image_source:
        return
    win = _windows.setdefault(user_id, _UserWindow(user_id))
    win.generation += 1
    gen = win.generation
    if win.task:
        win.task.cancel()
    win.bot = bot
    win.is_private = is_private
    win.target_id = target_id
    win.pending.append((text, image_source))
    win.task = asyncio.create_task(_process(win, gen))


async def _process(win: _UserWindow, gen: int) -> None:
    """读秒窗口：静默够时长才回复；被插话取消则静默退出。"""
    try:
        # 读秒窗口：随机 5~10s，模拟真人打字回复的不固定停顿
        await asyncio.sleep(random.uniform(READ_WINDOW_MIN_SECONDS, READ_WINDOW_MAX_SECONDS))
        await _flush(win)
    except asyncio.CancelledError:
        pass
    finally:
        if win.task is asyncio.current_task():
            win.task = None
        if gen == win.generation and not win.pending:
            _windows.pop(win.user_id, None)  # 空闲清理


async def _flush(win: _UserWindow) -> None:
    """回复前统一"读图 + 归纳"（作为读整批的一部分），再生成并分批发送回复。"""
    if not win.pending:
        return
    msgs, win.pending = win.pending, []

    # 读图：本批所有图片统一在回复前解析（不是收到就秒解析，避免节奏割裂）
    async def _parse(src: str) -> str:
        if not src:
            return ""
        desc = await _describe_image_src(win.bot, src)
        return desc or "灰泽满收到一张图片，但暂时没看清里面的内容"  # 兜底保证可回

    descs = await asyncio.gather(*[_parse(src) for _, src in msgs])
    parsed = [(t, d) for (t, _), d in zip(msgs, descs)]

    combined = _combine_text(parsed)
    vision_desc = "；".join(d for _, d in parsed if d)
    print(f"[读秒] user={win.user_id} 攒批 {len(msgs)} 条 → 回复")

    # 归纳：整批（含图片描述）一起给模型做理解提示
    batch_summary = await summarize_batch(parsed)
    reply = await handle_chat(win.user_id, combined, vision_desc=vision_desc,
                              batch_summary=batch_summary)
    reply = clean_reply(reply)  # 去括号前缀 + 整条至多 1 个括号
    parts = split_reply(reply) if SPLIT_REPLY_ENABLED else [reply]
    # 拆分后每段再清一次：内联括号可能落在某段开头（如 （刚醒，声音哑哑的）...），剥掉
    parts = [clean_reply(p) for p in parts]
    for p in parts[:-1]:
        await _send(win, p)
        await asyncio.sleep(split_delay(p))
    await _send(win, parts[-1])


async def _send(win: _UserWindow, content: str) -> None:
    """按私聊/群聊路由发送。"""
    if win.is_private:
        await win.bot.send_private_msg(user_id=win.target_id, message=Message(content))
    else:
        await win.bot.send_group_msg(group_id=win.target_id, message=Message(content))
