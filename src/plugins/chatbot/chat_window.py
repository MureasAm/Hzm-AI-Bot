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
from pathlib import Path

from nonebot.adapters.onebot.v11 import Message

from .core import (
    handle_chat, summarize_batch, _get_clients,
)
from .reply_style import (
    split_reply, split_delay, clean_reply,
)
from .voice import should_voice, send_voice
from .vision import describe_image, describe_image_bytes, _read_image_bytes
from .constants import (
    READ_WINDOW_MIN_SECONDS, READ_WINDOW_MAX_SECONDS, SPLIT_REPLY_ENABLED,
)


class _UserWindow:
    """会话窗口（私聊=用户 / 群聊=整个群）：缓冲 + 去抖/发送任务 + 代际计数。

    target_id = 会话标识（私聊=user_id，群聊=group_id）——群的多人发言攒同一个窗口，
    这样 bot 参与整场群聊，而不是只跟单个人说话。
    """

    def __init__(self, target_id: str):
        self.target_id = target_id
        self.pending: list[tuple[str, str, str, str]] = []   # (sender_id, text, image_url, image_file)
        self.task: asyncio.Task | None = None      # 当前"窗口等待/发送"任务
        self.generation = 0                        # 每次 enqueue 自增
        self.bot = None
        self.is_private = True


_windows: dict[str, _UserWindow] = {}

# 图片字节缓存：enqueue 时立刻下载（rkey 时效内），flush 时直接用。
# 根治"读秒窗口攒批后 rkey 过期 → gchat.qpic.cn 400"（QQ 图链 rkey 时效很短）。
_image_bytes_cache: dict[str, bytes] = {}
_IMAGE_CACHE_MAX = 16


async def _eager_download(source: str) -> None:
    """收到消息立刻下载图片字节存缓存。失败留空，flush 走原逻辑兜底。"""
    if not source or not source.startswith(("http://", "https://")):
        return
    if source in _image_bytes_cache:
        return
    try:
        data = await _read_image_bytes(source)
        if data:
            if len(_image_bytes_cache) > _IMAGE_CACHE_MAX:
                _image_bytes_cache.clear()
            _image_bytes_cache[source] = data
    except Exception:
        pass  # 下载失败不阻塞，flush 时再试


def _combine_text(msgs) -> str:
    """合并本批文本（图片源不参与文本合并）。首元素为文本，兼容 (text, url, file) 与 (text, desc)。"""
    return "\n".join(t for t, *_ in msgs if t.strip()).strip()


# 群聊发送者昵称缓存：(group:user) → 昵称
_sender_name_cache: dict[str, str] = {}


async def _get_sender_name(bot, group_id: str, user_id: str) -> str:
    """取群成员昵称（缓存 + 失败用短 id 兜底），群聊组装时标注"谁在说话"。"""
    key = f"{group_id}:{user_id}"
    if key in _sender_name_cache:
        return _sender_name_cache[key]
    name = f"绿冻{str(user_id)[-4:]}"  # 兜底：短 id
    try:
        info = await bot.get_group_member_info(group_id=int(group_id), user_id=int(user_id))
        n = (info or {}).get("card") or (info or {}).get("nickname") or ""
        if n:
            name = n
    except Exception:
        pass
    _sender_name_cache[key] = name
    if len(_sender_name_cache) > 300:
        _sender_name_cache.clear()
    return name


async def _describe_image_src(bot, image_url: str, image_file: str) -> str:
    """把图片源（url + NapCat file）解析成视觉描述。

    URL 优先（带 eager 缓存）；URL 下载失败（gchat.qpic.cn rkey 过期等）时，
    兜底读 NapCat 本地缓存文件——完全绕开 CDN 的 rkey 问题。
    """
    if not image_url and not image_file:
        return ""
    _, zhipu_client = _get_clients()
    # 1) URL 路径：优先 eager 缓存，否则现场下载
    if image_url:
        data = _image_bytes_cache.get(image_url)
        if data is None:
            try:
                print(f"[视觉] 下载图片: {image_url[:100]}")
                data = await _read_image_bytes(image_url)
            except Exception as e:
                print(f"⚠️ URL 下载失败（尝试本地缓存兜底）: {e}")
                data = None
        if data:
            desc = await describe_image_bytes(zhipu_client, data)
            if desc:
                return desc
    # 2) file 兜底：NapCat 本地缓存文件（绕开 CDN rkey）
    if image_file:
        try:
            info = await bot.get_image(file=image_file)
            local = (info or {}).get("file") or ""
            if local and Path(local).exists():
                print(f"[视觉] 读本地缓存: {local}")
                data = Path(local).read_bytes()
                return await describe_image_bytes(zhipu_client, data)
            u2 = (info or {}).get("url") or ""
            if u2 and u2 != image_url:
                print(f"[视觉] file→url 兜底: {u2[:100]}")
                data = await _read_image_bytes(u2)
                return await describe_image_bytes(zhipu_client, data)
        except Exception as e:
            print(f"⚠️ 图片 file 兜底失败: {e}")
    return ""


def enqueue(target_id: str, sender_id: str, text: str, image_url: str, image_file: str,
            bot, is_private: bool) -> None:
    """采集一条消息进缓冲，重置读秒窗口。

    窗口按 target_id（私聊=user_id，群聊=group_id）开——群的多人发言攒同一个窗口，
    bot 就能参与整场群聊。sender_id 记录发言者，群聊组装时带发送者标签。
    图片：url 优先（rkey 新鲜时急切缓存），file 留给 CDN 失败时读本地兜底。
    """
    if not text.strip() and not image_url and not image_file:
        return
    win = _windows.setdefault(target_id, _UserWindow(target_id))
    win.generation += 1
    gen = win.generation
    if win.task:
        win.task.cancel()
    win.bot = bot
    win.is_private = is_private
    win.pending.append((sender_id, text, image_url, image_file))
    # 图片立刻缓存下载（rkey 新鲜），读秒窗口后不因过期 400
    if image_url:
        asyncio.create_task(_eager_download(image_url))
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
            _windows.pop(win.target_id, None)  # 空闲清理


async def _flush(win: _UserWindow) -> None:
    """回复前统一"读图 + 归纳"（作为读整批的一部分），再生成并分批发送回复。"""
    if not win.pending:
        return
    msgs, win.pending = win.pending, []

    # 读图：本批所有图片统一在回复前解析（不是收到就秒解析，避免节奏割裂）
    async def _parse(u: str, f: str) -> str:
        if not u and not f:
            return ""
        desc = await _describe_image_src(win.bot, u, f)
        return desc or "灰泽满收到一张图片，但暂时没看清里面的内容"  # 兜底保证可回

    descs = await asyncio.gather(*[_parse(u, f) for _, _, u, f in msgs])
    parsed = [(s, t, d) for (s, t, _, _), d in zip(msgs, descs)]  # (sender, text, desc)

    # 组装：私聊=纯文本；群聊=每条带发送者名字，bot 知道谁在说话
    if win.is_private:
        combined = _combine_text([(t, d) for _, t, d in parsed])
    else:
        lines = []
        for s, t, _ in parsed:
            if t.strip():
                name = await _get_sender_name(win.bot, win.target_id, s)
                lines.append(f"{name}：{t.strip()}")
        combined = "\n".join(lines).strip()
    vision_desc = "；".join(d for _, _, d in parsed if d)
    print(f"[读秒] {'群' if not win.is_private else '私聊'}={win.target_id} 攒批 {len(msgs)} 条 → 回复")

    # 归纳：整批（含图片描述）一起给模型做理解提示
    batch_summary = await summarize_batch([(t, d) for _, t, d in parsed])
    reply = await handle_chat(win.target_id, combined, vision_desc=vision_desc,
                              batch_summary=batch_summary, is_group=not win.is_private)
    reply = clean_reply(reply)  # 去括号前缀 + 整条至多 1 个括号

    # 语音优先：成句回复(≥20字、无内心戏括号/链接)朗读成语音条——像真人"话多就录给你听"。
    # 成功就不刷文字分段（互斥不双发）；未启用/合成失败时 send_voice 返回 False，
    # 落回下方文字分段兜底，绝不影响收到回复。
    if should_voice(reply):
        if await send_voice(win.bot, win.target_id, win.is_private, reply):
            return

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
