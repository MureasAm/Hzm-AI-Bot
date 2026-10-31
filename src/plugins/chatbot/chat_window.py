"""读秒窗口（方案B）：把同一用户几秒内的多条消息攒批，静默后统一回复。

架构（参考 Koishi 三模块）：
- 采集：__init__ 的处理器把每条消息 (text, image_sources) 交给 enqueue()
- 会话：enqueue 重置去抖定时器；用户停手超过随机读秒窗口（5~10s）才触发一次回复
- 回复：_flush 在回复前统一"读图 + 归纳"（作为读整批的一部分），再交给 handle_chat，
        复用分批发送（A）；发送中途用户插话 → 取消未发送的分段，优先回新消息

generation 计数用于区分代际：插话取消旧任务后，旧任务不会误清掉新任务正在用的窗口。
参数在 constants.py 配置（READ_WINDOW_MIN/MAX_SECONDS）。
"""
import asyncio
import random
import re
import time
from pathlib import Path

from nonebot.adapters.onebot.v11 import Message, MessageSegment

from .bg import spawn
from .core import (
    handle_chat, summarize_batch, summarize_group_batch, _get_clients, _compose_record_msg,
    _user_name_for, name_budget_note,
)
from .config import set_llm_route, get_vision_client, aux_clients_or
from .memory import get_user_history, append_user_history, humanize_gap
from .group_gate import should_reply_in_group, _addressed_to_her
from .chatlog import log_turn
from .stickers import pick_sticker, log_event
from .reply_style import (
    split_reply, split_delay, clean_reply, maybe_call_name,
)
from .voice import should_voice, send_voice, voice_requested
from .vision import describe_image_bytes, _read_image_bytes, image_kind
from .constants import (
    READ_WINDOW_MIN_SECONDS, READ_WINDOW_MAX_SECONDS, SPLIT_REPLY_ENABLED,
    GROUP_EVENT_COOLDOWN, STICKER_COOLDOWN_TURNS, STICKER_RECENT_KEEP, PROJECT_ROOT,
    AT_SELF_MARK,
)
from . import group_memory
from . import session_memory


class _UserWindow:
    """会话窗口（私聊=用户 / 群聊=整个群）：缓冲 + 去抖/发送任务 + 代际计数。

    target_id = 会话标识（私聊=user_id，群聊=group_id）——群的多人发言攒同一个窗口，
    这样 bot 参与整场群聊，而不是只跟单个人说话。
    """

    def __init__(self, target_id: str):
        self.target_id = target_id
        self.pending: list[tuple[str, str, list[tuple[str, str]]]] = []   # (sender_id, text, [(image_url, image_file), ...])
        # 与 pending 按下标对齐的元信息（ts / message_id）。不并进 tuple，
        # 是为了不打断现有测试/调用方对 pending 三元组的约定。
        self.pending_meta: list[dict] = []
        self.task: asyncio.Task | None = None      # 当前"窗口等待/发送"任务
        self.generation = 0                        # 每次 enqueue 自增
        self.bot = None
        self.is_private = True
        # 表情包冷却**不放这里**——窗口空闲会被销毁，放这儿等于每次重置（见 _sticker_state 的注释）


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
    """合并本批文本（图片源不参与文本合并）。首元素为文本，兼容 (text, ...) 任意尾巴。"""
    return "\n".join(t for t, *_ in msgs if t.strip()).strip()


def _clean_group_record(text: str) -> str:
    """群聊记入短期记忆的版本：去掉内部 @ 标记和它后面的空格，别让模型从历史里抄标记。"""
    lines = []
    for line in (text or "").splitlines():
        line = line.replace(f"{AT_SELF_MARK} ", "").replace(AT_SELF_MARK, "").rstrip()
        if line.endswith("："):
            line += "（只 @ 了她）"
        lines.append(line)
    return "\n".join(lines).strip()


# 文字里**明确指图**的说法：这时候图片就是话题，不能丢。
_IMAGE_REF_RE = re.compile(
    r"这张图|这个图|这张截图|这个截图|这个表情包|表情包|图里|图中|图上|"
    r"上面这个|这个表情|看这个|你看这个|看图|这图"
)
# 问句/评价请求：文字在追问上一轮的话题，表情包通常只是反应。
_QUESTION_RE = re.compile(r"怎么|什么|为什么|如何|评价|觉得|呢|吗|哪|咋")


_MOOD_WORDS = (
    "无奈", "无语", "委屈", "得意", "卖萌", "撒娇", "尴尬", "害羞", "嫌弃", "崩溃",
    "开心", "难过", "生气", "震惊", "疑惑", "期待", "害怕", "疲惫", "敷衍", "得逞",
    "累", "困", "笑", "哭", "可爱",
)


def _mood_words(desc: str, limit: int = 2) -> str:
    """从视觉描述里按出现顺序抽情绪词（最多 limit 个）。抽不到返回空串。"""
    found = [(desc.find(w), w) for w in _MOOD_WORDS if w in desc]
    out = []
    for _, w in sorted(found):
        if w not in out:
            out.append(w)
        if len(out) >= limit:
            break
    return "/".join(out)


def _plain_note(note: str) -> str:
    """把 `_image_note` 的呈现还原成**记录/兜底**用的普通文字。

    "[图片：X]" → "X"；"[表情包：无奈]" → "表情包，无奈"。短期记忆里那行是
    「用户：…（附图：X）」，带着方括号会和记录模板的括号打架。
    """
    note = (note or "").replace(chr(10), " ")
    if note.startswith("[图片："):
        return note[len("[图片："):].rstrip("]")
    if note.startswith("[表情包"):
        return note.strip("[]").replace("：", "，")
    return note


def _image_note(text: str, desc: str) -> str:
    """一张图在**用户消息里**怎么呈现（2026-10-30 改成三档）。

    - 纯图 / 极短非问句 / 文字明确指了图 → 完整描述（表情包本身就是内容）
    - 文字 + 表情包 → `[表情包：无奈]` 这种**只带情绪词**的短标签
    - 图里是灰泽满本人 → 短标签写成 `[表情包·灰泽满本人：得意]`

    为什么不一刀切丢（旧规则 2026-10-19 的教训反过来）：文字+表情包整段丢掉，
    就看不见"用户是想给她看 / 让她评价"和"用户在用表情包表达自己心情"这两种真需求；
    而整段保留又会劫持话题（实测：聊小说时发「怎么评价」+虫子表情包，她去评价虫子）。
    折中点是**只留情绪词、不留描述里的名词**——旧翻车的根子正是那个名词变成了话题。
    """
    desc = (desc or "").replace(chr(10), " ").strip()   # 描述是两行（标签+内容），内联时压成一行
    if image_kind(desc) != "表情包":
        return f"[图片：{desc}]"
    # 图里是她自己：**不压缩**。实测（2026-10-30）同一句"怎么评价"+她的表情包，
    # 完整描述回来的那句是"不就是灰泽满周五下午的状态嘛"——比短标签的"这表情包吗"活多了；
    # 而且这种图本来就在聊她，不存在"把别人的话题劫走"的风险。
    if "灰泽满本人" in desc:
        return f"[图片：{desc}]"
    t = (text or "").strip()
    if not t or _IMAGE_REF_RE.search(t) or (len(t) <= 4 and not _QUESTION_RE.search(t)):
        return f"[图片：{desc}]"
    mood = _mood_words(desc)
    return f"[表情包：{mood}]" if mood else "[表情包]"


# 群聊发送者昵称缓存：(group:user) → 昵称
# ⚠️ 这行被误删过一次（2026-10-30 线上 NameError：`name '_sender_name_cache' is not defined`）——
# 当时把 `_keep_image_desc` 整段替换掉时，把函数**后面**的这行声明一起带走了，
# 单测没抓到是因为群聊用例都 monkeypatch 了 `_get_sender_name`。**改这段代码时别整段删到下一个 def。**
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
    vision_client = get_vision_client()
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
            desc = await describe_image_bytes(vision_client, data)
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
                return await describe_image_bytes(vision_client, data)
            u2 = (info or {}).get("url") or ""
            if u2 and u2 != image_url:
                print(f"[视觉] file→url 兜底: {u2[:100]}")
                data = await _read_image_bytes(u2)
                return await describe_image_bytes(vision_client, data)
        except Exception as e:
            print(f"⚠️ 图片 file 兜底失败: {e}")
    return ""


def enqueue(target_id: str, sender_id: str, text: str,
            image_sources: list[tuple[str, str]], bot, is_private: bool,
            *, ts: float | None = None, message_id: str = "") -> None:
    """采集一条消息进缓冲，重置读秒窗口。

    窗口按 target_id（私聊=user_id，群聊=group_id）开——群成员发言攒同一个窗口。
    群内闲聊也攒批：群现场注入 + 轻量群记忆帮她分辨在场和该接谁的话。
    sender_id 记录发言者，群聊组装时带发送者标签。
    图片：**一条消息可以带多张**（文字+多图一起发是 QQ 的常见形态），
    每张的 url 优先（rkey 新鲜时急切缓存），file 留给 CDN 失败时读本地兜底。
    """
    if not text.strip() and not image_sources:
        return
    win = _windows.setdefault(target_id, _UserWindow(target_id))
    win.generation += 1
    gen = win.generation
    win.bot = bot
    win.is_private = is_private
    # 私聊/群聊统一：整窗口攒批、读秒后一起回（群=回整场群聊）
    win.pending.append((sender_id, text, list(image_sources)))
    win.pending_meta.append({
        "ts": float(ts) if ts is not None else time.time(),
        "message_id": str(message_id or ""),
    })
    if len(win.pending) > 30:
        cut = len(win.pending) - 30
        del win.pending[:cut]
        del win.pending_meta[:cut]
    for url, _ in image_sources:
        if url:
            spawn(_eager_download(url))
    if win.task:
        win.task.cancel()
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
    # 用量分账：这条链路上的所有 LLM 调用（归纳 / 接话门 / 主回复 / 表情包 / 记忆提取…）
    # 都算在这条通道上。设在最前面是因为**归纳和接话门在 handle_chat 之前**——
    # 而它们恰恰是群聊多出来的那两次调用，正是要量的东西（见 config._ROUTE）。
    set_llm_route("群" if not win.is_private else "私")
    # ⚠️ 先取快照，**不要**把 pending 清空。这一批什么时候算“处理完”？
    #    回复真的生成并记入短期记忆之后才从 pending 里删（_commit_batch）。
    #    踩坑（2026-10-20，用户 2151983431）：旧代码一进来就 `msgs, win.pending = win.pending, []`，
    #    而新消息会取消正在进行的 _process；取消时这批已经被取走 → **静默丢失**。
    #    实况：「跑步呀」已经开始回复，6 秒后「你忘了吗」进来把它取消，
    #    于是「跑步呀」既没回复、也没进记忆/聊天记录，下一轮只看得到「早」。
    msgs = list(win.pending)
    # 与 msgs 对齐的元信息；测试里手工设置 pending 时没有 meta，用空 dict 兜底。
    metas = [win.pending_meta[i] if i < len(win.pending_meta) else {} for i in range(len(msgs))]
    my_gen = win.generation   # 起跑时的代际；后面用它判断"我是否已被新消息取代"

    def _commit_batch() -> None:
        """这一批已经生成并记录完毕，从 pending 删掉；中途被取消就永远走不到这里。"""
        del win.pending[:len(msgs)]
        del win.pending_meta[:len(msgs)]

    # 读图：本批所有消息里的**所有**图片统一在回复前解析（不是收到就秒解析，避免节奏割裂）
    async def _parse(src: tuple[str, str]) -> str:
        u, f = src
        if not u and not f:
            return ""
        desc = await _describe_image_src(win.bot, u, f)
        return desc or "灰泽满收到一张图片，但暂时没看清里面的内容"  # 兜底保证可回

    jobs = [(i, src) for i, (_, _, srcs) in enumerate(msgs) for src in srcs]
    results = await asyncio.gather(*[_parse(src) for _, src in jobs])
    descs = [[] for _ in msgs]
    for (i, _), d in zip(jobs, results):
        if d:
            descs[i].append(d)
    all_descs = [d for ds in descs for d in ds]
    # 给模型看的呈现（可能被压成情绪短标签）vs 给归纳/日志用的完整描述：
    parsed = [(s, t, [_image_note(t, d) for d in ds])
              for (s, t, _), ds in zip(msgs, descs)]     # (sender, text, [图片呈现...])
    full = [(s, t, ds) for (s, t, _), ds in zip(msgs, descs)]  # 完整描述（归纳/日志用）

    # 组装：
    # · 私聊 = 纯文本 transcript（图片按顺序内联）
    # · 群聊 = 纯「昵称：文本」（图片按顺序内联）+ 一条**系统元信息**说明本批谁点名了她。
    #   ⚠️ 2026-10-21 改：原来把 `[对灰泽满]/[未点名]` 直接写进用户消息，模型偶尔会把
    #   标记原样吐出来（实测 2/216 条群回复带了 `[未点名]`）。**标记不该出现在用户消息里**——
    #   换成系统侧的【本批点名情况】，用户消息保持干净，模型没有可抄的标记。
    #   点名判定仍复用接话门的 `_addressed_to_her`（@ / 引用她 / 直呼名字），确定性、可回归。
    id_to_name = {}
    combined = ""
    user_transcript = ""
    group_address_note = ""
    batch_addressed = False
    group_items: list[tuple[str, str, str]] = []   # (昵称, 文本, 图片描述) → 对话线归纳
    reply_to = ""                                  # 她这句在回谁（历史里标「灰泽满（回X）」）
    group_refs: dict[int, dict] = {}
    if win.is_private:
        combined = _combine_text([(t, ds) for _, t, ds in parsed])
        tlines = []
        for _, t, ds in parsed:
            if t.strip():
                tlines.append(t.strip())
            tlines.extend(ds)   # _image_note 已经拼好 "[图片：…]" / "[表情包：无奈]"，别再包
        user_transcript = "\n".join(tlines).strip()
    else:
        lines, plain_lines, addressed_lines = [], [], []
        addressed_speakers = []
        last_speaker = ""
        line_no = 0
        for idx, ((s, t, ds), (_, _, fds)) in enumerate(zip(parsed, full)):
            meta = metas[idx] if idx < len(metas) else {}
            text = t.strip()
            if not text and not ds:
                continue
            name = await _get_sender_name(win.bot, win.target_id, s)
            last_speaker = name
            if text:
                id_to_name[s] = name
            addressed = _addressed_to_her(text)   # @ 已被 __init__ 变成 AT_SELF_MARK
            batch_addressed = batch_addressed or addressed
            # 模型看到的用户消息也要干净：把内部的 @ 标记换成中性说法（点名情况另由系统元信息给）
            shown = text.replace(AT_SELF_MARK, "").strip() or ("（只 @ 了她）" if addressed else "")
            body = " ".join(([shown] if shown else []) + list(ds))
            line_no += 1
            # 每条带本地编号 + 相对时间：群聊里旧消息被唤醒后当成刚发生，是高频翻车点。
            # <90 秒不显示时间（同一批内没必要），只有跨批次/断线重连才会出现。
            ts = meta.get("ts")
            age = humanize_gap(time.time() - float(ts)) if ts else ""
            age_part = f"（{age}前）" if age else ""
            lines.append(f"[{line_no}] {name}{age_part}：{body}")
            group_refs[line_no] = meta
            # 归纳用**中性版**（`shown`）：内部 @ 标记不该进对话线，否则会被归纳成
            # 「有人@了灰泽满」再注入回去，模型有了可复述的标记来源。
            group_items.append((name, shown, "；".join(fds)))
            if text:
                plain_lines.append(f"{name}：{text}")
            if addressed:
                said = shown or "（只 @ 了她 / 发了图）"
                addressed_lines.append(f"{name}说「{said[:60]}」")
                addressed_speakers.append(name)
        combined = "\n".join(plain_lines).strip()
        user_transcript = "\n".join(lines).strip()
        if id_to_name:
            group_memory.upsert_members(win.target_id, id_to_name)
        # ⚠️ **未点名时不注入任何“没点名”的信息**（2026-10-23）。旧版会注一句
        #    「本批无人点名灰泽满：以上都是群友在互相聊」——模型于是把这句话复述成
        #    「没点我，灰泽满才不管」「（群里没人 @ 灰泽满，保持潜水）」这类元回复。
        #    安静与否由接话门决定；她这边只需要知道「有人在跟她说话」这个**正向**事实。
        if addressed_lines:
            group_address_note = "；".join(addressed_lines[:3])
        reply_to = addressed_speakers[-1] if addressed_speakers else last_speaker

    vision_desc = "；".join(_plain_note(n) for _, _, ds in parsed for n in ds)
    img_count = sum(len(ds) for _, _, ds in parsed)
    print(f"[读秒] {'群' if not win.is_private else '私聊'}={win.target_id} "
          f"攒批 {len(msgs)} 条 → 回复" + (f"（含 {img_count} 张图）" if img_count else ""))

    # 归纳：私聊整批（含图片描述）给模型做理解提示；群聊换成**带发言人的对话线**——
    # 群里有好几个人，只归纳“说了什么”不够，得知道谁在对谁说、停在哪句话头上，
    # 她才能接在同一条线上（2026-10-23，见 docs/历史/交接文档.md）。
    batch_summary = ""
    group_thread_note = ""
    if win.is_private:
        batch_summary = await summarize_batch([(t, "；".join(ds)) for _, t, ds in full])
    else:
        group_thread_note = await summarize_group_batch(group_items)
    # 群近况写入群记忆（有冷却，防碎碎念刷爆；只记有信息量的归纳）
    if not win.is_private and group_thread_note:
        g = group_memory.get_group(win.target_id)
        last_t = (g.get("events") or [{}])[0].get("t", 0.0) if g.get("events") else 0.0
        if time.time() - last_t >= GROUP_EVENT_COOLDOWN:
            group_memory.add_event(win.target_id, group_thread_note)

    # 群聊接话判定：群友不是每条都接，攒批安静下来时先问一句"这批要不要开口"。
    # **私聊不过门**——被人直接找来必回，那是另一回事。
    # 判定不接时：不生成、不发送，但**仍把这一批记进短期记忆**（只记用户侧）——
    # 她"看到了、只是没说"；不记的话下一轮判定看不到上下文，她就会开始"装作知道上文"。
    if not win.is_private:
        deepseek_client, _ = aux_clients_or(_get_clients)   # 接话门走辅助链（2026-10-29）
        hist = "\n".join(get_user_history(win.target_id)[-8:])
        if not await should_reply_in_group(deepseek_client, hist, combined,
                                           thread_note=group_thread_note):
            append_user_history(win.target_id, _clean_group_record(combined), "",
                                user_label="群聊")
            _commit_batch()   # 这批已经“处理”了（只记不发），不能留给下一轮重判
            print(f"[读秒] 群={win.target_id} 这批不接，只记不发")
            return

    # 群聊的 user_transcript 带批次标记（给模型看）；**短期记忆记干净版**——
    # 批次级标记不该被当成"对话历史"喂回去（record_text，见 core.handle_chat）。
    record_kwargs = {}
    if not win.is_private:
        # 内部 @ 标记也不进历史（模型会照抄）；历史只留干净的「昵称：文本」
        record_kwargs["record_text"] = _compose_record_msg(_clean_group_record(combined), vision_desc)
    group_batch_note = (f"本批 {len(msgs)} 条"
                        if not win.is_private and len(msgs) >= 2 else "")
    reply = await handle_chat(win.target_id, combined, vision_desc=vision_desc,
                              batch_summary=batch_summary, is_group=not win.is_private,
                              user_transcript=user_transcript,
                              group_addressed=(True if win.is_private else batch_addressed),
                              group_address_note=group_address_note,
                              group_thread_note=group_thread_note,
                              group_batch_note=group_batch_note,
                              reply_to=reply_to if not win.is_private else "",
                              **record_kwargs)
    # <reply:编号>：桥接式群聊的“接某一条”协议。只在群聊生效；
    # 编号映射到本批消息的 QQ message_id，发送时变成引用回复。
    # 标签必须在 clean_reply / 落盘之前剥掉，绝不能发到 QQ 上。
    reply_mid = ""
    if not win.is_private and reply:
        m = re.search(r"<reply:(\d+)>", reply)
        if m:
            n = int(m.group(1))
            reply_mid = str(group_refs.get(n, {}).get("message_id") or "")
            reply = (reply[:m.start()] + reply[m.end():]).strip()
        # 模型偶尔会把用户消息里的 [1] 当成自己的编号照抄；群聊开头剥掉。
        reply = re.sub(r"^\s*\[\d+\]\s*", "", reply).strip()
    reply = clean_reply(reply)  # 去括号前缀 + 整条至多 1 个括号
    if reply and win.is_private:
        # 称呼替换（2026-10-30）：模型要么每句都叫名字、要么一句都不叫；这里在**已经说出的话**上
        # 按概率把一次"你"换成对方要的名字，落在中间。用 name_budget 兜底：最近她叫过就直接跳过。
        _name = _user_name_for(win.target_id)
        if _name and not name_budget_note(win.target_id):
            reply = maybe_call_name(reply, _name)
    if not reply:
        # 未点名的群聊批次，模型输出的是“没点我/保持潜水”式元回复 → 这批就当没接。
        # handle_chat 已经只记了用户侧（不记这句元回复），这里不发送、不落聊天记录。
        _commit_batch()
        print(f"[读秒] {'群' if not win.is_private else '私聊'}={win.target_id} 这轮选择不说话")
        return

    # 聊天落盘（分析用，gitignore）。放这里而不是发完之后：下面**语音那条路会提前 return**，
    # 放到发送之后会漏掉所有语音回复。这里记的是"她这轮说了什么"（不管最终是语音还是文字）。
    log_turn(win.target_id, not win.is_private, combined, reply,
             images=all_descs, batch_messages=len(msgs))
    _commit_batch()   # 回复已生成并记录；从这里开始即使被取消也不能重答这一批

    # 语音优先：成句回复(≥30字、无内心戏括号/链接)朗读成语音条——像真人"话多就录给你听"。
    # 成功就不刷文字分段（互斥不双发）；未启用/合成失败时 send_voice 返回 False，
    # 落回下方文字分段兜底，绝不影响收到回复。
    # 用户点名要语音（消息里带"语音"）→ force：短句也读，别让人白问一句。
    if should_voice(reply, force=voice_requested(combined)):
        if win.generation != my_gen:
            # 已经被新消息取代：别再占 GPU 去合成（合成了也送不出去）
            print("[读秒] 语音：已被新消息取代，跳过合成")
            return
        # ⚠️ shield：合成+发送**一旦开始就让它做完**。
        # 不 shield 的话：用户在这几秒里插话 → enqueue 的 win.task.cancel() 会把
        # await 中的 send_voice 打断 → **wav 已经生成、消息没发出去**，而且文字兜底
        # 也不会走（CancelledError 是 BaseException，except Exception 抓不到）——
        # 她那边什么都收不到。实测症状就是"TTS 转出来了但没发出去"，且**时有时无**
        # （取决于插话有没有落在合成那几秒里）。
        # 语义上也说得通：文字分段是"发一段是一段"，语音是一条整消息，
        # 所以"已开始就等于第一段已发出"。
        if await asyncio.shield(send_voice(win.bot, win.target_id, win.is_private, reply)):
            # 语音之后**也补表情包**（用户 2026-10-05）：原来这里直接 return，
            # 表情包永远是语音轮的盲区。表情包自己有"十分对应才发"的闸门，不会硬凑。
            await _maybe_send_sticker(win, reply)
            return

    parts = split_reply(reply) if SPLIT_REPLY_ENABLED else [reply]
    # 拆分后每段再清一次：内联括号可能落在某段开头（如 （刚醒，声音哑哑的）...），剥掉
    parts = [clean_reply(p) for p in parts]
    for i, p in enumerate(parts[:-1]):
        if i == 0 and reply_mid:
            await _send(win, p, reply_mid=reply_mid)
        else:
            await _send(win, p)
        await asyncio.sleep(split_delay(p))
    if len(parts) == 1 and reply_mid:
        await _send(win, parts[-1], reply_mid=reply_mid)
    else:
        await _send(win, parts[-1])

    # 表情包：文字发完之后，若某张表情与这句话"十分对应"就补一张。
    # （语音那条路在上面单独调了一次——它提前 return，不在这里。）
    await _maybe_send_sticker(win, reply)


# 表情包冷却状态：**必须按 target_id 持久，不能挂在 _UserWindow 上**。
# 踩坑：放在 _UserWindow 上时，窗口一空闲就被销毁（_windows.pop），下次重建
# 两个字段都回到初始值 —— 而私聊里每轮之间**必然有静默**，所以每次都重建，
# **冷却形同虚设**。表现成"睡觉话题每次都会发那张眠了"。
# （同一个坑还让 recent 永远是空 —— 跨轮防重复也一起失效。）
# `recent` 里存的是**类别**（不是单张）：同一类可能有好几张图（委屈 3 张、
# 无语无奈 4 张），按单张去重挡不住"换一张还是同一个味道"（用户 2026-10-04 定）。
_sticker_state: dict[str, dict] = {}


def _sticker_st(target_id: str) -> dict:
    return _sticker_state.setdefault(
        target_id, {"turns": STICKER_COOLDOWN_TURNS, "recent": [],
                    "sent": False, "topic": None})


async def _maybe_send_sticker(win: _UserWindow, reply: str) -> None:
    """双闸：同一话题只发一张 + 发过之后 5 轮内不再发。

    - 话题闸：同一 `topic_since` 已经发过 → 跳过。只有换话题才解除。
    - 轮次闸：任何一次发送后 `STICKER_COOLDOWN_TURNS=5` 轮内不再问模型；
      即使几十秒内换了话题，也挡住（2026-10-24 用户定，实测话题探测会几十秒切一次）。
    - 两闸是 AND：两条都满足才可能发。
    """
    st = _sticker_st(win.target_id)
    kind = "private" if win.is_private else "group"
    topic = str(session_memory.get_session(win.target_id).get("topic_since") or "")
    if st.get("topic") != topic:
        # 新话题：解除“同话题一张”，但 **turns 不重置**，5 轮闸继续生效
        st["topic"] = topic
        st["sent"] = False
    if st.get("sent"):
        log_event("topic_used", session=win.target_id, kind=kind)
        return
    if st["turns"] < STICKER_COOLDOWN_TURNS:
        st["turns"] += 1
        log_event("cooldown", session=win.target_id, kind=kind)
        return
    deepseek_client, zhipu_client = _get_clients()
    # avoid 传**全部**最近发过的**类别**（不只最近 1 个）：光防"连着发"不够，
    # 用户报的是"同类"太频繁 —— 换个话题回来还会撞上同一类。
    hit = await pick_sticker(deepseek_client, reply, avoid_groups=st["recent"],
                             session=win.target_id, kind=kind, zhipu_client=zhipu_client)
    if not hit:
        st["turns"] += 1
        return
    img = MessageSegment.image(file=_to_qq_image(hit["file"]))
    try:
        if win.is_private:
            await win.bot.send_private_msg(user_id=win.target_id, message=img)
        else:
            await win.bot.send_group_msg(group_id=win.target_id, message=img)
        print(f"[表情包] 已发送 {hit['id']}［{hit.get('group', '')}］")
        log_event("sent", session=win.target_id, kind=kind, sticker=hit,
                  why=str(hit.get("_source") or "llm"))
    except Exception as e:
        print(f"⚠️ 表情包发送失败（忽略）: {e}")
        log_event("sent_failed", session=win.target_id, kind=kind, sticker=hit, why=str(e))
    st["turns"] = 0
    st["sent"] = True
    st["topic"] = topic
    st["recent"] = ([hit.get("group") or hit["id"]] + st["recent"])[:STICKER_RECENT_KEEP]


def _to_qq_image(rel_path: str) -> str:
    """本地图片 → 合法的 file URL。

    和语音同一个坑：必须 `/` 正斜杠 + `file:///` 三斜杠，反斜杠版不是合法 URL。
    """
    p = (PROJECT_ROOT / rel_path).resolve()
    return "file:///" + p.as_posix()


async def _send(win: _UserWindow, content: str, reply_mid: str = "") -> None:
    """按私聊/群聊路由发送；群聊带 reply_mid 时首段变成引用回复。"""
    message = Message(content)
    if reply_mid and not win.is_private:
        try:
            message = Message(MessageSegment.reply(int(reply_mid))) + message
        except (TypeError, ValueError):
            pass
    if win.is_private:
        await win.bot.send_private_msg(user_id=win.target_id, message=message)
    else:
        await win.bot.send_group_msg(group_id=win.target_id, message=message)
