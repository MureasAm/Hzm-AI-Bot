"""B站直播/动态监听 + 私聊广播（方向3）。

启动时注册后台任务，周期性轮询灰泽满本人的 B站账号：
- 开播：状态翻转（未播→直播）才推送一次，带上直播间标题
- 动态：出现新动态 ID 才推送，结构化通知（真实内容 + 个人空间链接），含配图时连图一起发
推送目标：灰泽满 QQ 号的所有好友（私聊），NOTIFY_FRIENDS_WHITELIST 可收窄为白名单。
去重状态（last_live_status / last_dynamic_id / 好友缓存）持久化到 data/bili_state.json，
重启不重复推送。所有外部调用失败都降级为日志，绝不让监听任务崩溃。
"""
import asyncio
import random
import re
import json
import tempfile
import time
from pathlib import Path

import httpx

from nonebot import get_bot, get_driver
from nonebot.adapters.onebot.v11 import Message, MessageSegment

from .constants import BILI_STATE_FILE
from .config import get_bili_uid, get_bili_sessdata, get_notify_whitelist, get_push_interval
from .vision import _read_image_bytes, _normalize_image

LIVE_STATUS_API = "https://api.live.bilibili.com/room/v1/Room/get_status_info_by_uids"

# B站 WAF 会拦 python-httpx 默认 UA，必须带浏览器 UA
_BILI_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

async def _download_image(url: str) -> Path:
    """下载动态配图并统一转 JPEG 到临时文件，返回路径；失败抛异常由调用方兜底。"""
    data = await _read_image_bytes(url)
    data = _normalize_image(data)  # WEBP/PNG 统一转 JPEG，QQ 显示更稳
    tmp = Path(tempfile.gettempdir()) / f"hzm_dyn_{time.time_ns()}.jpg"
    tmp.write_bytes(data)
    return tmp


# ==================== 状态持久化 ====================

def _load_state() -> dict:
    if BILI_STATE_FILE.exists():
        try:
            return json.loads(BILI_STATE_FILE.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(state: dict) -> None:
    try:
        BILI_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BILI_STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), "utf-8"
        )
    except OSError as e:
        print(f"⚠️ B站状态写入失败: {e}")


# ==================== 文案生成 ====================

def _live_open_message(room_title: str, room_id: int = 0) -> str:
    """开播播报：结构化通知（直播标题 + 直播间网址）。

    灰泽满宣布开播！

    今天的内容是：（直播标题）

    https://live.bilibili.com/房间号
    """
    title = room_title.strip() if room_title and room_title.strip() else "灰泽满的直播间"
    url = f"https://live.bilibili.com/{room_id}" if room_id else "https://live.bilibili.com"
    return f"灰泽满宣布开播！\n\n今天的内容是：{title}\n\n{url}"


def _extract_dynamic_text(item: dict) -> str:
    """从 bilibili-api 的动态 item 里提取正文（防御式，兼容多种类型）。

    旧格式（ARCHIVE/DRAW）：module_dynamic.desc.text；转发取 orig.desc.text。
    新格式（OPUS 图文/文字动态）：major.opus.summary.text。
    """
    modules = item.get("modules", {}) or {}
    md = modules.get("module_dynamic", {}) or {}
    text = (md.get("desc", {}) or {}).get("text", "") or ""
    if not text:
        orig = md.get("orig", {}) or {}
        text = (orig.get("desc", {}) or {}).get("text", "") or ""
    if not text:
        # OPUS 新格式
        opus = (md.get("major", {}) or {}).get("opus", {}) or {}
        summary = opus.get("summary")
        if isinstance(summary, dict):
            text = summary.get("text", "") or ""
        elif isinstance(summary, str):
            text = summary
    return text.strip()


# ==================== 表情解析 ====================
# B站动态文本里 [xxx] 是表情占位符（含 [UPOWER_uid_名字] 这种主播定制贴纸）。
# 通过 emote panel 接口解析成图片 URL；解析不到的占位符剥掉，避免推给 QQ 显示丑代码。
_emote_map = None


async def _get_emote_map() -> dict:
    """拉取 B站 emote panel 并缓存 text→url 映射（带 SESSDATA 才看得到完整表情）。"""
    global _emote_map
    if _emote_map is not None:
        return _emote_map
    try:
        url = "https://api.bilibili.com/x/emote/user/panel/web"
        async with httpx.AsyncClient(timeout=10.0,
                                     headers={"User-Agent": _BILI_UA,
                                              "Cookie": f"SESSDATA={get_bili_sessdata()}",
                                              "Referer": "https://t.bilibili.com/"}) as c:
            resp = await c.get(url, params={"business": "dynamic", "build": "0",
                                            "mobi_app": "web", "uid": get_bili_uid()})
            data = resp.json()
        m = {}
        for p in (data.get("data") or {}).get("packages") or []:
            for e in (p.get("emote") or []):
                if e.get("text") and e.get("url"):
                    m[e["text"]] = e["url"]
        _emote_map = m
        print(f"[B站] emote 表情表已加载: {len(m)} 个")
    except Exception as e:
        print(f"⚠️ emote panel 拉取失败（表情将剥除）: {e}")
        _emote_map = {}
    return _emote_map


def _parse_emotes(text: str, emote_map: dict) -> tuple[str, list]:
    """把 [xxx] 表情解析成图片 URL；返回 (清洗后文本, 图片URL列表)。

    解析到的表情从文本移除（图片单独附上）；解析不到的占位符也剥掉
    （避免 [UPOWER_...] 这种丑代码显示给 QQ 好友）。
    """
    if not text:
        return text, []
    urls = []

    def repl(m):
        token = m.group(0)
        u = emote_map.get(token) or ""
        if u:
            urls.append(u)
        return ""  # 占位符一律移除（图单独发）

    cleaned = re.sub(r"\[[^\]]{1,30}\]", repl, text)
    return cleaned.strip(), urls


def _extract_dynamic_images(item: dict) -> list:
    """从动态 item 提取配图 URL（图文/视频封面），无图返回空列表。

    OPUS 新格式图片在 major.opus.pics（不是 images）。
    """
    modules = item.get("modules", {}) or {}
    major = (modules.get("module_dynamic", {}) or {}).get("major", {}) or {}
    mtype = major.get("type", "")
    urls = []
    if mtype == "MAJOR_TYPE_OPUS":
        opus = major.get("opus", {}) or {}
        for pic in (opus.get("pics") or []):
            if pic.get("url"):
                urls.append(pic["url"])
        for im in (opus.get("images") or []):  # 兼容旧字段
            if im.get("url"):
                urls.append(im["url"])
    elif mtype == "MAJOR_TYPE_ARCHIVE":
        arc = major.get("archive", {}) or {}
        if arc.get("pic"):
            urls = [arc["pic"]]
    elif mtype == "MAJOR_TYPE_DRAW":
        draw = major.get("draw", {}) or {}
        urls = [im.get("src", "") for im in (draw.get("items") or []) if im.get("src")]
    return urls


# ==================== 监控主体 ====================

class BiliMonitor:
    def __init__(self):
        self.uid = get_bili_uid()
        self.sessdata = get_bili_sessdata()
        self.state = _load_state()
        self._warned_no_sessdata = False
        # 基线是否已建（进程内标志，不持久化——每次启动都重新对齐，
        # 避免停机期间的动态/开播在重启后被当新事件推送）
        self._primed = False

    async def poll_once(self, bot) -> None:
        if not self.uid:
            return
        if not self._primed:
            await self._prime()
        await self._check_live(bot)
        await self._check_dynamic(bot)

    async def _prime(self) -> None:
        """每次启动建基线：记录当前直播/动态状态，不推送"停机期间已发生"的事件。

        基线是进程内一次性的（self._primed），每次 bot 启动都重新对齐——
        否则重启后会把停机期间的开播/动态当新事件广播给所有好友。
        """
        try:
            info = await self._fetch_live_status()
            self.state["last_live_status"] = info["live_status"] == 1
        except Exception as e:
            print(f"⚠️ 基线-直播状态失败（忽略）: {e}")
        try:
            dyn = await self._fetch_latest_dynamic()
            self.state["last_dynamic_id"] = dyn.get("id", "")
        except Exception as e:
            print(f"⚠️ 基线-动态失败（忽略）: {e}")
        self._primed = True
        _save_state(self.state)
        print("[B站] 已建立基线（对齐当前直播/动态状态），之后检测到新的开播/动态才会推送")

    # ---- 开播 ----

    async def _fetch_live_status(self) -> dict:
        """查询直播状态，返回 {live_status, title, room_id}。失败抛异常由调用方兜底。"""
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                LIVE_STATUS_API,
                json={"uids": [int(self.uid)]},
                headers={"User-Agent": _BILI_UA, "Referer": "https://live.bilibili.com/"},
            )
            data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"B站直播接口返回 code={data.get('code')}")
        info = (data.get("data") or {}).get(str(self.uid))
        if not info:
            raise RuntimeError("B站直播接口未返回该 UID 数据")
        return {
            "live_status": int(info.get("live_status", 0)),
            "title": str(info.get("title", "")),
            "room_id": int(info.get("room_id", 0) or 0),
            # 直播封面：优先主播设置封面(cover_from_user)，空则回退关键帧(keyframe)
            "cover": str(info.get("cover_from_user", "") or info.get("keyframe", "") or ""),
        }

    async def _check_live(self, bot) -> None:
        try:
            info = await self._fetch_live_status()
        except Exception as e:
            print(f"⚠️ B站直播状态查询失败（忽略）: {e}")
            return

        is_live = info["live_status"] == 1  # 1=直播中, 2=轮播
        was_live = bool(self.state.get("last_live_status", False))
        if is_live and not was_live:
            content = _live_open_message(info["title"], info.get("room_id", 0))
            print(f"[B站] 检测到开播 -> {content}")
            # 开播带直播封面（下载失败不阻塞，仍发文字）
            image_paths = []
            if info.get("cover"):
                try:
                    p = await _download_image(info["cover"])
                    image_paths.append(p)
                    print(f"[B站] 直播封面已下载: {p.name}")
                except Exception as e:
                    print(f"⚠️ 直播封面下载失败（忽略，仍发文字）: {e}")
            await self._push(bot, content, image_paths=image_paths)
        self.state["last_live_status"] = is_live
        _save_state(self.state)

    # ---- 动态 ----

    async def _fetch_latest_dynamic(self) -> dict:
        """取最新一条动态，返回 {id, text, image_urls}。

        B站 2026 动态接口要求登录（SESSDATA），由 get_dynamic_page_info 处理 WBI/buvid。
        """
        from bilibili_api import dynamic  # 懒加载，避免无关场景的额外依赖开销
        from bilibili_api.utils.network import Credential

        # 每次轮询重新读 .env.prod，换 SESSDATA 不用重启（B站 SESSDATA 会过期，省去重启）
        cred = Credential(sessdata=get_bili_sessdata())
        page = await dynamic.get_dynamic_page_info(cred, host_mid=int(self.uid), pn=1)
        items = (page or {}).get("items") or []
        if not items:
            raise RuntimeError("该 UID 暂无动态")
        item = items[0]
        text = _extract_dynamic_text(item)
        emote_map = await _get_emote_map()
        # 动态自带的 emote 映射（若有）优先——UPOWER 定制贴纸往往在这提供 URL
        item_emotes = item.get("emote") or {}
        if item_emotes:
            merged = dict(emote_map)
            for k, v in item_emotes.items():
                u = (v or {}).get("url", "") or ""
                if u:
                    merged[k] = u
            emote_map = merged
        clean_text, emote_urls = _parse_emotes(text, emote_map)
        return {
            "id": str(item.get("id_str") or item.get("id") or ""),
            "text": clean_text,
            "image_urls": _extract_dynamic_images(item),
            "emote_urls": emote_urls,
        }

    async def _check_dynamic(self, bot) -> None:
        if not get_bili_sessdata():
            if not self._warned_no_sessdata:
                print("⚠️ BILI_SESSDATA 未配置，动态监控关闭（开播监控正常）。配置后下次轮询即生效。")
                self._warned_no_sessdata = True
            return
        try:
            dyn = await self._fetch_latest_dynamic()
        except Exception as e:
            print(f"⚠️ B站动态查询失败（忽略）: {e}")
            return
        if not dyn["id"]:
            return
        if dyn["id"] == str(self.state.get("last_dynamic_id", "") or ""):
            return  # 已推送过

        # 动态配图 + 表情图一起下载附上（失败不阻塞，仍发文字）；限 4 张防刷屏
        image_paths = []
        for u in (dyn.get("image_urls") or [])[:1] + (dyn.get("emote_urls") or [])[:4]:
            try:
                p = await _download_image(u)
                image_paths.append(p)
            except Exception as e:
                print(f"⚠️ 图片下载失败（忽略，仍发文字）: {e}")
        if image_paths:
            print(f"[B站] 图片已下载 {len(image_paths)} 张")

        content = self._format_dynamic_push(dyn["text"])
        print(f"[B站] 检测到新动态 -> {content}")
        await self._push(bot, content, image_paths=image_paths)
        self.state["last_dynamic_id"] = dyn["id"]
        _save_state(self.state)

    def _format_dynamic_push(self, text: str) -> str:
        """动态推送模板：结构化通知（真实内容 + 个人空间链接）。

        灰泽满刚刚发了动态哦！

        动态内容：（真正的内容）

        https://space.bilibili.com/1298779265/dynamic
        """
        content = text.strip() or "（图片动态）"
        space_url = f"https://space.bilibili.com/{self.uid}/dynamic"
        return f"灰泽满刚刚发了动态哦！\n\n动态内容：{content}\n\n{space_url}"

    # ---- 私聊广播 ----

    async def _get_friends(self, bot) -> list:
        """每次推送都拉最新好友列表（推送本就罕见，确保新加的好友立即能收到）。

        旧版缓存 1 天，导致当天新加的好友收不到推送。
        """
        lst = await bot.get_friend_list()
        friends = [{"user_id": f["user_id"]} for f in lst if f.get("user_id")]
        ids = [f["user_id"] for f in friends]
        print(f"[B站] 当前好友 {len(ids)} 人: {ids[:20]}{'…' if len(ids) > 20 else ''}")
        self.state["friends"] = friends
        self.state["friends_ts"] = time.time()
        _save_state(self.state)
        return friends

    async def _push(self, bot, content: str, image_paths: list | None = None) -> None:
        """私聊广播给全部好友（白名单非空则只发白名单）。单好友失败不中断。

        image_paths 提供时，消息附带这些图片（动态配图 + 表情图）。
        """
        friends = await self._get_friends(bot)
        whitelist = get_notify_whitelist()
        targets = [f for f in friends
                   if not whitelist or str(f.get("user_id")) in whitelist]
        if not targets:
            print("[B站] 无推送目标（好友为空或白名单过滤后为空）")
            return
        ok = 0
        for t in targets:
            try:
                msg = Message(content)
                for p in (image_paths or []):
                    msg += MessageSegment.image(file=str(p))
                await bot.send_private_msg(user_id=t["user_id"], message=msg)
                ok += 1
            except Exception as e:
                print(f"⚠️ 推送失败 user={t.get('user_id')}: {e}")
        print(f"[B站] 已推送 {ok}/{len(targets)} 位好友")


# ==================== 启动注册 ====================

async def _monitor_loop() -> None:
    monitor = BiliMonitor()
    interval = get_push_interval()
    print(f"[B站] 监听启动: uid={monitor.uid}, 轮询间隔={interval}s（±随机抖动防风控）")
    while True:
        try:
            bot = get_bot()
            await monitor.poll_once(bot)
        except Exception as e:
            print(f"⚠️ B站监听轮询异常（继续）: {e}")
        # 随机抖动：固定间隔是典型机器人特征，B站风控会据此识别并顶会话
        await asyncio.sleep(max(interval + random.uniform(-20, 30), 30))


# NoneBot 未初始化时（离线工具/单测直接 import 本模块）跳过启动注册，
# 仅当作为插件被 NoneBot 加载后才挂后台任务。
try:
    @get_driver().on_startup
    async def _start_bili_monitor() -> None:
        if not get_bili_uid():
            print("⚠️ BILI_UID 未配置，跳过 B站直播/动态监听")
            return
        asyncio.create_task(_monitor_loop())
except ValueError:
    pass
