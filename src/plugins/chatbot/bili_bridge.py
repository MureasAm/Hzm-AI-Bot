"""B站直播/动态监听 + 私聊广播（方向3）。

启动时注册后台任务，周期性轮询灰泽满本人的 B站账号：
- 开播：状态翻转（未播→直播）才推送一次，带上直播间标题
- 动态：出现新动态 ID 才推送，结构化通知（真实内容 + 个人空间链接），含配图时连图一起发
推送目标：灰泽满 QQ 号的所有好友（私聊），NOTIFY_FRIENDS_WHITELIST 可收窄为白名单。
去重状态（last_live_status / last_dynamic_id / 好友缓存）持久化到 data/bili_state.json，
重启不重复推送。所有外部调用失败都降级为日志，绝不让监听任务崩溃。
"""
import asyncio
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

    async def poll_once(self, bot) -> None:
        if not self.uid:
            return
        if not self.state.get("_primed"):
            await self._prime()
        await self._check_live(bot)
        await self._check_dynamic(bot)

    async def _prime(self) -> None:
        """首次启动建基线：记录当前直播/动态状态，不推送已发生的事件。

        避免 bot 重启后把"启动前就开播/发的动态"当新事件广播给所有好友。
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
        self.state["_primed"] = True
        _save_state(self.state)
        print("[B站] 首次启动已建立基线，之后检测到新的开播/动态才会推送")

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
            await self._push(bot, content)
        self.state["last_live_status"] = is_live
        _save_state(self.state)

    # ---- 动态 ----

    async def _fetch_latest_dynamic(self) -> dict:
        """取最新一条动态，返回 {id, text, image_urls}。

        B站 2026 动态接口要求登录（SESSDATA），由 get_dynamic_page_info 处理 WBI/buvid。
        """
        from bilibili_api import dynamic  # 懒加载，避免无关场景的额外依赖开销
        from bilibili_api.utils.network import Credential

        cred = Credential(sessdata=self.sessdata)
        page = await dynamic.get_dynamic_page_info(cred, host_mid=int(self.uid), pn=1)
        items = (page or {}).get("items") or []
        if not items:
            raise RuntimeError("该 UID 暂无动态")
        item = items[0]
        return {
            "id": str(item.get("id_str") or item.get("id") or ""),
            "text": _extract_dynamic_text(item),
            "image_urls": _extract_dynamic_images(item),
        }

    async def _check_dynamic(self, bot) -> None:
        if not self.sessdata:
            if not self._warned_no_sessdata:
                print("⚠️ BILI_SESSDATA 未配置，动态监控关闭（开播监控正常）。配置后重启即生效。")
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

        # 动态含图时下载配图一起发（失败不阻塞，仍发文字）
        image_path = None
        if dyn.get("image_urls"):
            try:
                image_path = await _download_image(dyn["image_urls"][0])
                print(f"[B站] 动态配图已下载: {image_path.name}")
            except Exception as e:
                print(f"⚠️ 动态配图下载失败（忽略，仍发文字）: {e}")

        content = self._format_dynamic_push(dyn["text"])
        print(f"[B站] 检测到新动态 -> {content}")
        await self._push(bot, content, image_path=image_path)
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

    async def _push(self, bot, content: str, image_path: Path | None = None) -> None:
        """私聊广播给全部好友（白名单非空则只发白名单）。单好友失败不中断。

        image_path 提供时，消息附带该图片（动态配图）。
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
                if image_path:
                    msg += MessageSegment.image(file=str(image_path))
                await bot.send_private_msg(user_id=t["user_id"], message=msg)
                ok += 1
            except Exception as e:
                print(f"⚠️ 推送失败 user={t.get('user_id')}: {e}")
        print(f"[B站] 已推送 {ok}/{len(targets)} 位好友")


# ==================== 启动注册 ====================

async def _monitor_loop() -> None:
    monitor = BiliMonitor()
    interval = get_push_interval()
    print(f"[B站] 监听启动: uid={monitor.uid}, 轮询间隔={interval}s")
    while True:
        try:
            bot = get_bot()
            await monitor.poll_once(bot)
        except Exception as e:
            print(f"⚠️ B站监听轮询异常（继续）: {e}")
        await asyncio.sleep(interval)


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
