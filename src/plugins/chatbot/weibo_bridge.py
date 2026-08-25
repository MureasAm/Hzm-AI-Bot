"""微博动态监听 + 私聊广播（镜像 bili_bridge 的 B站动态模式）。

启动时注册后台任务，周期性轮询灰泽满本人的微博账号，出现新微博就私聊广播给好友：
- 需要 WEIBO_UID（微博 UID）+ WEIBO_COOKIE（浏览器登录后的完整 Cookie，无登录会被微博风控挡 403/432）
- 去重状态 last_post_id 持久化到 data/weibo_state.json，重启不重复推送
- 推送目标：灰泽满 QQ 号的所有好友（私聊），NOTIFY_FRIENDS_WHITELIST 可收窄为白名单
所有外部调用失败都降级为日志，绝不让监听任务崩溃。
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

from .constants import WEIBO_STATE_FILE
from .config import get_weibo_uid, get_weibo_cookie, get_notify_whitelist, get_push_interval
from .vision import _read_image_bytes, _normalize_image

WEIBO_API = "https://weibo.com/ajax/statuses/mymblog"

# 微博会拦默认 UA / 无 Cookie 的请求，必须带浏览器 UA + 登录 Cookie
_WEIBO_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


async def _download_image(url: str) -> Path:
    """下载微博配图并统一转 JPEG 到临时文件，返回路径；失败抛异常由调用方兜底。"""
    data = await _read_image_bytes(url)
    data = _normalize_image(data)  # WEBP/PNG 统一转 JPEG，QQ 显示更稳
    tmp = Path(tempfile.gettempdir()) / f"wb_dyn_{time.time_ns()}.jpg"
    tmp.write_bytes(data)
    return tmp


# ==================== 状态持久化 ====================

def _load_state() -> dict:
    if WEIBO_STATE_FILE.exists():
        try:
            return json.loads(WEIBO_STATE_FILE.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(state: dict) -> None:
    try:
        WEIBO_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        WEIBO_STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), "utf-8"
        )
    except OSError as e:
        print(f"⚠️ 微博状态写入失败: {e}")


# ==================== 字段提取 ====================

def _extract_post_text(item: dict) -> str:
    """从微博 item 提取纯文本（优先 raw_text，否则 text 去 HTML 标签）。"""
    text = item.get("raw_text", "") or item.get("text", "") or ""
    return re.sub(r"<[^>]+>", "", text).strip()


def _extract_post_images(item: dict) -> list:
    """从微博 item 提取配图 URL（优先大图），无图返回空列表。"""
    urls = []
    for pic in (item.get("pics") or []):
        u = (pic.get("large") or {}).get("url") or pic.get("url") or ""
        if u:
            urls.append(u)
    return urls


def _mid_to_bid(mid: str) -> str:
    """微博 mid（10 进制）→ bid（base36），用于拼微博正文链接。"""
    try:
        n = int(mid)
    except (ValueError, TypeError):
        return ""
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    s = ""
    while n:
        n, r = divmod(n, 36)
        s = chars[r] + s
    return s or "0"


# ==================== 监控主体 ====================

class WeiboMonitor:
    def __init__(self):
        self.uid = get_weibo_uid()
        self.state = _load_state()
        self._warned_no_cookie = False
        # 基线是否已建（进程内标志，不持久化——每次启动重新对齐，
        # 避免停机期间的新微博在重启后被当新事件推送）
        self._primed = False

    async def poll_once(self, bot) -> None:
        if not self.uid:
            return
        if not self._primed:
            await self._prime()
        await self._check_posts(bot)

    async def _prime(self) -> None:
        """每次启动建基线：记录当前最新微博 id，不推送停机期间已发的。"""
        try:
            post = await self._fetch_latest_post()
            self.state["last_post_id"] = post.get("id", "")
        except Exception as e:
            print(f"⚠️ 微博基线失败（忽略）: {e}")
        self._primed = True
        _save_state(self.state)
        print("[微博] 已建立基线（对齐当前微博状态），之后检测到新微博才会推送")

    async def _fetch_latest_post(self) -> dict:
        """取最新一条微博，返回 {id, mblogid, text, image_urls, url}。失败抛异常。"""
        cookie = get_weibo_cookie()
        async with httpx.AsyncClient(timeout=10.0,
                                     headers={"User-Agent": _WEIBO_UA,
                                              "Referer": f"https://weibo.com/u/{self.uid}"},
                                     follow_redirects=False) as client:
            headers = {"User-Agent": _WEIBO_UA, "Referer": f"https://weibo.com/u/{self.uid}"}
            if cookie:
                headers["Cookie"] = cookie
            resp = await client.get(WEIBO_API,
                                    params={"uid": self.uid, "page": 1, "feature": 0},
                                    headers=headers)
            data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"微博接口返回 ok={data.get('ok')} msg={data.get('msg')}")
        lst = (data.get("data") or {}).get("list") or []
        if not lst:
            raise RuntimeError("该 UID 暂无微博")
        item = lst[0]
        mid = str(item.get("id", "") or "")
        mblogid = str(item.get("mblogid", "") or "") or _mid_to_bid(mid)
        url = f"https://weibo.com/{self.uid}/{mblogid}" if mblogid else f"https://weibo.com/u/{self.uid}"
        return {
            "id": mid,
            "text": _extract_post_text(item),
            "image_urls": _extract_post_images(item),
            "url": url,
        }

    async def _check_posts(self, bot) -> None:
        if not get_weibo_cookie():
            if not self._warned_no_cookie:
                print("⚠️ WEIBO_COOKIE 未配置，微博监控关闭。配置后下次轮询即生效。")
                self._warned_no_cookie = True
            return
        try:
            post = await self._fetch_latest_post()
        except Exception as e:
            print(f"⚠️ 微博查询失败（忽略）: {e}")
            return
        if not post["id"]:
            return
        if post["id"] == str(self.state.get("last_post_id", "") or ""):
            return  # 已推送过

        # 含图时下载配图一起发（失败不阻塞，仍发文字）
        image_path = None
        if post.get("image_urls"):
            try:
                image_path = await _download_image(post["image_urls"][0])
                print(f"[微博] 配图已下载: {image_path.name}")
            except Exception as e:
                print(f"⚠️ 微博配图下载失败（忽略，仍发文字）: {e}")

        content = self._format_post_push(post["text"], post["url"])
        print(f"[微博] 检测到新微博 -> {content}")
        await self._push(bot, content, image_path=image_path)
        self.state["last_post_id"] = post["id"]
        _save_state(self.state)

    def _format_post_push(self, text: str, url: str) -> str:
        """微博推送模板：结构化通知（真实内容 + 微博链接）。"""
        content = text.strip() or "（图片微博）"
        return f"灰泽满刚刚发了微博哦！\n\n微博内容：{content}\n\n{url}"

    # ---- 私聊广播（与 B站共用逻辑） ----

    async def _get_friends(self, bot) -> list:
        """每次推送都拉最新好友列表（推送本就罕见，确保新加的好友立即能收到）。"""
        lst = await bot.get_friend_list()
        friends = [{"user_id": f["user_id"]} for f in lst if f.get("user_id")]
        ids = [f["user_id"] for f in friends]
        print(f"[微博] 当前好友 {len(ids)} 人: {ids[:20]}{'…' if len(ids) > 20 else ''}")
        self.state["friends"] = friends
        self.state["friends_ts"] = time.time()
        _save_state(self.state)
        return friends

    async def _push(self, bot, content: str, image_path: Path | None = None) -> None:
        """私聊广播给全部好友（白名单非空则只发白名单）。单好友失败不中断。"""
        friends = await self._get_friends(bot)
        whitelist = get_notify_whitelist()
        targets = [f for f in friends
                   if not whitelist or str(f.get("user_id")) in whitelist]
        if not targets:
            print("[微博] 无推送目标（好友为空或白名单过滤后为空）")
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
        print(f"[微博] 已推送 {ok}/{len(targets)} 位好友")


# ==================== 启动注册 ====================

async def _monitor_loop() -> None:
    monitor = WeiboMonitor()
    interval = get_push_interval()
    print(f"[微博] 监听启动: uid={monitor.uid}, 轮询间隔={interval}s（±随机抖动防风控）")
    while True:
        try:
            bot = get_bot()
            await monitor.poll_once(bot)
        except Exception as e:
            print(f"⚠️ 微博监听轮询异常（继续）: {e}")
        # 随机抖动：固定间隔是典型机器人特征，微博风控会据此识别
        await asyncio.sleep(max(interval + random.uniform(-20, 30), 30))


# NoneBot 未初始化时（离线工具/单测直接 import 本模块）跳过启动注册
try:
    @get_driver().on_startup
    async def _start_weibo_monitor() -> None:
        if not get_weibo_uid():
            print("⚠️ WEIBO_UID 未配置，跳过微博监听")
            return
        asyncio.create_task(_monitor_loop())
except ValueError:
    pass
