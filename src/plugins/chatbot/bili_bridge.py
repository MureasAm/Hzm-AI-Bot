"""B站直播/动态监听 + 私聊广播（方向3）。

启动时注册后台任务，周期性轮询灰泽满本人的 B站账号：
- 开播：状态翻转（未播→直播）才推送一次，带上直播间标题
- **下播**（2026-11-10）：直播中→未播 推「灰泽满宣布直播结束~ / 本次时长：X小时Y分钟」，
  用接口的 `live_time` 算，重启也能补回来
- 动态：出现新动态 ID 才推送。**能转成“她会主动说的那句话”时，只发那句（语音优先），
  不再推动态原文/链接**（2026-10-23 用户定）；转不出来/不适合时才发结构化原文通知。
- **开播自动动态不推**（2026-11-07）：开播会生成一条 `DYNAMIC_TYPE_LIVE_RCMD`，
  正文为空、只有直播卡片；直播状态由 `_check_live` 单独播报，这类只记水位线。
- **置顶周表**（2026-11-09）：置顶动态图集里的周表图换了（CDN 文件名哈希变）就推图；
  「本周周表已更新」公告也当一个触发信号，同一张图只推一次。
- 动态走**公共接口**（buvid3/4 + WBI 签名），**不需要 SESSDATA**（2026-10-31 实测）；
  该接口对匿名请求**间歇性返回 `code=0 + 空列表`**（2026-11-04 实测约一半请求为空），
  且 **WBI 签名键过期同样返回空列表**——两种空无法区分，所以空列表时 buvid 和 WBI
  一起换、多试几次（实测单次成功 ~1/3–1/2，试 10 次失败率约 1–2%）。SESSDATA 只影响
  emote panel 的完整度（没有时退回动态自带表情映射 + 本地 assets/emotes）。
推送目标：灰泽满 QQ 号的所有好友（私聊），NOTIFY_FRIENDS_WHITELIST 可收窄为白名单。
去重状态（last_live_status / last_dynamic_id / 好友缓存）持久化到 data/bili_state.json，
重启不重复推送。所有外部调用失败都降级为日志，绝不让监听任务崩溃。
"""
import asyncio
import hashlib
import random
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx
from PIL import Image

from nonebot import get_bot, get_driver

from .bg import spawn
from .constants import BILI_STATE_FILE
from .proactive import compose_proactive, broadcast_proactive
from . import _bridge_common
from .config import get_bili_uid, get_bili_sessdata, get_notify_whitelist, get_bili_interval

LIVE_STATUS_API = "https://api.live.bilibili.com/room/v1/Room/get_status_info_by_uids"

# B站 WAF 会拦 python-httpx 默认 UA，必须带浏览器 UA
_BILI_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# 动态公共接口：匿名身份（buvid3/4）+ WBI 签名即可，不需要 SESSDATA。
BILI_SPI_API = "https://api.bilibili.com/x/frontend/finger/spi"
BILI_NAV_API = "https://api.bilibili.com/x/web-interface/nav"
BILI_DYNAMIC_API = "https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/space"

# WBI 签名的 mixin 表（与 bilibili-api 的 utils/network.py 同一张表）
_WBI_MIXIN_TAB = [46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43,
                  5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16,
                  24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59,
                  6, 63, 57, 62, 11, 36, 20, 34, 44, 52]
_BUVID_TTL = 12 * 3600
_WBI_TTL = 6 * 3600
_buvid_cache: dict = {}
_buvid_at = 0.0
_wbi_cache: dict = {}
_wbi_at = 0.0

# QQ 无法把自定义表情内联进文字，只能当独立图片发。表情图压到这么小再发，尽量接近"小表情"观感。
_EMOJI_MAX_SIDE = 64


def _wbi_sign(params: dict, mixin_key: str) -> dict:
    """按 B站 WBI 规则签名：过滤特殊字符 → 按 key 排序 → md5(query + mixin)。"""
    signed = {k: "".join(ch for ch in str(v) if ch not in "!'()*")
              for k, v in params.items()}
    signed["wts"] = int(time.time())
    query = urlencode(sorted(signed.items()))
    signed["w_rid"] = hashlib.md5((query + mixin_key).encode("utf-8")).hexdigest()
    return signed


async def _get_buvid() -> dict:
    """获取并缓存匿名身份 buvid3/4（12 小时刷新一次）。"""
    global _buvid_cache, _buvid_at
    if _buvid_cache and time.time() - _buvid_at < _BUVID_TTL:
        return dict(_buvid_cache)
    async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": _BILI_UA}) as c:
        resp = await c.get(BILI_SPI_API)
        data = (resp.json() or {}).get("data") or {}
    b3, b4 = data.get("b_3"), data.get("b_4")
    if not b3 or not b4:
        raise RuntimeError("B站 buvid 获取失败（finger/spi 未返回 b_3/b_4）")
    _buvid_cache = {"buvid3": str(b3), "buvid4": str(b4),
                    "b_nut": str(int(time.time()))}
    _buvid_at = time.time()
    print(f"[B站] buvid 已刷新: {str(b3)[:8]}…")
    return dict(_buvid_cache)


async def _get_wbi_mixin_key(cookies: dict) -> str:
    """取 WBI 的 img/sub key 并拼出 32 位 mixin key（6 小时缓存）。"""
    global _wbi_cache, _wbi_at
    if _wbi_cache and time.time() - _wbi_at < _WBI_TTL:
        return _wbi_cache["mixin"]
    async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": _BILI_UA}) as c:
        resp = await c.get(BILI_NAV_API, cookies=cookies,
                           headers={"Referer": "https://www.bilibili.com/"})
        data = (resp.json() or {}).get("data") or {}
    wbi = data.get("wbi_img") or {}
    img = (wbi.get("img_url") or "").rsplit("/", 1)[-1].split(".")[0]
    sub = (wbi.get("sub_url") or "").rsplit("/", 1)[-1].split(".")[0]
    if not img or not sub:
        raise RuntimeError("B站 WBI key 获取失败（nav 未返回 wbi_img）")
    mixin = "".join((img + sub)[i] for i in _WBI_MIXIN_TAB)[:32]
    _wbi_cache = {"mixin": mixin}
    _wbi_at = time.time()
    return mixin


def _invalidate_bili_caches() -> None:
    """签名/身份失效时强制下次重新拉取。"""
    global _buvid_at, _wbi_at
    _buvid_at = 0.0
    _wbi_at = 0.0


def _invalidate_buvid() -> None:
    """只换 buvid；WBI key 是全局的，不需要跟着换。"""
    global _buvid_at
    _buvid_at = 0.0


# SESSDATA 失效（-101）只提示一次——每轮轮询都打印一遍只会刷屏，
# 而且公共接口本来就能用，这不是致命故障。
_sessdata_warned = False


def _warn_sessdata_expired() -> None:
    global _sessdata_warned
    if _sessdata_warned:
        return
    _sessdata_warned = True
    print("⚠️ BILI_SESSDATA 已失效（-101 账号未登录）：公共接口仍可监控动态，"
          "登录态兜底/完整表情表不可用。需要时请更新 .env.prod 的 BILI_SESSDATA 后重启")


async def _fetch_dynamic_items_via_login(uid: str) -> list:
    """登录态兜底：公共接口被风控时用现有 SESSDATA 走 bilibili-api。"""
    from bilibili_api import dynamic
    from bilibili_api.utils.network import Credential
    page = await dynamic.get_dynamic_page_info(
        Credential(sessdata=get_bili_sessdata()), host_mid=int(uid), pn=1)
    return (page or {}).get("items") or []


# 同一轮轮询里「最新动态」和「置顶动态」共用一次接口结果：这个接口本身就有
# 约一半概率返回空列表、要靠 buvid/WBI 轮换重试（见 `_fetch_dynamic_items`），
# 每轮多打一次就多一轮重试，还可能把轮询周期拖长。
_ITEMS_CACHE: dict[str, tuple[float, list]] = {}
_ITEMS_CACHE_TTL = 120.0


def _cache_dynamic_items(uid: str, items: list) -> None:
    _ITEMS_CACHE[str(uid)] = (time.time(), items)


def _cached_dynamic_items(uid: str) -> list | None:
    hit = _ITEMS_CACHE.get(str(uid))
    if hit and time.time() - hit[0] < _ITEMS_CACHE_TTL:
        return hit[1]
    return None


async def _fetch_dynamic_items(uid: str, attempts: int = 10) -> list:
    """拉取指定 UID 的动态列表（公共接口，无需登录）。

    ⚠️ 该接口会间歇返回 `code=0 + items=[]`：**匿名请求本身就有约一半概率被
    降级成空列表**（2026-11-04 实测，换 buvid/换参数都一样，同一 buvid 连发
    8 次空 4 次），而且 **WBI 签名键过期时接口同样回 `code=0 + 空列表`、不报
    -352**（实测用错误 mixin 签名得到完全一样的空响应）——两种空**无法区分**。
    所以空列表时 **buvid 和 WBI 键一起换**，并多试几次（单次成功 ~1/3–1/2，试 10 次
    失败率约 1–2%）。只换 buvid 的旧逻辑遇到签名轮换会**永远空下去**（2026-11-04
    实际翻车：连续 4 次空列表 + 登录兜底 -101，监控整晚没动静）。
    **不要传 offset=0**——那会稳定返回空。
    公共接口重试仍失败、且配置了 SESSDATA 时，用登录态兜底（老路径）。
    """
    uid = str(uid)
    cookies = await _get_buvid()
    mixin = await _get_wbi_mixin_key(cookies)
    headers = {"User-Agent": _BILI_UA,
               "Referer": f"https://space.bilibili.com/{uid}/dynamic",
               "Accept": "application/json, text/plain, */*"}
    last_err = ""
    async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
        for i in range(attempts):
            # features=itemOpusStyle 必须带：否则 /feed/space 返回旧 DRAW 结构，
            # 正文（major.opus.summary）整个缺失，只剩配图。
            params = _wbi_sign({"host_mid": uid, "timezone_offset": -480,
                                "features": "itemOpusStyle", "page": 1}, mixin)
            resp = await client.get(BILI_DYNAMIC_API, params=params, cookies=cookies)
            need_new_buvid = False
            need_new_wbi = False
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}"
                need_new_buvid = need_new_wbi = resp.status_code in (412, 403)
            else:
                try:
                    data = resp.json() or {}
                except ValueError:
                    data = {}
                if not data:
                    last_err = "响应不是可解析的 JSON（可能被 WAF 拦截）"
                    need_new_buvid = need_new_wbi = True
                elif data.get("code") != 0:
                    last_err = f"code={data.get('code')} {data.get('message', '')}"
                    need_new_buvid = need_new_wbi = data.get("code") in (-352, -403, -412)
                else:
                    items = (data.get("data") or {}).get("items") or []
                    if items:
                        _cache_dynamic_items(uid, items)
                        return items
                    last_err = "空列表"
                    need_new_buvid = need_new_wbi = True
            if i < attempts - 1:
                if need_new_wbi:
                    _invalidate_bili_caches()
                    cookies = await _get_buvid()
                    mixin = await _get_wbi_mixin_key(cookies)
                elif need_new_buvid:
                    _invalidate_buvid()
                    cookies = await _get_buvid()
                # 递增退避，别把重试打成高频请求
                await asyncio.sleep(0.4 + 0.3 * min(i, 4))
    # 公共接口被风控时用登录态兜底（SESSDATA 失效会自然抛错，被上层忽略）
    if get_bili_sessdata():
        try:
            items = await _fetch_dynamic_items_via_login(uid)
            if items:
                print("[B站] 公共动态接口未拿到数据，已用 SESSDATA 登录态兜底")
                _cache_dynamic_items(uid, items)
                return items
        except Exception as e:
            msg = str(e)
            if "-101" in msg or "账号未登录" in msg:
                _warn_sessdata_expired()
            else:
                print(f"⚠️ B站动态登录态兜底失败: {msg[:120]}")
    raise RuntimeError(f"B站动态接口连续 {attempts} 次未拿到数据（{last_err}）")


def _is_live_notice(item: dict) -> bool:
    """这条动态是不是开播自动生成的直播通知（不是她手动发的动态）。

    实测（2026-11-07）开播会生成 `type=DYNAMIC_TYPE_LIVE_RCMD` /
    `major.type=MAJOR_TYPE_LIVE_RCMD` 的动态，正文为空、只有直播卡片。
    它会被当成“最新动态”抓进来，推送成“灰泽满刚刚发了动态哦！（图片动态）”。
    直播状态已经由 `_check_live` 单独播报，这类动态一律不推。
    """
    if str(item.get("type") or "") in {"DYNAMIC_TYPE_LIVE_RCMD", "DYNAMIC_TYPE_LIVE"}:
        return True
    md = (item.get("modules") or {}).get("module_dynamic") or {}
    major = md.get("major") or {}
    if str(major.get("type") or "") in {"MAJOR_TYPE_LIVE_RCMD", "MAJOR_TYPE_LIVE"}:
        return True
    return bool(major.get("live_rcmd") or major.get("live"))


async def fetch_latest_dynamic(uid: str, items: list | None = None) -> dict:
    """取最新一条动态（含解析），返回 {id, text, image_urls, ...}。

    ⚠️ 公共接口会把**置顶动态**排在 `items[0]`（比最新动态旧得多），
    不能直接取第一条；按 id 取最大的一条才是真正的最新。
    """
    if items is None:
        items = await _fetch_dynamic_items(uid)
    item = max(items, key=lambda it: int(it.get("id_str") or it.get("id") or 0))
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
    clean_text, emote_urls, local_emotes = _parse_emotes(text, emote_map)
    return {
        "id": str(item.get("id_str") or item.get("id") or ""),
        "text": clean_text,
        "image_urls": _extract_dynamic_images(item),
        "emote_urls": emote_urls,
        "local_emotes": local_emotes,
        # 转发别人的动态：顶层 `orig` 有东西就是转发（接口层取不全原文，所以整条不推）
        "is_repost": isinstance(item.get("orig"), dict),
        # 开播自动通知：只用于水位线，不推送（直播状态另有 _check_live）
        "is_live_notice": _is_live_notice(item),
    }


async def _download_image(url: str) -> Path:
    return await _bridge_common.download_image_to_tmp(url, "hzm_dyn")


# ==================== 置顶周表 ====================
# 灰泽满每周在**置顶动态的图集**里换一张周表（2026-11-09 实测：置顶项
# `module_tag.text == "置顶"`，周表是 `major.opus.pics[3]`，882×1080，
# 内容是 `SCHEDULE 11月9日-11月15日`）。换图 = 重新上传，CDN 文件名里的内容哈希会变，
# 所以按文件名哈希比对，比“看发布时间”可靠（置顶帖是编辑更新，pub_ts 一直是首次发布）。
_PINNED_TAG = "置顶"
_DEFAULT_WEEK_PIC_INDEX = 3   # 实测槽位；布局变了会用视觉重新找
_WEEK_PIC_MAX_TRIES = 6       # 视觉识别最多看几张，防止一次扫 12 张拖住轮询


def _pic_hash(url: str) -> str:
    """图片内容指纹：取 CDN 文件名（去掉扩展名），忽略域名/协议差异。

    同一个文件换 i0/i1 域名不该算“换了图”；重新上传会换文件名，那才是真变了。
    """
    name = str(url or "").split("?")[0].rstrip("/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0]


def _find_pinned_item(items: list) -> dict | None:
    """在动态列表里找置顶项。接口把置顶放第一个，但标记才是权威——
    没有置顶标记就不认，避免把普通的第一条当周表。"""
    for it in items or []:
        tag = ((it.get("modules") or {}).get("module_tag") or {}).get("text") or ""
        if str(tag).strip() == _PINNED_TAG:
            return it
    return None


async def fetch_pinned_dynamic(uid: str, items: list | None = None,
                               use_network: bool = True) -> dict | None:
    """取置顶动态（{id, text, image_urls, pic_hashes}）；没有置顶返回 None。

    `use_network=False` 时只读本轮 `_fetch_dynamic_items` 的缓存——监控循环每轮
    已经拉过一次动态列表，不要为了置顶再打一次容易空列表的接口。
    """
    if items is None:
        items = _cached_dynamic_items(uid)
        if items is None:
            if not use_network:
                return None
            items = await _fetch_dynamic_items(uid)
    item = _find_pinned_item(items)
    if not item:
        return None
    urls = _extract_dynamic_images(item)
    return {
        "id": str(item.get("id_str") or item.get("id") or ""),
        "text": _extract_dynamic_text(item),
        "image_urls": urls,
        "pic_hashes": [_pic_hash(u) for u in urls],
    }


# 「本周周表已更新」这类公告。图片可能先换、公告可能晚到，两个信号都要认。
_WEEK_TABLE_ANNOUNCE_RE = re.compile(
    r"(周表|时间表|日程表|直播安排).{0,16}(更新|更了|已更|换了|换新|上新)"
    r"|(更新|更了|已更|换了|换新|上新).{0,16}(周表|时间表|日程表|直播安排)"
)


def is_week_table_announcement(text: str) -> bool:
    return bool(_WEEK_TABLE_ANNOUNCE_RE.search(text or ""))


_WEEK_TABLE_MARKERS = ("SCHEDULE", "周表", "日程表", "时间表", "时刻表", "直播安排")
_WEEK_DAY_MARKERS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
                     "周一", "周二", "周三", "周四", "周五", "周六", "周日")


def _desc_looks_like_week_table(desc: str) -> bool:
    """视觉描述里有没有周表特征（画面文字会被原样写出来）。"""
    t = (desc or "").upper()
    if any(m.upper() in t for m in _WEEK_TABLE_MARKERS):
        return True
    return sum(1 for m in _WEEK_DAY_MARKERS if m.upper() in t) >= 3


async def _image_is_week_table(url: str) -> bool | None:
    """让视觉模型看一眼是不是周表；失败返回 None（调用方按“不确定”处理）。"""
    try:
        from .vision import describe_image
        from .config import get_vision_client
        desc = await asyncio.wait_for(
            describe_image(get_vision_client(), url), timeout=15.0)
    except Exception as e:
        print(f"⚠️ 周表视觉识别失败（跳过）: {type(e).__name__}")
        return None
    if not desc:
        return None
    return _desc_looks_like_week_table(desc)


def _resize_emote_if_large(path: Path, max_side: int = 200) -> Path:
    """若表情图边长超过 max_side，缩小到 max_side（保持比例）写临时文件，返回新路径。

    防止高清大图在 QQ 里显得过大；正常尺寸的表情原样返回（不折腾）。"""
    try:
        im = Image.open(path)
        w, h = im.size
        if max(w, h) <= max_side:
            return path
        ratio = max_side / max(w, h)
        im = im.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS)
        tmp = Path(tempfile.gettempdir()) / f"hzm_emote_{time.time_ns()}.png"
        im.save(tmp)
        print(f"[B站] 表情过大({w}x{h})已缩到 {im.size}")
        return tmp
    except Exception:
        return path


# ==================== 状态持久化 ====================

def _load_state() -> dict:
    return _bridge_common.load_state_file(BILI_STATE_FILE)
def _save_state(state: dict) -> None:
    _bridge_common.save_state_file(BILI_STATE_FILE, state)
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


def _format_live_duration(seconds) -> str:
    """直播时长说成人话：2小时35分钟 / 35分钟 / 不到1分钟；算不出来给「未知」。"""
    try:
        sec = int(float(seconds))
    except (TypeError, ValueError):
        return "未知"
    if sec <= 0:
        return "未知"
    if sec < 60:
        return "不到1分钟"
    hours, minutes = divmod(sec // 60, 60)
    if hours and minutes:
        return f"{hours}小时{minutes}分钟"
    if hours:
        return f"{hours}小时"
    return f"{minutes}分钟"


def _live_end_message(started_at=None, now=None) -> str:
    """下播播报（用户 2026-11-10 定）：

    灰泽满宣布直播结束~
    本次时长：2小时35分钟
    """
    duration = None
    if started_at:
        try:
            duration = float(now if now is not None else time.time()) - float(started_at)
        except (TypeError, ValueError):
            duration = None
    return f"灰泽满宣布直播结束~\n本次时长：{_format_live_duration(duration)}"


# 转发动态里，被转发原文最多带几个字进正文——
# 目的是让通知模板/主动发言"知道在说什么"，不是把整篇公告搬进来。
_FORWARD_TEXT_MAX = 150


def _own_text_of(item: dict) -> str:
    """只取**这条动态自己写的那部分**（不含转发内容）。

    两种格式：旧 `module_dynamic.desc.text`；新 OPUS `major.opus.summary.text`。
    """
    md = (item.get("modules") or {}).get("module_dynamic") or {}
    text = (md.get("desc") or {}).get("text", "") or ""
    if not text:
        opus = ((md.get("major") or {}).get("opus") or {})
        summary = opus.get("summary")
        if isinstance(summary, dict):
            text = summary.get("text", "") or ""
        elif isinstance(summary, str):
            text = summary
    return (text or "").strip()


def _extract_dynamic_text(item: dict) -> str:
    """从 bilibili-api 的动态 item 里提取正文（防御式，兼容多种类型）。

    ⚠️ **转发动态的被转发原文在 `item["orig"]`（顶层），不在 `module_dynamic["orig"]`。**
    旧代码查的是后者 → 永远取不到 → 转发类的动态只剩自己的半句话。
    实测踩坑（2026-09-25）：她转发一条游戏公告只写了"晚上九点～"，
    于是转出来的主动发言就真的只有"晚上九点"，跟打卡一样。
    （顶层 `orig` 还可能是**嵌套转发**，所以递归取。）

    转发时把两段拼起来：`{她写的话}（转发自：{原文前 N 字}）`——
    这样通知模板能显示原文、主动发言也有内容可讲。
    """
    text = _own_text_of(item)
    orig = item.get("orig")
    if not isinstance(orig, dict):
        return text
    inner = _extract_dynamic_text(orig)          # 递归：嵌套转发
    inner = inner.strip()[:_FORWARD_TEXT_MAX]
    if not inner:
        return text
    return f"{text}（转发自：{inner}）" if text else f"（转发自：{inner}）"


# ==================== 表情解析 ====================
# B站动态文本里 [xxx] 是表情占位符（含 [UPOWER_uid_名字] 这种主播定制贴纸）。
# 解析优先级：本地表情库（付费/UPOWER 定制，手动放 assets/emotes/）> emote panel > 剥除。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EMOTE_DIR = PROJECT_ROOT / "assets" / "emotes"
_emote_map = None


async def _get_emote_map() -> dict:
    """拉取 B站 emote panel 并缓存 text→url 映射（带 SESSDATA 才看得到完整表情）。"""
    global _emote_map
    if _emote_map is not None:
        return _emote_map
    try:
        url = "https://api.bilibili.com/x/emote/user/panel/web"
        params = {"business": "dynamic", "build": "0",
                  "mobi_app": "web", "uid": get_bili_uid()}
        sessdata = get_bili_sessdata()
        headers = {"User-Agent": _BILI_UA, "Referer": "https://t.bilibili.com/"}
        if sessdata:
            headers["Cookie"] = f"SESSDATA={sessdata}"
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as c:
            data = (await c.get(url, params=params)).json() or {}
        # SESSDATA 失效（-101）时退回匿名请求：匿名仍能拿到基础表情包。
        # 别把过期 cookie 换来的空表当成"表情表就这样"永久缓存。
        if sessdata and data.get("code") not in (0, None):
            headers.pop("Cookie", None)
            async with httpx.AsyncClient(timeout=10.0, headers=headers) as c:
                data = (await c.get(url, params=params)).json() or {}
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


def _parse_emotes(text: str, emote_map: dict) -> tuple[str, list, list]:
    """把 [xxx] 表情解析成图片；返回 (清洗后文本, 需下载的URL列表, 本地表情文件列表)。

    优先级：① 本地表情库（付费/UPOWER 定制贴纸，文件名=占位符名，放 assets/emotes/）
           ② emote panel / 动态自带映射
           ③ 剥除（避免 [UPOWER_...] 丑代码显示给 QQ 好友）
    """
    if not text:
        return text, [], []
    urls, locals_paths = [], []

    def repl(m):
        token = m.group(0)
        name = token.strip("[]")
        # ① 本地表情库
        for ext in (".png", ".gif", ".jpg", ".jpeg", ".webp"):
            p = EMOTE_DIR / f"{name}{ext}"
            if p.exists():
                locals_paths.append(p)
                return ""
        # ② emote panel / 动态自带
        u = emote_map.get(token) or ""
        if u:
            urls.append(u)
        return ""  # ③ 占位符一律移除（图单独发）

    cleaned = re.sub(r"\[[^\]]{1,30}\]", repl, text)
    return cleaned.strip(), urls, locals_paths


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
        self.state = _load_state()
        # 基线是否已建（进程内标志，不持久化——每次启动都重新对齐，
        # 避免停机期间的动态/开播在重启后被当新事件推送）
        # 直播/动态分开记：一个接口失败不该拖住另一个的实时监控。
        self._live_primed = False
        self._dynamic_primed = False

    @property
    def _primed(self) -> bool:
        """兼容旧测试/调用：两个基线都建好才算整体 primed。"""
        return self._live_primed and self._dynamic_primed

    @_primed.setter
    def _primed(self, value: bool) -> None:
        self._live_primed = self._dynamic_primed = bool(value)

    async def poll_once(self, bot) -> None:
        if not self.uid:
            return
        if not self._primed:
            await self._prime()
        if self._live_primed:
            await self._check_live(bot)
        if self._dynamic_primed:
            await self._check_dynamic(bot)

    async def _prime(self) -> None:
        """每次启动建基线：记录当前直播/动态状态，不推送"停机期间已发生"的事件。

        基线是进程内一次性的（`_live_primed` / `_dynamic_primed`），每次 bot 启动都重新对齐。
        **某一项拉取失败时不能把它标记为已建**：否则下一轮成功请求会把停机期间的事件
        当新事件补推（2026-11-01 微博「快递到了」事故的同型 bug）。失败的部分下轮重试。
        """
        if not self._live_primed:
            try:
                info = await self._fetch_live_status()
                is_live = info["live_status"] == 1
                self.state["last_live_status"] = is_live
                if is_live and info.get("live_time"):
                    # 重启时她可能已经在播：把本次开播时间补回来，下播才能算时长
                    self.state["live_started_at"] = info["live_time"]
                else:
                    self.state.pop("live_started_at", None)
                self._live_primed = True
            except Exception as e:
                print(f"⚠️ 基线-直播状态失败（下轮重试，本轮不检查）: {e}")
        if not self._dynamic_primed:
            try:
                dyn = await self._fetch_latest_dynamic()
                cur = dyn.get("id", "")
                # 水位线只升不降：停机期间她若撤回/置顶，别把基线调低
                if _bridge_common.is_newer_id(cur, self.state.get("last_dynamic_id", "")):
                    self.state["last_dynamic_id"] = cur
                self._dynamic_primed = True
            except Exception as e:
                print(f"⚠️ 基线-动态失败（下轮重试，本轮不检查）: {e}")
        _save_state(self.state)
        if self._primed:
            print("[B站] 已建立基线（对齐当前直播/动态状态），之后检测到新的开播/动态才会推送")
        else:
            print("[B站] 基线未完整建立：完成的部分正常监控，失败的部分下轮重试，不会补推停机事件")

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
            # 本次开播时间（epoch 秒）：下播时用它算时长；重启后也能从接口补回来
            "live_time": int(info.get("live_time", 0) or 0),
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
        live_time = info.get("live_time") or 0
        if is_live and live_time > 0:
            # 接口每次都会给本次开播时间；以它为准，重启/漏轮也能算对时长
            self.state["live_started_at"] = live_time
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
        elif (not is_live) and was_live:
            # 下播通知（用户 2026-11-10 定）：已结束 + 本次时长
            started_at = self.state.get("live_started_at") or live_time
            content = _live_end_message(started_at)
            print(f"[B站] 检测到下播 -> {content}")
            await self._push(bot, content)
            self.state.pop("live_started_at", None)
        self.state["last_live_status"] = is_live
        _save_state(self.state)

    # ---- 动态 ----

    async def _fetch_latest_dynamic(self) -> dict:
        """取最新一条动态（公共接口，无需 SESSDATA）。"""
        return await fetch_latest_dynamic(self.uid)

    async def _check_dynamic(self, bot) -> None:
        try:
            dyn = await self._fetch_latest_dynamic()
        except Exception as e:
            print(f"⚠️ B站动态查询失败（忽略）: {e}")
            return
        if not dyn["id"]:
            return
        is_new = _bridge_common.is_newer_id(
            dyn["id"], self.state.get("last_dynamic_id", ""))
        # 周表公告 = 一条新动态，正文说「本周周表已更新」。图片可能先换、公告晚到，
        # 两个信号都要认；公告能接住时就不再推一条光有文字的动态。
        is_week_announce = bool(is_new and is_week_table_announcement(dyn.get("text", "")))
        week_handled = False
        try:
            week_handled = await self._check_week_table(
                bot, announcement=dyn if is_week_announce else None)
        except Exception as e:
            print(f"⚠️ 周表检查失败（忽略）: {e}")
        if not is_new:
            # 不是"更新"的动态：已推过的、或比她已发过的更旧（她撤回/置顶后
            # 接口的"最新"会退回旧动态，只判"和上一条不同"会把旧的又推一遍）
            return

        # **转发别人的动态：整条不推**（用户 2026-09-29 决定）。
        # 原因：原动态的标题/封面在接口层取不全（转一条视频时正文只剩她自己那半句），
        # 推出来粉丝看不懂；转出来的主动发言也容易张冠李戴。
        # ⚠️ 仍要**更新水位线**，否则每轮轮询都会重新"发现"它、日志刷屏。
        if dyn.get("is_repost"):
            print("[B站] 这条是转发动态 → 不推送（已记水位线，避免每轮重复判定）")
            self.state["last_dynamic_id"] = dyn["id"]
            _save_state(self.state)
            return

        # **开播自动通知：整条不推**（2026-11-07）。
        # 开播时 B站会生成一条 DYNAMIC_TYPE_LIVE_RCMD，正文为空、只有直播卡片；
        # 它会被当成“最新动态”推送成“刚刚发了动态哦！（图片动态）”。
        # 直播状态已经由 _check_live 单独播报；这里只记水位线，避免每轮重复判定。
        if dyn.get("is_live_notice"):
            print("[B站] 这条是开播自动动态 → 不推送（已记水位线）")
            self.state["last_dynamic_id"] = dyn["id"]
            _save_state(self.state)
            return

        # 周表公告已经被「置顶周表图」那条推送接住 → 只记水位线，别再推一遍文字。
        if is_week_announce and week_handled:
            print("[B站] 周表公告已由置顶周表图接管，不再推文字动态")
            self.state["last_dynamic_id"] = dyn["id"]
            _save_state(self.state)
            return

        # B站表情在 QQ 无法内联变小（发出去就是一张独立大图），所以取舍：
        # 动态有真配图(照片) → 只发真配图(大)，跳过 emoji 装饰图，免得挤成一排大表情；
        # 纯 emoji 动态(无真配图) → 表情图压小(≤_EMOJI_MAX_SIDE)一起发，接近小贴纸观感。
        MAX_IMG = 8
        photos = (dyn.get("image_urls") or [])[:MAX_IMG]
        image_paths = []
        if photos:
            for u in photos:
                try:
                    image_paths.append(await _download_image(u))
                except Exception as e:
                    print(f"⚠️ 配图下载失败（忽略，仍发文字）: {e}")
            if image_paths:
                print(f"[B站] 图文动态：配图 {len(image_paths)} 张（emoji 表情跳过——QQ 无法内联小图）")
            elif photos:
                print(f"[B站] 本有 {len(photos)} 张配图，但全部下载失败 → 本次只发文字")
        else:
            for p in (dyn.get("local_emotes") or [])[:MAX_IMG]:
                image_paths.append(_resize_emote_if_large(p, _EMOJI_MAX_SIDE))
            for u in (dyn.get("emote_urls") or [])[:MAX_IMG - len(image_paths)]:
                try:
                    p = await _download_image(u)
                    image_paths.append(_resize_emote_if_large(p, _EMOJI_MAX_SIDE))
                except Exception as e:
                    print(f"⚠️ 表情图下载失败（忽略，仍发文字）: {e}")
            if image_paths:
                print(f"[B站] 纯表情动态：表情 {len(image_paths)} 张（已压到 ≤{_EMOJI_MAX_SIDE}px）")

        # **2026-10-23 用户改定**：主动发言一旦生成，就**只发她那句语音**，
        # 不再推动态原文/链接（与旧版“原文 + 她那句”相反；旧理由是她转写可能漏细节，
        # 用户现在接受这个取舍：有她那句就不看原文）。
        # 她不发/不适合发时，才退回结构化原文通知（信息不丢）。
        notice = self._format_dynamic_push(dyn["text"])
        # 配图传给她看：内容里的"⬇️/这个/图片里那个"指的就是它。
        # **只传真配图**（photos）——纯表情动态那一堆 emoji 图不是"配图"，
        # 传过去只会让视觉模型描述一个小表情，反而误导。
        said = await compose_proactive(dyn["text"], "B站",
                                       image_paths=image_paths if photos else None)
        if said:
            print(f"[B站] 检测到新动态 -> 主动发言（语音优先，不推原文）：{said}")
            await self._push(bot, said, image_paths=image_paths, voice=True)
        else:
            print("[B站] 检测到新动态 -> 主动发言未生成，只发原文")
            await self._push(bot, notice, image_paths=image_paths, record=True)
        self.state["last_dynamic_id"] = dyn["id"]
        _save_state(self.state)

    # ---- 置顶周表 ----

    async def _discover_week_index(self, urls: list, candidates: list) -> int | None:
        """视觉识别哪一张是周表：先看候选槽位，再看其它图。

        只在“变了但不是已知槽位”或“第一次遇到这张置顶帖”时才调用——正常每周换图
        走已知槽位，不花这次视觉调用。最多看 `_WEEK_PIC_MAX_TRIES` 张。
        """
        order = []
        for i in list(candidates or []) + list(range(len(urls))):
            if 0 <= i < len(urls) and i not in order:
                order.append(i)
        for i in order[:_WEEK_PIC_MAX_TRIES]:
            ok = await _image_is_week_table(urls[i])
            if ok:
                return i
        return None

    async def _push_week_table(self, bot, url: str, pic_hash: str) -> bool:
        """下载并推送周表图；同一个图只推一次（哈希去重）。返回是否已接手这条公告。"""
        if pic_hash and pic_hash == self.state.get("last_pushed_week_pic"):
            return True
        try:
            path = await _download_image(url)
        except Exception as e:
            print(f"⚠️ 周表图下载失败（本轮不推）: {e}")
            return False
        said = None
        try:
            said = await compose_proactive("本周周表更新了", "B站",
                                           image_paths=[path])
        except Exception as e:
            print(f"⚠️ 周表主动发言生成失败（只发通知）: {type(e).__name__}")
        notice = ("灰泽满的周表更新了！\n\n本周直播安排都在这张图里：\n"
                  f"https://space.bilibili.com/{self.uid}/dynamic")
        await self._push(bot, said or notice, image_paths=[path],
                         voice=bool(said), record=True)
        self.state["last_pushed_week_pic"] = pic_hash
        print(f"[B站] 周表已更新 -> 推送（{pic_hash[:10]}…）")
        return True

    async def _check_week_table(self, bot, announcement: dict | None = None) -> bool:
        """检查置顶动态里的周表图有没有换；返回 True = 这条公告已由周表推送接住。

        - 每周换图：置顶帖 id 不变、图集文件名哈希变 → 推变化的那张。
        - 公告先到：正文出现「周表已更新」→ 推当前已知槽位的那张（哈希去重）。
        - 别的图变了（月度大航海礼物/立绘）→ 不动，除非公告同时到。
        - 第一次见到置顶帖：只建基线，不推（避免每次部署都把本周周表当“新”）。
        """
        pinned = await fetch_pinned_dynamic(self.uid, use_network=False)
        if not pinned or not pinned.get("image_urls"):
            return False
        urls = pinned["image_urls"]
        hashes = pinned["pic_hashes"]

        old_id = self.state.get("pinned_id")
        old_hashes = list(self.state.get("pinned_pic_hashes") or [])
        pinned_replaced = bool(old_id and old_id != pinned["id"])
        first_seen = not old_hashes or pinned_replaced

        if first_seen:
            self.state["pinned_id"] = pinned["id"]
            self.state["pinned_pic_hashes"] = hashes
            if pinned_replaced or not isinstance(self.state.get("week_pic_index"), int):
                self.state["week_pic_index"] = (
                    _DEFAULT_WEEK_PIC_INDEX if len(urls) > _DEFAULT_WEEK_PIC_INDEX else 0)
            if not announcement:
                _save_state(self.state)
                print(f"[B站] 已记录置顶图集基线（{len(urls)} 张，"
                      f"周表槽位第 {self.state['week_pic_index'] + 1} 张）")
                return False
        else:
            old_set = set(old_hashes)
            changed = [i for i, h in enumerate(hashes) if h not in old_set]
            idx_state = self.state.get("week_pic_index")
            candidate = None
            if changed:
                if isinstance(idx_state, int) and changed == [idx_state]:
                    candidate = idx_state
                elif announcement or len(changed) > 1:
                    # 槽位可能因为插入/删除/重排而位移：让视觉重新找。
                    # 单张且不是已知槽位（月度礼物/立绘）不找——那不是周表。
                    candidate = await self._discover_week_index(urls, changed)
                    if candidate is None and isinstance(idx_state, int) and idx_state < len(urls):
                        candidate = idx_state
                else:
                    print(f"[B站] 置顶图集有变化但不是周表槽位"
                          f"（第 {[i + 1 for i in changed]} 张），忽略")
            elif announcement:
                if isinstance(idx_state, int) and 0 <= idx_state < len(urls):
                    candidate = idx_state
                else:
                    candidate = await self._discover_week_index(urls, [])
            if candidate is None or candidate >= len(urls):
                # 只在真有变化时才落盘，别每 3 分钟写一次状态文件
                if changed or pinned_replaced:
                    self.state["pinned_pic_hashes"] = hashes
                    self.state["pinned_id"] = pinned["id"]
                    _save_state(self.state)
                return False
            self.state["week_pic_index"] = candidate

        idx = self.state.get("week_pic_index") or 0
        if idx >= len(urls):
            return False
        pushed = await self._push_week_table(bot, urls[idx], hashes[idx])
        self.state["pinned_id"] = pinned["id"]
        self.state["pinned_pic_hashes"] = hashes
        _save_state(self.state)
        return pushed

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

    async def _push(self, bot, content: str, image_paths: list | None = None,
                    record: bool = True, voice: bool = False) -> None:
        """私聊广播给全部好友（白名单非空则只发白名单）。

        `voice=True`：TTS **只合成一次**，再按并发上限发给所有人；单好友失败不中断。
        记忆写入也在 `broadcast_proactive` 里按发送成功逐人做。
        """
        friends = await self._get_friends(bot)
        whitelist = get_notify_whitelist()
        targets = [f for f in friends
                   if not whitelist or str(f.get("user_id")) in whitelist]
        if not targets:
            print("[B站] 无推送目标（好友为空或白名单过滤后为空）")
            return
        stats = await broadcast_proactive(
            bot, [t["user_id"] for t in targets], content,
            image_paths=image_paths, voice=voice, record=record)
        mode = "语音/文字" if voice else "文字"
        print(f"[B站] 已推送 {stats['voice'] + stats['text']}/{len(targets)} 位好友（{mode}）")


# ==================== 启动注册 ====================

async def _monitor_loop() -> None:
    monitor = BiliMonitor()
    interval = get_bili_interval()
    print(f"[B站] 监听启动: uid={monitor.uid}, 轮询间隔={interval}s（±随机抖动防风控）")
    while True:
        try:
            bot = get_bot()
            await monitor.poll_once(bot)
        except Exception as e:
            print(f"⚠️ B站监听轮询异常（继续）: {e}")
        # 随机抖动：固定间隔是典型机器人特征；B站用独立的 BILI_INTERVAL（默认 60s）
        await asyncio.sleep(max(interval + random.uniform(-10, 10), 30))


# NoneBot 未初始化时（离线工具/单测直接 import 本模块）跳过启动注册，
# 仅当作为插件被 NoneBot 加载后才挂后台任务。
try:
    @get_driver().on_startup
    async def _start_bili_monitor() -> None:
        if not get_bili_uid():
            print("⚠️ BILI_UID 未配置，跳过 B站直播/动态监听")
            return
        spawn(_monitor_loop())
except ValueError:
    pass
