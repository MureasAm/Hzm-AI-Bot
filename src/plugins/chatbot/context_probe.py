"""轻量感知：时间 / 农历 / 节日 / 天气。

get_now_context() 生成一行注入文本，让灰泽满知道"现在是几点、什么日子、天气如何"。
时间/农历/节气全部本地同步计算（免费、零依赖请求）；天气走和风天气免费版，
进程内缓存 1 小时（免费额度 1000 次/天）。天气 key/city 未配置或请求失败时静默
跳过天气段，绝不阻塞聊天主流程。
"""
import asyncio
import json
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from lunar_python import Solar

from . import injection as inj
from .constants import WEATHER_CACHE_SECONDS, WEATHER_GEO_CACHE_SECONDS, BILI_STATE_FILE
from .config import get_weather_base_url, get_weather_city, get_weather_key

_CST = ZoneInfo("Asia/Shanghai")
_MEL = ZoneInfo("Australia/Melbourne")  # 灰泽满所在地
_WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# 天气进程内缓存：LocationID -> { "ts": 上次请求时间戳, "text": 天气行文本 }
_weather_cache = {}
# 城市名 → LocationID 解析缓存：{ "名称": {"ts": ..., "id": ...} }
_location_cache = {}


def _weather_now_url() -> str:
    return get_weather_base_url().rstrip("/") + "/v7/weather/now"


def _geo_url() -> str:
    # 新格式专属域名下 geo 接口带 /geo 前缀（实测）
    return get_weather_base_url().rstrip("/") + "/geo/v2/city/lookup"


def _format_time(dt: datetime) -> str:
    """8月10日 周一 21:35"""
    return f"{dt.month}月{dt.day}日 {_WEEKDAYS[dt.weekday()]} {dt.hour:02d}:{dt.minute:02d}"


def _format_lunar(dt: datetime) -> str:
    """农历部分：农历七月初七 七夕 / 立秋。不是节气日、无节日时只返回农历日期。"""
    lunar = Solar.fromDate(dt).getLunar()
    parts = [f"农历{lunar.getMonthInChinese()}月{lunar.getDayInChinese()}"]
    festivals = [f for f in (lunar.getFestivals() or []) if f]
    others = [f for f in (lunar.getOtherFestivals() or []) if f]
    all_festivals = festivals + others
    if all_festivals:
        parts.append("、".join(all_festivals))
    jieqi = lunar.getJieQi()
    if jieqi:
        parts.append(jieqi)
    return " ".join(parts)


def _resolve_location(name: str) -> str:
    """城市名 → 和风 LocationID。纯数字（已是 ID）原样返回；否则查 geo 接口并缓存 1 天。

    失败返回空串（天气行静默降级），不影响时间/农历。
    """
    if not name:
        return ""
    name = name.strip()
    if name.isdigit():
        return name

    entry = _location_cache.get(name)
    if entry and time.time() - entry["ts"] < WEATHER_GEO_CACHE_SECONDS:
        return entry["id"]

    try:
        resp = httpx.get(
            _geo_url(),
            params={"location": name, "key": get_weather_key()},
            headers={"Accept": "application/json"},
            timeout=4.0,
        )
        data = json.loads(resp.text)
        if str(data.get("code")) == "200":
            locs = data.get("location") or []
            if locs:
                loc_id = str(locs[0].get("id", ""))
                _location_cache[name] = {"ts": time.time(), "id": loc_id}
                return loc_id
    except Exception as e:
        print(f"⚠️ 和风城市解析失败（{name}）: {e}")

    _location_cache[name] = {"ts": time.time(), "id": ""}
    return ""


def _fetch_weather(loc_id: str) -> str:
    """同步请求和风实时天气（带浏览器 UA）；失败返回空串。"""
    try:
        resp = httpx.get(
            _weather_now_url(),
            params={"location": loc_id, "key": get_weather_key()},
            headers={
                "Accept": "application/json",
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
            },
            timeout=3.0,
        )
        data = json.loads(resp.text)
        if str(data.get("code")) != "200":
            print(f"⚠️ 和风天气业务失败: code={data.get('code')}")
            return ""
        now = data.get("now", {})
        text = now.get("text", "")
        temp = now.get("temp", "")
        feels = now.get("feelsLike", "")
        if not text or temp == "":
            return ""
        return f"天气：{text} {temp}℃" + (f"（体感 {feels}℃）" if feels else "")
    except Exception as e:
        print(f"⚠️ 和风天气请求失败（已忽略）: {e}")
        return ""


def _weather_line(city: str) -> str:
    """天气行——**只读缓存**，不再同步发网络请求（防卡事件循环）。

    缓存里没有 / 过期 → 这轮先不报天气（返回空，等 warm_weather 后台预热补上）。
    """
    if not city:
        city = get_weather_city()
    if not city or not get_weather_key():
        return ""

    if str(city).isdigit():
        loc_id = city            # 本身就是 LocationID
    else:
        e = _location_cache.get(city)
        loc_id = (e or {}).get("id", "") if e else ""
    if not loc_id:
        return ""  # 还没预热出 LocationID → 不阻塞

    entry = _weather_cache.get(loc_id)
    if entry and time.time() - entry["ts"] < WEATHER_CACHE_SECONDS:
        return entry["text"]
    return ""  # 缓存过期/缺失 → 等后台预热


async def warm_weather(city: str = "") -> None:
    """异步预热：确保该城市的 LocationID + 天气已进缓存（网络走 asyncio.to_thread）。

    组装消息前在异步上下文里调用它，等真正 build 时 _weather_line 命中缓存、零阻塞。
    """
    if not city:
        city = get_weather_city()
    if not city or not get_weather_key():
        return
    try:
        loc_id = await asyncio.to_thread(_resolve_location, city)
        if not loc_id:
            return
        now = time.time()
        entry = _weather_cache.get(loc_id)
        if entry and now - entry["ts"] < WEATHER_CACHE_SECONDS:
            return  # 已新鲜，别每轮都打
        text = await asyncio.to_thread(_fetch_weather, loc_id)
        if text:
            _weather_cache[loc_id] = {"ts": time.time(), "text": text}
    except Exception:
        pass  # 预热失败下轮再试，不影响回复


def _live_status_text() -> str:
    """读取 B站 monitor 最近更新的真实直播状态，返回一句话；无状态返回空串。

    让灰泽满知道自己当前真实的直播状态，避免编造"刚下播/在直播"。
    """
    if not BILI_STATE_FILE.exists():
        return ""
    try:
        state = json.loads(BILI_STATE_FILE.read_text("utf-8"))
        is_live = bool(state.get("last_live_status", False))
    except (json.JSONDecodeError, OSError):
        return ""
    return "灰泽满现在正在直播中" if is_live else "灰泽满现在没有在直播"


def get_now_context(city: str = "") -> str:
    """生成一行感知注入文本。含北京时间、墨尔本当地时间（灰泽满所在地）、农历/节气/节日、
    真实直播状态、天气（city 传用户城市则按用户城市，否则默认墨尔本）。如：
    【当前时间】北京时间 8月10日 周一 02:36（墨尔本当地时间 04:36），农历六月廿八，灰泽满所在地：澳洲，灰泽满现在没有在直播，天气：Light Rain 11℃。（农历和节气是中国历法的日期参考，……）

    ⚠️ 结尾那两句是**读取这一行数据的守则**，必须挂在这一行自己身上（2026-10-16 从骨架搬来）：
    骨架是最前面那块、这行是最后一块，中间隔着十几块——守则写在骨架里等于写在离它最远的地方。
    """
    dt_cn = datetime.now(_CST)
    dt_mel = datetime.now(_MEL)
    parts = [
        f"{_format_time(dt_cn)} 北京时间（澳洲当地时间 {dt_mel.hour:02d}:{dt_mel.minute:02d}）",
        _format_lunar(dt_cn),
        "灰泽满所在地：澳洲",
    ]
    live = _live_status_text()
    if live:
        parts.append(live)
    weather = _weather_line(city)
    if weather:
        parts.append(weather)
    return inj.T("now", "head") + "，".join(parts) + inj.T("now", "tail")


def _cst(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(_CST)
    return now.astimezone(_CST) if now.tzinfo else now.replace(tzinfo=_CST)


def _period_label(dt: datetime) -> str:
    h = dt.hour
    if 5 <= h < 11:
        return "早上"
    if 11 <= h < 14:
        return "中午"
    if 14 <= h < 18:
        return "下午"
    if 18 <= h < 23:
        return "晚上"
    if h >= 23 or h < 5:
        return "深夜" if h >= 23 else "凌晨"
    return "深夜"


_TIME_HINT_RE = re.compile(
    r"(^早|早安|早上|晚安|睡了|睡了没|还没睡|没睡|这么晚|几点|熬夜|通宵|凌晨|起床|醒了)"
)


def frame_now(now: datetime | None = None) -> str:
    """常驻 frame：日期 + 星期 + 直播状态。**不含精确时间/农历/天气**。

    精确时间、农历、天气都走 `recall(kind=perception/weather)`；这样每轮不会有一个
    现成的“都凌晨了”脚本摆在模型面前（2026-10-27 定）。
    """
    dt = _cst(now)
    date = f"{dt.month}月{dt.day}日 {_WEEKDAYS[dt.weekday()]}"
    live = _live_status_text()
    return inj.T("now", "frame", date=date, live=(f"，{live}" if live else ""))


def session_time_hint(now: datetime | None = None, user_msg: str = "",
                      is_new_session: bool = False) -> str:
    """本场时段：只在会话开场、问候/作息/问时间时注一次；不是每轮。"""
    if not is_new_session and not _TIME_HINT_RE.search(user_msg or ""):
        return ""
    period = _period_label(_cst(now))
    return inj.T("now", "time_hint", period=period)


def weather_now(city: str = "") -> str:
    """只读缓存，返回天气行；没有/过期返回空串（由 warm_weather 预热）。"""
    return _weather_line(city)


def perception_now(now: datetime | None = None) -> str:
    """精确时间 + 农历/节气；查询驱动，不进常驻 frame。"""
    dt_cn = _cst(now)
    dt_mel = dt_cn.astimezone(_MEL)
    return (f"北京时间 {_format_time(dt_cn)}"
            f"（澳洲当地时间 {dt_mel.hour:02d}:{dt_mel.minute:02d}），"
            f"{_format_lunar(dt_cn)}")
