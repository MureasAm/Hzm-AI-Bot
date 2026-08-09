"""轻量感知：时间 / 农历 / 节日 / 天气。

get_now_context() 生成一行注入文本，让灰泽满知道"现在是几点、什么日子、天气如何"。
时间/农历/节气全部本地同步计算（免费、零依赖请求）；天气走和风天气免费版，
进程内缓存 1 小时（免费额度 1000 次/天）。天气 key/city 未配置或请求失败时静默
跳过天气段，绝不阻塞聊天主流程。
"""
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from lunar_python import Solar

from .constants import WEATHER_CACHE_SECONDS, WEATHER_GEO_CACHE_SECONDS
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
    """按城市（LocationID 或城市名）带 1h 缓存的天气行；city 空则用全局默认 WEATHER_CITY。"""
    if not city:
        city = get_weather_city()
    if not city or not get_weather_key():
        return ""

    loc_id = _resolve_location(city)
    if not loc_id:
        return ""

    now = time.time()
    entry = _weather_cache.get(loc_id)
    if entry and now - entry["ts"] < WEATHER_CACHE_SECONDS:
        return entry["text"]

    text = _fetch_weather(loc_id)
    _weather_cache[loc_id] = {"ts": now, "text": text}
    return text


def get_now_context(city: str = "") -> str:
    """生成一行感知注入文本。含北京时间、墨尔本当地时间（灰泽满所在地）、农历/节气/节日、
    天气（city 传用户城市则按用户城市，否则默认墨尔本）。如：
    【当前时间】北京时间 8月10日 周一 02:36（墨尔本当地时间 04:36），农历六月廿八，灰泽满所在地：澳洲墨尔本，天气：Light Rain 11℃
    """
    dt_cn = datetime.now(_CST)
    dt_mel = datetime.now(_MEL)
    parts = [
        f"{_format_time(dt_cn)} 北京时间（澳洲当地时间 {dt_mel.hour:02d}:{dt_mel.minute:02d}）",
        _format_lunar(dt_cn),
        "灰泽满所在地：澳洲",
    ]
    weather = _weather_line(city)
    if weather:
        parts.append(weather)
    return "【当前时间】" + "，".join(parts)
