"""NoneBot 全局配置读取助手（避免各模块重复 get_driver 样板）。

.env.prod 里的 `WEATHER_KEY` 会被 NoneBot 小写化为 config 属性 `weather_key`。
此处统一提供带默认值的读取，供感知模块（context_probe / vision / bili_bridge）使用。
"""
from nonebot import get_driver

from .constants import VISION_MODEL, PUSH_INTERVAL, WEATHER_BASE_URL


def get_config(name: str, default=None):
    """读取 NoneBot 全局配置项，未配置或异常时返回 default。"""
    try:
        value = getattr(get_driver().config, name, None)
        return value if value not in (None, "") else default
    except Exception:
        return default


def get_weather_base_url() -> str:
    """和风 API Host：每个控制台项目有专属域名，需在控制台查看并配置 WEATHER_BASE_URL。"""
    return get_config("weather_base_url", WEATHER_BASE_URL)


def get_weather_city() -> str:
    return get_config("weather_city", "")


def get_weather_key() -> str:
    return get_config("weather_key", "")


def get_vision_model() -> str:
    return get_config("vision_model", VISION_MODEL)


def get_bili_uid() -> str:
    return str(get_config("bili_uid", "") or "").strip()


def get_bili_sessdata() -> str:
    """B站登录 cookie SESSDATA（动态监控必需，可留空只监控开播）。"""
    return str(get_config("bili_sessdata", "") or "").strip()


def get_notify_whitelist() -> list:
    raw = get_config("notify_friends_whitelist", "")
    if not raw:
        return []
    return [str(x).strip() for x in str(raw).replace("，", ",").split(",") if str(x).strip()]


def get_push_interval() -> int:
    try:
        return int(get_config("push_interval", PUSH_INTERVAL))
    except (TypeError, ValueError):
        return PUSH_INTERVAL
