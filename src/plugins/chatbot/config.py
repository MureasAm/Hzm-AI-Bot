"""NoneBot 全局配置读取助手 + API 客户端工厂。

.env.prod 里的 `WEATHER_KEY` 会被 NoneBot 小写化为 config 属性 `weather_key`。
此处统一提供带默认值的读取，供感知模块（context_probe / vision / bili_bridge）使用；
API 客户端工厂（_get_clients / _get_model_name）也在此，core / routing 共用，
避免 core ↔ routing 循环依赖。
"""
from nonebot import get_driver
from openai import AsyncOpenAI

from .constants import (
    VISION_MODEL, PUSH_INTERVAL, WEATHER_BASE_URL,
    DEEPSEEK_BASE_URL, ZHIPU_BASE_URL, DEFAULT_MODEL,
)


def get_config(name: str, default=None):
    """读取 NoneBot 全局配置项，未配置或异常时返回 default。"""
    try:
        value = getattr(get_driver().config, name, None)
        return value if value not in (None, "") else default
    except Exception:
        return default


# ==================== API 客户端（惰性初始化） ====================
# 延迟到首次使用时才读取 config 并创建客户端，
# 保证模块可被独立导入（便于测试），不依赖 NoneBot 已初始化。
_clients_cache = None


def _get_clients():
    """返回 (deepseek_client, zhipu_client)，首次调用时创建。"""
    global _clients_cache
    if _clients_cache is not None:
        return _clients_cache

    _global_config = get_driver().config
    deepseek_api_key = getattr(_global_config, "openai_api_key", None)
    deepseek_api_base = getattr(_global_config, "openai_api_base", DEEPSEEK_BASE_URL)
    zhipu_api_key = getattr(_global_config, "zhipu_api_key", None)

    if not deepseek_api_key:
        raise ValueError("❌ 未检测到 OPENAI_API_KEY")
    if not zhipu_api_key:
        raise ValueError("❌ 未检测到 ZHIPU_API_KEY")

    _clients_cache = (
        AsyncOpenAI(api_key=deepseek_api_key, base_url=deepseek_api_base),
        AsyncOpenAI(api_key=zhipu_api_key, base_url=ZHIPU_BASE_URL),
    )
    return _clients_cache


def _get_model_name() -> str:
    """解析对话模型名：优先 config.openai_model，回退到 DEFAULT_MODEL。"""
    try:
        return getattr(get_driver().config, "openai_model", None) or DEFAULT_MODEL
    except Exception:
        return DEFAULT_MODEL


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


def get_weibo_uid() -> str:
    """微博 UID（监控该账号的新微博）。留空关闭微博监听。"""
    return str(get_config("weibo_uid", "") or "").strip()


def get_weibo_cookie() -> str:
    """微博登录 Cookie（浏览器登录后复制的完整 Cookie，无登录会被风控挡 403/432）。"""
    return str(get_config("weibo_cookie", "") or "").strip()


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


def get_voice_enabled() -> bool:
    """语音总开关：voice_enabled=1/true/yes/on 才开。

    NoneBot 会把 .env 值当字符串读进来，"0" 会被 bool("0") 误判成 True，
    所以这里显式按字符串解析。默认关（GPT-SoVITS 通了再把 .env 改 1）。
    """
    return str(get_config("voice_enabled", "") or "").strip().lower() in ("1", "true", "yes", "on")
