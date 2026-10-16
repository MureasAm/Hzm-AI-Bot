"""NoneBot 全局配置读取助手 + API 客户端工厂。

.env.prod 里的 `WEATHER_KEY` 会被 NoneBot 小写化为 config 属性 `weather_key`。
此处统一提供带默认值的读取，供感知模块（context_probe / vision / bili_bridge）使用；
API 客户端工厂（_get_clients / _get_model_name）也在此，core / routing 共用，
避免 core ↔ routing 循环依赖。
"""
import contextvars
import json
import os
import time

from nonebot import get_driver
from openai import AsyncOpenAI, RateLimitError, APIConnectionError

from .constants import (
    VISION_MODEL, PUSH_INTERVAL, WEATHER_BASE_URL,
    DEEPSEEK_BASE_URL, ZHIPU_BASE_URL, DEFAULT_MODEL, PROJECT_ROOT, LLM_TIMEOUT_SEC,
    MODEL_EXTRA_PARAMS,
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

# 降级链：某一层失败之后，**隔多久才再回头试它**（秒）。两档，因为两种失败的代价差几十倍：
#
# - **线路挂了**（服务端 5xx / 超时）→ `LLM_CHAIN_RETRY_SEC`（默认 180 = 3 分钟）回头**探一次**。
#   探一次的代价只有 ~0.3~1s（`max_retries=0`，见 `_mk_client`），所以值得探勤一点，线路一好就自己回来。
#   ⚠️ **原来是 600 秒**，2026-10-15 用户反馈"十多分钟都走下一层，是不是太长了"——是的：
#   一次失败探一次的代价才 1 秒，锁 10 分钟纯属浪费，而且**这 10 分钟的调用全落到更贵的层上**。
# - **抖了一下**（HTTP 200 但响应不能用：没有 choices / 正文空）→ `LLM_CHAIN_FLAKY_SEC`
#   （默认 60 = 1 分钟）**马上再试**。这一档代价大得多（可能要等到超时，实测最长 60s+），
#   但这一层**多半是好的**（实测 6 次里 1 次卡死、其余毫秒级正常），
#   所以别让它因为抖一下就离场十分钟。
#
# ⚠️ 两档都**不是**"每次调用都去撞"（用户 2026-10-14 否掉的方案）：
# 同一层在窗口内只会被试一次。设成 0 = 本次进程内**永不回头**。
_CHAIN_RETRY_SEC = float(os.environ.get("LLM_CHAIN_RETRY_SEC", "180") or 180)
_CHAIN_FLAKY_SEC = float(os.environ.get("LLM_CHAIN_FLAKY_SEC", "60") or 60)


def _bad_shape_reason(resp) -> str:
    """响应形状不能用就返回一句人话原因；正常返回空串。

    判据跟 `extract_chat_content` 保持一致（那边抛什么，这里就当哪类坏）：
    没有 choices / content 是 None / 正文是空的（思考吃光 max_tokens 那种）。
    """
    choices = getattr(resp, "choices", None)
    if not choices:
        return "上游没返回 choices"
    content = getattr(getattr(choices[0], "message", None), "content", None)
    if content is None:
        return "content 是 None"
    if not str(content).strip():
        return "正文是空的"
    return ""


def _mk_client(api_key, base_url):
    """建一个对话客户端——**参数全仓只有这一处**，因为有两个反直觉的默认值要改：

    - `max_retries=0`：SDK 默认自己重试 2 次。**我们的链就是重试机制**，
      这两层内部重试纯属白等：实测同一条死线路 3.7s →（retries=1）2.0s →（retries=0）**1.1s**。
      也正是这两秒多，当初逼出了"粘性不回头"这个更糟的方案。
    - `timeout=90`：SDK 默认 600s。上游偶发**只是卡住不返回**（实测 luna 6 次里 1 次挂到 60s+），
      不设超时的话，这一轮会**干等十分钟**才轮到降级——用户只看到"她不回话"。
      （90s 是给正常长回复留的余量：真实主回复实测都在 30s 内。）
    """
    return AsyncOpenAI(api_key=api_key, base_url=base_url,
                       max_retries=0, timeout=LLM_TIMEOUT_SEC)


# **这一轮到底是哪一层答的**（整条标签，形如 `gemini-3.1-flash-lite（linkapi.ai）`）。
# 为什么需要（2026-10-15）：链变成三个端点之后，光看模型名分不出是哪个站——
# 而两个站可能挂着**同名不同价**的模型（比如 3.8-flash 两家都有）。日志和账本都得带上它。
_SERVED_BY = contextvars.ContextVar("llm_served_by", default="")


class _ChainCompletions:
    """`client.chat.completions.create` 的包装：**按顺序试一串模型/端点，自动降级**。

    为什么要它（2026-10-14 实测）：中转站的 gemini 线路从 13:00 起**整条 500 挂了 75 分钟**，
    而 bot 只会一轮一轮地回兜底文案——**端点/模型级故障不该让整只 bot 变哑巴**。

    链的形状（用户 2026-10-14 定）：**gemini → luna → deepseek 官方**。
    - 前两层是**同一个端点**的不同模型（中转站上换 model 名就行）；
    - 最后一层是**另一个端点**（deepseek 官方，最稳）。
    备用用**同一套参数**喂（`THINKING_DISABLED` + `reasoning_effort=none` 三家实测都吃），
    只是换个 model 名。全链都挂了 → 照原样把最后一个异常抛出去（调用方照旧兜底成一句人话）。

    ⚠️ **一次失败就"全线切走"，但会自己回来**（用户 2026-10-14 两轮定稿）：
    某一层挂掉后，后续调用先绕开它（否则按真实流量的轮次间隔，几乎每次都要白等）；
    绕开多久见 `_CHAIN_RETRY_SEC` / `_CHAIN_FLAKY_SEC` 两档——**默认 10 分钟回头探一次**，
    探通了就用回来。中间那个"永远不回头"的版本只活了一轮：
    用户当天就发现"滑到 deepseek 之后就一直在 deepseek"，而**最贵的恰恰是那一层**。
    """

    def __init__(self, layers, cooldown=None):
        # [(client, model, label, extra_params?), ...]，第 0 层是主。
        # 三元组也收（`extra_params` 补 None）——测试和"没有特殊参数"的层都这么写。
        self._layers = [tuple(l) + (None,) * (4 - len(l)) if len(l) < 4 else tuple(l)
                        for l in layers]
        # cooldown = **硬失败**（抛异常）后隔多久再回头试这一层；0 = 本次进程内不回头。
        # 软失败（形状坏）走 `_CHAIN_FLAKY_SEC`，与这里无关——两档的理由见文件头那段。
        self._cooldown = cooldown if cooldown is not None else _CHAIN_RETRY_SEC
        self._down_until: dict[int, float] = {}
        self._last_i = 0             # 上次实际用的是第几层（用来打"降级 / 回到"那行日志）

    def _mark_down(self, i: int, soft: bool = False) -> None:
        """把第 i 层标成"暂时别用"，到点了自动再试。

        `soft=True` = 它只是**抖了一下**（200 但响应不能用），用短窗口；否则按"线路挂了"算。
        （两档的取值与理由见文件头 `_CHAIN_RETRY_SEC` 那段。）
        """
        sec = _CHAIN_FLAKY_SEC if soft else self._cooldown
        self._down_until[i] = float("inf") if sec <= 0 else time.monotonic() + sec

    def _pick(self):
        """挑这次要试的层：跳过已标记"挂了"的；全挂了就只试**最后一层**（通常是最稳的官方端点）。"""
        now = time.monotonic()
        alive = [(i, l) for i, l in enumerate(self._layers) if now >= self._down_until.get(i, 0.0)]
        return alive or [(len(self._layers) - 1, self._layers[-1])]

    async def create(self, **kwargs):
        last_err = None
        for i, (client, model, label, extra) in self._pick():
            # ⚠️ `model is None` = **别动调用方传进来的模型名**（第 1 层就是这么标的）。
            # 踩坑（2026-10-14 当场撞到）：写成 `{**kwargs, "model": model}` 会把调用方的
            # 模型名覆盖成 None → 上游收到 `model=null` → 400 invalid_request_error，
            # 而那条错误会被当成"主模型挂了"，白白降级。
            payload = kwargs if model is None else {**kwargs, "model": model}
            if extra:
                # 这一层自己的参数（见 `constants.MODEL_EXTRA_PARAMS`）——**按层**盖，
                # 不动别的层：换个模型可能就换一套"关思考"的写法。
                payload = {**payload, "extra_body": {**(payload.get("extra_body") or {}), **extra}}
            try:
                resp = await client.chat.completions.create(**payload)
            except Exception as e:
                last_err = e
                # **"抖一下"和"真挂了"要分开**——两类都不该用 10 分钟的长窗口：
                #   · 429 = "你问太快了"（实测中转站 luna 连发 5 次后桶就空，之后全 0.2s 被拒）；
                #   · **APIConnectionError = 网络层连不上**，请求**可能根本没到站**
                #     （2026-10-15 用户发现："我这其中转站没收到失败日志，但代码报了
                #     APIConnectionError"——**两边对得上**，因为那是本地/链路的问题）。
                #     这种错**一点也不能说明那条渠道坏了**，锁它十分钟等于白花钱走贵的那层。
                soft = isinstance(e, (RateLimitError, APIConnectionError))
                self._mark_down(i, soft=soft)
                nxt = "全线切到下一层" if i < len(self._layers) - 1 else "没有下一层了"
                if isinstance(e, RateLimitError):
                    kind = "限流（慢一点，不是坏了）"
                elif isinstance(e, APIConnectionError):
                    kind = "连不上（网络层，不是它坏了）"
                else:
                    kind = type(e).__name__
                print(f"⚠️ 模型 {label} 调用失败（{kind}: {str(e)[:60]}）→ {nxt}")
                continue
            # ⚠️ **HTTP 成功 ≠ 拿到能用的响应**（2026-10-14 真实撞到）：
            # 上游偶发返回 `choices` 为空/null 的"200 响应"，链以前只认异常、把它当成功返回，
            # 于是下游 `extract_chat_content` 才发现形状不对——**那时已经在链外了，不会再降级**，
            # 整轮就掉进 "小满现在不在哦~"。现在形状不对也当这一层失败，继续往下试。
            reason = _bad_shape_reason(resp)
            if reason and i < len(self._layers) - 1:
                self._mark_down(i, soft=True)      # 抖了一下 ≠ 挂了，一分钟后就再来试
                print(f"⚠️ 模型 {label} 返回异常（{reason}）→ 全线切到下一层")
                continue
            self._down_until.pop(i, None)          # 它活着，清掉冷却
            _SERVED_BY.set(label)                  # 记下"这轮是谁答的"（日志/账本都要用）
            if i != self._last_i:
                # 降级和**回升**都报一句——不然"回到 gemini 了"在日志里看不出来
                print(f"[模型] {'降级到' if i > self._last_i else '回到'}第 {i + 1} 层：{label}")
                self._last_i = i
            # 最后一层就算形状不对也**照原样返回**——让调用方照旧报一句能看懂的错
            return resp
        raise last_err


class _ClientProxy:
    """把 AsyncOpenAI 包一层，**只拦 chat.completions.create**；其它属性（embeddings…）原样透传。"""

    def __init__(self, client, chain=None):
        self._client = client
        self.chat = _ChatProxy(_ChainCompletions(chain or [(client, None, "主")]))

    def __getattr__(self, name):        # 没显式定义的一律透传（注意：只对"找不到的属性"生效）
        return getattr(self._client, name)


class _ChatProxy:
    def __init__(self, completions):
        self.completions = completions


# 最多认几个"另一个端点"（`FALLBACK1_*` / `FALLBACK2_*` / …）。
# 3 够用：实测的链是 **linkapi → krill → deepseek 官方**，三个不同端点。
_MAX_EXTRA_ENDPOINTS = 3


def _more_endpoints(global_config):
    """按顺序读出"另一个端点"那几层：`FALLBACK1_*`、`FALLBACK2_*`、…

    为什么要编号（2026-10-15）：链变成了**三个不同端点**（linkapi → krill → deepseek 官方），
    而原来只支持"主端点 + 一个备用端点"。**不带编号的 `FALLBACK_*` 仍当第 1 个读**，
    老配置不用改。
    """
    out = []
    for i in range(1, _MAX_EXTRA_ENDPOINTS + 1):
        key = getattr(global_config, f"fallback{i}_api_key", None)
        if not key and i == 1:
            key = getattr(global_config, "fallback_api_key", None)      # 老写法：不带编号
        if not key:
            continue
        base = getattr(global_config, f"fallback{i}_api_base", None) or (
            getattr(global_config, "fallback_api_base", None) if i == 1 else None)             or DEEPSEEK_BASE_URL
        model = getattr(global_config, f"fallback{i}_model", None) or (
            getattr(global_config, "fallback_model", None) if i == 1 else None)             or DEFAULT_MODEL
        host = str(base).split("//")[-1].split("/")[0]
        out.append((key, base, model, f"{model}（{host}）"))
    return out


def _chain_layers(global_config):
    """拼出降级链：主模型 → 同端点的其它模型 → 一个一个"另一个端点"。

    为什么分两截：中转站上换模型只要改 model 名（同一个 client），而"换端点"要另建 client。
    端点可以有**好几个**（见 `_more_endpoints`），顺序就是 `FALLBACK1_*` → `FALLBACK2_*` → …
    """
    primary_client = _mk_client(
        getattr(global_config, "openai_api_key", None) or "missing",
        getattr(global_config, "openai_api_base", DEEPSEEK_BASE_URL),
    )
    layers = [(primary_client, None, "")]      # model=None = 用调用方传进来的那个（= _get_model_name()）
    for m in str(getattr(global_config, "llm_fallback_models", "") or "").split(","):
        m = m.strip()
        if m:
            layers.append((primary_client, m, m, MODEL_EXTRA_PARAMS.get(m)))
    for key, base, model, label in _more_endpoints(global_config):
        layers.append((_mk_client(key, base), model, label, MODEL_EXTRA_PARAMS.get(model)))
    # 给每层贴个好看的名字：主层用真实模型名，方便日志一眼看出"这轮是谁答的"
    if layers:
        client, m, label = layers[0]
        # 主层的名字要等 `_get_model_name()` 才拿得到，所以它的标签和参数覆盖都只能在这补。
        # **端点名要跟其它层一样带上**——三个端点之后，只有第 1 层不带的话，
        # 日志上根本分不出"这轮是哪个站答的"。
        if not label:
            name = _get_model_name()
            host = str(getattr(global_config, "openai_api_base", "") or "").split("//")[-1].split("/")[0]
            label = f"{name}（{host}）" if host else name
        layers[0] = (client, m, label, MODEL_EXTRA_PARAMS.get(_get_model_name()))
    return layers


def _get_clients():
    """返回 (对话客户端, zhipu_client)，首次调用时创建。

    第一个客户端是 `_ClientProxy`——**它的 create() 会沿降级链自动换模型/换端点**
    （见 `_ChainCompletions`）。没配 `LLM_FALLBACK_MODELS`/`FALLBACK_*` 时行为与原来完全一样。
    """
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
        _ClientProxy(_mk_client(deepseek_api_key, deepseek_api_base),
                     _chain_layers(_global_config)),
        _mk_client(zhipu_api_key, ZHIPU_BASE_URL),
    )
    return _clients_cache


def parse_json_block(content: str):
    """把 LLM 返回的 JSON 正文解析成对象（自动剥 ``` 围栏 + json 语言标签）。

    踩坑：模型常返回 ```json\\n{...}\\n```，只取 ``` 之间那段的话，
    剩下的 `json\\n{...}` 会让 json.loads 直接失败——**语言标签要一起剥**。

    解析不了就抛 json.JSONDecodeError（各判官的调用方本来就都有 try/except 兜底）。
    """
    text = (content or "").strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
    if text[:4].lower() == "json":
        text = text[4:].strip()
    return json.loads(text)


def log_cache_usage(resp, tag: str) -> None:
    """打一行 DeepSeek 缓存用量（命中/未命中/命中率 + 输出），并把这次调用记进账本。

    为什么要它（2026-10-09）：缓存命中是**按前缀**算的，只有打出来才能看见——
    "感觉命中低"没法排查。实测教训：提示词里**变量放前面**会让后面整段静态正文
    永远不命中（同一条判据实测 0% vs 68%），而这种问题在日志里一眼就能看出来。

    上游不返回这两个字段时静默跳过打印（不同模型/网关不一定有）——但**记账照记**
    （prompt/输出 是标准字段，分账不依赖缓存字段）。
    """
    u = getattr(resp, "usage", None)
    if u is None:
        return
    _record_usage(tag, resp)               # 记账**先做**：不依赖上游给不给缓存字段
    pt = getattr(u, "prompt_tokens", 0) or 0
    out = getattr(u, "completion_tokens", 0) or 0
    # ⚠️ **缓存字段两套写法都要认**（2026-10-14 换模型后才撞到）：
    #   · DeepSeek 直接在 usage 上给 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`；
    #   · OpenAI / Gemini 那套把它们藏在 `prompt_tokens_details.cached_tokens` 里。
    #   以前取不到就 `return`——于是换成 luna 之后**这行一个字都不打**，日志上看着像没在跑。
    hit = getattr(u, "prompt_cache_hit_tokens", None)
    if hit is None:
        details = getattr(u, "prompt_tokens_details", None)
        hit = getattr(details, "cached_tokens", None) if details is not None else None
    miss = getattr(u, "prompt_cache_miss_tokens", None)
    if miss is None:
        miss = max(pt - (hit or 0), 0) if hit is not None else "?"
    rate = f"{hit / max(pt, 1):.0%}" if hit is not None else "—"
    route = _ROUTE.get()
    # **这轮是谁答的**：降级链跑起来之后，"用的是哪一层"是排查的第一信息
    # （取响应回显的名字——那才是真正干活的模型，不是我们请求时写的那个）。
    model = str(getattr(resp, "model", "") or "")
    # **优先打整条层标签**（带端点）——光有模型名的话，三个端点挂着同名模型时分不出来。
    # 没有链信息时（单测里直接调）退回模型名。
    who = _SERVED_BY.get() or model
    print(f"[缓存] {tag}{'·' + route if route else ''}{f' [{who}]' if who else ''}: "
          f"prompt={pt} 命中={hit if hit is not None else '—'} 未命中={miss} ({rate}) 输出={out}")


# ==================== LLM 用量分账（2026-10-13） ====================
# **为什么要它**：用户要量清"群聊到底占多少花费"（bot 10-12 单日 ¥8.43，假设是群聊读
# 全群发言占了大头）。stdout 那行 `[缓存] …` 看完就滚没了，而且**不带输出 token、
# 也分不出私聊/群聊**——聚合不出"哪条路贵"。所以每次调用追加一行 JSONL 落到
# `data/llm_usage.jsonl`（data/ 不入库），聚合看 `scripts/llm_cost.py`。
#
# 记账**不改变任何行为**：写失败静默，缺字段按 0 算。
USAGE_LOG_FILE = PROJECT_ROOT / "data" / "llm_usage.jsonl"

# 当前这条链路属于哪条通道。由**入口**设置（chat_window._flush / proactive）。
# 用 contextvar 而不是加参数：调用点有 12 处（routing/corpus_judge/memory_v2/session_memory…），
# 挨个加参数等于把"记账"这个横切关注点捅进每个人格模块。
# asyncio.create_task 会**复制**当前 context，所以 handle_chat 里起的记忆提取任务也带得上。
_ROUTE = contextvars.ContextVar("hzm_llm_route", default="")


def set_llm_route(route: str) -> None:
    """标记"接下来这条链路的 LLM 调用算在哪条通道上"（私/群/主动）。"""
    _ROUTE.set(route)


def usage_log_enabled() -> bool:
    """环境变量 LLM_USAGE=0 可关掉记账（缺省开，和 CHATLOG 一个约定）。"""
    return os.environ.get("LLM_USAGE", "1") != "0"


def _cache_hit(u):
    """这次调用的缓存命中量——**两套写法都认**（DeepSeek 直给 / OpenAI·Gemini 藏在 details 里）。"""
    hit = getattr(u, "prompt_cache_hit_tokens", None)
    if hit is not None:
        return hit
    details = getattr(u, "prompt_tokens_details", None)
    return getattr(details, "cached_tokens", None) if details is not None else None


def _record_usage(tag: str, resp) -> None:
    """把一次调用的用量追加进账本。**失败静默**——记账绝不能影响聊天。

    `model` 取**响应自己回显的名字**（不是 `_get_model_name()`）——降级链跑起来之后，
    这次调用可能走的不是主模型，记成主模型会让账本对不上。
    """
    if not usage_log_enabled():
        return
    u = getattr(resp, "usage", None)
    if u is None:
        return
    try:
        rec = {
            "t": time.time(),
            "tag": tag,
            "route": _ROUTE.get(),
            "model": str(getattr(resp, "model", "") or _get_model_name()),
            # 哪一层答的（带端点）。**model 分不出同名不同价的两个站**，靠这个分。
            "layer": _SERVED_BY.get(),
            "prompt": int(getattr(u, "prompt_tokens", 0) or 0),
            "hit": int(_cache_hit(u) or 0),
            "miss": int(getattr(u, "prompt_cache_miss_tokens", 0) or 0),
            "output": int(getattr(u, "completion_tokens", 0) or 0),
        }
        USAGE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(USAGE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def extract_chat_content(resp) -> str:
    """从 chat completion 响应里安全取正文；取不到就抛一个**看得懂**的错误。

    为什么必须有这个：7 个调用点以前都直接写 `resp.choices[0].message.content.strip()`，
    而这只在"响应形状永远正确"时才成立。实测踩过：上游偶发返回 `choices: null`，
    三个调用点**同时**报 `'NoneType' object is not subscriptable`——看不出是哪个环节、
    也看不出是上游形状问题，只能一行行猜。这里把三种坏形状统一翻译成人话。

    顺带挡一个**静默**的坑（更危险，因为它不报错）：
    DeepSeek 部分模型（如 `deepseek-flash`）**默认开思考模式**，推理内容会先吃掉
    max_tokens，于是 `content` 是**空串**、`finish_reason=length`。以前这种情况会
    静默返回 ""，调用方当"这次没结果"处理——**功能悄悄失灵，日志上什么都看不到**。
    所以空正文也抛错（调用方本来都有 try/except，行为不变，只是能看见了）。
    """
    choices = getattr(resp, "choices", None)
    if not choices:
        raise RuntimeError("LLM 响应没有 choices（上游返回了异常形状？）")
    choice = choices[0]
    content = getattr(getattr(choice, "message", None), "content", None)
    finish = getattr(choice, "finish_reason", "?")
    if content is None:
        raise RuntimeError(f"LLM 响应 content 为 None（finish_reason={finish}）")
    text = content.strip()
    if not text:
        reasoning = getattr(getattr(choice, "message", None), "reasoning_content", None) or ""
        raise RuntimeError(
            f"LLM 响应正文为空（finish_reason={finish}，思考内容 {len(reasoning)} 字）"
            f"——多半是思考模式吃光了 max_tokens；该模型不支持禁用思考时要把额度调大")
    return text


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
