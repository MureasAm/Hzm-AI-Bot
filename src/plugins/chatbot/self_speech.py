"""把她自己说过的话，在注入前改写成中性转述。

为什么要有这一层（2026-10-21 实测，n=20/组）：
- 引用原句进用户消息 → 复读同一句式 **20/20**；
- 原句前加「这是你之前说过的话」→ 18/20，再加「不要沿用」→ 16/20（提醒基本无效）；
- 同样的原句放进 system 消息 → 7/20（有用但不够）；
- **换成一句不带原句的大意 → 0/20**。
短期历史里的原句同样驱动（历史里有 3 条「少来」→ 10/10），加前缀无效。
所以两条通道统一改成「转述」：保留她说了什么事实/态度，去掉自称、称呼、口头禅和句式。

失败一律原样返回（fail-open）：转述写错比复读更伤上下文，宁可退回原句。
"""
import hashlib
import io
import json
import os
import re

from . import config
from .constants import THINKING_DISABLED

_CACHE_PATH = "data/self_speech_cache.json"
_CACHE_MAX = 3000
_CACHE: dict[str, str] = {}
_CACHE_LOADED = False

# 历史行：[3小时前] 灰泽满：…… ／ 群聊：[3小时前] 灰泽满（回甲）：……
_HER_LINE_RE = re.compile(
    r"^(?P<lead>(?:\[[^\]]*\]\s*)*)灰泽满(?:（[^）]*）)?：(?P<text>.*)$", re.S
)
# 引用块：[引用灰泽满自己说的：「……」] / [引用灰泽满说的：「……」（带图）]
_SELF_QUOTE_RE = re.compile(
    r"\[引用灰泽满(?:自己)?说的：「(?P<text>.*?)」(?P<tail>[^\]]*)\]", re.S
)
# 她主动发消息时可能带图：记忆行写成「灰泽满：[发了1张图片：描述] 正文」。
# 这个标记是**事实**（她确实发了图），不能像句式/自称那样被转述掉；
# 只转述标记后面的正文，转述完再把标记原样接回去。
_IMAGE_NOTE_RE = re.compile(r"^\[发了[^\]]*图片[^\]]*\]\s*")


def _split_image_note(text: str) -> tuple[str, str]:
    t = (text or "").strip()
    m = _IMAGE_NOTE_RE.match(t)
    if not m:
        return "", t
    return m.group(0).strip(), t[m.end():].strip()

PROMPT = (
    "你是文本转述器。下面每一条都是某个角色过去说过的话。请把每一条改写成一句"
    "**第三人称中性转述**：\n"
    "- 只保留她说了什么事实、表达了什么态度、答应了什么、让对方做什么；寒暄和口水话丢掉；\n"
    "- **保留**地点、物品、课程、事件等专名（如虚拟澳洲、晚自习、机械制图、直播间）；\n"
    "- **保留时间、数字、日期**，不得改动：「七点」「周三」「两遍」要原样留下；\n"
    "- **去掉**她的自称（灰泽满/我）、对对方的称呼（温宝/绿冻/宝宝/饱饱等）、口头禅和句式；\n"
    "- **分清事实属于谁**：对方的事写「对方」，她自己的事写「她」；"
    "她替对方确认/提醒的事，要写成「她提醒对方……」，不要写成她自己的事；\n"
    "- **分清句子类型**：她在提问就写「她问对方……」；她在答应就写「她答应……」；"
    "她在开玩笑/敷衍就保留这个性质，不要写成承诺；\n"
    "- **「他/她/他们/别人」不是「对方」**：聊天对象一律写「对方」，"
    "第三方写「另一个人」或按上下文写清是谁（如「对方的室友」），不要混成一个人；\n"
    "- **双重否定还原成肯定**：「没有不爱」= 她是喜欢的；\n"
    "- **反话写成本意**：「你可真行」= 她在嘲讽对方，不要照字面写成夸奖；\n"
    "- **保留不确定**：「可能/大概/说不定」不要写成确定；\n"
    "- **句首的 [发了…图片…] 是事实**（她确实发了图，可能还带描述）：原样保留，不要改写/删除，只转述它后面的正文；\n"
    "- **环境/外部状态不写成事实**：她随口说的网速、网络、天气、时间、设备/平台卡顿，"
    "只写她当时的说法/态度（如「她拿网速当借口敷衍对方」），不要把“她那边网速不好”本身写成事实；"
    "**但如果句子里有明确的计划/安排（马上要直播、在调设备、今晚八点开播），计划和动作必须保留**，"
    "只把环境部分写成她的说法；\n"
    "- **不要加解释、不要评价、不要补因果**；一条只写主要意思；\n"
    "- **绝对不要**在转述里出现「才不」「才没」「少来」「别来这套」「突然说这个」这些原话；\n"
    "- 每条不超过 35 字。\n"
    "例：「少来，灰泽满才不和绿冻命中注定呢」→「她否认和对方是命中注定的一对。」\n"
    "例：「灰泽满才没忘呢，这不是等着看绿冻急嘛」→「她没忘，只是在等对方着急。」\n"
    "例：「芒果，最爱的水果是芒果」→「她告诉对方，对方最爱吃的水果是芒果。」\n"
    "例：「要，明天早上空腹去体检」→「她确认对方需要空腹去体检。」\n"
    "例：「你明天几点的课来着？」→「她问对方明天几点有课。」\n"
    "例：「行吧，下次给你唱一段」→「她答应下次给对方唱一段。」\n"
    "例：「开玩笑的呀，你还当真了」→「她说自己是开玩笑，对方却当真了。」\n"
    "例：「他昨天说他要来，谁知道呢」→「她提到另一个人昨天说要来，但不确定。」\n"
    "例：「你室友不是挺好的吗，怎么又闹了」→「她问对方，既然和室友关系不错，为什么又闹矛盾了。」\n"
    "例：「你可真行啊，这都能搞错」→「她嘲讽对方连这都能搞错。」\n"
    "例：「大概明天就能弄完吧」→「她说大概明天能弄完。」\n"
    "例：「马上要直播，在调设备」→「她告诉对方自己马上要直播，正在调设备。」\n"
    "例：「设备又卡了，今晚可能播不了」→「她拿设备卡顿当理由，说今晚可能播不了。」\n"
    "写每一条前自查：谁做的事？肯定还是否定？时间和数字对不对？是问句还是陈述？\n"
    "只输出 JSON：{\"gists\": [\"...\", ...]}，条数和顺序与输入完全一致。\n\n"
)
_JSON_RE = re.compile(r"\{.*\}", re.S)
_BAD_GIST_RE = re.compile(r"才不|才没|少来|突然说这个|别来这套")

# 她自己的“环境/外部状态”说法：网速、天气、时间、设备……不能变成可引用的事实。
# 否则会像“虚拟澳洲风很大/网不好”一样，被反复当成她那边真实发生的事。
_ENV_CLAIM_RE = re.compile(
    r"(网速|网络|网卡|断网|信号|天气|气温|下雨|下雪|风大|风很大|风好大|台风|降温|升温|"
    r"卡顿|加载半天|连不上|平台|设备|电脑|手机)")
_SELF_ENV_RE = re.compile(
    r"(虚拟[一-龥]{2,6}|她那边|她这边|自己那边|她家|她的网|她的设备|她那边的)")
# 计划/安排：它和“环境抱怨”不是一回事。
# 「马上要直播，在调设备」里的“设备”是**具体计划的一部分**，不能降级成
# 「她提到环境状态，可能只是随口一说，不是事实」——那会把下一步的事实锚点删掉。
# 只匹配“未来/当前 + 开播/调设备”这种计划句；「上次连直播差点断掉」不匹配。
_PLAN_RE = re.compile(
    r"(?:马上|待会|等会|今晚|现在|正在|准备|就要|快要|刚).{0,12}(?:直播|开播|调设备|连麦|采集卡)"
    r"|(?:直播|开播).{0,12}(?:马上|待会|等会|今晚|现在|正在|准备|就要|快要)"
)
_ENV_SPEECH_MARKERS = ("抱怨", "吐槽", "敷衍", "借口", "解释", "否认", "反问", "表示", "提到")


def _sanitize_gist(source: str, gist: str) -> str:
    """她自己的环境/外部状态说法 → 只保留“她在说/找借口”，不保留世界状态。

    ⚠️ 只处理**她自己的**；gist 明确说的是“对方/他”的网/天气，原样保留。
    这是防止她自己的即兴（网速/天气/时间/设备）被历史回喂成事实。
    """
    g = (gist or "").strip()
    if not g or not _ENV_CLAIM_RE.search(g):
        return g
    # 有明确的计划/安排时，保留计划本身；只把“设备状态”写成她的说法，
    # 不能整句降级成“不是事实”（「马上要直播」不是环境 claim）。
    if _PLAN_RE.search(g):
        return g
    other = ("对方" in g) or ("他" in g and "她" not in g)
    if other and not _SELF_ENV_RE.search(g):
        return g
    if not (_SELF_ENV_RE.search(g) or (g.startswith("她") and not other)):
        return g
    if any(x in g for x in ("抱怨", "吐槽")):
        return "她拿环境/设备状态当借口抱怨，可能只是随口一说。"
    if "否认" in g:
        return "她拿环境/设备状态解释/否认，可能只是随口一说。"
    if "解释" in g or "表示" in g:
        return "她拿环境/设备状态当解释，可能只是随口一说。"
    return "她提到环境/设备状态，可能只是随口一说，不是事实。"


def enabled() -> bool:
    return os.environ.get("SELF_SPEECH_ANNOTATE", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _load_cache() -> None:
    global _CACHE_LOADED
    if _CACHE_LOADED:
        return
    _CACHE_LOADED = True
    try:
        if os.path.exists(_CACHE_PATH):
            with io.open(_CACHE_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(k, str) and isinstance(v, str) and len(_CACHE) < _CACHE_MAX:
                        _CACHE[k] = v
    except Exception:
        pass


def _save_cache() -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH) or ".", exist_ok=True)
        with io.open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_CACHE, f, ensure_ascii=False)
    except Exception:
        pass


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _parse_gists(content: str) -> list[str] | None:
    if not content:
        return None
    m = _JSON_RE.search(content)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    gists = data.get("gists") if isinstance(data, dict) else None
    if not isinstance(gists, list) or not gists:
        return None
    out = [str(g).strip() for g in gists]
    return out if all(out) else None


async def gist_texts(texts: list[str], client=None) -> dict[str, str]:
    """批量转述；返回 {原文: 转述}。失败或转述仍带原句式的条目不出现在结果里。"""
    uniq: list[str] = []
    for t in texts:
        t = (t or "").strip()
        if t and t not in uniq:
            uniq.append(t)
    if not uniq:
        return {}
    _load_cache()
    missing = [t for t in uniq if _key(t) not in _CACHE]
    for start in range(0, len(missing), 10):
        chunk = missing[start:start + 10]
        try:
            if client is None:
                client, _ = config.aux_clients_or(config._get_clients)
            payload = "".join(f"{i + 1}. {t}\n" for i, t in enumerate(chunk))
            resp = await client.chat.completions.create(
                model=config.aux_model(config._get_model_name()),
                messages=[{"role": "user", "content": PROMPT + payload}],
                temperature=0.0,
                max_tokens=min(1600, 110 * len(chunk) + 120),
                **THINKING_DISABLED,
            )
            content = config.extract_chat_content(resp)
            config.log_cache_usage(resp, "自述转述")
            gists = _parse_gists(content)
            if gists and len(gists) == len(chunk):
                bad = 0
                for src, g in zip(chunk, gists):
                    g = _sanitize_gist(src, g)
                    if not (0 < len(g) <= 80) or _BAD_GIST_RE.search(g):
                        bad += 1
                        continue
                    _CACHE[_key(src)] = g
                if bad:
                    print(f"⚠️ 自述转述里仍有 {bad} 条带原句式，按原样注入")
                _save_cache()
            else:
                print(f"⚠️ 自述转述条数不符（{len(gists or [])}/{len(chunk)}），这批原样注入")
        except Exception as e:
            print(f"⚠️ 自述转述失败（原样注入）: {type(e).__name__}")
    return {t: _sanitize_gist(t, _CACHE[_key(t)]) for t in uniq if _key(t) in _CACHE}


async def prewarm(text: str, client=None) -> None:
    """后台预转述一条她刚发出去的话，让下一轮注入直接命中缓存。"""
    text = (text or "").strip()
    if not text or not enabled():
        return
    try:
        await gist_texts([text], client)
    except Exception:
        pass


async def annotate_turn(history_lines: list[str], user_msg: str, client=None):
    """把历史里她的句子和本轮的自引用改写成转述。返回 (新历史, 新用户消息)。

    找不到可改写的内容时不发请求；失败时原样返回。
    """
    if not enabled():
        return history_lines, user_msg
    hist = list(history_lines or [])
    her_texts = []
    for ln in hist:
        m = _HER_LINE_RE.match(ln or "")
        if m and m.group("text").strip():
            _, core = _split_image_note(m.group("text"))
            if core:
                her_texts.append(core)
    quote_texts = []
    for m in _SELF_QUOTE_RE.finditer(user_msg or ""):
        _, core = _split_image_note(m.group("text"))
        if core:
            quote_texts.append(core)
    if not her_texts and not quote_texts:
        return hist, user_msg
    try:
        gists = await gist_texts(her_texts + quote_texts, client)
    except Exception as e:
        print(f"⚠️ 自述转述异常（原样注入）: {type(e).__name__}")
        return hist, user_msg
    if not gists:
        return hist, user_msg

    new_hist = []
    for ln in hist:
        m = _HER_LINE_RE.match(ln or "")
        if not m:
            new_hist.append(ln)
            continue
        text = m.group("text").strip()
        note, core = _split_image_note(text)
        g = gists.get(core)
        if g:
            new_text = f"{note} {g}" if note else g
            new_hist.append(f"{m.group('lead')}灰泽满（转述）：{new_text}")
        else:
            new_hist.append(ln)

    def _quote_sub(m):
        text = m.group("text").strip()
        note, core = _split_image_note(text)
        g = gists.get(core)
        if not g:
            return m.group(0)
        new_text = f"{note} {g}" if note else g
        return f"[对方引用了你之前说过的话，大意是：{new_text}{m.group('tail')}]"

    return new_hist, _SELF_QUOTE_RE.sub(_quote_sub, user_msg or "")
