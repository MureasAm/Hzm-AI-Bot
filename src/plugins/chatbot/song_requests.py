"""把已确认的私聊点歌意图变成本轮可选的本地音频。"""

import json
import random
import re
from dataclasses import dataclass
from unicodedata import normalize

from . import injection as inj
from .songs import Song, SongDelivery, song_key

_FILE_REQUEST_WORDS = re.compile(r"文件|附件|(?<![a-z])(?:m4a|mp3|wav|aac|flac|ogg|opus)(?![a-z])", re.I)
_FILE_REFUSAL = re.compile(r"(?:别|不要|不用)(?:再)?(?:给我)?(?:发|用|传)?\s*(?:音频)?(?:文件|附件)")
_HINT = re.compile(r"唱|哼|听|歌|曲|首|完整版|再来两句|音频文件|文件形式|《")
_FOLLOWUP = re.compile(
    r"想听|要听|要不要.{0,12}(?:唱|听|整首|完整版|换)|选哪首|(?:完整版|整首).{0,5}[？?吗么]"
)
_TALKING_ABOUT_SINGING = re.compile(
    r"(?:我?(?:有点|突然|好|又)?想听(?:你|灰泽满|小满)唱歌"
    r"|(?:你|灰泽满|小满)(?:今天|刚才)?唱(?:歌|得)(?:真|太|很|好)?好听)[了啊呀诶哦呢。！!～~ ]*"
)
_CANCELLED = re.compile(r"(?:^|[，,。！？!?\n])\s*(?:算了|不用了|(?:还是)?(?:先)?(?:别|不用)唱了?)[吧啊呀。！!～~ ]*$")
_MAX_OPTIONS = 8
OFFER_MAX_AGE = 600
_ORDINAL = re.compile(
    r"(?:那(?:就)?|就选|就|我选|我想听|选|听|要|来)?\s*第\s*(\d+|[一二三四五六七八九十零两]+)\s*(?:首|个)"
)
_CHOICE_END = re.compile(r"[吧呀啊呢喵。！!～~ ]*")
_CHOICE_FILLER = re.compile(r"(?:嗯+|好[的嘞]?|ok|喵)[啊呀。！! ]*", re.I)
_OFFER = re.compile(r"(?:换一首|(?:想听|要听|选|要发)哪首(?P<full>完整版(?P<file>文件)?)?)[吧呀啊吗]?\s*[？?：:]\s*")
_OFFER_SEPARATOR = re.compile(r"[、，,\r\n]+|还是|或者")
_QUOTED_TITLE = re.compile(r'《([^》]+)》|“([^”]+)”|"([^"]+)"')
_MODES = {"clip", "full", "recent_full", "another", "other_part"}
_CLAUSE_START = r"(?:^|[，,。！？!?\n])\s*(?:(?:灰泽满|小满|我|今天|这次|这回|现在|这会儿|先|才|就|还是|真)\s*){0,6}"
_CLAUSE_END = r"[了呢啦呀啊哦吧]*\s*(?=[，,。！？!?\n]|$)"
_IMMEDIATE_PROMISE = re.compile(
    _CLAUSE_START + r"(?:这就|现在就|马上|立刻)(?:给你)?(?:唱|哼|发(?:歌|这首|这段|录音))"
)
_ACCEPTED_PROMISE = re.compile(
    _CLAUSE_START + r"(?:唱(?:完就|就是了)|给你(?:唱|哼)(?:两句|一首|这首))"
)
_REPLY_CLAUSE = re.compile(r"([^，,。！？!?\n]+)([，,。！？!?\n]*)")
_DECLINES_SINGING = re.compile(
    _CLAUSE_START + r"(?:这(?:一)?首(?:歌)?(?:就|先|还是)?)?"
    r"(?:不唱(?:不唱)?|不想(?:给你)?唱|不会唱|不给你唱|唱不了|没法唱|不哼)"
    r"(?:歌|这首歌?)?" + _CLAUSE_END
)
_DECLINES_FULL = re.compile(
    _CLAUSE_START + r"(?:(?:整首|完整版)(?:就|先|还是)?(?:不唱|不发|不给你发|没法发)"
    r"|(?:不唱|不发|不给你发|没法发)(?:这首的)?(?:整首|完整版)(?:录音|文件)?)" + _CLAUSE_END
)


def may_request_song(text: str, catalog: dict[str, Song], previous_reply: str) -> bool:
    """决定是否让已有的意图分类器多看一个字段。此处不授权播放。"""
    text = text.strip()
    if not text or _TALKING_ABOUT_SINGING.fullmatch(text) or _CANCELLED.search(text):
        return False
    return bool(
        _HINT.search(text) or find_song(text.strip("《》"), catalog) is not None
        or _FOLLOWUP.search(previous_reply)
    )


def find_song(title: str, catalog: dict[str, Song]) -> Song | None:
    """歌名以文件名为准。只有兼容写法唯一时才回退匹配。"""
    key = song_key(title)
    if key in catalog:
        return catalog[key]
    compact = _compatible_title(key)
    if not compact:
        return None
    matches = [song for name, song in catalog.items() if _compatible_title(name) == compact]
    return matches[0] if len(matches) == 1 else None


def _compatible_title(key: str) -> str:
    return "".join(normalize("NFKC", key).casefold().split())


def song_choice_index(text: str) -> int | None:
    """无序号返回 None，序号含糊或超出备选范围返回 -1。"""
    text = "\n".join(line.strip() for line in text.splitlines()
                     if line.strip() and not _CHOICE_FILLER.fullmatch(line.strip()))
    match = _ORDINAL.match(text)
    if match is None:
        return None
    if not _CHOICE_END.fullmatch(text[match.end():]):
        return -1
    index = match[1]
    if index in "一二三":
        return "一二三".index(index)
    if len(index) == 1 and index.isdecimal() and 1 <= int(index) <= 3:
        return int(index) - 1
    return -1


@dataclass(frozen=True)
class SongOffer:
    prefix: str
    mode: str
    as_file: bool
    titles: tuple[str, ...]

    def render(self, *, prefix: str, as_file: bool) -> str:
        heading = _offer_heading(self.mode, as_file=as_file)
        if prefix and prefix[-1] not in "，,。！!？?：:":
            prefix += "，"
        return prefix + heading + "、".join(f"《{title}》" for title in self.titles)


def _offer_heading(mode: str, *, as_file: bool) -> str:
    if mode == "clip":
        return "换一首？"
    return "要发哪首完整版文件？" if as_file else "要听哪首完整版？"


def read_song_offer(text: str, catalog: dict[str, Song]) -> SongOffer | None:
    """解析一份能按实发顺序选的名单，逐项验证所承诺的版本。"""
    match = _OFFER.search(text)
    if match is None:
        return None
    tail = text[match.end():].strip()
    quoted = _QUOTED_TITLE.findall(tail)
    unquoted = _QUOTED_TITLE.sub("", tail)
    if quoted and not _OFFER_SEPARATOR.sub("", unquoted).strip(" \t\r\n。！!？?～~"):
        labels = [next(group for group in item if group) for item in quoted]
    else:
        labels = [part.strip() for part in _OFFER_SEPARATOR.split(tail) if part.strip()]
    if not 1 <= len(labels) <= 3:
        return None
    mode = "full" if match["full"] else "clip"
    titles = []
    for i, label in enumerate(labels):
        label = label.strip(' \t\r\n"“”《》')
        song = find_song(label, catalog)
        if i == len(labels) - 1:
            # 先试原曲名，最后一项可能又带了句末标点。
            while song is None and label and label[-1] in "。！!？?～~":
                label = label[:-1].rstrip()
                song = find_song(label, catalog)
        if song is None or getattr(song, mode) is None:
            return None
        titles.append(song.title)
    if len(set(titles)) != len(titles):
        return None
    return SongOffer(text[:match.start()], mode, bool(match["file"]), tuple(titles))


def confirmed_choice(index: int, catalog: dict[str, Song], previous_reply: str) -> dict | None:
    """按上一轮实发名单的顺序承接明确序号。"""
    offer = read_song_offer(previous_reply, catalog) if index >= 0 else None
    if offer is None or index >= len(offer.titles):
        return None
    request = {"mode": offer.mode, "title": offer.titles[index]}
    if offer.as_file:
        request["as_file"] = True
    return request


def parse_song_request(raw: object, song_text: str = "") -> dict | None:
    if not isinstance(raw, dict):
        return None
    mode, title = raw.get("mode"), raw.get("title")
    if not isinstance(mode, str) or mode not in _MODES or not isinstance(title, str):
        return None
    title = title.strip().strip("《》〈〉").strip()
    if len(title) > 120 or "\n" in title or "\r" in title:
        return None
    request = {"mode": mode, "title": title}
    if (mode in ("full", "recent_full") and raw.get("as_file") is True
            and _FILE_REQUEST_WORDS.search(song_text) and not _FILE_REFUSAL.search(song_text)):
        request["as_file"] = True
    return request


def declined_song(text: str, mode: str) -> bool:
    """模型同时明确拒唱并带发送标记时，尊重它实际说出的拒绝。"""
    return bool(_DECLINES_SINGING.search(text) or (mode == "full" and _DECLINES_FULL.search(text)))


def remove_song_promise(text: str, *, confirmed_request: bool) -> str | None:
    """返回去掉即时开唱承诺后的正文。疑问句和将来邀约不构成承诺。"""
    retained = []
    found = False
    for clause, punctuation in _REPLY_CLAUSE.findall(text):
        immediate = _IMMEDIATE_PROMISE.match(clause)
        accepted = confirmed_request and _ACCEPTED_PROMISE.match(clause)
        if ((immediate or accepted) and "？" not in punctuation and "?" not in punctuation
                and not clause.rstrip().endswith(("吗", "么", "嘛"))):
            found = True
        else:
            retained.append(clause + punctuation)
    return "".join(retained).strip().rstrip("，,").rstrip() if found else None


def plan_song(request: dict, catalog: dict[str, Song],
              recent: str) -> tuple[str, dict[str, SongDelivery]]:
    """编号只在本轮有效。只有被请求的曲目和版本能有可执行编号。"""
    mode, title = request["mode"], request["title"]
    if mode == "recent_full":
        title = title or recent
        if not title:
            return inj.T("actions", "song_which"), {}
        mode = "full"
    if mode == "other_part":
        song = find_song(title or recent, catalog)
        if song and (song.clip or song.full):
            return inj.T("actions", "song_other_part", title=_display(song.title),
                         available=_display_available(song)), {}
        return inj.T("actions", "song_other_unknown"), {}

    variant = "full" if mode == "full" else "clip"
    as_file = variant == "full" and request.get("as_file") is True
    format_name = "完整版" if variant == "full" else "经典两句"
    available = [s for s in catalog.values() if getattr(s, variant)]
    song = find_song(title, catalog) if title else None
    if mode == "another" and recent:
        available = [s for s in available if song_key(s.title) != song_key(recent)]
        if song is not None and song_key(song.title) == song_key(recent):
            title = ""
    if title:
        if song is None:
            alternatives = "、".join(f"《{s.title}》" for s in available[:3])
            return inj.T("actions", "song_missing", title=_display(title),
                         question=_offer_heading(variant, as_file=as_file),
                         alternatives=alternatives or "暂无"), {}
        if not getattr(song, variant):
            return inj.T("actions", "song_variant_missing", title=_display(song.title),
                         wanted=format_name, available=_display_available(song)), {}
        available = [song]
    if not available:
        if mode == "clip" and catalog:
            alternatives = "、".join(_display(s.title) for s in list(catalog.values())[:3] if s.full)
            return inj.T("actions", "song_only_full", alternatives=alternatives), {}
        key = "song_no_alternative" if mode == "another" else "song_no_variant"
        return inj.T("actions", key, wanted=format_name), {}
    if len(available) > _MAX_OPTIONS:
        available = random.sample(available, _MAX_OPTIONS)

    options = {}
    labels = []
    for number, song in enumerate(available, 1):
        path = getattr(song, variant)
        options[str(number)] = SongDelivery(song.title, variant, path, as_file=as_file)
        labels.append(f"{number} {_display(song.title)}（{format_name}）")
    return inj.T("actions", "song_options", options="、".join(labels),
                 transport="音频文件" if as_file else "QQ 语音条"), options


def unavailable_song_reply(request: dict | None, catalog: dict[str, Song]) -> str:
    """模型给了无法兑现的承诺时，按曲库状态收住这一轮。"""
    if not request:
        return "这次先不唱啦。"
    title = request["title"]
    song = find_song(title, catalog) if title else None
    if title and song is None:
        return f"《{title}》没收录，换一首？"
    if request["mode"] in ("full", "recent_full") and song and not song.full:
        return f"《{song.title}》现在只有经典两句。"
    return "这次先不唱啦。"


def _display(title: str) -> str:
    return json.dumps(title, ensure_ascii=False)


def _display_available(song: Song) -> str:
    return "经典两句和完整版" if song.clip and song.full else ("经典两句" if song.clip else "完整版")
