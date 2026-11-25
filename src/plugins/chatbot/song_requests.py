"""把已确认的私聊点歌意图变成本轮可选的本地音频。"""

import json
import random
import re

from . import injection as inj
from .songs import Song, SongDelivery, song_key

_FILE_REQUEST_WORDS = re.compile(r"文件|附件|(?<![a-z])(?:m4a|mp3|wav|aac|flac|ogg|opus)(?![a-z])", re.I)
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
_MODES = {"clip", "full", "recent_full", "another", "other_part"}
_CLAUSE_START = r"(?:^|[，,。！？!?\n])\s*(?:(?:灰泽满|小满|我|今天|这次|这回|现在|这会儿|先|才|就|还是|真)\s*){0,6}"
_CLAUSE_END = r"[了呢啦呀啊哦吧]*\s*(?=[，,。！？!?\n]|$)"
_DECLINES_SINGING = re.compile(
    _CLAUSE_START + r"(?:不唱(?:不唱)?|不想(?:给你)?唱|不会唱|不给你唱|唱不了|没法唱|不哼)"
    r"(?:歌|这首歌?)?" + _CLAUSE_END
)
_DECLINES_FULL = re.compile(
    _CLAUSE_START + r"(?:整首|完整版)(?:就|先|还是)?(?:不唱|不发|不给你发|没法发)" + _CLAUSE_END
)


def may_request_song(text: str, catalog: dict[str, Song], previous_reply: str) -> bool:
    """决定是否让已有的意图分类器多看一个字段。此处不授权播放。"""
    text = text.strip()
    if not text or _TALKING_ABOUT_SINGING.fullmatch(text) or _CANCELLED.search(text):
        return False
    return bool(
        _HINT.search(text) or song_key(text.strip("《》")) in catalog
        or _FOLLOWUP.search(previous_reply)
    )


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
    if mode in ("full", "recent_full") and raw.get("as_file") is True and _FILE_REQUEST_WORDS.search(song_text):
        request["as_file"] = True
    return request


def declined_song(text: str, mode: str) -> bool:
    """模型同时明确拒唱并带发送标记时，尊重它实际说出的拒绝。"""
    return bool(_DECLINES_SINGING.search(text) or (mode == "full" and _DECLINES_FULL.search(text)))


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
        song = catalog.get(song_key(title or recent))
        if song and (song.clip or song.full):
            return inj.T("actions", "song_other_part", title=_display(song.title),
                         available=_display_available(song)), {}
        return inj.T("actions", "song_other_unknown"), {}

    variant = "full" if mode == "full" else "clip"
    as_file = variant == "full" and request.get("as_file") is True
    format_name = "完整版" if variant == "full" else "经典两句"
    available = [s for s in catalog.values() if getattr(s, variant)]
    if mode == "another" and recent:
        available = [s for s in available if song_key(s.title) != song_key(recent)]
        if title and song_key(title) == song_key(recent):
            title = ""
    if title:
        song = catalog.get(song_key(title))
        if song is None:
            alternatives = "、".join(_display(s.title) for s in available[:3])
            return inj.T("actions", "song_missing", title=_display(title),
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


def _display(title: str) -> str:
    return json.dumps(title, ensure_ascii=False)


def _display_available(song: Song) -> str:
    return "经典两句和完整版" if song.clip and song.full else ("经典两句" if song.clip else "完整版")
