"""把本轮相关的记忆来源还原成同一私聊的历史回复。"""
import json
import re
from datetime import datetime, timedelta, timezone

from . import event_memory, reply_evidence
from .addressing import author_text
from .memory_v2 import _NAME_KEYS, _identity_record_supported, _redact_sensitive_excerpt

_CST = timezone(timedelta(hours=8))
_PAST = re.compile(
    r"刚才|刚刚|上一次|上次|上回|上周|上个月|之前|以前|先前|当时|曾经|昨天|昨晚|前天|"
    r"(?:说|讲|聊|提|答应|同意|承诺)过|原话"
)
_QUESTION = re.compile(r"[？?]|吗|么|啥|来着|没有|没[呀啊]?[。！!]*$|记得|到底|什么时候|什么|怎么|几点|哪[天次句]$")
_PROMISE = re.compile(r"答应|承诺|约定|说好|说过要|提醒")
_DECISION = re.compile(r"答应|同意|拒绝|承认|允许|说|回应|回[的了我]|怎么叫|叫过|原话")
_STATUS = re.compile(
    r"还(?:有效|算数|算不算数|作数|在吗)|现在还(?:算|能|会)|"
    r"(?:发|提醒|通知)(?:过我?了?|我?了)(?:吗|没)|有没有(?:发|提醒|通知)(?:过|我|到)"
)
_REMINDER = re.compile(r"提醒|通知|到点叫")
_REMINDER_DELIVERY = re.compile(r"(?:发|发出|提醒|通知)(?:过|了)|有没有(?:发|提醒|通知)|怎么提醒")
_NICKNAME = re.compile(r"(?:叫|喊|称呼)你|(?:我|他)叫过|称呼|昵称")
_QUOTED_NICKNAME = re.compile(r'[「『“"]([\w·-]{1,20})[」』”"]')
_COMMON_NAMES = ("姐姐", "妈妈", "灰泽满")
_ADDRESS_HISTORY = re.compile(
    r"(?:怎么|怎样|如何)(?:称呼|叫|喊)(我|你|灰泽满)(?!自己)"
    r"|(?:称呼|叫|喊)(?:过)?(我|你|灰泽满)(?:为)?(?:什么|啥|哪些)"
)
_ALIAS_QUESTION = re.compile(
    r"^(?:那(?:个)?|所以|请问|(?:你|我)(?:刚才)?说的)?\s*"
    r"[「『“\"]?([\w·-]{1,20}?)[」』”\"]?\s*"
    r"(?:是谁|(?:是)?指(?:的)?谁)(?:呀|啊|呢|来着)?[？?。]*$",
    re.I,
)
_ALIAS_CONFIRMATION = re.compile(
    r"^(?:那|所以|请问|(?:你|我)(?:刚才)?说的)?\s*"
    r"[「『“\"]?([\w·-]{1,20}?)[」』”\"]?\s*"
    r"(?:是(?:指|在叫)?|指的是|是不是(?:指)?)\s*(?:hzm|hazel|灰泽满|小满|你|我)"
    r"(?:(?:呀|啊|吗|嘛|么|呢|来着)[？?。]*|[？?]+)$",
    re.I,
)
_DIRECT_HISTORY = re.compile(r"原话|聊天记录|对话记录|(?:上|前)一(?:句|条|轮)")
_QUOTE_HEADER = re.compile(r"\[引用[^\]\n]*\]")
_SPEECH = re.compile(
    r"(?:(?:跟|和|对|给|向)(?:我|你))?(?:"
    r"(?<![小听解传据])说(?!唱|服|明书|法)|(?<!无)聊(?!天记录)(?:到|过|了)?|讲(?:了|过|到|的)|"
    r"提(?:过|到|起|了|的)|问[了过到的我你]|告诉(?:过|了)?|发(?:过|了|的)(?!烧)|"
    r"回[我你的了过]|(?:回复|回答|回覆)(?:了|过|的)?)"
)
_MEMORY_QUESTION = re.compile(r"(?:记得|记不记得)(我们|灰泽满|[我你])(?=[^\s吗么呢呀啊嘛？?。！!])")
_PUBLIC_CONTEXT = re.compile(r"直播(?:里|时|的时候|上)|录播里|视频里|动态里|微博上")
_QUOTED_TEXT = re.compile(r'[「『“"]([^」』”"\n]{2,80})[」』”"]')
_ACTOR = re.compile(r"(?:我们|我俩|灰泽满|[你我])")
_ABS_DATE = re.compile(r"(?:(\d{4})[年/-])?(\d{1,2})(?:月|[-/])(\d{1,2})(?:日|号)?")
_REL_DATE = re.compile(r"今天|今日|昨晚|昨天|昨日|前天")
_NEAR = re.compile(r"刚才|刚刚|(?:上|前)一(?:条|句|轮)")
_LAST = re.compile(r"上一次|上次|上回|刚才|刚刚|(?:上|前)一(?:条|句|轮)")
_APPROXIMATE = re.compile(r"左右|前后|大概|大约|差不多")
_NUMERALS = "零一二三四五六七八九"
_CLOCK = re.compile(
    r"(?<!\d)(\d{1,2}|[零〇一二两三四五六七八九十]{1,3})"
    r"(?:[:：](\d{2})(?!\d)|点(?:钟)?(半|一刻|三刻|\d{1,2}|[零〇一二两三四五六七八九十]{1,3})?分?)"
)
_HOURS = {
    "凌晨": (0, 6), "清晨": (5, 9), "早上": (6, 12), "上午": (6, 12),
    "中午": (11, 14), "下午": (12, 19), "傍晚": (17, 20), "晚上": (18, 24), "夜里": (18, 24),
}
_PERIOD = re.compile("|".join(_HOURS))
_TOPIC_WORDS = re.compile(
    r"能不能|还记得|记不记得|记得|你说|我说|是不是|有没有|有没|你还|我还|"
    r"帮我|翻一下|找一下|翻看|查查|看一下|给我|来着|说了啥|啥(?:事|内容)?|"
    r"聊天记录|对话记录|这句话|那句话|这件事|那件事|原话|哪一句|哪一条|哪[句条天儿]|"
    r"为什么|什么时候|什么意思|怎么回事|什么(?:东西|内容)?|怎么[讲说回答复]*|几点|到底|"
    r"刚才|刚刚|上一次|上次|上回|上周|上个月|(?:上|前)一(?:句|条|轮)|之前|以前|先前|当时|曾经|"
    r"今天|今日|昨晚|昨天|昨日|前天|凌晨|清晨|早上|上午|中午|下午|傍晚|晚上|夜里"
)
_TOPIC_NOISE = re.compile(
    r"还记得|是不是|有没有|什么时候|一件事|这件事|那件事|什么(?:事|内容)?|哪件事|"
    r"还有效|还算不算数|还算数|还作数|还在吗|还会|发了吗|几点钟|几点|"
    r"刚刚|刚才|上次|上回|之前|以前|先前|当时|曾经|原话|"
    r"答应|承诺|约定|说好|说过|提醒|通知|到点叫|到底|"
    r"[你我她他它过的了是吗么呢啊呀吧嘛在会要给说没]|[^\w]"
)
_NEW_REQUEST = re.compile(
    r"(?:^|[，,。！!；;？?\n])\s*(?:那(?:就)?|还是|不过|顺便)?\s*"
    r"(?:(?:改成|改到|提前到|推迟到|取消|不用了)|"
    r"(?:再|帮我|麻烦|记得|请|别|不用|明天|后天|今天|今晚)"
    r"[^，,。！!；;？?\n]*(?:提醒|通知|到点叫))"
)
_STATES = {
    "active": "仍在有效期内",
    "confirmed": "仍在有效期内",
    "done": "已发送提醒",
    "legacy_done": "已结束，旧记录未区分发送和跳过",
    "missed": "到期后跳过，未发送",
    "expired": "已过有效期",
    "superseded": "后来被更新或撤回",
    "retracted": "已撤回",
    "cancelled": "已撤回",
}


def _looking_back(query: str) -> bool:
    return bool(_QUESTION.search(query) and (_PAST.search(query) or _STATUS.search(query)))


def _clarified_alias(query: str) -> str:
    query = author_text(query)
    match = _ALIAS_QUESTION.fullmatch(query) or _ALIAS_CONFIRMATION.fullmatch(query)
    return match[1] if match and match[1] not in {"你", "我", "他", "她", "那", "这"} else ""


def _historical_address_role(query: str) -> str:
    query = author_text(query)
    match = _ADDRESS_HISTORY.search(query) if _looking_back(query) else None
    if not match:
        return ""
    recipient = match[1] or match[2]
    actor = _speaker(query[:match.start()])
    if recipient == "我":
        return "assistant" if actor in {"", "你", "灰泽满"} else ""
    return "user" if actor in {"", "我"} else ""


def is_address_question(query: str) -> bool:
    """识别方向明确的历史称呼追问，以及精确到一个称呼的指代问题。"""
    return bool(_historical_address_role(query) or _clarified_alias(query))


def _quoted_question(query: str, reference_ids) -> bool:
    return bool(reference_ids and (_QUESTION.search(_QUOTE_HEADER.sub("", query)) or _clarified_alias(query)))


def needs_recall(query: str, *, reference_ids=()) -> bool:
    """明确复述或追问私聊历史时打开回查。"""
    if _quoted_question(query, reference_ids):
        return True
    query = _QUOTE_HEADER.sub("", query)
    if _historical_address_role(query):
        return True
    speech = _SPEECH.search(query)
    if _PUBLIC_CONTEXT.search(query) and not re.search(r"(?:跟|和|对|给|向)(?:我|你)(?:说|讲|聊|提)", query):
        return False
    if _DIRECT_HISTORY.search(query) and (
        _QUESTION.search(query) or _REL_DATE.search(query) or _ABS_DATE.search(query)
        or re.search(r"翻|查|复述|给我看|找一下|不是", query)
    ):
        return True
    if not (_QUESTION.search(query) or re.search(r"明明|你不是说|你不是.*说过", query)):
        return False
    if _looking_back(query) and (_NICKNAME.search(query) or _PROMISE.search(query)):
        return True
    if _MEMORY_QUESTION.search(query):
        return True
    return bool(speech and (_PAST.search(query) or _ABS_DATE.search(query)
                            or _REL_DATE.search(query) or _PERIOD.search(query) or _CLOCK.search(query)))


def is_commitment_question(query: str) -> bool:
    """纯回查提醒时避免 L3 将历史中的请求再次当作新提醒。"""
    return bool(_looking_back(query) and _REMINDER.search(query) and not _NEW_REQUEST.search(query))


def _quote(text: str, limit: int, terms=()) -> str:
    text = _redact_sensitive_excerpt(text)
    lowered = text.casefold()
    hit = min((index for term in terms if (index := lowered.find(term.casefold())) >= 0), default=0)
    offset = max(0, hit - limit // 3) if hit >= limit else 0
    quoted = json.dumps(text[offset:offset + limit], ensure_ascii=False)
    return quoted + ("（节选）" if len(text) > limit else "")


def _stamp(timestamp: float | None) -> str:
    try:
        return datetime.fromtimestamp(timestamp, _CST).strftime("%Y-%m-%d %H:%M")
    except (TypeError, OverflowError, OSError, ValueError):
        return "时间未确认"


def _render_turn(turn: dict) -> str:
    stamp = _stamp(turn["t"])
    if turn.get("received") and not turn["reply"]:
        return f"{stamp}（QQ 用户消息）用户 {_quote(turn['user'], 140)}"
    if not turn["verified"]:
        origin = "旧日志线索，发送未确认"
    elif any(part["modality"] == "voice" for part in turn["parts"]):
        origin = "QQ 已确认发送，语音以合成文本为准"
    else:
        origin = "QQ 已确认发送"
    return (f"{stamp}（{origin}）用户 {_quote(turn['user'], 140)}"
            f"，灰泽满 {_quote(turn['reply'], 320)}")


def _clock_number(text: str) -> int:
    if text.isdecimal():
        return int(text)
    text = text.replace("两", "二").replace("〇", "零")
    if "十" in text:
        tens, _, ones = text.partition("十")
        return 10 * (_NUMERALS.index(tens) if tens else 1) + (_NUMERALS.index(ones) if ones else 0)
    return int("".join(str(_NUMERALS.index(digit)) for digit in text))


def _history_window(prefix: str, now: datetime) -> tuple[float | None, float | None]:
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    date_match = _ABS_DATE.search(prefix)
    relative = _REL_DATE.search(prefix)
    period = _PERIOD.search(prefix)
    clock = _CLOCK.search(prefix)
    if date_match:
        year, month, date = date_match.groups()
        try:
            day = day.replace(year=int(year) if year else day.year, month=int(month), day=int(date))
        except ValueError:
            # 日期无效时避免扩大为无日期检索。
            return day.timestamp(), day.timestamp()
        if not year and day.date() > now.date():
            try:
                day = day.replace(year=day.year - 1)
            except ValueError:
                return day.timestamp(), day.timestamp()
    elif relative:
        delta = {"昨天": 1, "昨日": 1, "昨晚": 1, "前天": 2}.get(relative[0], 0)
        day -= timedelta(days=delta)
    elif "上周" in prefix:
        end = day - timedelta(days=day.weekday())
        return (end - timedelta(days=7)).timestamp(), end.timestamp()
    elif "上个月" in prefix:
        end = day.replace(day=1)
        return (end - timedelta(days=1)).replace(day=1).timestamp(), end.timestamp()
    elif _NEAR.search(prefix):
        return (now - timedelta(hours=2)).timestamp(), now.timestamp()
    elif not (period or clock):
        return None, None

    first, last = _HOURS[period[0]] if period else ((18, 24) if relative and relative[0] == "昨晚" else (0, 24))
    if clock:
        hours, colon_minutes, chinese_minutes = clock.groups()
        raw_minutes = colon_minutes or chinese_minutes
        try:
            hour = _clock_number(hours)
            minute = {"半": 30, "一刻": 15, "三刻": 45}.get(raw_minutes)
            if minute is None:
                minute = _clock_number(raw_minutes) if raw_minutes else 0
        except ValueError:
            return day.timestamp(), day.timestamp()
        if first >= 12 and 0 < hour < 12:
            hour += 12
        if hour > 23 or minute > 59:
            return day.timestamp(), day.timestamp()
        start = day + timedelta(hours=hour, minutes=minute)
        if _APPROXIMATE.search(prefix):
            margin = timedelta(minutes=15)
            return (start - margin).timestamp(), (start + margin).timestamp()
        end = start + (timedelta(minutes=1) if raw_minutes else timedelta(hours=1))
        return start.timestamp(), end.timestamp()
    return (day + timedelta(hours=first)).timestamp(), (day + timedelta(hours=last)).timestamp()


def _speaker(prefix: str) -> str:
    last_clause = re.split(r"[。！？?!；;\n]", prefix)[-1]
    actors = _ACTOR.findall(last_clause)
    return actors[-1] if actors else ""


def _history_query(query: str, now: datetime) -> tuple[str, tuple[str, ...], float | None, float | None]:
    # 日期只从说话动作之前取。「我之前说昨晚发烧」的昨晚描述发烧。
    query = _QUOTE_HEADER.sub("", query)
    bare = _QUOTED_TEXT.sub("", query)
    speech = _SPEECH.search(bare)
    anchor = speech or _MEMORY_QUESTION.search(bare)
    prefix = bare[:anchor.start()] if anchor else bare
    actor = anchor[1] if anchor and not speech else _speaker(prefix)
    if not anchor and re.search(r"聊天记录|对话记录", bare):
        actor = ""
    role = "user" if actor == "我" else "assistant" if actor in {"你", "灰泽满"} else ""

    terms = tuple(_QUOTED_TEXT.findall(query)[:3])
    if not terms:
        tail = bare[anchor.end():] if anchor else _CLOCK.sub("", _ABS_DATE.sub("", bare))
        cleaned = _TOPIC_WORDS.sub(" ", tail)
        words = [part.strip("你我他她们说的是有过了么吗呢吧嘛啊又就曾请")
                 for part in re.findall(r"[\w·-]+", cleaned)]
        terms = tuple(dict.fromkeys(part[:80] for part in words
                                   if len(part) >= 2 and part not in {"喜欢", "聊聊", "讨论"}))[:3]
    since, until = _history_window(prefix, now)
    return role, terms, since, until


_MEMORY_STATE = {
    "current": "目前仍在记忆中",
    "past": "历史事实",
    "superseded": "已被更新或撤回",
    "expired": "已过有效期",
}


def _address_records(store, user_id: str):
    for memory in store.snapshot(user_id).get("memories", []):
        key = memory.get("key")
        if memory.get("kind") != "fact" or (key != "her_nickname" and key not in _NAME_KEYS):
            continue
        if memory.get("status") not in {"confirmed", "active", "superseded", "retracted"}:
            continue
        if not _identity_record_supported(memory):
            continue
        aliases = [part.strip().strip("「」『』\"“”'") for part in
                   re.split(r"[、,，/]", str(memory.get("value") or ""))]
        yield ("user" if key == "her_nickname" else "assistant"), aliases, memory


def _alias_identity_context(store, user_id: str, alias: str) -> str:
    directions = {"user": "用户对灰泽满的历史称呼", "assistant": "对用户的姓名或称呼记录"}
    lines = []
    refs = []
    for role, aliases, record in _address_records(store, user_id):
        if alias.casefold() not in {name.casefold() for name in aliases}:
            continue
        description = f"{directions[role]}提到 {_quote(alias, 40)}。"
        if description not in lines:
            lines.append(description)
        refs.extend(response["turn_id"] for response in record.get("nickname_responses", [])
                    if response.get("turn_id"))
        if turn_id := (record.get("source") or {}).get("turn_id"):
            refs.append(turn_id)
    if not lines:
        return ""
    turns = reply_evidence.recall_turns(user_id, turn_ids=refs, limit=3) if refs else []
    lines.extend(_render_turn(turn) for turn in turns)
    if not turns:
        lines.append("该称呼的对应原话未核验。")
    return (
        "【原话回查】\n按这些资料里的说话人和称呼对象理解称呼。"
        "同一个词可用于多个对象，解释具体一句时以那句话为准。\n" + "\n".join(lines)
    )


def _general_history(store, user_id: str, query: str, now: datetime, reference_ids=()) -> tuple[list[str], bool]:
    role, terms, since, until = ("", (), None, None) if reference_ids else _history_query(query, now)
    literal = bool(_QUOTED_TEXT.search(query))
    address_role = _historical_address_role(query) if not reference_ids else ""
    if address_role:
        role = address_role
        terms = tuple(_QUOTED_NICKNAME.findall(query)[:3]) or tuple(dict.fromkeys(
            name for owner, aliases, _ in _address_records(store, user_id) if owner == role
            for name in aliases if name
        ))[:6]
    topic = " ".join(terms)
    memories = []
    events = []
    if topic and role != "assistant" and not (literal or address_role):
        memories = store.history_references(user_id, topic, limit=2)
        events = event_memory.history_references(user_id, topic, limit=2)
    refs = reference_ids or tuple(dict.fromkeys(
        item["turn_id"] for item in (*memories, *events) if item.get("turn_id")))
    turns = reply_evidence.recall_turns(
        user_id, turn_ids=refs, terms=terms, role=role, since=since, until=until,
        limit=1 if not terms and not reference_ids and _LAST.search(query) else 3,
        include_legacy=not reference_ids, fuzzy=not (literal or address_role),
        use_recent=not terms and not reference_ids and bool(
            since is not None or _DIRECT_HISTORY.search(query) or _SPEECH.search(query) or address_role),
    )
    lines = []
    if since is None and until is None:
        lines.extend(f"记忆摘要 {_quote(item['text'], 110)}（{_MEMORY_STATE[item['status']]}）。"
                     for item in memories)
        if not turns:
            lines.extend(f"事件摘要 {_quote(item['text'], 110)}。" for item in events)
    grounded = False
    for turn in turns:
        incoming = turn.get("incoming") or []
        parts = turn["parts"]
        if reference_ids:
            incoming = [event for event in incoming if event.get("message_id") in reference_ids]
            parts = [part for part in parts if part.get("message_id") in reference_ids]
        if role != "assistant":
            matching = [event for event in incoming if event.get("text")
                        and _in_window(event.get("t"), since, until)]
            if terms:
                matching.sort(key=lambda event: any(term.casefold() in event["text"].casefold() for term in terms),
                              reverse=True)
            for event in matching[:3]:
                lines.append(f"{_stamp(event['t'])}（QQ 用户消息）用户 {_quote(event['text'], 220, terms)}")
            if matching:
                grounded = True
            elif (not reference_ids and not turn.get("incoming") and turn["user"]
                  and _in_window(turn["t"], since, until)):
                origin = "同轮记录" if turn["verified"] else "旧日志线索，时间/发送未单独确认"
                lines.append(f"{_stamp(turn['t'])}（{origin}）用户 {_quote(turn['user'], 180, terms)}")
        if role != "user" and turn["reply"]:
            parts = [part for part in parts if _in_window(part["t"], since, until)]
            if parts:
                origin = ("QQ 已确认发送，语音以合成文本为准"
                          if any(part["modality"] == "voice" for part in parts) else "QQ 已确认发送")
                reply = "\n".join(part["text"] for part in parts)
                lines.append(f"{_stamp(parts[0]['t'])}（{origin}）灰泽满 {_quote(reply, 320, terms)}")
                grounded = True
            elif not turn["verified"]:
                lines.append(f"{_stamp(turn['t'])}（旧日志线索，发送未确认）灰泽满 {_quote(turn['reply'], 320, terms)}")
    return lines, grounded


def _in_window(timestamp: float | None, since: float | None, until: float | None) -> bool:
    return (since is None and until is None) or (
        timestamp is not None and (since is None or timestamp >= since) and (until is None or timestamp < until)
    )


def build_recall_context(store, user_id: str, query: str, *,
                         nicknames=(), nickname_turn_ids=(), reference_ids=(),
                         now: datetime | None = None) -> str:
    """仅在本轮明确追问时附上相关来源及独立的当前状态。"""
    looking_back = _looking_back(query)
    quoted = _quoted_question(query, reference_ids)
    alias = _clarified_alias(query)
    address_role = _historical_address_role(query)
    if alias and not quoted:
        return _alias_identity_context(store, user_id, alias)
    if not (needs_recall(query, reference_ids=reference_ids) or (looking_back and nicknames)):
        return ""
    nickname_query = looking_back and not (address_role or is_commitment_question(query)) and (
        _NICKNAME.search(query) or (nicknames and _DECISION.search(query))
    )
    speech = _SPEECH.search(query)
    user_speech = speech and _speaker(query[:speech.start()]) == "我"
    lines = []
    if nickname_query and not quoted:
        refs = nickname_turn_ids
        terms = nicknames or (*_QUOTED_NICKNAME.findall(query),
                              *(name for name in _COMMON_NAMES if name in query))
    elif looking_back and _PROMISE.search(query) and not (quoted or user_speech or address_role):
        topic = _TOPIC_NOISE.sub("", query)
        records = store.commitment_history(user_id, topic)
        refs = tuple(m["source"]["turn_id"] for m in records if (m.get("source") or {}).get("turn_id"))
        if _REMINDER_DELIVERY.search(query):
            delivery_refs = tuple(m["reminder_turn_id"] for m in records if m.get("reminder_turn_id"))
            refs = (*delivery_refs, *refs)
        terms = tuple(record["value"] for record in records) or ((topic,) if len(topic) >= 2 else ())
        for record in records:
            origin = ("约定记录" if (record.get("source") or {}).get("delivery_verified")
                      else "旧约定记录（原回复发送未确认）")
            status = _STATES.get(record.get("status"), "状态未确认")
            due = str(record.get("due_at") or "").replace("T", " ")[:16]
            due_text = f"（约定时间 {due}）" if due else ""
            lines.append(f"{origin} {_quote(record['value'], 150)}{due_text}，{status}。")
    else:
        current = now or datetime.now(_CST)
        current = current.replace(tzinfo=_CST) if current.tzinfo is None else current.astimezone(_CST)
        lines, grounded = _general_history(store, user_id, query, current, reference_ids if quoted else ())
        if not grounded:
            lines.append("未找到可核验的对应原话。可以说明能查到的范围，避免凭摘要复述原话。")
        return (
            "【原话回查】\n这些是同一私聊的历史资料。按日期和说话人引用原话。"
            "摘要和旧日志只作线索，当前状态以对应的有效记录为准。"
            + ("按引文里的说话人、称呼对象和当时的直接用法理解称呼。"
               "同一个词可用于多个对象。" if alias or address_role else "")
            + "\n" + "\n".join(lines)
        )

    turns = reply_evidence.recall_turns(
        user_id, turn_ids=refs, terms=terms if not refs else (),
        limit=3 if nickname_query and refs else 2, include_legacy=bool(terms) and not refs,
    ) if refs or terms else []
    lines.extend(_render_turn(turn) for turn in turns)
    if not any(turn["verified"] for turn in turns):
        lines.append("未找到能确认发送的对应原话。可以说明能查到的范围，避免断言以前从未说过。")
    return (
        "【原话回查】\n"
        "以下对话是历史引文。只用与当前问题对应的内容作答，旧日志只作线索。"
        "以前的同意按当时的范围理解，承诺目前的状态以约定记录为准，回查本身不会续约。\n"
        + "\n".join(lines)
    )
