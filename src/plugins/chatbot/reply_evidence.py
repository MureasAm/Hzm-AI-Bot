"""保存同一私聊中收到的发言与经 QQ 接口确认发送的回复。"""
import json
import math
import re
import threading
import time
from pathlib import Path

from . import chatlog

EVIDENCE_DIR: Path | None = None
MAX_BYTES = 1024 * 1024
KEEP_ARCHIVES = 3
_MAX_ROW_BYTES = 64 * 1024
_LEGACY_SCAN_BYTES = 256 * 1024
_PRIVATE_ID = re.compile(r"[A-Za-z0-9_-]{1,80}")
_HAN_WORD = re.compile(r"[\u3400-\u9fff]{2,}")
_HISTORY_FILLER = re.compile(
    r"[你我她他]?(?:以前|之前|上次|刚刚|刚才|当时|曾经)?"
    r"(?:(?:跟|和|对)[我你她他])?(?:说过|说了|提过|还记得|记得|聊过)|"
    r"是不是|有没有|怎么说|什么来着|[那这](?:个|件事)|原话|之前|上次|刚才|以前|当时|后来"
)
_PARTICLES = re.compile(r"[你我她他它的了的是吗么呢啊呀吧嘛也还过]+")
_WRITE_LOCK = threading.Lock()


def _evidence_file(user_id: str) -> Path:
    directory = EVIDENCE_DIR if EVIDENCE_DIR is not None else chatlog.LOG_DIR / "reply_evidence"
    return directory / f"private-{user_id}.jsonl"


def _archives(path: Path) -> list[Path]:
    files = []
    for candidate in path.parent.glob(f"{path.stem}.*.jsonl"):
        suffix = candidate.stem.removeprefix(f"{path.stem}.")
        if suffix.isascii() and suffix.isdecimal():
            files.append((int(suffix), candidate))
    return [candidate for _, candidate in sorted(files, reverse=True)]


def _rotate(path: Path) -> None:
    if path.exists() and path.stat().st_size >= MAX_BYTES:
        path.rename(path.with_suffix(f".{time.time_ns()}.jsonl"))
        for archive in _archives(path)[KEEP_ARCHIVES:]:
            archive.unlink(missing_ok=True)


def record_part(user_id: str, turn_id: str, part_index: int, text: str, *,
                user_text: str = "", incoming=(), modality: str = "text", message_id: str = "") -> bool:
    """在发送回执后写一段。日志关闭、数据无效、磁盘不可写时返回 False。"""
    if (type(part_index) is not int or part_index < 0
            or not isinstance(text, str) or not text.strip()
            or modality not in ("text", "voice", "song")):
        return False
    return _record(user_id, turn_id, user_text, incoming, {
        "part_index": part_index, "text": text, "modality": modality, "message_id": str(message_id or ""),
    })


def record_incoming(user_id: str, turn_id: str, user_text: str, *, incoming=()) -> bool:
    """保存本轮已收到的用户发言，允许本轮没有成功发送的回复。"""
    if not isinstance(user_text, str) or not (user_text.strip() or incoming):
        return False
    return _record(user_id, turn_id, user_text, incoming, {"kind": "incoming"})


def _record(user_id: str, turn_id: str, user_text: str, incoming, fields: dict) -> bool:
    user_id = str(user_id)
    if (not chatlog.chatlog_enabled() or not _PRIVATE_ID.fullmatch(user_id)
            or not isinstance(turn_id, str) or not turn_id or len(turn_id) > 120):
        return False
    try:
        row = {"user_id": user_id, "turn_id": turn_id, "t": time.time(),
               "user": user_text, "incoming": list(incoming or ()), **fields}
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(encoded) > _MAX_ROW_BYTES:
            return False
        path = _evidence_file(user_id)
        with _WRITE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                _rotate(path)
            except OSError:
                pass
            with path.open("a+b") as stream:
                if stream.tell():
                    stream.seek(-1, 2)
                    if stream.read(1) != b"\n":
                        stream.write(b"\n")    # 崩溃留下的半行不能吞掉下一条有效记录
                stream.write(encoded)
    except (OSError, UnicodeError, TypeError, ValueError):
        return False
    return True


def _rows(path: Path, max_bytes: int):
    try:
        with path.open("rb") as stream:
            offset = max(0, stream.seek(0, 2) - max_bytes)
            if offset:
                stream.seek(offset - 1)
                if stream.read(1) != b"\n":
                    stream.readline()           # 跳过从字节窗口中间切开的 JSON 行
            else:
                stream.seek(0)
            for line in stream:
                try:
                    row = json.loads(line)
                except (UnicodeError, ValueError):
                    continue
                if isinstance(row, dict):
                    yield row
    except OSError:
        return


def _timestamp(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        stamp = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return stamp if math.isfinite(stamp) else None


def _archive_turns(user_id: str):
    path = _evidence_file(user_id)
    grouped: dict[str, dict[int, dict]] = {}
    received: dict[str, dict] = {}
    for source in [path, *_archives(path)[:KEEP_ARCHIVES]]:
        for row in _rows(source, MAX_BYTES + _MAX_ROW_BYTES):
            turn_id = row.get("turn_id")
            stamp = _timestamp(row.get("t"))
            if (row.get("user_id") != user_id or not isinstance(turn_id, str) or not turn_id
                    or stamp is None):
                continue
            if row.get("kind") == "incoming":
                if turn_id not in received or stamp < received[turn_id]["t"]:
                    received[turn_id] = dict(row, t=stamp)
                continue
            index, text = row.get("part_index"), row.get("text")
            if (type(index) is not int or index < 0 or not isinstance(text, str) or not text.strip()
                    or row.get("modality") not in ("text", "voice", "song")):
                continue
            parts = grouped.setdefault(turn_id, {})
            if index not in parts or stamp < parts[index]["t"]:
                parts[index] = dict(row, t=stamp)

    for turn_id in grouped.keys() | received.keys():
        parts = grouped.get(turn_id, {})
        ordered = [parts[index] for index in sorted(parts)]
        inbound = received.get(turn_id)
        rows = [inbound, *ordered] if inbound else ordered
        references = {turn_id}
        references.update(str(part["message_id"]) for part in ordered if part.get("message_id"))
        incoming = []
        seen = set()
        for row in rows:
            events = row.get("incoming")
            for event in events if isinstance(events, list) else ():
                if not isinstance(event, dict):
                    continue
                message_id = str(event.get("message_id") or "")
                stamp = _timestamp(event.get("t", event.get("ts")))
                text = event.get("text")
                text = text if isinstance(text, str) else ""
                if not (message_id or stamp is not None or text):
                    continue
                key = message_id or (stamp, text)
                if key not in seen:
                    seen.add(key)
                    incoming.append({"message_id": message_id, "t": stamp, "text": text})
                if message_id:
                    references.add(message_id)
        turn_time = (min(part["t"] for part in ordered) if ordered
                     else min((row["t"] for row in incoming if row["t"] is not None), default=inbound["t"]))
        turn = {
            "turn_id": turn_id,
            "t": turn_time,
            "user": next((row["user"] for row in rows if isinstance(row.get("user"), str) and row["user"]), ""),
            "reply": "\n".join(part["text"] for part in ordered),
            "parts": [{key: part.get(key, "") for key in ("part_index", "t", "text", "modality", "message_id")}
                      for part in ordered],
            "incoming": incoming,
            "received": bool(inbound or incoming),
            "verified": bool(ordered),
        }
        yield turn, references


def _legacy_turns(user_id: str):
    path = chatlog.LOG_DIR / f"{user_id}.jsonl"
    for source in [path, *_archives(path)[:KEEP_ARCHIVES]]:
        for row in _rows(source, _LEGACY_SCAN_BYTES):
            if (str(row.get("session")) != user_id or row.get("kind") != "private"
                    or row.get("reply_evidence_turn_id")):
                continue
            stamp = _timestamp(row.get("t"))
            if stamp is not None and isinstance(row.get("reply"), str) and row["reply"].strip():
                user = row.get("user")
                yield {"turn_id": "", "t": stamp, "user": user if isinstance(user, str) else "",
                       "reply": row["reply"], "parts": [], "incoming": [],
                       "received": False, "verified": False}


def _search_text(turn: dict, role: str, since: float | None, until: float | None) -> tuple[str, float] | None:
    messages = []
    if role != "assistant":
        incoming = turn["incoming"]
        if incoming:
            messages.extend((message["text"], message["t"]) for message in incoming)
        elif turn["user"]:
            # 旧日志只有整轮时间可以参考。现代记录用接收时间筛，回复回执不能代替它。
            stamp = None if turn["verified"] or turn["received"] else turn["t"]
            messages.append((turn["user"], stamp))
    if role != "user":
        if turn["parts"]:
            messages.extend((part["text"], part["t"]) for part in turn["parts"])
        elif turn["reply"]:
            messages.append((turn["reply"], turn["t"]))
    if since is not None:
        messages = [(text, stamp) for text, stamp in messages if stamp is not None and since <= stamp]
    if until is not None:
        messages = [(text, stamp) for text, stamp in messages if stamp is not None and stamp < until]
    if not messages:
        return None
    return ("\n".join(text for text, _ in messages).casefold(),
            max((stamp for _, stamp in messages if stamp is not None), default=turn["t"]))


def _han_bigrams(text: str) -> set[str]:
    return {word[index:index + 2]
            for part in _HISTORY_FILLER.split(text)
            for word in _HAN_WORD.findall(part)
            for index in range(len(word) - 1)}


def recall_turns(user_id: str, *, turn_ids=(), terms=(), limit: int = 2,
                 include_legacy: bool = False, role: str = "",
                 since: float | None = None, until: float | None = None,
                 use_recent: bool = True, fuzzy: bool = False) -> list[dict]:
    """查同一私聊的轮次。先按 ID，其次按限定说话人与时间的文本命中。"""
    user_id = str(user_id)
    if (not chatlog.chatlog_enabled() or not _PRIVATE_ID.fullmatch(user_id)
            or limit <= 0 or role not in ("", "user", "assistant")):
        return []
    refs = {ref: i for i, ref in enumerate(dict.fromkeys(str(ref) for ref in turn_ids if ref))}
    raw_tokens = {str(term).strip().casefold() for term in terms if str(term).strip()}
    tokens = ({token for text in raw_tokens for part in _HISTORY_FILLER.split(text)
               if len(token := part.strip(" ？?。，、：:!！\n")) >= 2 and not _PARTICLES.fullmatch(token)}
              if fuzzy else raw_tokens)
    grams = [_han_bigrams(token) for token in tokens] if fuzzy else []
    recent = not (refs or raw_tokens) and (use_recent or since is not None or until is not None)
    if not (refs or tokens or recent):
        return []

    def score(body: str) -> tuple[int, int, int]:
        lengths = [len(token) for token in tokens if token in body]
        if lengths:
            return 2, max(lengths), len(lengths)
        if grams:
            body_grams = _han_bigrams(body)
            matching = [(len(term_grams & body_grams), len(term_grams)) for term_grams in grams]
            best = max((count for count, total in matching if count >= 2 and count * 2 >= total), default=0)
            if best:
                return 1, best, 0
        return 0, 0, 0

    exact = []
    relevant = []
    for turn, references in _archive_turns(user_id):
        searchable = _search_text(turn, role, since, until)
        if searchable is None:
            continue
        body, stamp = searchable
        rank = min((refs[reference] for reference in references if reference in refs), default=None)
        if rank is not None:
            exact.append((rank, stamp, turn))
            continue
        match = score(body)
        if recent or match[0]:
            relevant.append((match, True, stamp, turn))

    exact.sort(key=lambda item: (item[0], -item[1]))
    selected = [turn for _, _, turn in exact[:limit]]
    if len(selected) >= limit:
        return selected
    if include_legacy and (tokens or recent):
        for turn in _legacy_turns(user_id):
            searchable = _search_text(turn, role, since, until)
            if searchable is None:
                continue
            body, stamp = searchable
            match = score(body)
            if recent or match[0]:
                relevant.append((match, False, stamp, turn))

    relevant.sort(key=lambda item: item[:3], reverse=True)
    seen = {(turn["user"], turn["reply"]) for turn in selected}
    modern = seen | {(turn["user"], turn["reply"]) for _, current, _, turn in relevant if current}
    for _, current, _, turn in relevant:
        key = (turn["user"], turn["reply"])
        if not current and (key in modern or key in seen):
            continue
        selected.append(turn)
        seen.add(key)
        if len(selected) >= limit:
            break
    return selected
