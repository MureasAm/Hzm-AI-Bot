"""群级记忆（user_memory/groups.json）：按群号存两样东西——

- members: 群成员身份 {user_id: 昵称}（稳定 id → 昵称，不再每次现拉卡片）
- events:  群里发生过的事/梗/近况摘要（带时间戳，倒序，上限 GROUP_EVENT_MAX）

私聊不回这里；这里的内容也只在本群回复时注入，不跨群、不进私聊个人卡。
写操作带进程内锁，防与短期记忆并发写坏文件。
"""
import json
import threading
import time
from pathlib import Path

from .constants import GROUP_MEMORY_FILE, GROUP_EVENT_MAX

_lock = threading.Lock()


def _load() -> dict:
    if GROUP_MEMORY_FILE.exists():
        try:
            return json.loads(GROUP_MEMORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save(data: dict) -> None:
    try:
        GROUP_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        GROUP_MEMORY_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"⚠️ 群记忆写入失败: {e}")


def get_group(group_id: str) -> dict:
    """取某群的记忆 {members:{uid:name}, events:[{t, text}]}，没有则返回空壳。"""
    with _lock:
        g = _load().get(str(group_id)) or {}
    g.setdefault("members", {})
    g.setdefault("events", [])
    return g


def upsert_members(group_id: str, id_to_name: dict) -> None:
    """批量更新成员 id→昵称 映射（只在拿到昵称的地方调）。"""
    if not id_to_name:
        return
    with _lock:
        data = _load()
        g = data.setdefault(str(group_id), {})
        g.setdefault("members", {})
        for uid, name in id_to_name.items():
            if uid and name and g["members"].get(uid) != name:
                g["members"][uid] = name
        _save(data)


def add_event(group_id: str, text: str) -> None:
    """追加一条群近况/梗（丢弃空、去重相邻、倒序、封顶 GROUP_EVENT_MAX）。"""
    text = (text or "").strip()
    if not text:
        return
    with _lock:
        data = _load()
        g = data.setdefault(str(group_id), {})
        ev = g.setdefault("events", [])
        if ev and ev[0].get("text") == text:   # 相邻重复（同一批归纳重放）跳过
            return
        ev.insert(0, {"t": time.time(), "text": text[:160]})
        del ev[GROUP_EVENT_MAX:]
        _save(data)
