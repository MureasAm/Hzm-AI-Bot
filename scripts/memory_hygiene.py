# -*- coding: utf-8 -*-
"""记忆库卫生：去重、补 key、修掉一次性状态占着习惯槽的坏数据。

只做**确定性**动作，不调模型：
  1. 同一个单值 (kind, key) 有多条活着的记录 -> 只留最新，其余标 superseded；
  2. 裸 `habit` 按内容改成具体 `habit_*`，并补结构化字段；
  3. 两个已知坏槽：215 的「睡前要摸摸」被「已经洗完澡躺床上了」顶掉；
  4. 215 的旧 `morning_running` 与 `habit_morning_run` 重复。

默认只看不改；`--write` 才落盘（先备份）。

    python scripts/memory_hygiene.py
    python scripts/memory_hygiene.py --write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATE = ROOT / "user_memory" / "memory_v2_shadow.json"

_BARE_HABIT_MAP = (
    (("药", "助眠", "吃药"), "habit_medication"),
    (("周五", "周末", "回家", "出租屋"), "habit_weekend_home"),
    (("跑", "操"), "habit_running"),
    (("熬夜", "晚睡", "深夜", "凌晨"), "habit_late_sleep"),
    (("拿铁", "咖啡"), "habit_coffee"),
    (("健身", "健身房"), "habit_gym"),
)


def _mid(kind: str, key: str, value: str) -> str:
    return hashlib.sha256(f"{kind}\0{key}\0{value}".encode("utf-8")).hexdigest()[:20]


def _alive(m: dict) -> bool:
    return m.get("status") in {"confirmed", "active"}


def _stamp(m: dict) -> str:
    return str(m.get("last_observed_at") or m.get("created_at") or "")


def _single_slot(m: dict) -> bool:
    from src.plugins.chatbot import memory_v2 as mv
    kind = m.get("kind")
    key = str(m.get("key") or "")
    if kind in {"interaction_preference", "request", "commitment"}:
        return True
    return (kind == "fact" and key != "her_nickname" and (key in mv._SINGLE_VALUED_FACT_KEYS
                                or key.startswith("habit_")
                                or m.get("temporal_scope") == "current"))


def _bare_habit_key(value: str) -> str:
    text = str(value or "")
    for words, key in _BARE_HABIT_MAP:
        if any(w in text for w in words):
            return key
    return ""


def repair(state: dict, *, now: datetime | None = None) -> dict:
    from src.plugins.chatbot import memory_v2 as mv
    now = now or datetime.now()
    now_iso = now.isoformat()
    changes: list[dict] = []

    for uid, user in (state.get("users") or {}).items():
        memories = user.get("memories") or []

        # 1) 裸 habit -> 具体 key（包括已经因同 key 被顶掉的：两条其实往往是两件事）
        active_keys = {str(m.get("key") or "") for m in memories if _alive(m)}
        for m in memories:
            if m.get("kind") != "fact" or m.get("key") != "habit":
                continue
            new_key = _bare_habit_key(m.get("value"))
            if not new_key:
                changes.append({"user": uid, "action": "bare_habit_unmapped",
                                "value": m.get("value")})
                continue
            old_id = m.get("id")
            revived = False
            if m.get("status") == "superseded" and new_key not in active_keys:
                m["status"] = "confirmed"
                m.pop("invalidated_at", None)
                active_keys.add(new_key)
                revived = True
            m["key"] = new_key
            m["habit"] = mv.normalize_habit_metadata(None, key=new_key, value=m.get("value"))
            m["id"] = _mid("fact", new_key, str(m.get("value") or ""))
            changes.append({"user": uid, "action": "bare_habit_renamed",
                            "old_key": "habit", "new_key": new_key,
                            "old_id": old_id, "new_id": m["id"], "revived": revived})

        # 3) 同 (kind,key) 多条 -> 只留最新
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for m in memories:
            if _alive(m) and _single_slot(m):
                groups[(str(m.get("kind")), str(m.get("key")))].append(m)
        for (kind, key), group in groups.items():
            if len(group) < 2:
                continue
            keep = max(group, key=_stamp)
            for m in group:
                if m is keep:
                    continue
                m["status"] = "superseded"
                m["invalidated_at"] = now_iso
                changes.append({"user": uid, "action": "same_key_superseded",
                                "kind": kind, "key": key, "value": m.get("value")})

    return {"changes": changes, "n": len(changes)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="落盘（默认只看）")
    ap.add_argument("--state", default=str(STATE))
    args = ap.parse_args(argv)

    import nonebot
    try:
        nonebot.get_driver()
    except ValueError:
        nonebot.init()

    path = Path(args.state)
    state = json.loads(path.read_text(encoding="utf-8"))
    report = repair(state)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.write and report["n"]:
        backup = path.with_name(path.name + f".bak-hygiene-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(path, backup)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[hygiene] 已写入；备份：{backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
