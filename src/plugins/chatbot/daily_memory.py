#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日记忆：**用户当天很单纯的日常**（吃了什么、几点上课、什么时候睡的）。

为什么要单开一层（2026-10-17 定，设计讨论见 `docs/下一步.md` §3.1）：
短期是原文、会话级是"这一场聊什么"（3 小时就散）、长期是稳定事实——**没有一层能说
"他今天几点做了什么"**。他的会话事件是 `list[str]`、逐条不带时间，只有话题级的
`last_active`。而群记忆的事件**是带 `t` 的**（`core.py` 里渲染成「（X前）」），
照那个模式做即可。

三条硬规矩（都是设计时定死的，别改回去）：
1. **纯事件**。只收"有事发生"的句子；状态/情绪/喜好/关系归 `memory_v2`，不在这里。
2. **带时点**。每条存**绝对时刻**（他自己不带时间词也没关系，系统知道这句是什么时候说的），
   渲染时才换算成「今天早上9点」。存相对词会在注入时刻漂移——早上存的"今天早上"，
   到晚上还叫"今天早上"就错了。
3. **宁可不提**。入库有代码侧闸门（见 `_looks_like_event`），不靠模型自觉；条目太少就不注。

过期是**惰性**的：不挂定时器，读或写的时候顺手判一次"跨没跨过凌晨 3 点"。
理由——245 个会话里大多数天不活跃，定时器会空跑一整天；而且日期边界本来就该在
"要用这份数据的时候"判。
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .constants import PROJECT_ROOT

DAILY_DIR = PROJECT_ROOT / "user_memory" / "daily"

# 日界：凌晨 3 点。⚠️ 它只管**清理**，不管措辞——"今天/昨天"按自然日判（见 `_day_label`）。
# 分开的理由：按 3 点划日界的话，凌晨 1 点说的话会被算进"前一天"，第二天上午提起来
# 会渲染成"昨天凌晨1点"，可实际只隔了 9 小时，中文里该说"今天凌晨1点"。
DAY_BOUNDARY_HOUR = 3

MAX_ENTRIES_PER_DAY = 6
MAX_CONTEXT_CHARS = 120
MIN_ENTRIES_TO_INJECT = 2
MAX_ENTRY_CHARS = 40

# 入库闸门①：句子得**有事发生**。没有动词短语的（"他和室友关系一般"）直接丢。
_EVENT_MARKERS = (
    "吃", "喝", "上课", "下课", "去", "回", "睡", "起床", "起晚", "跑", "买", "写", "看",
    "做", "开", "结束", "考试", "考完", "放", "参加", "打", "玩", "走", "到", "离开",
    "洗", "收拾", "搬", "坐", "上班", "下班", "加班", "约", "见", "拿", "送", "排队",
    "点", "订", "复习", "刷", "遛", "运动", "健身", "出门", "到家", "开会", "交", "补",
    "取消", "推迟", "提前", "请假", "迟到", "改了",
)
# 入库闸门②：**状态/情绪/喜好/关系**不是"有事发生"，一律丢。
# ⚠️ 单靠①会漏："不喜欢吃辣"里有"吃"、会被当事件放进来——这正是本项目反复踩的
#    "状态混进事件"那类错（长期记忆里"最近很累"被存成 fact 同一个病）。宁可错杀。
_STATE_MARKERS = (
    "喜欢", "讨厌", "觉得", "认为", "感觉", "希望", "习惯", "性格", "关系",
    "心情", "开心", "难过", "焦虑", "压力",
)
# 入库闸门③：讲的是她自己/机器人的，一律不收（长期记忆里也有同一条铁律）。
_SELF_MARKERS = ("灰泽满", "hzm", "机器人", "主播")
# 明显是"网上的段子/模板"的起手，别当真（长期记忆踩过：段子被存成 occupation 五条）。
_MEME_HEADS = ("我重新介绍一下自己", "丫头", "家人们", "兄弟们我")

# 时段词：只给"打在点上"的那一段。
# ⚠️ 不设"上午"——9:00 中文里说"早上9点"（用户给的就是这个例子），
#    设了"上午"就会渲染成"上午9点"，跟平时说法不一致。
_DAY_PARTS = (
    (5, "早上"), (11, "中午"), (13, "下午"), (18, "晚上"), (23, "深夜"),
)


def _day_key(when: datetime) -> date:
    """这条记录属于"哪一天"——按 3 点日界把凌晨那几个小时算进前一天。"""
    return (when - timedelta(hours=DAY_BOUNDARY_HOUR)).date()


def _day_part(hour: int) -> str:
    part = "凌晨"
    for start, name in _DAY_PARTS:
        if hour >= start:
            part = name
    return part


def _day_label(when: datetime, now: datetime) -> str:
    """「今天」/「昨天」/「N天前」——**按自然日判**，跟清理用的 3 点日界解耦。"""
    delta = (now.date() - when.date()).days
    if delta <= 0:
        return "今天"
    if delta == 1:
        return "昨天"
    return f"{delta}天前"


def render_entry(entry: dict, now: datetime) -> str:
    """一条记录 → 「今天早上9点：用户早上吃了包子」。时刻是**用户说的那一刻**。

    ⚠️ **主语必须写"用户"**（2026-10-18 验收发现）：只写「今天中午12点：中午吃了炸鸡」
    时，模型会把事件认成灰泽满自己的（实测回出"灰泽满今天中午才吃完兰州拉面"）。
    块头写"用户/他"不够——事件自己不带主语，模型就按人设默认成她自己。

    ⚠️ 下午/晚上/深夜用**12 小时制**：说"下午4点"，不说"下午16点"；
    "晚上10点"，不说"晚上22点"。这跟用户给的两个例子一致。
    """
    try:
        when = datetime.fromisoformat(str(entry.get("t")))
    except (TypeError, ValueError):
        return ""
    part = _day_part(when.hour)
    hour = when.hour - 12 if part in ("下午", "晚上", "深夜") else when.hour
    return f"{_day_label(when, now)}{part}{hour}点：用户{entry.get('text', '')}"


def _looks_like_event(text: str) -> bool:
    """入库闸门：是不是一条"有事发生"的日常。

    ⚠️ 这是**代码侧的硬闸门**，不是提示词里的建议。长期记忆的教训：光在提示词里写反例
    会被违反（某用户发网上的段子，被当真存成 occupation/location/age/income 五条）。
    """
    t = (text or "").strip()
    if not t or len(t) > MAX_ENTRY_CHARS:
        return False
    if any(m in t for m in _SELF_MARKERS):
        return False
    if any(m in t for m in _STATE_MARKERS):
        return False
    if any(t.startswith(h) for h in _MEME_HEADS):
        return False
    return any(m in t for m in _EVENT_MARKERS)


class DailyStore:
    """一天一批的用户日常。一个用户一个文件（存绝对时刻 + 一句事件）。"""

    def __init__(self, base_dir: Path | str = DAILY_DIR) -> None:
        self.base_dir = Path(base_dir)
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 文件

    def _path(self, user_id: str) -> Path:
        safe = re.sub(r"[^0-9A-Za-z_\-]", "", str(user_id))[:64]
        return self.base_dir / f"{safe}.json"

    def _load(self, user_id: str) -> dict:
        path = self._path(user_id)
        if not path.exists():
            return {"entries": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"entries": []}
        if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
            return {"entries": []}
        return {"entries": [e for e in data["entries"] if isinstance(e, dict)]}

    def _save(self, user_id: str, data: dict) -> None:
        path = self._path(user_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass

    # ---------------------------------------------------------------- 清理

    @staticmethod
    def _prune(entries: list[dict], now: datetime) -> list[dict]:
        """只留"今天"（3 点日界）的条目——跨过凌晨 3 点，前一天整批消失。"""
        today = _day_key(now)
        kept = []
        for e in entries:
            try:
                when = datetime.fromisoformat(str(e.get("t")))
            except (TypeError, ValueError):
                continue
            if _day_key(when) == today:
                kept.append(e)
        return kept

    # ---------------------------------------------------------------- 读写

    def add(self, user_id: str, texts: list[str], *, now: datetime | None = None) -> int:
        """写入一批事件句。**入库闸门在这儿**，不合格的直接丢。返回真正写进去的条数。"""
        now = now or datetime.now()
        fresh = [t.strip() for t in (texts or []) if _looks_like_event(t)]
        if not fresh:
            return 0
        with self._lock:
            data = self._load(user_id)
            entries = self._prune(data["entries"], now)
            seen = {str(e.get("text")) for e in entries}
            added = 0
            for text in fresh:
                if text in seen:
                    continue
                entries.append({"t": now.isoformat(), "text": text})
                seen.add(text)
                added += 1
            # 一天留不下太多：超了丢最旧的
            if len(entries) > MAX_ENTRIES_PER_DAY:
                entries = entries[-MAX_ENTRIES_PER_DAY:]
            data["entries"] = entries
            self._save(user_id, data)
        return added

    def context(self, user_id: str, *, now: datetime | None = None) -> str:
        """渲染"他今天的日常"。**条目太少就不注**（一条孤零零的日常没资格占位置）。"""
        now = now or datetime.now()
        if not user_id:
            return ""
        with self._lock:
            data = self._load(user_id)
            entries = self._prune(data["entries"], now)
            if len(entries) != len(data["entries"]):
                data["entries"] = entries
                self._save(user_id, data)
        if len(entries) < MIN_ENTRIES_TO_INJECT:
            return ""
        lines = [render_entry(e, now) for e in entries[-MAX_ENTRIES_PER_DAY:]]
        body = "；".join(x for x in lines if x)
        return body[:MAX_CONTEXT_CHARS]


def enabled() -> bool:
    """默认开启。要临时关掉（回退排查）在 `.env.prod` 设 `DAILY_MEMORY=0`，不用改代码。"""
    return os.environ.get("DAILY_MEMORY", "1") != "0"


# 全进程共用**一个**实例：写路径（提取完入库）和读路径（每轮注入）必须共用同一把锁，
# 各自 new 一个的话会各持一把锁去写同一个文件。
_DEFAULT_STORE = DailyStore()


def add_events(user_id: str, texts: list[str], *, now: datetime | None = None) -> int:
    """给某个用户记一批当天的事件（写路径；不合格的由入库闸门丢掉）。"""
    return _DEFAULT_STORE.add(user_id, texts, now=now)


def context(user_id: str, *, now: datetime | None = None) -> str:
    """渲染"他今天的日常"，顺便惰性清理过期条目（读路径）。"""
    return _DEFAULT_STORE.context(user_id, now=now)