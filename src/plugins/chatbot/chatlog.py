# -*- coding: utf-8 -*-
"""聊天落盘：把每轮"用户说了什么 / 她回了什么"追加进 JSONL，供事后分析用。

**为什么要它**：短时记忆（`short_term.json`）只有 **10 条/会话**的滚动窗口，超出就没了。
于是"她哪里答得不对"只能靠人肉记得住的那几条——这一轮做的覆盖度、回归 case，
全是从别处硬凑出来的证据。

有了全量记录才能：
- 事后挖"她答得不好的时刻" → **那才是"该补什么素材"的依据**
- 覆盖度 / 同质率这类指标**随时间看变化**
- 回归 case 有真实来源（不必每次靠手贴）

⚠️ **这是真实用户对话，仓库是 PUBLIC**：
- 只写 `data/chat_log/`，**必须留在 .gitignore 里**
- 只在**真的发出去了**之后才记（兜底文案、没接话的批次都不记）
- 写失败绝不影响主流程（分析用的东西，不能拖累聊天）

**按会话分文件**（2026-10-08）：`chat_log/<QQ号或群号>.jsonl`。
原来全挤在 `chat.jsonl` 一个文件里 —— 1.4MB 混着所有人的对话，想"看某个人聊了啥"
只能翻全文；写侧倒是没成本（追加），所以拆它纯粹为了**能看**：
一个人一个文件，删某人的记录＝删一个文件。
分析脚本要看全量请走 `chatlog.iter_records()`，别自己拼路径。

轮转：单个会话文件超 `MAX_BYTES` 就改名加时间戳，最多留 `KEEP_FILES` 个
（正常单人不会到 20MB；留着是防机器人群那种极端量）。
"""
import json
import os
import re
import time
from pathlib import Path

from .constants import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "data" / "chat_log"
MAX_BYTES = 20 * 1024 * 1024     # 单文件 20MB 就轮转
KEEP_FILES = 10                  # 最多留 10 个历史文件


def chatlog_enabled() -> bool:
    """环境变量 CHATLOG=0 可关掉（缺省开）。"""
    return os.environ.get("CHATLOG", "1") != "0"


def _rotate(path: Path) -> None:
    """超过上限就改名存档，并清掉最老的几个。"""
    try:
        if not path.exists() or path.stat().st_size < MAX_BYTES:
            return
        path.rename(path.with_suffix(f".{int(time.time())}.jsonl"))
        # 只清**同一个会话**的历史归档（2026-10-09 起按会话分文件：
        # 原来的 `chat.*.jsonl` 匹配不到 `<QQ号>.<时间戳>.jsonl` 这种归档，会永久堆积）
        stem = path.name[: -len(".jsonl")]
        olds = sorted(LOG_DIR.glob(f"{stem}.*.jsonl"), key=lambda p: p.stat().st_mtime)
        for p in olds[:-KEEP_FILES]:
            p.unlink(missing_ok=True)
    except OSError:
        pass


def _file_for(session: str) -> Path:
    """某个会话的记录文件。id 只可能是 QQ 号/群号，仍然过滤一遍防路径穿越。"""
    return LOG_DIR / f"{re.sub(r'[^\w.-]', '_', str(session))}.jsonl"


_migrated = False


def _ensure_migrated() -> None:
    """把旧的单文件 `chat.jsonl` 按会话拆开（只跑一次；**原件改名存档，不删**）。

    旧的单文件保留着不动也能读（`iter_records` 会 glob 到它），拆分只是为了
    "看某个人聊了啥"能一个文件看完。
    """
    global _migrated
    if _migrated:
        return
    _migrated = True
    legacy = LOG_DIR / "chat.jsonl"
    if not legacy.exists():
        return
    try:
        buckets = {}
        for line in legacy.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            buckets.setdefault(str(rec.get("session") or "unknown"), []).append(line)
        n = 0
        for sess, lines in buckets.items():
            f = _file_for(sess)
            if f.exists():          # 已经拆过/已有新记录 → 别覆盖
                continue
            f.write_text("\n".join(lines) + "\n", encoding="utf-8")
            n += 1
        legacy.rename(legacy.with_suffix(".jsonl.migrated"))
        print(f"[聊天落盘] 旧单文件已按会话拆分：{n} 个会话 → {LOG_DIR}")
    except OSError as e:
        print(f"⚠️ 聊天落盘拆分失败（退回单文件读，不影响聊天）: {e}")


def iter_records():
    """**跨所有会话**逐条产出记录（dict，里面的 `session` 字段照旧有）。

    分析脚本统一走这个 —— 按会话分文件后，"路径"不再是一个文件，
    谁都不该再去拼 `chat_log/chat.jsonl`（2026-10-08 拆分）。
    坏行/坏文件跳过，绝不因为一条脏数据让整个分析挂掉。
    """
    _ensure_migrated()
    if not LOG_DIR.exists():
        return
    for f in sorted(LOG_DIR.glob("*.jsonl")):
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def log_turn(target_id: str, is_group: bool, user_text: str, reply: str) -> None:
    """追加一轮。**只在消息真的发出去之后调用。**

    失败静默——分析用的记录，绝不能因为它写不进去而影响聊天。
    """
    if not chatlog_enabled():
        return
    try:
        _ensure_migrated()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = _file_for(target_id)      # 一个会话一个文件（见模块头「为什么按会话分」）
        _rotate(path)
        rec = {
            "t": time.time(),
            "session": str(target_id),
            "kind": "group" if is_group else "private",
            "user": (user_text or "").strip(),
            "reply": (reply or "").strip(),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ 聊天记录写入失败（忽略）: {e}")
