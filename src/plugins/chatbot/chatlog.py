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

轮转：单文件超 `MAX_BYTES` 就改名加时间戳，最多留 `KEEP_FILES` 个。
"""
import json
import os
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
        olds = sorted(LOG_DIR.glob("chat.*.jsonl"), key=lambda p: p.stat().st_mtime)
        for p in olds[:-KEEP_FILES]:
            p.unlink(missing_ok=True)
    except OSError:
        pass


def log_turn(target_id: str, is_group: bool, user_text: str, reply: str) -> None:
    """追加一轮。**只在消息真的发出去之后调用。**

    失败静默——分析用的记录，绝不能因为它写不进去而影响聊天。
    """
    if not chatlog_enabled():
        return
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / "chat.jsonl"
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
