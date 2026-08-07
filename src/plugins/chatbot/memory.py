"""短期记忆（memory.json）读写，带进程内文件锁。

NoneBot 单进程运行，同一时刻可能有多条消息触发写入，
用 threading.Lock 保证「读-改-写」原子化，防止并发互相覆盖。
"""
import json
import threading
import sys
from pathlib import Path

from .constants import PROJECT_ROOT, MEMORY_FILE, SHORT_MEMORY_LINES

# 确保能导入项目根目录下的 memory_manager 模块
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_manager import (  # noqa: E402  # 长期记忆：记忆卡/关系等级/LLM提取
    get_user_memory,
    update_user_memory,
    build_memory_context,
    MEMORY_EXTRACT_PROMPT,
)

# 短期记忆文件锁（进程内）
_memory_lock = threading.Lock()


def load_short_memory() -> dict:
    """读取 memory.json；文件缺失/空/格式错误时返回空字典。"""
    if not MEMORY_FILE.exists():
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, OSError):
        return {}


def get_user_history(user_id: str) -> list:
    """获取某用户的短期对话历史（最近 3 轮）。"""
    memory = load_short_memory()
    history = memory.get(user_id, [])
    return list(history) if isinstance(history, list) else []


def append_user_history(user_id: str, user_msg: str, reply: str) -> None:
    """追加一轮对话到短期记忆，保留最近 N 条。全程持锁。"""
    with _memory_lock:
        memory = load_short_memory()
        history = memory.get(user_id, [])
        if isinstance(history, str):
            history = [history] if history else []
        elif not isinstance(history, list):
            history = []
        history.append(f"用户：{user_msg}")
        history.append(f"灰泽满：{reply}")
        if len(history) > SHORT_MEMORY_LINES:
            history = history[-SHORT_MEMORY_LINES:]
        memory[user_id] = history
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)


__all__ = [
    "load_short_memory",
    "get_user_history",
    "append_user_history",
    "get_user_memory",
    "update_user_memory",
    "build_memory_context",
    "MEMORY_EXTRACT_PROMPT",
    "PROJECT_ROOT",
]
