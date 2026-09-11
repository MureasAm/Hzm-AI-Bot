"""短期记忆（short_term.json）读写，带进程内文件锁。

NoneBot 单进程运行，同一时刻可能有多条消息触发写入，
用 threading.Lock 保证「读-改-写」原子化，防止并发互相覆盖。

长期记忆实现放在项目根 memory_manager.py（被 watchdog/评测脚本独立使用，
不适合塞进插件包触发 NoneBot 初始化），这里按文件路径直接加载它——不再改 sys.path。
"""
import importlib.util
import json
import threading
import time

from .constants import PROJECT_ROOT, MEMORY_FILE, SHORT_MEMORY_LINES

_mm_spec = importlib.util.spec_from_file_location("memory_manager", PROJECT_ROOT / "memory_manager.py")
_mm = importlib.util.module_from_spec(_mm_spec)
_mm_spec.loader.exec_module(_mm)
# 长期记忆：记忆卡/关系等级/LLM提取（从根模块 re-export，语义与原一致）
get_user_memory = _mm.get_user_memory
update_user_memory = _mm.update_user_memory
build_memory_context = _mm.build_memory_context
_format_profile_summary = _mm._format_profile_summary
MEMORY_EXTRACT_PROMPT = _mm.MEMORY_EXTRACT_PROMPT
# 时长措辞（"3天"/"2小时"）：长期/短期两处注入共用一份，别各写一套
humanize_gap = _mm.humanize_gap

# 短期记忆文件锁（进程内）
_memory_lock = threading.Lock()


def load_short_memory() -> dict:
    """读取 short_term.json；文件缺失/空/格式错误时返回空字典。"""
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


def _entry_text(item) -> str:
    """取一行历史的纯文本。

    历史行有两种形态：旧数据的裸字符串，和新数据的 {"t": epoch, "text": "…"}。
    对外一律还原成字符串，调用方（core 的 ln[4:] 剥前缀、\n.join 拼接）无需知道存法。
    """
    if isinstance(item, dict):
        return str(item.get("text", ""))
    return str(item)


def get_user_history(user_id: str) -> list:
    """获取某用户的短期对话历史（最近 3 轮），一律返回字符串行。"""
    raw = _load_raw_history(user_id)
    return [_entry_text(it) for it in raw]


def _load_raw_history(user_id: str) -> list:
    """读原始历史（保留 {t,text} 结构），供需要时间戳的调用方用。"""
    memory = load_short_memory()
    history = memory.get(user_id, [])
    if isinstance(history, str):
        return [history] if history else []
    return list(history) if isinstance(history, list) else []


def get_last_turn_gap_seconds(user_id: str) -> float | None:
    """距"上一轮对话"过去多少秒；没有历史或旧数据无时间戳 → None。

    用于注入"距上次聊天 X"——模型因此能分清"刚刚说的"和"三天前说的"。
    注意调用时机：本轮自己的话要等回复后才 append，所以此刻最后一条就是上一轮。
    """
    raw = _load_raw_history(user_id)
    if not raw:
        return None
    last = raw[-1]
    if not isinstance(last, dict):
        return None  # 旧数据（裸字符串）没有时间戳，这轮先不注入，等她再说一句就有
    try:
        return max(0.0, time.time() - float(last.get("t", 0)))
    except (TypeError, ValueError):
        return None


def append_user_history(user_id: str, user_msg: str, reply: str) -> None:
    """追加一轮对话到短期记忆（带时间戳），保留最近 N 条。全程持锁。"""
    with _memory_lock:
        memory = load_short_memory()
        history = memory.get(user_id, [])
        if isinstance(history, str):
            history = [history] if history else []
        elif not isinstance(history, list):
            history = []
        now = time.time()
        history.append({"t": now, "text": f"用户：{user_msg}"})
        history.append({"t": now, "text": f"灰泽满：{reply}"})
        if len(history) > SHORT_MEMORY_LINES:
            history = history[-SHORT_MEMORY_LINES:]
        memory[user_id] = history
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)


__all__ = [
    "load_short_memory",
    "get_user_history",
    "get_last_turn_gap_seconds",
    "append_user_history",
    "humanize_gap",
    "get_user_memory",
    "update_user_memory",
    "build_memory_context",
    "_format_profile_summary",
    "MEMORY_EXTRACT_PROMPT",
    "PROJECT_ROOT",
]
