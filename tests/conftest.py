"""pytest 公共配置。

在 conftest 导入期就初始化 NoneBot 驱动，因为测试模块导入
src.plugins.chatbot 包时会执行 __init__.py → core.py → get_driver()。
core.py 的 API 客户端是惰性初始化的，测试不会真正调用外部 API。
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# —— 模块级初始化：必须先于任何测试模块导入 ——
import nonebot
from nonebot.adapters.onebot.v11 import Adapter as ONEBOT_V11Adapter

nonebot.init()
_driver = nonebot.get_driver()
_driver.register_adapter(ONEBOT_V11Adapter)
nonebot.load_plugins("src/plugins")


@pytest.fixture(autouse=True)
def _isolate_local_stores(tmp_path, monkeypatch):
    """所有测试一律不碰真实运行数据（短期记忆 / 聊天落盘）。

    为什么必须全局兜底（2026-10-08）：短期记忆改成**按会话分文件**后，
    `_ensure_migrated()` 首次运行会去拆真实那份 `user_memory/short_term.json`
    （拆完还改名存档）——只要有一个用例顺手写一条记忆，就会动到线上数据。
    """
    from src.plugins.chatbot import chatlog, memory, session_memory

    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path / "short_term")
    monkeypatch.setattr(memory, "MEMORY_FILE", tmp_path / "short_term.json")
    monkeypatch.setattr(memory, "_migrated", False)
    monkeypatch.setattr(chatlog, "LOG_DIR", tmp_path / "chat_log")
    # 会话记忆 / 长期记忆也按会话分文件了（2026-10-09）——同一个坑：
    # 只改文件路径的话，写会落到真实 user_memory/ 里去。
    monkeypatch.setattr(session_memory, "SESSION_DIR", tmp_path / "session")
    monkeypatch.setattr(session_memory, "SESSION_MEMORY_FILE", tmp_path / "session.json")
    monkeypatch.setattr(session_memory, "_migrated", False)
    monkeypatch.setattr(memory._mm, "MEMORY_DIR", tmp_path / "long_term")
    monkeypatch.setattr(memory._mm, "MEMORY_FILE", tmp_path / "long_term.json")
    monkeypatch.setattr(memory._mm, "_migrated", False)
