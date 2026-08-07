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
