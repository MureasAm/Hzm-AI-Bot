import sys

# ⚠️ 必须在任何 print 之前执行。
# stdout 被重定向到管道/文件时，Windows 下编码会退回 cp936，带 ⚠️/emoji 的 print 直接
# UnicodeEncodeError。而 voice.py 里"多次合成仍不达标"那条 print 在 try 之外，异常会一路
# 冒到 chat_window._flush —— **那一轮回复一条都发不出去**（2026-10-11 实跑撞到，见
# docs/下一步.md）。真控制台（启动.bat）走 PEP 528 本来就是 utf-8，这里等于没动它。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import nonebot  # noqa: E402
from nonebot.adapters.onebot.v11 import Adapter as ONEBOT_V11Adapter  # noqa: E402

nonebot.init()
driver = nonebot.get_driver()
driver.register_adapter(ONEBOT_V11Adapter)

nonebot.load_plugins("src/plugins")
# nonebot.load_plugin("nonebot_plugin_gocqhttp")  ← 注释掉

if __name__ == "__main__":
    nonebot.run()