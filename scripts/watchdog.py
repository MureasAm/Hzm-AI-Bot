#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""watchdog 自愈：NapCat 假死（显示在线但收不到消息）时自动重启。

原理：
- bot 每 30s touch data/heartbeat（见 __init__.py 心跳协程）
- 本脚本循环检测 heartbeat 是否过期：
  · 过期 → bot 假死（多半是 NapCat 假死）→ 杀掉 bot.py 进程 → 重启 bot（+可选 NapCat）

用法（Windows，推荐用 pythonw 后台无窗口跑）：
    python scripts/watchdog.py                  # 前台跑
    pythonw scripts/watchdog.py                 # 后台跑（推荐，最小化无窗口）

配置（改下面常量）：
    BOT_CMD         启动 bot 的命令（改盘符/venv 路径）
    NAPCAT_CMD      启动 NapCat 的命令（每个安装方式不同，必填才能重启 NapCat；
                    留空则只重启 bot）
    HEARTBEAT_TTL   心跳过期秒数（默认 180s = bot 连续 6 次没心跳）
"""
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HEARTBEAT_FILE = PROJECT_ROOT / "data" / "heartbeat"
LOG_FILE = PROJECT_ROOT / "logs" / "watchdog.log"

# ==================== 配置（按你的环境改） ====================
HEARTBEAT_TTL = 180            # 心跳过期秒数
CHECK_INTERVAL = 30            # 检测间隔秒数
BOT_CMD = [sys.executable, "-u", str(PROJECT_ROOT / "bot.py")]  # 启动 bot 的命令
NAPCAT_CMD = []                # 启动 NapCat 的命令，如 ["D:/napcat/napcat.bat"]。留空则只重启 bot。
# ===============================================================


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _heartbeat_age() -> float:
    if not HEARTBEAT_FILE.exists():
        return float("inf")
    return time.time() - HEARTBEAT_FILE.stat().st_mtime


def _kill_bot() -> None:
    """精确杀掉跑 bot.py 的 python 进程（不影响 watchdog 自身与其他 python）。"""
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like '*bot.py*' -and $_.Name -eq 'python.exe' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, timeout=15)
        _log("💀 已杀掉 bot.py 进程")
    except Exception as e:
        _log(f"⚠️ 杀 bot 进程失败: {e}")


def _start(cmd: list, name: str) -> None:
    """后台启动新进程（不等待），输出重定向到 watchdog.log。"""
    try:
        DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0)
        subprocess.Popen(cmd, cwd=str(PROJECT_ROOT),
                         stdout=open(LOG_FILE, "a", encoding="utf-8"),
                         stderr=subprocess.STDOUT,
                         creationflags=DETACHED | subprocess.CREATE_NEW_PROCESS_GROUP)
        _log(f"✅ 已启动 {name}")
    except Exception as e:
        _log(f"❌ 启动 {name} 失败: {e}")


def _restart() -> None:
    _log("🔁 触发自愈重启...")
    _kill_bot()
    if NAPCAT_CMD:
        _start(NAPCAT_CMD, "NapCat")
    _start(BOT_CMD, "bot")


def _run() -> None:
    _log("🚀 watchdog 启动（TTL=%ss, NapCat 重启=%s）"
         % (HEARTBEAT_TTL, "开" if NAPCAT_CMD else "关(只重启bot)"))
    while True:
        try:
            age = _heartbeat_age()
            if age > HEARTBEAT_TTL:
                _log(f"💀 心跳过期 {age:.0f}s（> {HEARTBEAT_TTL}s）→ 判定假死")
                _restart()
                # 重启后给 bot 足够时间起心跳，避免反复重启
                time.sleep(HEARTBEAT_TTL + 60)
            else:
                _log(f"✅ 正常（心跳 {age:.0f}s 前）")
        except Exception as e:
            _log(f"⚠️ watchdog 异常（继续）: {e}")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    _run()
