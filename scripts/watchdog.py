#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""watchdog 自愈 v2：假死（QQ 会话静默断开）检测 + SnowLuma 重启 + 被踢扫码发邮箱。

信号（都是 bot 侧写的文件，见 src/plugins/chatbot/__init__.py）：
- data/heartbeat   bot 进程心跳（bot.py 每 30s touch）→ 死了说明 bot 崩了
- data/qq_alive    QQ 会话在线标记（只有 OneBot WS 连接 + 账号在线才 touch）→ 停更
                    说明会话断了（"显示在线但收不到消息"的假死）
- data/qq_offline  掉线标记：内容 bot_offline=被踢（权威信号，WS 可能还连着），
                    disconnect=WS 断开（基础设施层）

流程：
1. heartbeat 过期           → bot 进程崩溃 → 重启 bot
2. QQ 会话掉线（qq_offline 存在 或 qq_alive 过期）：
   a. 被踢（bot_offline）    → 重启没用（登录已失效）→ 等二维码渲染 → 截屏发邮箱
   b. 静默掉线（disconnect）  → 先重启 SnowLuma（多数重启能自动重连）→ 宽限期后仍
                               未恢复 → 截屏发邮箱
   c. 进入 waiting_scan      → 停止反复重启；恢复后发"已上线"通知并复位

用法（Windows，推荐 pythonw 后台无窗口跑）：
    pythonw scripts/watchdog.py            # 后台跑（推荐）
    python scripts/watchdog.py --once      # 跑一轮检查后退出（调试）
    python scripts/watchdog.py --dry-run   # 只判断不执行任何动作（调试）
    python scripts/watchdog.py --send-test # 发一封测试邮件到 SMTP_TO

配置（改下面常量 / .env.prod）：
    BOT_CMD          启动 bot 的命令
    SNOWLUMA_DIR     SnowLuma 安装目录
    SNOWLUMA_CMD     重启 SnowLuma 的命令（默认 node index.mjs）
"""
import json
import subprocess
import sys
import time
from pathlib import Path

# Windows 控制台默认 GBK，打 emoji 会崩；统一成 UTF-8（pythonw 下 stdout 是 None，忽略）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HEARTBEAT_FILE = DATA_DIR / "heartbeat"
QQ_ALIVE_FILE = DATA_DIR / "qq_alive"
QQ_OFFLINE_FILE = DATA_DIR / "qq_offline"
STATE_FILE = DATA_DIR / "watchdog_state.json"
LOG_FILE = PROJECT_ROOT / "logs" / "watchdog.log"
QR_DIR = DATA_DIR / "qr"

# ==================== 配置（按你的环境改） ====================
HEARTBEAT_TTL = 180            # bot 心跳过期秒数（bot 连续 6 次没心跳 = 崩了）
QQ_ALIVE_TTL = 120             # QQ 会话在线标记过期秒数（约 4 次没更新 = 掉线）
CHECK_INTERVAL = 20            # 检测间隔秒数
RESTART_COOLDOWN = 300         # 两次重启 SnowLuma 的最小间隔（秒），防 flap 循环
GRACE_SECONDS = 90             # 静默掉线后等待自动恢复的时间，过了判定"要扫码"
QR_RENDER_DELAY = 10           # 被踢后等二维码窗口渲染出来的时间，再截屏发码
BOT_CMD = [sys.executable, "-u", str(PROJECT_ROOT / "bot.py")]  # 启动 bot 的命令
SNOWLUMA_DIR = Path(r"D:\SnowLuma-v1.14.7-win-x64")             # SnowLuma 安装目录
SNOWLUMA_CMD = [str(SNOWLUMA_DIR / "node.exe"), "index.mjs"]    # 重启 SnowLuma
# ===============================================================

_DRY_RUN = "--dry-run" in sys.argv
_ONCE = "--once" in sys.argv


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _file_age(path: Path) -> float:
    if not path.exists():
        return float("inf")
    return time.time() - path.stat().st_mtime


# ==================== 状态持久化（watchdog 自己重启不丢状态） ====================

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(st: dict) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(st, ensure_ascii=False, indent=2), "utf-8")
    except OSError as e:
        _log(f"⚠️ 状态写入失败: {e}")


# ==================== 进程控制 ====================

def _kill_processes(name_pattern: str, cmdline_keywords: list[str]) -> None:
    """按进程名（*通配）+ 命令行关键字精确杀进程（PowerShell 模糊匹配）。

    name_pattern 如 'python*.exe'（覆盖 python.exe / pythonw.exe）、'node.exe'。
    """
    name_cond = "$_.Name -like '%s'" % name_pattern if name_pattern else "$true"
    cmd_cond = " -or ".join("$_.CommandLine -like '*%s*'" % k for k in cmdline_keywords)
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { %s -and (%s) } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
        % (name_cond, cmd_cond)
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, timeout=15)
    except Exception as e:
        _log(f"⚠️ 杀进程失败: {e}")


def _kill_bot() -> None:
    _kill_processes("python*.exe", ["bot.py"])
    _log("💀 已杀掉 bot.py 进程")


def _kill_snowluma() -> None:
    # SnowLuma 是 node 跑 index.mjs；NapCatQQ-Desktop 是管理壳，不动它。
    # 若 NapCatQQ-Desktop 会自动拉起 SnowLuma，可把 SNOWLUMA_CMD 置空（只杀不拉）。
    _kill_processes("node.exe", ["index.mjs", "SnowLuma"])
    _log("💀 已杀掉 SnowLuma 进程")


def _start(cmd: list, name: str, cwd: Path | None = None) -> None:
    """后台启动新进程（不等待），输出重定向到 watchdog.log。"""
    try:
        DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0)
        subprocess.Popen(cmd, cwd=str(cwd) if cwd else str(PROJECT_ROOT),
                         stdout=open(LOG_FILE, "a", encoding="utf-8"),
                         stderr=subprocess.STDOUT,
                         creationflags=DETACHED | subprocess.CREATE_NEW_PROCESS_GROUP)
        _log(f"✅ 已启动 {name}")
    except Exception as e:
        _log(f"❌ 启动 {name} 失败: {e}")


def _restart_bot() -> None:
    _log("🔁 重启 bot...")
    _kill_bot()
    _start(BOT_CMD, "bot")


def _restart_snowluma() -> None:
    _log("🔁 重启 SnowLuma...")
    _kill_snowluma()
    time.sleep(2)  # 等进程彻底退出再拉起
    _start(SNOWLUMA_CMD, "SnowLuma", cwd=SNOWLUMA_DIR)


# ==================== 截屏 + 邮件通知 ====================

def _capture_qr() -> str:
    """截取整屏（被踢时 QQ 登录二维码就在屏幕上），返回 PNG 路径；失败返回空串。"""
    if _DRY_RUN:
        _log("[dry-run] 📷 截屏(不实际执行)")
        return ""
    try:
        from PIL import ImageGrab
        QR_DIR.mkdir(parents=True, exist_ok=True)
        p = QR_DIR / f"qr_{time.strftime('%Y%m%d_%H%M%S')}.png"
        ImageGrab.grab().save(p)
        return str(p)
    except Exception as e:
        _log(f"⚠️ 截屏失败: {e}")
        return ""


def _notify(subject: str, body: str, images: list[str] | None = None) -> None:
    if _DRY_RUN:
        _log(f"[dry-run] 📧 邮件(不实际发送): {subject}")
        return
    try:
        import notifier  # scripts/notifier.py
        ok = notifier.send_email(subject, body, attachments=images or [])
        _log(f"📧 邮件 {'✅已发' if ok else '❌失败'}: {subject}")
    except Exception as e:
        _log(f"⚠️ 通知异常: {e}")


# ==================== 主循环 ====================

def _run() -> None:
    st = _load_state()
    last_action = float(st.get("last_action", 0))
    waiting_scan = bool(st.get("waiting_scan", False))
    qr_sent = bool(st.get("qr_sent", False))
    down_since = float(st.get("down_since", 0))

    _log("🚀 watchdog v2 启动（heartbeat TTL=%ss, qq_alive TTL=%ss%s）"
         % (HEARTBEAT_TTL, QQ_ALIVE_TTL, ", dry-run" if _DRY_RUN else ""))

    while True:
        try:
            hb_age = _file_age(HEARTBEAT_FILE)
            qa_age = _file_age(QQ_ALIVE_FILE)

            # ---- 1) bot 进程崩溃 ----
            if hb_age > HEARTBEAT_TTL:
                _log(f"💀 bot 心跳过期 {hb_age:.0f}s（> {HEARTBEAT_TTL}s）→ 重启 bot")
                if not _DRY_RUN:
                    _restart_bot()
                    last_action = time.time()
                    time.sleep(HEARTBEAT_TTL + 30)  # 给 bot 时间起心跳，防反复重启
                else:
                    time.sleep(5)  # dry-run：模拟动作耗时，避免空转
                if _ONCE:
                    break
                continue

            # ---- 2) QQ 会话掉线（假死 / 被踢）----
            # 被踢（qq_offline=bot_offline）或静默掉线（qq_alive 过期）都算掉线。
            # 被踢时 WS 可能还连着、qq_alive 仍被更新，所以必须同时看 qq_offline。
            kicked = QQ_OFFLINE_FILE.exists()
            if kicked or qa_age > QQ_ALIVE_TTL:
                if down_since == 0:
                    # 掉线起点：被踢用标记写入时间（更准），否则用检测时间
                    down_since = (QQ_OFFLINE_FILE.stat().st_mtime if kicked
                                  else time.time())
                    _log(f"💤 QQ 会话掉线（kicked={kicked}, qq_alive 已 {qa_age:.0f}s 无更新）")
                down_elapsed = time.time() - down_since

                if kicked:
                    # 被踢下线：重启没用（登录已失效），等二维码渲染出来直接发码
                    if not qr_sent and down_elapsed >= QR_RENDER_DELAY:
                        qr_sent = True
                        waiting_scan = True
                        _log(f"📷 被踢下线 {down_elapsed:.0f}s → 截屏并发邮箱")
                        img = _capture_qr()
                        _notify("🔑 灰泽满需要扫码登录",
                                f"账号被踢下线，需要扫码重新登录（{time.strftime('%H:%M:%S')}）。\n"
                                f"请看附件截图里的二维码，用手机 QQ 扫码；登录成功后会自动发通知。",
                                [img] if img else [])
                elif not waiting_scan:
                    # 静默掉线：先尝试重启 SnowLuma（多数重启能自动重连）
                    if time.time() - last_action > RESTART_COOLDOWN:
                        _log("🔁 掉线未恢复 → 重启 SnowLuma")
                        _notify("⚠️ 灰泽满掉线",
                                f"QQ 会话掉线（{time.strftime('%H:%M:%S')}），正在尝试重启 SnowLuma。")
                        if not _DRY_RUN:
                            _restart_snowluma()
                            last_action = time.time()

                    # 宽限期后仍未恢复 → 判定需要扫码 → 截屏发邮箱
                    if down_elapsed > GRACE_SECONDS and not qr_sent:
                        qr_sent = True
                        waiting_scan = True
                        _log(f"📷 掉线 {down_elapsed:.0f}s 未恢复 → 判定需要扫码，截屏并发邮箱")
                        img = _capture_qr()
                        _notify("🔑 灰泽满需要扫码登录",
                                f"掉线未恢复，需要扫码登录才能恢复（{time.strftime('%H:%M:%S')}）。\n"
                                f"请看附件截图里的二维码，用手机 QQ 扫码；登录成功后会自动发通知。",
                                [img] if img else [])
            else:
                # ---- 3) 恢复在线 ----
                if waiting_scan or down_since:
                    _log("✅ QQ 会话已恢复在线")
                    _notify("✅ 灰泽满已恢复在线",
                            f"已重新上线（{time.strftime('%H:%M:%S')}）。")
                waiting_scan = False
                qr_sent = False
                down_since = 0

            _save_state({"last_action": last_action, "waiting_scan": waiting_scan,
                         "qr_sent": qr_sent, "down_since": down_since})
        except Exception as e:
            _log(f"⚠️ watchdog 异常（继续）: {e}")
        if _ONCE:
            break
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    if "--send-test" in sys.argv:
        _notify("📮 测试邮件", "如果你在手机 QQ 邮箱看到这条，说明 SMTP 通知链路是通的。")
        sys.exit(0)
    try:
        _run()
    except KeyboardInterrupt:
        _log("👋 watchdog 退出")
