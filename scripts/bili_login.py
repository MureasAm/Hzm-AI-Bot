#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B站扫码登录：生成二维码 → 手机哔哩哔哩 App 扫码 → 自动把新 SESSDATA 写进 .env.prod。

替代手动复制浏览器 cookie。SESSDATA 过期（日志报 -101 账号未登录）时跑一次，扫码即刷新。

用法:
    python scripts/bili_login.py          # 生成二维码 PNG + 等扫码
    python scripts/run_tool.py bili-login # 或走统一入口

成功后写入：BILI_SESSDATA / BILI_BILI_JCT / BILI_BUVID3 / BILI_DEDEUSERID
（bot 用 SESSDATA 已够，其余一并存下备用）
"""
import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env.prod"
QR_PNG = PROJECT_ROOT / "outputs" / "bili_qrcode.png"


def _save_env(key: str, value: str) -> None:
    if not value:
        return
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    line = f"{key}={value}"
    replaced = False
    for i, ln in enumerate(lines):
        if ln.startswith(key + "="):
            lines[i] = line
            replaced = True
            break
    if not replaced:
        lines.append(line)
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  ✅ 已写入 {key}")


async def main() -> int:
    from bilibili_api import login_v2

    print("🚀 正在获取 B站 登录二维码…")
    qr = login_v2.QrCodeLogin(platform=login_v2.QrCodeLoginChannel.WEB)
    await qr.generate_qrcode()

    # 终端二维码（可选，依赖 qrcode_terminal）
    try:
        qr.get_qrcode_terminal()
    except Exception as e:
        print(f"（终端二维码不可用，用图片）: {e}")

    # 二维码图片 → PNG
    try:
        pic = qr.get_qrcode_picture()
        QR_PNG.parent.mkdir(parents=True, exist_ok=True)
        pic.to_file(str(QR_PNG))
        print(f"📱 二维码已保存: {QR_PNG}")
        print(f"   用哔哩哔哩 App 扫一扫 → 确认登录")
    except Exception as e:
        print(f"⚠️ 二维码图片生成失败: {e}")

    print("⏳ 等待扫码确认…（Ctrl+C 取消）")
    while not qr.has_done():
        st = await qr.check_state()
        if st == login_v2.QrCodeLoginEvents.SCAN:
            print("  ⏳ 等待扫码…")
        elif st == login_v2.QrCodeLoginEvents.CONF:
            print("  ✅ 已扫码，请在手机上确认登录！")
        elif st == login_v2.QrCodeLoginEvents.TIMEOUT:
            print("❌ 二维码已过期，请重跑一次。")
            return 1
        await asyncio.sleep(2)

    cred = qr.get_credential()
    _save_env("BILI_SESSDATA", cred.sessdata or "")
    _save_env("BILI_BILI_JCT", cred.bili_jct or "")
    _save_env("BILI_BUVID3", cred.buvid3 or "")
    _save_env("BILI_DEDEUSERID", cred.dedeuserid or "")
    print(f"\n🎉 登录成功！新 SESSDATA 已写入 {ENV_FILE}")
    print("   bot 下次轮询即生效（已支持免重启），B站动态监控恢复。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n已取消")
        sys.exit(130)
