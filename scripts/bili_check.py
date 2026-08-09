#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bili-check：手动验证 B站 UID 的直播状态 + 最新动态。

不依赖 bot 运行，用于确认：
1. UID 对不对（接口是否返回该用户）
2. 直播状态接口通不通
3. 动态接口（含 WBI 签名）通不通、正文/配图解析对不对

用法：
    python scripts/bili_check.py --uid 1298779265
    python scripts/bili_check.py            # 缺省读 .env.prod 的 BILI_UID
"""
import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import ensure_utf8_stdout  # noqa: E402


def _load_env(name: str):
    try:
        for line in (PROJECT_ROOT / ".env.prod").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


async def main(uid: int) -> None:
    import httpx

    sessdata = _load_env("BILI_SESSDATA")

    # ---- 直播状态 ----
    print("== 直播状态（无需登录） ==")
    _UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                "https://api.live.bilibili.com/room/v1/Room/get_status_info_by_uids",
                json={"uids": [uid]},
                headers={"User-Agent": _UA, "Referer": "https://live.bilibili.com/"},
            )
            data = resp.json()
        print(f"接口 code: {data.get('code')}")
        info = (data.get("data") or {}).get(str(uid))
        if info:
            status = info.get("live_status")
            status_text = {0: "未开播", 1: "直播中", 2: "轮播中"}.get(status, f"未知({status})")
            print(f"直播状态: {status_text}")
            print(f"房间标题: {info.get('title')}")
            print(f"房间号: {info.get('room_id') or info.get('roomid')}")
        else:
            print("❌ 接口未返回该 UID 数据——UID 可能不对")
    except Exception as e:
        print(f"❌ 直播状态查询失败: {e}")

    # ---- 最新动态（需 SESSDATA） ----
    print("\n== 最新动态 ==")
    if not sessdata:
        print("⚠️ .env.prod 未配置 BILI_SESSDATA——动态接口需登录，跳过。\n"
              "   想启用动态监控：B站网页版登录你自己的号 → F12 → Application → Cookies →\n"
              "   复制 SESSDATA 的值填进 .env.prod 的 BILI_SESSDATA=")
        return
    try:
        from bilibili_api import dynamic
        from bilibili_api.utils.network import Credential
        from src.plugins.chatbot.bili_bridge import (
            _extract_dynamic_text, _extract_dynamic_images,
        )
        cred = Credential(sessdata=sessdata)
        page = await dynamic.get_dynamic_page_info(cred, host_mid=uid, pn=1)
        items = (page or {}).get("items") or []
        print(f"动态接口: 获取到 {len(items)} 条")
        if items:
            item = items[0]
            print(f"最新动态 id: {item.get('id_str') or item.get('id')}")
            text = _extract_dynamic_text(item)
            imgs = _extract_dynamic_images(item)
            print(f"动态正文: {text[:120]}")
            print(f"配图数: {len(imgs)}")
        else:
            print("该 UID 暂无可见动态")
    except Exception as e:
        print(f"❌ 动态查询失败: {e}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="验证 B站 UID 直播状态与最新动态")
    parser.add_argument("--uid", default=None, help="B站 UID（缺省读 .env.prod 的 BILI_UID）")
    args = parser.parse_args()
    uid = args.uid or _load_env("BILI_UID")
    if not uid:
        print("❌ 未提供 --uid 且 .env.prod 无 BILI_UID")
        sys.exit(1)
    asyncio.run(main(int(uid)))
