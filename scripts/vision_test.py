#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vision-test：验证视觉模型描述（本地图片文件或 http URL）。

默认视觉模型跟主模型同一条线路（gemini）；传 --model glm-4.6v 可验证智谱。

用法：
    python scripts/vision_test.py -u 本地图片路径或http图链
    python scripts/vision_test.py -u 图链 --model glm-4.6v
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


async def main(source: str, model: str) -> None:
    import nonebot

    nonebot.init()   # config/vision 都是插件包内模块，导入前必须先初始化驱动
    from src.plugins.chatbot.config import get_vision_client, get_vision_model
    from src.plugins.chatbot.vision import describe_image

    model = model or get_vision_model()
    client = get_vision_client()
    print(f"视觉模型: {model} @ {client.base_url}")
    desc = await describe_image(client, source, model=model)
    print(f"视觉描述: {desc if desc else '（空——看上方是否有⚠️警告）'}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="验证视觉模型描述")
    parser.add_argument("-u", "--image", required=True, help="本地图片路径或 http(s) 图链")
    parser.add_argument("--model", default=None,
                        help=f"视觉模型（默认 {_load_env('VISION_MODEL') or 'gemini-3.1-flash-lite'}）")
    args = parser.parse_args()
    asyncio.run(main(args.image, args.model))
