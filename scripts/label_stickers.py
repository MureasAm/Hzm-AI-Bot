#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 assets/stickers/ 里的表情包打标，产出 persona/media/stickers.json。

**为什么要打标**：表情包文件名是一串哈希，模型不知道每张画的是什么、什么时候该发。
运行时要用「回复内容 ↔ 表情含义」匹配，所以先离线把每张的**画面描述 + 语义标签 +
什么时候用**问出来存好。这样运行时只需检索，不用每次让模型看图（贵且慢）。

用法：
    python scripts/label_stickers.py            # 只打没打过的（幂等，可反复跑）
    python scripts/label_stickers.py --force    # 全部重打

加了新表情：丢进 assets/stickers/ 再跑一次本脚本即可。

注意：调用视觉模型（glm-4.6v），会花钱；24 张量级很便宜。
"""
import argparse
import asyncio
import base64
import io
import json
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env.prod"
STICKER_DIR = PROJECT_ROOT / "assets" / "stickers"
OUT_FILE = PROJECT_ROOT / "persona" / "media" / "stickers.json"

ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_VISION_MODEL = "glm-4.6v"

PROMPT = """这是一张 QQ 聊天里用的表情包。请判断它的**含义和用法**，供一个聊天机器人挑选使用。

只输出 JSON，不要别的：
{"desc": "画面上是什么（谁、什么动作/表情、有没有文字，30字内）",
 "tags": ["3~6个语义标签，如 无语/嘲笑/赞同/无奈/得意/崩溃/催更/晚安"],
 "use_when": "什么情况下适合发这张（一句话，如'对方说了离谱的话、她想表达无语时'）"}

要求：
- 标签用**聊天里会出现的情绪/意图词**，不要用"卡通""可爱"这种画风词
- use_when 写"什么时候发"，不是"画面是什么"
- 如果画面上有文字，把文字也写进 desc"""


def _key(name: str) -> str:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(name):
                return line.split("=", 1)[1].replace('"', "").strip()
    return ""


def _vision_model() -> str:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("VISION_MODEL"):
                return line.split("=", 1)[1].replace('"', "").strip()
    return DEFAULT_VISION_MODEL


def _data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def _parse(text: str) -> dict:
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        if len(parts) >= 2:
            t = parts[1]
    t = t.strip()
    if t[:4].lower() == "json":
        t = t[4:].strip()
    return json.loads(t)


async def label_one(client, model: str, path: Path) -> dict | None:
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _data_uri(path)}},
                {"type": "text", "text": PROMPT},
            ]}],
            max_tokens=300,
            extra_body={"thinking": {"type": "disabled"}},
        )
        data = _parse(r.choices[0].message.content or "")
        return {
            "id": path.stem,
            "file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "desc": str(data.get("desc", "")).strip(),
            "tags": [str(x).strip() for x in (data.get("tags") or []) if str(x).strip()],
            "use_when": str(data.get("use_when", "")).strip(),
        }
    except Exception as e:
        print(f"  ❌ {path.name}: {e}")
        return None


async def run(force: bool = False, out_file: str | None = None):
    out_path = Path(out_file) if out_file else OUT_FILE
    files = sorted(STICKER_DIR.glob("*.png")) if STICKER_DIR.exists() else []
    if not files:
        raise SystemExit(f"[ERR] {STICKER_DIR} 里没有 png")

    existing = {}
    if out_path.exists() and not force:
        try:
            existing = {s["id"]: s for s in json.loads(out_path.read_text(encoding="utf-8"))["stickers"]}
        except Exception:
            existing = {}

    todo = [f for f in files if f.stem not in existing]
    print(f"共 {len(files)} 张，已有 {len(existing)} 张，待打标 {len(todo)} 张")
    if not todo:
        print("没有要打标的，退出。")
        return

    key = _key("ZHIPU_API_KEY")
    if not key:
        raise SystemExit("[ERR] .env.prod 里没有 ZHIPU_API_KEY")
    client = AsyncOpenAI(api_key=key, base_url=ZHIPU_BASE_URL)
    model = _vision_model()

    result = list(existing.values())
    for i, f in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {f.name} ...", end=" ", flush=True)
        rec = await label_one(client, model, f)
        if rec:
            print(rec["desc"][:30])
            result.append(rec)

    result.sort(key=lambda x: x["id"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"_readme": "表情包标注（scripts/label_stickers.py 生成）。运行时按 desc+tags+use_when "
                    "与回复内容匹配；改动后无需重算向量（未走向量，见 media/stickers.py 的说明）。",
         "stickers": result},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n✅ 写入 {out_path}（{len(result)} 张）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="全部重打（默认只打没打过的）")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()
    asyncio.run(run(force=args.force, out_file=args.output))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
