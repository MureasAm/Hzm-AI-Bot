#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""半自动更新灰泽满周表：识图 OCR 一张"每周直播时间表"图 → 写 persona/injection/02_schedule/schedule.json。

识图走阿里云百炼千问（DASHSCOPE，D:\\CLAUDE CODE\\.env 里配好的 key，和 vision.js 同一套），
不用手动把图上的字抄下来。

用法：
    python scripts/update_schedule.py <图片路径> [--preview]   # --preview 只看结果不写文件
    不带参数跑：从 data/schedule_inbox/ 取最新一张图（双击「更新周表.bat」拖图就是这个模式）

本脚本只覆盖 `weekly` 字段，同文件里的其它字段（`_readme`）原样保留。
写完后要【重启 bot】才生效（周表有进程内缓存）。

## 一周一条长什么样

    {"day": "周一", "time": "18:00", "topic": "唱歌", "note": ""}

`time` **照抄图上的原话**——图上写"动态通知"就存"动态通知"，**不要**换成"无固定时间"。
（旧版把两者归一成"无固定时间"，等于把"还没定、看动态"这条信息扔掉，模型只会当成"这天不播"。）
`topic` 是这天播什么；`note` 兜其它列。后三个字段都可以是空串。

## 为什么字段是这四个

`day` 固定中文星期（模型要按周算"明天播不播"）；`time` 保留原话不归一（见上）；
`topic` / `note` 是**照抄**不是归纳——图上写了什么就是什么，别自己总结成"常规直播"。
"""
import argparse
import base64
import json
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "persona" / "injection" / "02_schedule" / "schedule.json"
VISION_ENV = Path(r"D:\CLAUDE CODE\.env")
_DAY_ORDER = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

PROMPT = """这是一张虚拟主播的每周直播时间表图。请**逐行照抄**，不要归纳、不要改写用词、不要翻译。

输出一个 JSON 数组，每行一个对象，字段固定为这四个：
[{"day": "周一", "time": "18:00", "topic": "唱歌", "note": ""}, ...]

- day：转成中文星期（周一 / 周二 / … / 周日）。共 7 项，按周一到周日排序。
- time：**照抄"时间"那一格里的原话**。有时钟就写时钟（如 18:00）；没有时钟就照抄格子里写的字，
  例如"动态通知""待定""休"——**原样写，不要换成"无固定时间"或别的说法**。
- topic：**照抄这一行"播什么 / 主题"那一格的原话**（如 唱歌、杂谈、游戏、学习、杂谈+点歌）；没有就写空串。
- note：图上**其它列**（备注之类）的原话，用"／"连接；没有就写空串。
- 图上没有出现的星期，time 写"无"。

只输出 JSON，不要其他文字。"""


def _key():
    txt = VISION_ENV.read_text(encoding="utf-8")
    k = re.search(r"DASHSCOPE_API_KEY=(\S+)", txt)
    m = re.search(r"VISION_MODEL=(\S+)", txt)
    return (k.group(1) if k else None, m.group(1) if m else "qwen3.5-omni-plus")


def ocr(image: Path) -> list:
    key, model = _key()
    if not key:
        raise SystemExit("[ERR] 没找到 DASHSCOPE_API_KEY（在 D:\\CLAUDE CODE\\.env）")
    b64 = base64.b64encode(image.read_bytes()).decode()
    r = httpx.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ]}],
            "max_tokens": 900,
        }, timeout=120,
    )
    body = r.json()
    if "choices" not in body:
        raise SystemExit(f"[ERR] 识图接口没返回结果：{str(body)[:300]}")
    txt = re.sub(r"```json|```", "", body["choices"][0]["message"]["content"]).strip()
    arr = json.loads(txt)
    if not isinstance(arr, list) or not arr:
        raise SystemExit(f"[ERR] OCR 结果不是数组：{txt[:200]}")
    clean = []
    for x in arr:
        if not isinstance(x, dict) or not x.get("day"):
            continue
        clean.append({
            "day": str(x.get("day")).strip(),
            "time": str(x.get("time") or "").strip(),
            "topic": str(x.get("topic") or "").strip(),
            "note": str(x.get("note") or "").strip(),
        })
    missing = [d for d in _DAY_ORDER if d not in {c["day"] for c in clean}]
    for d in missing:                       # 图上缺的那天补一条空的，别让周表少一格
        clean.append({"day": d, "time": "无", "topic": "", "note": ""})
    clean.sort(key=lambda x: _DAY_ORDER.index(x["day"]) if x["day"] in _DAY_ORDER else 99)
    return clean


def show(weekly: list) -> None:
    print(f"{'星期':<4} {'时间':<12} {'主题':<16} 备注")
    print("-" * 52)
    for x in weekly:
        print(f"{x['day']:<4} {x['time'] or '—':<12} {x['topic'] or '—':<16} {x['note']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="识图更新周表")
    ap.add_argument("image", nargs="?", help="周表图路径；不给就从 data/schedule_inbox/ 取最新一张")
    ap.add_argument("--preview", action="store_true", help="只看结果，不写文件")
    args = ap.parse_args()

    if args.image:
        img = Path(args.image)
        if not img.exists():
            raise SystemExit(f"[ERR] 图片不存在: {img}")
        from_inbox = False
    else:
        inbox = ROOT / "data" / "schedule_inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        cands = [p for p in inbox.iterdir()
                 if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")]
        if not cands:
            raise SystemExit(f"收图夹里没有图片。请把周表图丢进: {inbox}\n"
                             f"然后重跑本脚本（或双击「更新周表.bat」）")
        img = max(cands, key=lambda p: p.stat().st_mtime)
        from_inbox = True

    weekly = ocr(img)
    print(f"识图: {img.name}\n")
    show(weekly)

    if args.preview:
        print("\n（--preview：没写文件）")
        return 0

    cur = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    cur["weekly"] = weekly
    SCHEDULE.write_text(json.dumps(cur, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[OK] 已只更新 weekly（{SCHEDULE.name} 的其它字段原样保留）")
    print("     记得【重启 bot】生效（周表有进程内缓存）。")
    if from_inbox:
        done = ROOT / "data" / "schedule_inbox" / "done"
        done.mkdir(parents=True, exist_ok=True)
        img.rename(done / img.name)
        print(f"[OK] 已把处理完的图移进: {done}")
    return 0


if __name__ == "__main__":
    sys.exit(main())