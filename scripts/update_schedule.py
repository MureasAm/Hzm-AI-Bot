#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""半自动更新灰泽满周表：识图 OCR 一张"每周直播时间表"图 → 更新 persona/world/schedule.json 的 weekly。

识图走阿里云百炼千问（DASHSCOPE，D:\\CLAUDE CODE\\.env 里配好的 key，和 vision.js 同一套），
不用手动把图上的字抄下来。

用法：
    python scripts/run_tool.py ...  # 或直接：
    python scripts/update_schedule.py <图片路径> [--preview]   # --preview 只看结果不写文件

半自动的含义：weekly（固定周表）由 OCR 填好；「近况」那一行是临时安排，OCR 不猜，
需要的话自己顺手在 schedule.json 里改，并把 近况_updated 更新成当天。
改完 / 脚本写完后要【重启 bot】才生效（周表有进程内缓存）。
"""
import json
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "persona" / "world" / "schedule.json"
VISION_ENV = Path(r"D:\CLAUDE CODE\.env")
_DAY_ORDER = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

PROMPT = """这是虚拟主播的每周直播时间表图。把表逐行转成 JSON 数组，格式严格如下：
[{"day":"周一","time":"19:00"},{"day":"周二","time":"19:00"},...]
day 用 周一~周日 中文，time 写图上的时间（没有固定时间的写"无固定时间"，休播写"休"），
共 7 项、按周一到周日的顺序。只输出 JSON，不要其他文字。"""


def _key():
    txt = VISION_ENV.read_text(encoding="utf-8")
    k = re.search(r"DASHSCOPE_API_KEY=(\S+)", txt)
    m = re.search(r"VISION_MODEL=(\S+)", txt)
    return (k.group(1) if k else None, m.group(1) if m else "qwen3.5-omni-plus")


def ocr(image: Path):
    import base64
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
            "max_tokens": 500,
        }, timeout=90,
    )
    txt = r.json()["choices"][0]["message"]["content"]
    txt = re.sub(r"```json|```", "", txt).strip()
    arr = json.loads(txt)
    if not isinstance(arr, list) or len(arr) != 7:
        raise SystemExit(f"[ERR] OCR 结果不是 7 项：{txt[:200]}")
    return arr


def main():
    preview = "--preview" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        img = Path(args[0])
        if not img.exists():
            raise SystemExit(f"[ERR] 图片不存在: {img}")
    else:
        # 收图夹模式：把周表图丢进 data/schedule_inbox/，不带参数跑即处理最新一张
        inbox = ROOT / "data" / "schedule_inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        cands = [p for p in inbox.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")]
        if not cands:
            raise SystemExit(f"收图夹里没有图片。请把周表图丢进: {inbox}\n然后运行:\n  python scripts/update_schedule.py")
        img = max(cands, key=lambda p: p.stat().st_mtime)
    weekly = ocr(img)
    # 按周一~周日排好序
    weekly.sort(key=lambda x: _DAY_ORDER.index(x.get("day")) if x.get("day") in _DAY_ORDER else 99)

    cur = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    cur["weekly"] = weekly
    print(f"识图: {img.name}\nweekly:")
    for x in weekly:
        print(f"  {x['day']}  {x['time']}")
    print(f"\n原近况仍保留: {cur.get('近况','')!r}（如需改临时安排/请假，编辑 {SCHEDULE} 那行）")
    if not preview:
        SCHEDULE.write_text(json.dumps(cur, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] 已写入 {SCHEDULE} —— 记得【重启 bot】生效（周表有缓存）。")
        if args:
            print("（传入的是文件参数，未移动原图）")
        else:
            done = ROOT / "data" / "schedule_inbox" / "done"
            done.mkdir(parents=True, exist_ok=True)
            img.rename(done / img.name)
            print(f"[OK] 已把处理完的图移进: {done}")


if __name__ == "__main__":
    main()
