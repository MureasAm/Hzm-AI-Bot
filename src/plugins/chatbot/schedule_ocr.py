# -*- coding: utf-8 -*-
"""周表图 OCR：把"每周直播时间表"图写成 persona/injection/02_schedule/schedule.json 的 weekly。

**单一真值**：手动脚本 `scripts/update_schedule.py`（双击「更新周表.bat」）和
B 站置顶周表自动检测（`bili_bridge._push_week_table`）都走这里，别再各写一份 OCR。

识图走阿里云百炼千问（DASHSCOPE），key 默认从 `D:\\CLAUDE CODE\\.env` 读，
也可用环境变量 `SCHEDULE_OCR_API_KEY` / `SCHEDULE_OCR_MODEL` 覆盖。
写完会清掉 `persona` 的周表缓存，**运行中的 bot 下一轮就能用新表，不需要重启**。
"""
import asyncio
import base64
import json
import os
import re
from pathlib import Path

import httpx

from .constants import SCHEDULE_FILE

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VISION_ENV = Path(r"D:\CLAUDE CODE\.env")
_DAY_ORDER = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
_DAY_ALIASES = {
    "周一": "周一", "星期一": "周一", "礼拜一": "周一", "MON": "周一", "MONDAY": "周一",
    "周二": "周二", "星期二": "周二", "礼拜二": "周二", "TUE": "周二", "TUESDAY": "周二",
    "周三": "周三", "星期三": "周三", "礼拜三": "周三", "WED": "周三", "WEDNESDAY": "周三",
    "周四": "周四", "星期四": "周四", "礼拜四": "周四", "THU": "周四", "THURSDAY": "周四",
    "周五": "周五", "星期五": "周五", "礼拜五": "周五", "FRI": "周五", "FRIDAY": "周五",
    "周六": "周六", "星期六": "周六", "礼拜六": "周六", "SAT": "周六", "SATURDAY": "周六",
    "周日": "周日", "周天": "周日", "星期日": "周日", "星期天": "周日", "礼拜天": "周日",
    "SUN": "周日", "SUNDAY": "周日",
}

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


def _ocr_config() -> tuple[str, str]:
    key = os.environ.get("SCHEDULE_OCR_API_KEY") or ""
    model = os.environ.get("SCHEDULE_OCR_MODEL") or ""
    if not key and VISION_ENV.exists():
        try:
            txt = VISION_ENV.read_text(encoding="utf-8")
            m = re.search(r"DASHSCOPE_API_KEY=(\S+)", txt)
            key = m.group(1) if m else ""
            if not model:
                m = re.search(r"VISION_MODEL=(\S+)", txt)
                model = m.group(1) if m else ""
        except OSError:
            pass
    return key, model or "qwen3.5-omni-plus"


def _image_bytes_and_mime(image) -> tuple[bytes, str]:
    if isinstance(image, (bytes, bytearray)):
        data = bytes(image)
    else:
        data = Path(image).read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8"):
        mime = "image/jpeg"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    elif data.startswith(b"BM"):
        mime = "image/bmp"
    else:
        mime = "image/png"
    return data, mime


def _normalize_day(raw) -> str:
    s = str(raw or "").strip()
    return _DAY_ALIASES.get(s.upper(), _DAY_ALIASES.get(s, ""))


def clean_weekly(arr) -> list[dict]:
    """把 OCR 的数组收口成 7 天；缺的补"无"，并拒绝明显无效的结果。"""
    if not isinstance(arr, list) or not arr:
        raise ValueError(f"OCR 结果不是数组：{str(arr)[:200]}")
    clean = []
    for x in arr:
        if not isinstance(x, dict):
            continue
        day = _normalize_day(x.get("day"))
        if not day:
            continue
        clean.append({
            "day": day,
            "time": str(x.get("time") or "").strip(),
            "topic": str(x.get("topic") or "").strip(),
            "note": str(x.get("note") or "").strip(),
        })
    if not clean:
        raise ValueError("OCR 结果里没有一个可识别的星期")
    for d in _DAY_ORDER:
        if d not in {c["day"] for c in clean}:
            clean.append({"day": d, "time": "无", "topic": "", "note": ""})
    clean.sort(key=lambda x: _DAY_ORDER.index(x["day"]) if x["day"] in _DAY_ORDER else 99)
    useful = [c for c in clean if c["time"] not in ("", "无", "休", "—", "-")]
    if not useful:
        raise ValueError("OCR 结果里没有任何有效时间，拒绝覆盖旧周表")
    return clean


async def ocr_schedule_image(image) -> list[dict]:
    key, model = _ocr_config()
    if not key:
        raise RuntimeError("没找到 DASHSCOPE_API_KEY（环境变量或 D:\\CLAUDE CODE\\.env）")
    data, mime = _image_bytes_and_mime(image)
    b64 = base64.b64encode(data).decode()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": PROMPT},
        ]}],
        "max_tokens": 900,
    }
    # trust_env=False：百炼是国内直连，别被 VPN 代理带偏
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        r = await client.post(
            DASHSCOPE_BASE + "/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )
        body = r.json()
    if "choices" not in body:
        raise RuntimeError(f"识图接口没返回结果：{str(body)[:300]}")
    txt = re.sub(r"```json|```", "", body["choices"][0]["message"]["content"]).strip()
    return clean_weekly(json.loads(txt))


def write_schedule_weekly(weekly: list[dict], path: Path = SCHEDULE_FILE) -> Path:
    """只覆盖 `weekly`，其它字段（_readme）原样保留；写完清缓存。"""
    path = Path(path)
    cur = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cur, dict):
        raise ValueError(f"{path} 不是对象，拒绝覆盖")
    cur["weekly"] = weekly
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    try:
        from .persona import invalidate_schedule_cache
        invalidate_schedule_cache()
    except Exception:
        pass
    return path


async def auto_update_schedule(image, timeout: float = 90.0) -> bool:
    """B 站检测到新周表图后调用：识图→写 weekly→清缓存。失败保留旧表，返回 False。"""
    try:
        weekly = await asyncio.wait_for(ocr_schedule_image(image), timeout=timeout)
        write_schedule_weekly(weekly)
        print(f"[周表] 已自动识图并写入 schedule.json（{Path(SCHEDULE_FILE).name}，无需重启）")
        return True
    except Exception as e:
        print(f"⚠️ 周表自动识图失败（保留旧表）: {type(e).__name__}: {e}")
        return False
