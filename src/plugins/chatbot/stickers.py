# -*- coding: utf-8 -*-
"""表情包：判断她这句话配哪张表情，十分对应时才发。

**为什么要先打标**：表情包文件名是一串哈希，模型不知道每张画的是什么、什么时候该发。
`scripts/label_stickers.py` 离线用视觉模型问出 desc/tags/use_when，存
`persona/media/stickers.json`。加新表情：丢进 assets/stickers/ 再跑一次那个脚本。

**为什么用 LLM 判、不用向量阈值**：要的是"**十分**对应"。阈值调不准——
群聊接话门那边已经吃过教训（同一份数据、判据稍动就从 0% 跳到 74%）。
24 张的清单塞进提示词很便宜，让模型直接判"有没有十分对应的、没有就 null"更稳。

**频率**：真人也不会每轮都甩表情包。调用方负责冷却（见 chat_window 的 `_UserWindow`）。
"""
import json
import os

from .constants import PROJECT_ROOT, THINKING_DISABLED
from .config import _get_model_name, extract_chat_content

STICKER_FILE = PROJECT_ROOT / "persona" / "media" / "stickers.json"

_stickers_cache = None

STICKER_PROMPT = """灰泽满刚说了一句话，判断要不要给她配一张表情包。

【她刚说的】
{reply}

【可选表情包】
{options}{avoid}

规则：
- **只有表情和这句话十分对应时才发。宁可不发**——发错比不发难看得多。
- 大多数回复都不配表情包（真人也是聊好几轮才用一张）。
- 表情是**配她自己这句话的情绪**，不是评论别人。

只输出 JSON：{{"id": "选中的表情 id" 或 null, "why": "不超过15字"}}"""


def sticker_enabled() -> bool:
    """环境变量 STICKER=0 可整体关掉（快速回滚用），缺省开。"""
    return os.environ.get("STICKER", "1") != "0"


def load_stickers() -> list:
    """加载表情标注（模块级缓存）。文件缺失/写坏时返回空列表，调用方当作没表情。"""
    global _stickers_cache
    if _stickers_cache is not None:
        return _stickers_cache
    _stickers_cache = []
    if STICKER_FILE.exists():
        try:
            data = json.loads(STICKER_FILE.read_text(encoding="utf-8"))
            items = data.get("stickers", []) if isinstance(data, dict) else []
            # 文件不存在于磁盘的条目要剔掉：标注是离线生成的，图可能被手动删过
            _stickers_cache = [
                s for s in items
                if isinstance(s, dict) and s.get("id") and s.get("file")
                and (PROJECT_ROOT / s["file"]).exists()
            ]
        except (json.JSONDecodeError, OSError):
            _stickers_cache = []
    return _stickers_cache


def _options_text(items: list) -> str:
    lines = []
    for s in items:
        tag = "/".join(s.get("tags") or [])
        lines.append(f"- {s['id']}：{s.get('desc', '')}（{tag}）适合：{s.get('use_when', '')}")
    return "\n".join(lines)


async def pick_sticker(deepseek_client, reply: str, avoid_ids: list | None = None) -> dict | None:
    """给这句话挑一张表情；没有十分对应的返回 None。

    avoid_ids：最近发过的，提示模型别连着用同一张。
    失败一律返回 None（**不发**）——发错表情比不发难看得多，这里和接话门相反，
    宁可少发不可发错。
    """
    if not sticker_enabled() or not (reply or "").strip():
        return None
    items = load_stickers()
    if not items:
        return None

    avoid = ""
    if avoid_ids:
        names = "、".join(avoid_ids)
        avoid = f"\n\n【最近已经发过的，别再选】{names}"

    try:
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": STICKER_PROMPT.format(
                reply=reply.strip(), options=_options_text(items), avoid=avoid)}],
            temperature=0,
            max_tokens=80,
            **THINKING_DISABLED,
        )
        content = extract_chat_content(resp)
        if "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
        content = content.strip()
        if content[:4].lower() == "json":
            content = content[4:].strip()
        data = json.loads(content)
        sid = data.get("id")
        if not sid:
            return None
        hit = next((s for s in items if s["id"] == sid), None)  # 拦住模型编的 id
        if hit is None:
            print(f"⚠️ 表情判定给了不存在的 id={sid!r}，忽略")
            return None
        print(f"[表情包] 选 {sid}（{str(data.get('why') or '')[:20]}）")
        return hit
    except Exception as e:
        print(f"⚠️ 表情包判定失败（不发）: {e}")
        return None
