"""注入块的加载与渲染：20 块，一块一个文件夹。

## 为什么有它

在 2026-10-16 之前，**「一个注入块」这个概念在代码里根本不存在**——它是
`core.build_message_list` 里几十个 `append` 的副产物。文案散在四个 py 文件
（core / memory_v2 / session_memory / context_probe），数据散在 persona 的
world/behavior 两个目录。想看"第 7 块到底注了什么"得同时翻三处；骨架里说
`【行为指令】`、代码里叫`【当前情境下的行为指令】`这种脱节就是这么来的
（改名时只改了骨架，代码那边没人知道）。

现在：`persona/injection/<序号_块名>/` 一个文件夹 = 一个注入块，
里面 `block.json` 装着**段头 + 文案模板 + 它从哪拿内容 + 什么时候注**。
目录名的序号**就是注入顺序**（顺序是有讲究的：情绪紧挨用户消息、当前时间放最后
是为了前缀缓存），所以顺序的唯一真值也是文件夹名，不再散在 append 的先后里。

## 这个模块只做「取文案」

**什么时候注、注哪条变体、参数从哪来，仍然由调用方决定**。这样"文字"和"逻辑"
各归各：改一句话不用碰代码，改判断不用碰文案。

## 加一块要做什么

建一个 `NN_块名/` 文件夹，里面放 `block.json`，然后让调用方在合适的时机
`inj.T("块名", 键, **参数)`。`_readme` 写清楚这块管什么、别把什么写进来。
"""
import json

from .constants import INJECTION_DIR, BLOCK_FILE, SYSTEM_PROMPT_FILE

_blocks_cache: dict | None = None
_raw_cache: dict = {}


def load_blocks() -> dict:
    """扫描 `persona/injection/`，返回 {id: block}；**按文件夹名排序**（序号即顺序）。

    文件坏掉/缺 `id` 的块会被跳过而不是抛异常——**注入链不能因为一个块写坏了就整条挂掉**。
    真正的检查交给 `tests/test_injection.py`，那里会红。
    """
    global _blocks_cache
    if _blocks_cache is not None:
        return _blocks_cache
    blocks: dict = {}
    if INJECTION_DIR.exists():
        for d in sorted(INJECTION_DIR.iterdir(), key=lambda p: p.name):
            if not d.is_dir():
                continue
            p = d / BLOCK_FILE
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and data.get("id"):
                data.setdefault("dir", d.name)
                blocks[data["id"]] = data
    _blocks_cache = blocks
    return blocks


def block(block_id: str) -> dict:
    """取一个块的元数据；没有就抛——**打错块名要立刻炸**，不要静默注个空的。"""
    blocks = load_blocks()
    if block_id not in blocks:
        raise KeyError(f"没有这个注入块：{block_id}（现有：{', '.join(blocks)}）")
    return blocks[block_id]


def order() -> list:
    """按注入顺序返回块 id。"""
    return sorted(load_blocks(), key=lambda k: load_blocks()[k].get("order", 999))


def T(block_id: str, key: str = "main", **kwargs) -> str:
    """渲染一个块的文案模板。`key` 是 `strings` 里的名字（单条模板的块用默认的 "main"）。

    占位符用 `str.format` 的风格（`{weekly}`）。**缺参数会直接 KeyError**，
    这是故意的：宁可炸在测试里，也不要注一句缺了半截的话给她。
    """
    b = block(block_id)
    strings = b.get("strings") or {}
    if key not in strings:
        raise KeyError(f"块 {block_id} 里没有字符串 {key}（现有：{', '.join(strings)}）")
    return strings[key].format(**kwargs)


def raw(block_id: str) -> str:
    """`kind=raw_file` 的块：整块内容就是那个文件（目前只有骨架）。"""
    b = block(block_id)
    if b.get("kind") != "raw_file":
        raise TypeError(f"块 {block_id} 不是 raw_file，用 T() 取它的文案")
    if block_id not in _raw_cache:
        p = (INJECTION_DIR / b["dir"] / b["source"]).resolve()
        _raw_cache[block_id] = p.read_text(encoding="utf-8")
    return _raw_cache[block_id]


def behavior_sentence(key: str, value: str) -> str:
    """第 3 块（相处方式）的「键 → 行为句」查表；没有对应句子返回空串。

    键形如 `("support_style", "comfort")`，在 block.json 里写成 `support_style.comfort`。
    """
    table = (block("profile").get("strings") or {}).get("behaviors") or {}
    return table.get(f"{key}.{value}", "")


def weekly_row(x: dict) -> str:
    """周表一行 → 注入文本。`周一 18:00（唱歌）`。

    `time` **不归一**：图上写"动态通知"就原样进来——换成"无固定时间"等于把
    "还没定、看动态"这条信息扔掉（2026-10-16 用户提的，旧版识图脚本就是这么丢的）。
    """
    head = f"{x.get('day', '')} {x.get('time', '')}".strip()
    extra = [str(x.get(k) or "").strip() for k in ("topic", "note")]
    extra = [e for e in extra if e]
    return f"{head}（{'／'.join(extra)}）" if extra else head


def format_weekly(weekly: list) -> str:
    """周表整体 → 一行注入文本。"""
    return "、".join(weekly_row(x) for x in weekly if x.get("day"))


def system_prompt() -> str:
    """骨架全文（第 1 块）。`core.py` 启动时读的就是它。"""
    return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")


def overview() -> str:
    """把 20 块列成一页——给人看「到底注入了什么」。"""
    lines = ["序号  块名                何时注入", "-" * 68]
    for bid in order():
        b = load_blocks()[bid]
        lines.append(f"{b.get('order', '?'):>3}  {b.get('title', bid):<18} {b.get('when', '')}")
    return "\n".join(lines)


__all__ = ["load_blocks", "block", "order", "T", "raw", "behavior_sentence",
           "weekly_row", "format_weekly", "system_prompt", "overview"]