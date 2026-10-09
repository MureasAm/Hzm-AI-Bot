"""短期记忆读写，带进程内文件锁。**一个会话一个文件**（`user_memory/short_term/<id>.json`）。

为什么按会话拆文件（2026-10-08）：
- 原来全挤在 `short_term.json` 一个文件里 —— 每条消息都要**把全部会话读出来、整份写回去**，
  写放大随用户数线性增长（实测已经 175 会话 / 223KB，还在涨）。
- 单文件也不方便人工查看：谁的对话都糊在一起。
- 拆开之后：一次读写只碰一个会话（O(1)）；**删某人的历史 = 删一个文件**。
- 旧单文件首次运行会自动拆分，原件改名为 `.json.migrated` 留下（不删数据）。

NoneBot 单进程运行，同一时刻可能有多条消息触发写入，
用 threading.Lock 保证「读-改-写」原子化，防止并发互相覆盖。

长期记忆实现放在项目根 memory_manager.py（被 watchdog/评测脚本独立使用，
不适合塞进插件包触发 NoneBot 初始化），这里按文件路径直接加载它——不再改 sys.path。
"""
import importlib.util
import json
import os
import re
import threading
import time
from pathlib import Path

from .constants import PROJECT_ROOT, MEMORY_FILE, MEMORY_DIR, SHORT_MEMORY_LINES

_mm_spec = importlib.util.spec_from_file_location("memory_manager", PROJECT_ROOT / "memory_manager.py")
_mm = importlib.util.module_from_spec(_mm_spec)
_mm_spec.loader.exec_module(_mm)
# 长期记忆：记忆卡/关系等级/LLM提取（从根模块 re-export，语义与原一致）
get_user_memory = _mm.get_user_memory
update_user_memory = _mm.update_user_memory
build_memory_context = _mm.build_memory_context
_format_profile_summary = _mm._format_profile_summary
MEMORY_EXTRACT_PROMPT = _mm.MEMORY_EXTRACT_PROMPT
# 时长措辞（"3天"/"2小时"）：长期/短期两处注入共用一份，别各写一套
humanize_gap = _mm.humanize_gap

# 短期记忆文件锁（进程内）
_memory_lock = threading.Lock()

_migrated = False


def _file_for(user_id: str) -> Path:
    """某个会话的历史文件。id 只可能是 QQ 号/群号，仍然过滤一遍防路径穿越。"""
    return MEMORY_DIR / f"{re.sub(r'[^\w.-]', '_', str(user_id))}.json"


def _read_json(path: Path):
    """读一个 JSON 文件；缺失/空/坏 → None（调用方决定怎么兜底）。"""
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8").strip()
        return json.loads(content) if content else None
    except (json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, data) -> None:
    """写 JSON 文件（先写 .tmp 再 replace：避免写一半被读到半个文件）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _ensure_migrated() -> None:
    """把旧的单文件 short_term.json 拆成一人一个文件（只跑一次，拆完改名存档）。"""
    global _migrated
    if _migrated:
        return
    _migrated = True
    if not MEMORY_FILE.exists():
        return
    data = _read_json(MEMORY_FILE)
    if not isinstance(data, dict):
        return
    n = 0
    try:
        for uid, hist in data.items():
            # 老数据两种形态都要搬：行列表，或整个 key 存成一个字符串（更早的遗留形态）
            if isinstance(hist, str):
                hist = [hist] if hist.strip() else []
            if not isinstance(hist, list) or not hist:
                continue
            f = _file_for(uid)
            if f.exists():        # 已经拆过就别覆盖（防分几次迁移时丢新数据）
                continue
            _write_json(f, hist)
            n += 1
        MEMORY_FILE.rename(MEMORY_FILE.with_suffix(".json.migrated"))
        print(f"[记忆] 旧单文件已拆分：{n} 个会话 → {MEMORY_DIR}"
              f"（原件存为 {MEMORY_FILE.name}.migrated）")
    except OSError as e:
        print(f"⚠️ 旧记忆拆分失败（会退回旧单文件读，不影响聊天）: {e}")


def use_storage(root) -> None:
    """把短期记忆整体指到 `root` 目录下（**测试/离线脚本专用**）。

    一次改齐三样（目录 / 旧单文件路径 / 迁移标志）——历史上只改 `MEMORY_FILE`
    等于没隔离；**只改 `MEMORY_DIR` 更糟**：`_ensure_migrated()` 会去拆真实那份
    `short_term.json`（拆完还改名）。要隔离就调这个，别自己 assign。
    """
    global MEMORY_DIR, MEMORY_FILE, _migrated
    root = Path(root)
    MEMORY_DIR = root / "short_term"
    MEMORY_FILE = root / "short_term.json"
    _migrated = True          # 临时目录里没有旧单文件要拆


def load_short_memory() -> dict:
    """**调试/兼容**用：把所有会话合并成一个 dict。

    生产路径**不用它**（那正是被拆掉的原因：每次都要读写全部）。分析脚本要看全量
    聊天记录请用 `chatlog`，这里只在排查记忆问题时手动看一眼。
    """
    _ensure_migrated()
    out = {}
    if MEMORY_DIR.exists():
        for f in MEMORY_DIR.glob("*.json"):
            v = _read_json(f)
            if isinstance(v, list) and v:
                out[f.stem] = v
    return out


def _entry_text(item) -> str:
    """取一行历史的纯文本。

    历史行有两种形态：旧数据的裸字符串，和新数据的 {"t": epoch, "text": "…"}。
    对外一律还原成字符串，调用方（core 的 ln[4:] 剥前缀、\n.join 拼接）无需知道存法。
    """
    if isinstance(item, dict):
        return str(item.get("text", ""))
    return str(item)


def get_user_history(user_id: str) -> list:
    """获取某用户的短期对话历史（最近 3 轮），一律返回字符串行。"""
    raw = _load_raw_history(user_id)
    return [_entry_text(it) for it in raw]


def get_user_history_timed(user_id: str) -> list:
    """和 `get_user_history` 同样返回字符串行，但**每行前面带「距上一条多久」**。

    例：
        用户：还有十五分钟就九点了，等着等
        [3小时前] 灰泽满：睡了，你咋还不睡🧑‍🌾
        [5分钟前] 用户：在吗

    为什么需要它（2026-10-05 用户提出）：
      原来只把「距上一轮多久」当**一行汇总**拼在整块开头，而**每条消息自己没有时间**。
      结果 gap=10 秒 和 gap=3 小时 注入出来的【当前会话】/历史**长得一模一样**——
      她分不出"刚说完"和"三小时前说的"，只能一律当成"正在聊"。
      这跟「上一场会话」那条不一样：那条只在 >12h 时出现，且自带时间定性。

    ⚠️ 这里**只给事实（过了多久），不给任何"该怎么理解"的说明**——按项目实测，
    "管怎么用"的包装语一律无效，事实类信息才有效。判断交给模型。

    没有时间戳的老数据（裸字符串）不给前缀，不猜。
    """
    from memory_manager import humanize_gap

    raw = _load_raw_history(user_id)
    if not raw:
        return []
    out = []
    prev_t = None
    for it in raw:
        text = _entry_text(it)
        t = it.get("t") if isinstance(it, dict) else None
        prefix = ""
        if t is not None and prev_t is not None:
            try:
                g = humanize_gap(max(0.0, float(t) - float(prev_t)))
                if g:
                    prefix = f"[{g}前] "
            except (TypeError, ValueError):
                prefix = ""
        elif t is not None and prev_t is None:
            # 第一行：相对"现在"（它可能已经是几小时前的了，正是要让她知道的那条）
            try:
                g = humanize_gap(max(0.0, time.time() - float(t)))
                if g:
                    prefix = f"[{g}前] "
            except (TypeError, ValueError):
                prefix = ""
        out.append(prefix + text)
        prev_t = t if t is not None else prev_t
    return out


def _load_raw_history(user_id: str) -> list:
    """读某个会话的原始历史（保留 {t,text} 结构），供需要时间戳的调用方用。"""
    _ensure_migrated()
    hist = _read_json(_file_for(user_id))
    if hist is None and MEMORY_FILE.exists():
        # 兜底：万一旧单文件还没拆成功，仍能从里面读到这个会话（不影响聊天）
        legacy = _read_json(MEMORY_FILE)
        if isinstance(legacy, dict):
            hist = legacy.get(str(user_id))
    if isinstance(hist, str):
        return [hist] if hist else []
    return list(hist) if isinstance(hist, list) else []


def get_last_turn_gap_seconds(user_id: str) -> float | None:
    """距"上一轮对话"过去多少秒；没有历史或旧数据无时间戳 → None。

    用于注入"距上次聊天 X"——模型因此能分清"刚刚说的"和"三天前说的"。
    注意调用时机：本轮自己的话要等回复后才 append，所以此刻最后一条就是上一轮。
    """
    raw = _load_raw_history(user_id)
    if not raw:
        return None
    last = raw[-1]
    if not isinstance(last, dict):
        return None  # 旧数据（裸字符串）没有时间戳，这轮先不注入，等她再说一句就有
    try:
        return max(0.0, time.time() - float(last.get("t", 0)))
    except (TypeError, ValueError):
        return None


def _append_lines(user_id: str, lines: list) -> None:
    """往某会话的短期记忆尾部追加若干行（持锁 + 截断到窗口）。空串行自动丢掉。

    只读写**这一个会话的文件**（O(1)）——原来是把全部会话读出来整份写回去。
    """
    lines = [ln for ln in lines if ln]
    if not lines:
        return
    with _memory_lock:
        history = _load_raw_history(user_id)
        now = time.time()
        history.extend({"t": now, "text": ln} for ln in lines)
        _write_json(_file_for(user_id), history[-SHORT_MEMORY_LINES:])


def append_user_history(user_id: str, user_msg: str, reply: str) -> None:
    """追加一轮对话到短期记忆（带时间戳），保留最近 N 条。全程持锁。

    `reply` 传空串 = **她这批没开口**（群聊接话门判定安静）。这时只记"群里说了什么"，
    不记一条空的"灰泽满："——她看到了、只是没接，下一轮判定和生成仍该看得到上下文。
    """
    _append_lines(user_id, [f"用户：{user_msg}", f"灰泽满：{reply}" if reply else ""])


def count_consecutive_requests(user_id: str, current_msg: str, max_look: int = 10,
                               threshold: float = 0.6) -> int:
    """用户是不是在**连续第几次**要同一件事？返回"连上当前这条共几轮"（1 = 不是重复）。

    为什么需要它（2026-10-04 从真实聊天记录里发现）：
      她的招牌动作是「数遍数」（"第六遍""你今晚把'永远'用了三遍了"），
      但那只能数**同一条消息内**的重复；**跨轮的连续索要她完全看不见**——
      上下文只带最近 10 条，每轮她都当成"第一次被要"。
      后果：同一条弧线里她换着理由挡了 5 轮，然后在第 12 轮突然松口
      （3076669330 要"宝宝"、要语音那两段都是这样），
      松口之后又失去"我从没答应过"的立场。

    做法：拿当前消息和最近的历史里**用户侧**的每一条比（字符 bigram 重合），
          从最近往回数，连续相似的算一轮；一旦掉下阈值就停。
          **纯本地计算，不调 API。**

    ⚠️ 只回答"第几次"，**不回答"该不该答应"**——那是 behaviors 的事。
    """
    hist = get_user_history(user_id)
    if not hist or not (current_msg or "").strip():
        return 1
    prev = []
    for ln in hist:
        if ln.startswith("用户："):
            prev.append(ln[3:])
    if not prev:
        return 1

    def bg(s: str) -> set:
        s = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", s)
        return {s[i:i + 2] for i in range(len(s) - 1)}

    cur = bg(current_msg)
    if not cur:
        return 1
    n = 1
    for line in reversed(prev[-max_look:]):
        other = bg(line)
        if not other:
            break
        if len(cur & other) / min(len(cur), len(other)) < threshold:
            break
        n += 1
    return n


def append_bot_message(user_id: str, text: str) -> None:
    """只追加一条「她主动说的话」——**没有对应的用户消息**（主动发言/动态推送用）。

    为什么必须记（2026-09-29 用户反馈）：主动发出去的那条消息是机器人**自己用 API
    发出去的**，不会作为事件回到 `_handle_chat`，所以它**从来没进过记忆**。
    粉丝顺着那句回她（"这个好玩吗""你说的是哪个"）时，她的【最近对话记录】里
    根本没有那一句 → 完全不知道对方在说什么。
    """
    text = (text or "").strip()
    if not text:
        return
    _append_lines(user_id, [f"灰泽满：{text}"])


__all__ = [
    "load_short_memory",
    "get_user_history",
    "get_last_turn_gap_seconds",
    "append_user_history",
    "append_bot_message",
    "humanize_gap",
    "get_user_memory",
    "update_user_memory",
    "build_memory_context",
    "_format_profile_summary",
    "MEMORY_EXTRACT_PROMPT",
    "PROJECT_ROOT",
]
