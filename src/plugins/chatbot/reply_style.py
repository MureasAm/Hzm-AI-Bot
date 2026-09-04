"""回复风格后处理：拆句 / 分批延迟 / 输出清洗 / 复读检测 / 纯情绪查询判定。

纯函数，不依赖 NoneBot 运行时状态——chat_window 发送前、core 生成后都用它。
从 core.py 拆出，让 core 专注"消息组装 + 生成"。
"""
import random
import re

from .constants import (
    SPLIT_MIN_LEN, SPLIT_MAX_PARTS, SPLIT_MERGE_MIN_CHARS,
    SPLIT_DELAY_BASE_MS, SPLIT_DELAY_PER_CHAR_MS,
    SPLIT_DELAY_MIN_MS, SPLIT_DELAY_MAX_MS, SPLIT_DELAY_JITTER,
)
from .persona import load_terms


# 角色以外的人（第三人称名字）：回复里出现这些名字时，"她/他"可能指别人，不动
_OTHER_PERSON_NAMES_CACHE = None


def _get_other_person_names() -> set:
    """取"可能是'她/他'先行词"的第三人称指代集合。

    用于 clean_reply 的"她/他自指兜底"保护——回复里出现这些词时，"她/他"可能指这个人
    而不是角色自己，故不替换。兜底原则：**宁可不替换（留一句自指"她"，小瑕疵），
    也不误替换（把别人的行为安到灰泽满头上，改变语义）**——所以保护名单宁宽勿窄。

    来源：terms 里所有提到的人/群体（绿冻/满区/前辈/同期…含 aliases）+ 常用人称名词。
    动态取自 terms，新增人物不用记两处。
    """
    global _OTHER_PERSON_NAMES_CACHE
    if _OTHER_PERSON_NAMES_CACHE is not None:
        return _OTHER_PERSON_NAMES_CACHE
    names = {"女同学", "女仆女同学", "弥希", "真绯瑠", "瑞雅", "塔菲"}  # 灰泽满专属：不在 terms 的第三人称人物
    # 常用人称名词（"她/他"的常见先行词——粉丝/观众/同学这类没有专名的指代）
    names.update({"粉丝", "观众", "水友", "同学", "室友", "阿姨", "姐姐", "妹妹",
                  "女生", "女孩", "老师", "邻居", "朋友", "同事", "主播", "家人们", "绿冻"})
    # terms 里所有人/群体（含 aliases）。绿冻/满区 是 world 分类也是人，一并纳入
    for t in load_terms():
        if t.get("category") in ("person", "family", "relation", "world") and t.get("keyword"):
            names.add(t["keyword"])
            names.update(str(a) for a in t.get("aliases", []) if a)
    _OTHER_PERSON_NAMES_CACHE = names
    return names


def _trim_text(text: str, max_chars: int) -> str:
    """裁剪长文本，超长加省略号。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "……"


def _lcs_len(a: str, b: str) -> int:
    """最长公共子串长度（DP，O(n*m)）。复读检测用。"""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def is_echo_reply(reply: str, recent_bot_replies: list, min_ratio: float = 0.6,
                  window: int = 3) -> bool:
    """判断新回复是否复读了最近自己说过的话（复读机防护）。

    复读是模型对情境相关句的执着（会在短期记忆里看到自己的原话再引用），
    提示词的【防复读】拦不住，只能确定性检测：与最近回复完全一致，
    或最长公共子串覆盖较短者的 min_ratio 以上。太短的句子（<8字）不判，避免误伤"晚安"类。
    window：回看最近几条 bot 消息。隔几句后才重复（用户重复同一抱怨 → 上上轮的解释被翻出来）
    是常见漏网，core 调用时把窗口放宽到 8。
    """
    reply = (reply or "").strip()
    if not reply:
        return False
    for old in (recent_bot_replies or [])[-window:]:
        old = (old or "").strip()
        if not old:
            continue
        if reply == old:
            return True
        shorter = min(len(reply), len(old))
        if shorter < 8:
            continue
        if _lcs_len(reply, old) >= shorter * min_ratio:
            return True
    return False


def _split_sentences(text: str) -> list:
    """按句末标点切分（。！？）；省略号仅在"前后都有内容"时才当边界。

    省略号常表"无语/语气"（如"啊……这……"）和犹豫前缀，不能乱切；
    只有省略号后面跟着新内容（≥5 字）**且前面也有足够内容**（≥4 字）才当停顿边界。
    """
    parts = []
    i = 0
    for m in re.finditer(r'[。！？]|…+', text):
        end = m.end()
        if m.group(0).startswith("…"):
            rest = text[end:].lstrip()
            before = text[i:m.start()].rstrip()
            if len(rest) < 5 or len(before) < 4:
                continue
        parts.append(text[i:end])
        i = end
    if i < len(text):
        parts.append(text[i:])
    return [p for p in parts if p.strip()]


def _limit_commas(parts: list) -> list:
    """每段至多 1 个逗号；超了从最后一个逗号拆开（逗号去掉），避免长串逗号连句。"""
    result = []
    for p in parts:
        while True:
            commas = [idx for idx, ch in enumerate(p) if ch in "，,"]
            if len(commas) < 2:
                break
            idx = commas[-1]
            result.append(p[:idx].strip())
            p = p[idx + 1:].strip()
        result.append(p)
    return [x for x in result if x]


def split_reply(reply: str, min_len: int = SPLIT_MIN_LEN,
                max_parts: int = SPLIT_MAX_PARTS) -> list:
    """把长回复按句子断开发送（打字感）。短回复/单句不拆，返回单元素列表。

    切分规则：
    - 句末标点（。！？）切分；省略号仅在后面还有新内容（≥5 字）时才切
    - 逗号限制：每段至多 1 个逗号，超了从最后一个逗号拆开
    - 聊天习惯不打句号：切分后去掉句尾"。"（保留？！…）
    - 短碎片并入下一段；纯括号段并入上一段；超 max_parts 并入最后一段
    """
    text = reply.strip().rstrip("。")
    if not text or len(text) < min_len:
        return [text]

    parts = _split_sentences(text)
    parts = _limit_commas(parts)
    parts = [p.strip().rstrip("。") for p in parts if p and p.strip()]

    # 纯括号段（（小声嘀咕））并入上一段，避免单独发一条"舞台说明"
    merged = []
    for p in parts:
        if merged and re.fullmatch(r'[（(][^）)]*[）)]', p):
            merged[-1] += p
        else:
            merged.append(p)
    parts = merged

    # 短段并入下一段（防"哦？""那倒是稀奇……"这种微消息单独成条）
    merged_short = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and len(parts[i]) < SPLIT_MERGE_MIN_CHARS:
            parts[i + 1] = parts[i] + parts[i + 1]
        else:
            merged_short.append(parts[i])
        i += 1
    parts = merged_short
    if len(parts) > 1 and len(parts[-1]) < SPLIT_MERGE_MIN_CHARS:
        parts[-2] += parts[-1]
        parts.pop()

    if len(parts) <= 1:
        return [text]

    if len(parts) > max_parts:
        # 超限分段用换行连接（防 run-on）
        merged = "\n".join(p for p in parts[max_parts - 1:] if p).strip()
        parts = parts[:max_parts - 1] + [merged]
    return parts


def split_delay(part_text: str) -> float:
    """句间发送延迟（秒）：按段落长度模拟打字 + ±15% 随机抖动，避免机械等长。"""
    ms = SPLIT_DELAY_BASE_MS + SPLIT_DELAY_PER_CHAR_MS * len(part_text)
    ms = max(SPLIT_DELAY_MIN_MS, min(ms, SPLIT_DELAY_MAX_MS))
    ms *= random.uniform(1 - SPLIT_DELAY_JITTER, 1 + SPLIT_DELAY_JITTER)
    return round(ms / 1000.0, 3)


def clean_reply(reply: str) -> str:
    """输出清洗：去括号前缀、整条至多 1 个括号、省略号归一并限频、第三人称自指兜底。

    人设规则是"每轮回复至多一个括号、省略号是例外"，但声音样本里带（小声）（心虚），
    模型 few-shot 会学着用。这里做确定性过滤。
    """
    text = reply.strip()
    # 去掉开头的连续括号前缀，如 （咽口水）你看...
    while text.startswith("（"):
        idx = text.find("）")
        if idx == -1:
            break
        text = text[idx + 1:].lstrip()
    if not text:
        return reply  # 剥光了就回原样，避免空回复
    # 若仍有 ≥2 个括号，只保留第一个，其余删除
    matches = list(re.finditer(r'（[^）]*）', text))
    if len(matches) >= 2:
        first = matches[0]
        before = text[:first.start()]
        kept = first.group(0)
        after = re.sub(r'（[^）]*）', '', text[first.end():])
        text = before + kept + after
    # 省略号纪律：归一连续省略号；剥掉开头省略号；开头"词+省略号"犹豫 → 逗号；整条至多 2 个
    text = re.sub(r'\.{3,}|…+', '……', text)
    text = re.sub(r'^……(?=\S)', '', text)
    text = re.sub(r'^([一-鿿]{1,5})……(?=.{4,})', r'\1，', text)
    if not text:
        text = '……'
    ell_pos = [m.start() for m in re.finditer('……', text)]
    if len(ell_pos) > 2:
        cut = ell_pos[2]
        text = text[:cut] + text[cut:].replace('……', '')
    # 结巴消融："那、那"→"那那"
    text = re.sub(r'(.)、\1', r'\1\1', text)
    # 自指"她/他"——折中版（不再"无他名就全换成灰泽满"，那会把真指别人的她/他也改坏）：
    # 仅当回复里出现自称名(灰泽满/hzm)且没出现别的第三人称名时，才去掉跟在句读标点后
    # 复指自己的"她/他"（保留前文的灰泽满，不重复堆名也不串成别人）：
    #   灰泽满到点下播了，她准备睡了 → 灰泽满到点下播了，准备睡了
    # 没自称名的"她/他"多半指对话里的第三人 → 一律保留，别硬安到灰泽满头上。
    # 她俩/她们/她的/她和=复数/所有格/并列，指别人，不碰。
    if ("灰泽满" in text or "hzm" in text.lower()) and \
            not any(n in text for n in _get_other_person_names()):
        text = re.sub(r"(?<=[，,。；;！!？?、…\n])\s*[她他](?!们|俩|的|和|家)", "", text)
    return text


def is_emotion_only_query(query: str) -> bool:
    """判断 probe 补全的检索 query 是否为"纯情绪"描述（如'用户发了个偷笑的表情'）。

    表情/情绪为主的短消息（如'可惜🤭'）会被 probe 按表情规则补全成这类纯情绪句——
    它没有话题词，做语义检索会误命中无关样本。对齐纯表情消息的既有设计：
    跳过语义检索，补全句只作语气提示（query_hint）注入。
    """
    q = (query or "").strip()
    return q.startswith("用户发") and "表情" in q
