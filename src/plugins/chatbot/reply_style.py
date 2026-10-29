"""回复风格后处理：拆句 / 分批延迟 / 输出清洗 / 复读检测 / 纯情绪查询判定。

纯函数，不依赖 NoneBot 运行时状态——chat_window 发送前、core 生成后都用它。
从 core.py 拆出，让 core 专注"消息组装 + 生成"。
"""
import os
import random
import re
from collections import Counter

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


# 「时间钩子」：她在熬夜场景下会反复拿"现在几点了/该睡了"当切入点（实测某用户 10 条回复里
# 5 条如此）。逐字复读检测抓不到（说法不同、公共子串不够长），所以单列一条**只认时间词**的窄判据。
# 窄是故意的：历史上"防措辞固化"就是死在误伤她自己的口癖（75% 触发都在拦自称"灰泽满"），
# 所以这条只管"又提时间"，别的词一概不认。
_TIME_HOOK_RE = re.compile(r"凌晨|半夜|深夜|大半夜|大清早|一大早|[0-9]{1,2}\s*点")
# 开头的一整组括号（可连着几组）：「（心虚）」「(笑)」「【小声】」
_LEADING_PAREN_RE = re.compile(r"^(?:[（(【\[][^）)】\]]*[）)】\]]\s*)+")
# 「说话人标注」：`灰泽满：……好啦，那那记着呢` 这种——**带冒号的名字是标注，不是她的话**。
# 出处：注入里给的例句是 `粉丝说：{用户那句} → 灰泽满：{她的话}`（persona.py），
# 模型偶尔把标注一起吐出来；历史记忆行也是 `灰泽满：{内容}` 这个格式（memory.py）。
# 这类东西在 QQ 里一眼出戏，所以确定性兜底剥掉（黄金律 3：后处理只兜底，不改人设）。
# ⚠️ 必须要求**冒号紧跟**名字：`她说：…`「灰泽满刚刚…」这类正常句子不能被误伤。
_SPEAKER_LABEL_RE = re.compile(r"^(?:灰泽满|hzm|小满|满姐|满神|粉丝说|对方|用户|她|我)\s*[：:]\s*")


def uses_time_hook(reply: str) -> bool:
    """这条回复是不是又在拿"时间"当切入点（用于拦熬夜场景下的反复开场）。"""
    return bool(_TIME_HOOK_RE.search(reply or ""))


# 「空壳起手」：`灰泽满看看` 这种**只剩壳、没有宾语**的固定开头。
# 实测它的退化轨迹（某用户 2026-10-07）："看看时间"（正常动词）→"看看这个mua"→"看看这张"
# →"看看你"→"看看，企鹅笔盒？"（连宾语都没了）——然后**自强化**：她的回复进【最近对话记录】
# → 被当样本学回去 → 下一轮又说。537 条里 108 条（20%）这么开头，最长连 29 条。
# ⚠️ 别改成通用的"起手两字重复"：全库实测那样会打中 18.3%（"今天…""嗯，…"全是正常开头）。
_HOLLOW_OPENER_RE = re.compile(r"^(?:灰泽满|hzm|小满|满姐|我)?[，,、：:\s]*看看")


def uses_hollow_opener(reply: str) -> bool:
    """这条回复是不是又拿"看看"当空壳起手（用于拦自强化出来的固定开头）。"""
    return bool(_HOLLOW_OPENER_RE.match(_LEADING_PAREN_RE.sub("", (reply or "").lstrip())))


# 「固定模板起手」：她最近反复用同一个否定/转移句式开场。
# 2026-10-20 实测：`灰泽满才没` 全库 2.6%，但 10-15 起从 ~0.3% 跳到 8–10%
# （gemini-3.1-flash-lite；同批 prompt 下 gemini-3.8 是 0/20、deepseek 1/20），
# 是**模型级模板**，不是 behaviors 例句。
# 用户 2026-10-20 定：核心就是「才不」「才没」两个词——不限定主语/前缀，
# 只要**第一句**里出现这两个词就算同一模板；`突然说这个` 保留为第二模板。
def formula_opener(reply: str) -> str:
    """回复第一句里有没有 `才不/才没`，或是否以 `突然说这个` 开头。"""
    text = _LEADING_PAREN_RE.sub("", (reply or "").lstrip())
    first_sentence = re.split(r"[。！？!?]", text, maxsplit=1)[0]
    if "才不" in first_sentence or "才没" in first_sentence:
        return "才不/才没"
    if re.match(
        r"^(?:(?:啊|哎呀|好端端的?|干嘛|怎么)\s*[，,、：:\s]*)?突然说这个",
        text,
    ):
        return "突然说这个"
    return ""


def repeated_formula(reply: str, recent_bot_replies: list,
                     window: int = 4, min_prev: int = 2) -> str:
    """当前回复 + 最近 window 条里有 >=min_prev 条同模板时，返回该模板；否则空串。

    只用于"最近反复用同一个固定句式"的窄场景；单条或偶尔出现不拦。
    """
    opener = formula_opener(reply)
    if not opener:
        return ""
    count = sum(1 for prev in (recent_bot_replies or [])[-window:]
                if formula_opener(prev) == opener)
    return opener if count >= min_prev else ""


def starts_with_name(reply: str, name: str) -> bool:
    """这条回复是不是**拿用户的名字当开头**。

    为什么要单独判它（2026-10-11 实测）：她一旦开始每句都叫名字就会**自我强化**——
    她的回复进了【最近对话记录】→ 被当作样本学回去。某用户**前 100 轮 0 次**名字开头，
    从第 103 轮（用户第一次说出那个名字）起一路涨到**最近 14 轮 14/14**。
    名字进记忆只是"种子"，真正让它停不下来的是这个正反馈。
    前导括号/情绪括号是内心戏、不算"开头说的是什么"，先剥掉再比。
    """
    if not name:
        return False
    # 剥掉开头的括号组（内心戏，可能连着好几组）再比——但**只剥完整的一组**，
    # 不能只剥括号字符（否则「（心虚）X」会变成「心虚）X」，名字判不出来）。
    text = _LEADING_PAREN_RE.sub("", (reply or "").lstrip())
    return text.startswith(name)


def _metric_opener(reply: str) -> str:
    """回复开头（剥掉开头括号后的前 8 个字，到第一个标点为止）——量开头同质用。"""
    t = _LEADING_PAREN_RE.sub("", (reply or "").lstrip())
    m = re.match(r"[^，,。；;！!？?…\s]{1,8}", t)
    return m.group(0) if m else ""


def self_reference_metrics(replies: list) -> dict:
    """自称方式改写的**确定性体检指标**（供 `skeleton-ab --check` 用）。

    为什么需要它（2026-10-23 用户提出）：他试过好几版「自称方式」提示词，都出现两种副作用——
    ① 名字起手（每句都用“灰泽满”开头）；② 固定否定句式（“灰泽满才不/才没”很频繁）。
    改这段提示词时必须能量化“有没有把这两样带回来”，不能只看单条像不像。

    指标（都是比例，0–1）：
      · 第一人称率：出现“我”（不含“我们”）的比例——她应该用第三人称自称，越低越好
      · 第三人称自指率：出现“灰泽满/hzm”的比例——改提示词不能把它改没
      · 名字起手率：剥掉开头括号后以“灰泽满/hzm”开头的比例
      · 才不才没率：出现“才不/才没”的比例
      · 代词错位率：无主语“真拿灰泽满没办法/没辙”的比例（指对方的“你”被自称名顶掉）
      · 开头同质率：最常见开头（前 8 字）占全部回复的比例
    """
    rs = [str(r or "") for r in replies]
    if not rs:
        return {}

    def rate(pred) -> float:
        return round(sum(1 for r in rs if pred(r)) / len(rs), 3)

    openers = Counter(op for op in (_metric_opener(r) for r in rs) if op)
    top = openers.most_common(1)[0][1] if openers else 0
    return {
        "第一人称率": rate(lambda r: bool(re.search(r"我(?!们)", r))),
        "第三人称自指率": rate(lambda r: bool(re.search(r"灰泽满|hzm", r, re.I))),
        "名字起手率": rate(lambda r: bool(re.match(r"(灰泽满|hzm)", _LEADING_PAREN_RE.sub(
            "", r.lstrip()), re.I))),
        "才不才没率": rate(lambda r: "才不" in r or "才没" in r),
        "代词错位率": rate(lambda r: bool(_SELF_AS_OBJECT_RE.search(r))),
        "开头同质率": round(top / len(rs), 3) if rs else 0.0,
    }


def self_reference_check(baseline: dict, candidate: dict,
                         max_limits: dict, min_limits: dict,
                         rel_warn: float = 0.15) -> tuple:
    """把两版 `self_reference_metrics` 的结果比对，返回 (fails, warns)。

    纯函数：阈值由调用方给（`scripts/skeleton_ab.py` 的 `--check`），这里只做比较，
    这样阈值调整和单测都不需要起 NoneBot / 调模型。
    """
    fails, warns = [], []
    for key, limit in max_limits.items():
        val = candidate.get(key)
        if val is not None and val > limit:
            fails.append(f"{key} {val:.0%} > 上限 {limit:.0%}")
    for key, floor in min_limits.items():
        val = candidate.get(key)
        if val is not None and val < floor:
            fails.append(f"{key} {val:.0%} < 下限 {floor:.0%}")
    # 相对基线只提醒不判死：这些是模型级口癖，单次采样噪声本来就大。
    for key in ("第一人称率", "名字起手率", "才不才没率", "开头同质率"):
        a, b = baseline.get(key, 0), candidate.get(key, 0)
        if b - a > rel_warn:
            warns.append(f"{key} 比基线高 {b - a:.0%}（{a:.0%}→{b:.0%}）")
    drop = baseline.get("第三人称自指率", 0) - candidate.get("第三人称自指率", 0)
    if drop > rel_warn:
        warns.append(f"第三人称自指率 比基线低 {drop:.0%}")
    return fails, warns


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


# 群聊“元回复”：把**内部点名标记**说成回复（2026-10-20 实测群 901907410）。
# 接话门允许她对没点名的情绪批次插一句（判“情绪→接”），但群聊注入写着
# “只有点名才是在跟你说话”+ 每条带 [未点名]，模型于是把路由说出来：
#   「没点灰泽满，灰泽满才不管。」「（群里没人 @ 灰泽满，保持潜水状态，没有回复。）」
# 这类句子发出去等于“用一条消息说明自己不回”——对没点名的批次应视为“这批不说话”。
_META_SILENCE_RE = re.compile(
    r"没(?:有人)?(?:点|叫|喊)(?:我|灰泽满|hzm)|没被\s*@|没有被\s*@|"
    r"保持潜水|潜水状态|静静看着|不插话|不发言|这轮不(?:说|回)|这批不(?:说|回)|"
    r"（[^）]{0,30}(?:不说话|潜水|没点|没叫|没被\s*@)[^）]{0,30}）"
)


def is_meta_silence(reply: str) -> bool:
    """这句是不是在**解释“我不该说话/我在潜水”**，而不是真的回了一句话。

    ⚠️ 只在**未点名的群聊批次**里用（见 `core.handle_chat` 的闸）：
    点名之后的“关我什么事”是正常斗嘴，别误杀。
    """
    t = (reply or "").strip()
    return bool(t) and bool(_META_SILENCE_RE.search(t))


# 「真拿灰泽满没办法」→「真拿你没办法」：她偶尔把“真拿你没办法”的对象“你”写成自己的名字
# （2026-10-23，真实日志 4 例；“拿灰泽满没办法呢，真受不了你”同一句里“你”还用对了，
# 是**局部槽位被自称名挤掉**，不是把对方认错）。
# ⚠️ 只在**没有主语**的“真(是)拿…”固定句式里改：
#    “你也拿灰泽满没办法”“你们拿灰泽满没办法”是对的（对方拿她没办法），别动；
#    句首裸露的“拿灰泽满没办法”也可能是“（你）拿她没办法”，有歧义，同样不动。
_SELF_AS_OBJECT_RE = re.compile(
    r"(?P<lead>^|[，,。；;！!？?…\s])(?P<adv>可?真(?:是)?)拿灰泽满(?P<tail>没办法|没辙)"
)


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


def _join_fragments(frags: list) -> str:
    """把小段接回一条消息：左段没有收尾标点时补逗号，防"两句焊成 run-on"。

    切分时句尾"。"已被去掉（聊天习惯不打句号），直接相连会读成一句话：
    "真的是这样"+"明天还要早起呢" → "真的是这样明天还要早起呢"。
    """
    out = ""
    for f in frags:
        if out and not re.search(r"[，,。；;：:！!？?、…～~）)】\]]$", out):
            out += "，"
        out += f
    return out


_INTERJECTION_RE = re.compile(
    r"^(?:(?:好啦|行了|好吧|好的|好嘞|成吧|得嘞|哎呀|哎哟|诶呀|行啦|对啦|是啊|是吧|真的吗)[！!？?~～。…]*)+$")


def _comma_sentences(text: str) -> list:
    """按句末（。！）切；省略号在前后都有内容且后面 ≥5 字时才当边界。**问号不切。**"""
    parts, i = [], 0
    for m in re.finditer(r"[。！]|…+", text):
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


def _split_on_commas(text: str, max_parts: int) -> list:
    """逗号分段（2026-10-27，2026-10-28 调整）：

    - 句末只有 。！ 切；**问号不处理**（不切、不删）。
    - 逗号处：**两边都 ≥5 字**才切，切时丢掉逗号；
      任一边不足 5 字就保留逗号、不切（避免“少来”“好啦”被单发或焊成 run-on）。
    - 例外：左边是多字语气词（好啦/行了/好吧…）时并进下一句并丢掉逗号。
    - 括号/引号内的逗号不切；纯括号段并入上一段；超 max_parts 用空格接。
    """
    out = []
    for s in _comma_sentences(text):
        s = s.strip().rstrip("。").strip()
        if not s:
            continue
        buf, depth, quote = [], 0, False
        for i, ch in enumerate(s):
            buf.append(ch)
            if ch in "（(【[":
                depth += 1
            elif ch in "）)】]":
                depth = max(0, depth - 1)
            elif ch in "“「『":
                quote = True
            elif ch in "”」』":
                quote = False
            elif ch == "，" and depth == 0 and not quote:
                left = "".join(buf[:-1]).strip()
                rest = s[i + 1:]
                nxt = rest.split("，", 1)[0].strip()
                if _INTERJECTION_RE.match(left):
                    buf.pop()                      # 语气词：并进下一句
                elif len(left) >= 5 and len(nxt) >= 5:
                    buf.pop()                      # 两边都够长：切成两条
                    if left:
                        out.append(left)
                    buf = []
                # 其余情况：保留逗号，继续拼（不切）
        piece = "".join(buf).strip().rstrip("，").strip()
        if piece:
            out.append(piece)

    merged = []
    for p in out:
        if merged and re.match(r"^[（(]", p):
            merged[-1] += p
        else:
            merged.append(p)
    if len(merged) > max_parts:
        merged = merged[:max_parts - 1] + [" ".join(merged[max_parts - 1:])]
    return merged


def split_reply(reply: str, min_len: int = SPLIT_MIN_LEN,
                max_parts: int = SPLIT_MAX_PARTS) -> list:
    """把长回复断开发送（打字感）。短回复/单句不拆，返回单元素列表。

    默认走**逗号分段**（2026-10-27 用户定，10-28 调整）：句末只有 。！ 切，问号不处理；
    逗号两边都 ≥5 字才切并丢掉逗号，短的一侧保留逗号不切（“好啦”这类语气词例外）。
    括号/引号内的逗号不切；纯括号段并入上一段；超 max_parts 用空格接。
    `SPLIT_ON_COMMA=0` 可回退旧逻辑。

    旧逻辑（回退时）：句末标点切分；每段至多 1 个逗号；短碎片并入下一段；
    超 max_parts 用逗号接。
    """
    text = reply.strip().rstrip("。")
    if not text:
        return [""]

    comma_mode = os.environ.get("SPLIT_ON_COMMA", "1").strip().lower() not in (
        "0", "false", "no", "off")
    if comma_mode:
        parts = _split_on_commas(text, max_parts)
        return parts or [text]
    if len(text) < min_len:
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
            parts[i + 1] = _join_fragments([parts[i], parts[i + 1]])
        else:
            merged_short.append(parts[i])
        i += 1
    parts = merged_short
    if len(parts) > 1 and len(parts[-1]) < SPLIT_MERGE_MIN_CHARS:
        parts[-2] = _join_fragments([parts[-2], parts[-1]])
        parts.pop()

    if len(parts) <= 1:
        return [text]

    if len(parts) > max_parts:
        # 超限分段接成一条（补逗号防 run-on）。原用换行连接：换行在 QQ 里显示成
        # "一条消息里的多行"（剧本感），语音也会被读成上气不接下气，改用逗号接。
        merged = _join_fragments([p for p in parts[max_parts - 1:] if p])
        parts = parts[:max_parts - 1] + [merged]
    return parts


def split_delay(part_text: str) -> float:
    """句间发送延迟（秒）：按段落长度模拟打字 + ±15% 随机抖动，避免机械等长。"""
    ms = SPLIT_DELAY_BASE_MS + SPLIT_DELAY_PER_CHAR_MS * len(part_text)
    ms = max(SPLIT_DELAY_MIN_MS, min(ms, SPLIT_DELAY_MAX_MS))
    ms *= random.uniform(1 - SPLIT_DELAY_JITTER, 1 + SPLIT_DELAY_JITTER)
    return round(ms / 1000.0, 3)


def clean_reply(reply: str) -> str:
    """输出清洗：换行归一、去括号前缀、整条至多 1 个括号、省略号归一并限频、第三人称自指兜底。

    人设规则是"每轮回复至多一个括号、省略号是例外"，但声音样本里带（小声）（心虚），
    模型 few-shot 会学着用。这里做确定性过滤。
    """
    text = reply.strip()
    # 换行归一：她的输出永远是"一条消息、一口气"，不分段。
    # 模型偶尔写成 RP 式分段（"没有。\n\n（小声）mua又不是糖，说给就给的"），
    # 在 QQ 里就是"一条消息里的多行"（剧本感），语音也会被读成上气不接下气。
    # 规则：断行跟在标点后 → 直接接上；跟在实词后 → 补逗号（两行焊死会读成 run-on）。
    text = re.sub(r"[ \t]*(?:\r?\n[ \t]*)+", "\n", text)
    text = re.sub(r"(?<=[，,。；;：:！!？?、…～~）)】\]])\n+", "", text)
    text = re.sub(r"\n+", "，", text)
    # 去掉开头的括号前缀和说话人标注，如 （咽口水）你看… / 灰泽满：……好啦
    # ⚠️ 两者会交替出现（「（心虚）灰泽满：X」），所以循环剥到剥不动为止。
    stripped = text
    for _ in range(4):
        before = stripped
        while stripped.startswith("（"):
            idx = stripped.find("）")
            if idx == -1:
                break
            stripped = stripped[idx + 1:].lstrip()
        stripped = _SPEAKER_LABEL_RE.sub("", stripped)
        if stripped == before:
            break
    if not stripped:
        return text or reply  # 剥光了就回原样（用归一后的，别把换行又带出去）
    text = stripped
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
    # 代词错位兜底（2026-10-23）：只在“真拿灰泽满没办法”这一种无主语句式里改，
    # 见 _SELF_AS_OBJECT_RE 的说明——带主语的“你们拿灰泽满没办法”是对的。
    text = _SELF_AS_OBJECT_RE.sub(r"\g<lead>\g<adv>拿你\g<tail>", text)
    return text


def is_emotion_only_query(query: str) -> bool:
    """判断 probe 补全的检索 query 是否为"纯情绪"描述（如'用户发了个偷笑的表情'）。

    表情/情绪为主的短消息（如'可惜🤭'）会被 probe 按表情规则补全成这类纯情绪句——
    它没有话题词，做语义检索会误命中无关样本。对齐纯表情消息的既有设计：
    跳过语义检索，补全句只作语气提示（query_hint）注入。
    """
    q = (query or "").strip()
    return q.startswith("用户发") and "表情" in q
