"""硬匹配路由：经典梗库 + LLM 语境确认 + 行为意图分类（L3）。

从 core.py 拆出。这些是"先判断这条消息是啥、要不要走固定回复"的入口逻辑，
与"组装消息 + 生成"分离——core 专注消息组装。

梗库数据化：LEGENDARY_REPLIES / LEGENDARY_CONFIRMS 从 persona/world/legendary.json 加载，
使用者填 JSON 即可，不用改代码。改后重启生效。
"""
import json
import re
from pathlib import Path

from .constants import THINKING_DISABLED, PROJECT_ROOT
from .config import log_cache_usage, _get_clients, _get_model_name, extract_chat_content

LEGENDARY_FILE = PROJECT_ROOT / "persona" / "world" / "legendary.json"

_legendary_cache = None


def load_legendary() -> dict:
    """加载经典梗库：{replies: {关键词: [候选回复]}, confirms: {关键词: 确认prompt}}。"""
    global _legendary_cache
    if _legendary_cache is not None:
        return _legendary_cache
    if not LEGENDARY_FILE.exists():
        _legendary_cache = {"replies": {}, "confirms": {}}
        return _legendary_cache
    try:
        data = json.loads(LEGENDARY_FILE.read_text(encoding="utf-8"))
        raw = data.get("replies", {}) or {}
        # 两种写法都接受：
        #   "关键词": ["应答", …]                        （纯子串触发）
        #   "关键词": {"replies": [...], "pattern": "…"} （子串 **或** 正则触发）
        # ⚠️ 为什么要 pattern：`trigger in user_msg` 是**精确子串**，它假设用户会说出那个固定词。
        #    实测（scripts/entry_audit.py）legendary 只有 5/20 能被「用户可能会怎么说」勾出来——
        #    问年龄的 5 条触发词（你是谁/你多大/几岁啦/多少岁）**一条都没匹配上**用户实际说的
        #    「你到底几岁啊」。照抄 terms.json 的 pattern 机制来治。
        replies = {}
        for k, v in raw.items():
            if isinstance(v, list):
                replies[k] = {"replies": v, "pattern": None}
            elif isinstance(v, dict) and isinstance(v.get("replies"), list):
                replies[k] = {"replies": v["replies"], "pattern": v.get("pattern")}
        _legendary_cache = {
            "replies": replies,
            "confirms": data.get("confirms", {}) or {},
        }
    except (json.JSONDecodeError, OSError):
        _legendary_cache = {"replies": {}, "confirms": {}}
    return _legendary_cache


# ==================== 💬 经典梗硬匹配库（数据化） ====================
LEGENDARY_REPLIES = load_legendary()["replies"]
LEGENDARY_CONFIRMS = load_legendary()["confirms"]


def legendary_hit(trigger: str, msg: str) -> bool:
    """这条梗是否被消息命中。

    **配了 pattern 就以 pattern 为准；没配才回落到子串。**
    为什么不是"子串或正则"：那样 pattern 只能**放宽**（加更多说法），没法**收紧**——
    而裸触发词（如「没感觉」）会先被子串命中（"这歌我没感觉"），pattern 写的再细也没用。
    改成"pattern 优先"后，一个词既能放宽也能收窄，由数据说了算。
    """
    entry = LEGENDARY_REPLIES.get(trigger) or {}
    pat = entry.get("pattern")
    if pat:
        try:
            return re.search(pat, msg or "") is not None
        except re.error:
            return False
    return trigger in (msg or "")


async def legendary_confirmed(user_msg: str, prompt_template: str, history: str = "") -> bool:
    """LLM 判断关键词命中的消息是否真是目标梗的语境（双路由第二层，防误触发）。

    关键词命中是低频事件，为它加一次便宜 LLM 判断成本可控；确认失败默认放行（不阻塞）。
    """
    try:
        content = prompt_template
        if "{context}" in content:
            content = content.replace("{context}", history or "（无）").replace("{msg}", user_msg)
        else:
            content = content.replace("{msg}", user_msg)
        deepseek_client, _ = _get_clients()
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=10,
            **THINKING_DISABLED,
        )
        log_cache_usage(resp, "行为分类")
        c = extract_chat_content(resp)
        # 确认模板要求只答"是/否"：以"是"开头且不是"不是/是不是"这类否定/疑问才放行
        return c.startswith("是") and not c.startswith("不是") and not c.startswith("是不是")
    except Exception as e:
        print(f"⚠️ 梗确认失败（默认放行）: {e}")
        return True


# ==================== 🎭 L3 意图分类（LLM 判意图，不再用 embedding 猜） ====================
# 背景：embedding 聚的是"句式"不是"意图"——'灰泽满你唱歌好听'（夸）和
# '灰泽满你怎么又迟到了'（质问）句式相同，在向量空间挤成一团，余弦匹配会把
# 夸奖误判成质疑。治本：归属交给 LLM 理解，判别性词汇仍作关键词兜底。
#
# 2026-09-26 扩展：**行为 + 措辞共用这一次调用**。
# 原因：两者是同一个问题的两半——"这条消息落在哪个已知情境"。
#   · behavior 回答"该怎么做"（11 条）
# 而措辞原本走向量检索，实测 100% 开火（trigger 是 6~13 字的**类别标签**，不是句子，
# 余弦根本分不开）——用检索工具干分类的活。并进来 = 零额外调用 + 判据统一。
BEHAVIOR_CLASSIFY_PROMPT = """你是{role_name}的意图分类器。判断用户刚发的这条消息落入哪些「已知情境」。只有明确匹配才选，拿不准一律不放（宁可不触发，不误触发）。

【行为】用户这条消息触发哪个行为场景？**最多选一个**（它决定{role_name}该怎么做）：
{behavior_defs}

行为判定要点：
- 只看用户这条消息本身的内容和语气，结合最近对话判断语境。
- **判据以每条 trigger 的描述为准**，下面是容易判错的地方，别再按关键词硬映射：
- 「被夸」：用户在夸她或认可她本人（声音/表现/才能/用心），**不是随口道贺**。
  ⚠️ 「生日快乐」「祝你开心」这类纯祝福、以及「我考上了」这类用户自己的好消息，都不要判成被夸。
- 「被质疑戳穿 / 失约被催」：用户在质问、戳穿、翻旧账、传她没说过的话，或催她没做到的事。
  如果消息里**既有催问又有质疑**，按哪个更重选一个（拿不准就不选）。
- 「低俗擦边」与「关系向越界」是**两个不同情景**，别混：
  · 低俗/擦边/黄段子/XP话题、以及**要她说一句带颜色的话**（用户会这么说：'给我们造个黄桃吧''说句骚话'） → 「被说低俗或擦边时冷脸挡回」
  · 求交往、要当女友、喊老公老婆、恋人昵称、'我们分手吧'、**要啵啵/摸摸头/叫宝宝这类亲昵动作或称呼** → 「被索要亲密动作或关系时推脱」
- 「被起哄编故事加码」与「低俗擦边」的**区别在内容的性质，不在谁在起哄**：
  · 内容是**色情/擦边** → 低俗那条（她冷脸挡回）
  · 内容是**荒诞编造、往她身上安离谱的假事**（说她怀孕了、说她昨晚没在家、把她口误往歪处带）→ 起哄编故事那条（她顺着玩）
  两者应对**相反**，判错就很假，拿不准时按「内容是不是色情」来分。
- 「被真心对待」：用户**很认真地**感谢她、说在意她、关心她的情绪、深夜惦记她——是真心，不是玩笑。
  随口一句「谢谢」、普通寒暄不算。
- 「说了真心话就立刻缩回」：判的是**她自己**刚在上一轮说了句真心话，用户这轮顺着接。**用户这轮没提供新信息时也可能是它。**
- 普通闲聊、提问、寒暄、表情、玩梗、单纯道贺、用户聊自己的事 → null。
- 拿不准 → null。

★ 另外判一件事：**这是不是「连续第几轮在要同一件事」**（same_request）。
  · 看【最近对话】里用户那几句：如果**当前这句和前面几句是在要同一样东西**
    （换个说法也算，如「我要听你的声音」→「那我要听语音」→「发个语音条」），
    就说出是**同一件事**、以及**从几轮前开始连着**（含当前这轮，最小 1）。
  · **要的是「诉求」相同，不是「话题」相同**。用户连着问三个不同的生活问题 → 不算。
  · ⚠️ **注意方括号里的时间**（如 `[3小时前] 用户：…`）：如果用户那几句**中间隔了很久**
    （隔了几小时），那**不算"连着"**——只从**同一场对话内**的连续几轮算起。
    上一句在几小时前 → 这轮就是 `count` 1。
  · `what` 写「在要什么」（如「要语音」「要被叫宝宝」「要一个表态」），不要知道就不写。
最近对话：
{history}

用户消息：{user_msg}

只输出 JSON：{{"behavior": "<行为name>" 或 null, "same_request": {{"what": 「在要什么」, "count": 连续轮数}}}}"""



async def classify_l3(deepseek_client, user_msg: str, history_text: str,
                      behaviors: list) -> dict:
    """一次调用判出：行为名（0/1 个）+ 连续索要轮数。

    返回 {"behavior": str, "same_request": {"what": str, "count": int}}；
    **失败/拿不准返回空**（不触发任何东西）。返回的名字必须是数据里的真值（防模型编造）。

    ⚠️ 判据措辞是本模块最要紧的东西（改词=改行为，实测过判据稍动结果就从 0% 跳到 74%）。

    ⚠️ 原「措辞组」那一路已随 phrases.json 一起删除（2026-10-04）：
       输出的 `phrases` 字段、`phrase_groups` 参数、`PHRASE_SECTION` 全部拆掉。
       留着死字段只会浪费 token、并给 JSON 输出添乱（踩过：加字段后 40 token 截断过 JSON）。
       现在 `history_text` 由调用方传**带时间标签**的版本（`[3小时前] 用户：…`），
       same_request 靠它判断"是不是同一场对话内的连续索要"。
    """
    if not user_msg:
        return {"behavior": "", "same_request": {"what": "", "count": 1}}

    defs = []
    for b in behaviors or []:
        if not b.get("name"):
            continue
        line = f"- {b['name']}：{b.get('trigger', '')}"
        # 行为定义带真实粉丝话样例：领域黑话光靠 trigger 描述 LLM 认不出，
        # 给真实样例当参照（素材驱动），分类更准。
        for s in b.get("samples", [])[:2]:
            u = (s.get("user") or "").strip()
            if u:
                line += f"\n    例：{u}"
        defs.append(line)

    prompt = BEHAVIOR_CLASSIFY_PROMPT.format(
        behavior_defs="\n".join(defs) or "（无）",
        history=history_text or "（无）", user_msg=user_msg,
        role_name="灰泽满",
    )
    try:
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            # ⚠️ 别调小：加了 same_request 字段后输出变长，40 token 会把 JSON 从中间截断
            # （实测 raw = '…"same_request": {"what": "要她说带颜色的话（造黄桃）' 断在这儿）
            # → json.loads 报 "Expecting ',' delimiter"，而**每轮都调**，影响所有消息。
            max_tokens=200,
            **THINKING_DISABLED,
        )
        log_cache_usage(resp, "行为分类")
        content = extract_chat_content(resp)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        parsed = json.loads(_repair_llm_json(content))
        names = {b.get("name") for b in (behaviors or [])}
        behavior = str(parsed.get("behavior") or "").strip()
        return {"behavior": behavior if behavior in names else "",
                "same_request": _parse_same_request(parsed.get("same_request"))}
    except Exception as e:
        print(f"⚠️ L3 意图分类失败（降级不触发）: {e}")
        return {"behavior": "", "same_request": {"what": "", "count": 1}}


def _repair_llm_json(text: str) -> str:
    """修复 LLM 常见的不规范 JSON（DeepSeek 偶发），尽力让 json.loads 能过。

    常见病：键没加双引号（{name: "x"}）、单引号键/值、尾逗号、markdown 围栏、前后杂质。
    修不好的原样返回，交给调用方兜底（重试/丢弃）。

    ⚠️ 从 core.py 挪到这里（2026-10-04）：L3 也需要它——
    实测输入带 emoji（🥺）时模型会输出坏 JSON，L3 整个降级 →
    **不只丢"连续索要计数"，连行为/措辞一起丢**。
    放这里的另一个原因：本模块是 JSON 解析器，没有下层依赖，core 反向 import 会循环。
    """
    if not text:
        return text
    t = text.strip()
    # 剥 markdown 代码围栏
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    # 只取最外层 {…} / […]（剥掉前后杂质，如模型先写"好的"）
    start = min((i for i in (t.find("{"), t.find("[")) if i != -1), default=-1)
    end = max(t.rfind("}"), t.rfind("]"))
    if start != -1 and end > start:
        t = t[start:end + 1]
    # 键补双引号：单引号键 {'a': …} 和裸键 {a: …}
    t = re.sub(r"([{,]\s*)'([^']+)'(\s*:)", r'\1"\2"\3', t)
    t = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', t)
    # 单引号字符串值 → 双引号
    t = re.sub(r":\s*'([^']*)'", lambda m: ': "' + m.group(1).replace('"', '\\"') + '"', t)
    # 尾逗号 ,} / ,]
    t = re.sub(r",\s*([}\]])", r"\1", t)

    # ⚠️ 上面这些修不了两类"值里出问题"的坏 JSON（L3 实测踩到）：
    #   ① 值里有**裸换行**（模型把字段写成多行）→ "Invalid control character"
    #   ② 值里有**内层双引号**。L3 提示词里全是双引号（"被夸"、"低俗擦边"），
    #      模型写 behavior 字段时会照抄进去 → "被说"低俗"或擦边时…" → 解析失败。
    # 这两类无法靠正则修（引号是边界还是内层，位置上看不出来），用状态机：
    t2 = _fix_strings(t)
    if _try_load(t2) is not None:
        return t2
    return t


def _fix_strings(t: str) -> str:
    """按「后一个非空字符」判断双引号是**字符串边界**还是**值里的内层引号**。

    边界：后面跟 : , } ] 或结尾
    内层：后面跟字母/中文/空格等（如 "被说"低俗"时…" 里第二个引号后面是"低"）
    内层引号转义掉；同时把字符串内部的裸换行转成 \\n。
    """
    out, i, inside = [], 0, False
    n = len(t)
    while i < n:
        ch = t[i]
        if not inside:
            if ch == '"':
                inside = True
            out.append(ch)
            i += 1
            continue
        # 在字符串内部
        if ch == "\\" and i + 1 < n:          # 已经是转义序列，原样带走
            out.append(t[i:i + 2])
            i += 2
            continue
        if ch == '"':
            j = i + 1
            while j < n and t[j] in " \t\r\n":
                j += 1
            nxt = t[j] if j < n else ""
            if nxt in (":", ",", "}", "]", ""):   # 是边界
                inside = False
                out.append(ch)
            else:                                  # 是值里的内层引号
                out.append('\\"')
            i += 1
            continue
        if ch in "\r\n":                       # 字符串里的裸换行
            out.append("\\n")
            i += 2 if ch == "\r" and i + 1 < n and t[i + 1] == "\n" else 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _try_load(s: str):
    """能解析就返回结果，不能就返回 None。"""
    try:
        return json.loads(s)
    except Exception:
        return None


def _parse_same_request(raw) -> dict:
    """解析 L3 的 same_request（"连续第几轮在要同一件事"）。

    为什么放在 L3 里而不是本地算：实测字符 bigram 分不开"换着说法的同一诉求"
    （"我要听你的声音"→"那我要听语音"→"发个语音条"，两两重合都很低）。
    而 L3 本来就每轮跑、本来就是判"这句落在哪个处境"的最强工具，
    顺带判这一项 = **零额外调用**。

    容错：拿不准一律 count=1（= 不是重复），与全模块"拿不准不触发"的取向一致。
    """
    if not isinstance(raw, dict):
        return {"what": "", "count": 1}
    what = str(raw.get("what") or "").strip()
    try:
        n = int(raw.get("count") or 1)
    except (TypeError, ValueError):
        n = 1
    return {"what": what, "count": max(1, min(n, 99))}


async def classify_behavior(deepseek_client, user_msg: str, history_text: str, behaviors: list) -> str:
    """只取行为名（薄包装，评测/旧调用方用）。拿不准或失败返回空串。"""
    return (await classify_l3(deepseek_client, user_msg, history_text, behaviors))["behavior"]
