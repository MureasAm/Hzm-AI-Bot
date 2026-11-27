"""硬匹配路由：经典梗库 + LLM 语境确认 + 行为意图分类（L3）。

从 core.py 拆出。这些是"先判断这条消息是啥、要不要走固定回复"的入口逻辑，
与"组装消息 + 生成"分离——core 专注消息组装。

梗库数据化：LEGENDARY_REPLIES / LEGENDARY_CONFIRMS 从 persona/decision/legendary.json 加载，
使用者填 JSON 即可，不用改代码。改后重启生效。
"""
import json
import re
import datetime
from .constants import THINKING_DISABLED, LEGENDARY_FILE
from .config import log_cache_usage, _get_clients, _get_model_name, extract_chat_content
from .config import aux_clients_or, aux_model  # 辅助链（2026-10-29）
from .llm_json import parse_json_object
from .retrieval_context import CONTEXT_GUIDANCE, MATERIAL_SOURCES, RetrievalContext
from .song_requests import parse_song_request
from .term_hints import terms_hint_block
from .turn_understanding import parse_understanding

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


async def legendary_confirmed(user_msg: str, prompt_template: str, history: str = "",
                              context: RetrievalContext | None = None) -> bool:
    """LLM 判断关键词命中的消息是否真是目标梗的语境（双路由第二层，防误触发）。

    关键词命中是低频事件，为它加一次便宜 LLM 判断成本可控。
    共用语境下确认失败走普通生成；旧调用保留默认放行的兼容策略。
    """
    try:
        content = prompt_template
        if "{context}" in content:
            content = content.replace("{context}", history or "（无）").replace("{msg}", user_msg)
        else:
            content = content.replace("{msg}", user_msg)
        content += terms_hint_block(user_msg)
        if context is not None:
            content += "\n\n【本轮共用语境】\n" + context.note()
            content += "\n" + CONTEXT_GUIDANCE
        deepseek_client, _ = aux_clients_or(_get_clients)
        resp = await deepseek_client.chat.completions.create(
            model=aux_model(_get_model_name()),
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=120,
            **THINKING_DISABLED,
        )
        log_cache_usage(resp, "梗确认")
        c = extract_chat_content(resp)
        if context is not None:
            # 问号保留：疑问或回显选项不构成肯定确认。
            return c.strip().rstrip("。.!！").strip() == "是"
        # 旧调用保留原来的前缀解析，兼容原确认策略。
        return c.startswith("是") and not c.startswith("不是") and not c.startswith("是不是")
    except Exception as e:
        strategy = "走普通生成" if context is not None else "默认放行"
        print(f"⚠️ 梗确认失败（{strategy}）: {e}")
        return context is None


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
_BEHAVIOR_CLASSIFY_BODY = """你是{role_name}的意图分类器。判断用户刚发的这条消息落入哪些「已知情境」。只有明确匹配才选，拿不准一律不放（宁可不触发，不误触发）。

【行为】用户这条消息触发哪个行为场景？**最多选一个**（它决定{role_name}该怎么做）：
{behavior_defs}

行为判定要点：
- 只看用户这条消息本身的内容和语气，结合最近对话判断语境。
- 先确认谁在说话、评价谁、要求谁做什么，再按数据trigger绑定行为的主体与互动对象。
  对方在与灰泽满交谈，不等于灰泽满在与第三人互动。需要同期、前辈或熟人的trigger，
  必须有明确提及或可由最近对话追溯的对应人物；不能凭熟悉语气补出未出现的第三人。
- **判据以每条 trigger 的描述为准**，下面是容易判错的地方，别再按关键词硬映射：
- 「被夸」：用户在夸她或认可她本人（声音/表现/才能/用心），**不是随口道贺**。
  ⚠️ 「生日快乐」「祝你开心」这类纯祝福、以及「我考上了」这类用户自己的好消息，都不要判成被夸。
- 「被质疑戳穿 / 失约被催」：用户在质问、戳穿、翻旧账、传她没说过的话，或催她没做到的事。
  如果消息里**既有催问又有质疑**，按哪个更重选一个（拿不准就不选）。
- 「被说低俗或擦边时冷脸挡回」：内容带**低俗/擦边/黄段子/XP性质**，或用户**要她说一句带颜色的话**
  （用户会这么说：'给我们造个黄桃吧''说句骚话'）。求交往、喊老公老婆、要啵啵这类**不含色情内容**，不算这一格。
- 「被起哄编故事加码」与「低俗擦边」的**区别在内容的性质，不在谁在起哄**：
  · 内容是**色情/擦边** → 低俗那条（她冷脸挡回）
  · 内容是**荒诞编造、往她身上安离谱的假事**（说她怀孕了、说她昨晚没在家、把她口误往歪处带）→ 起哄编故事那条（她顺着玩）
  两者应对**相反**，判错就很假，拿不准时按「内容是不是色情」来分。
- 「被真心对待」：用户**很认真地**感谢她、说在意她、关心她的情绪、深夜惦记她——是真心，不是玩笑。
  随口一句「谢谢」、普通寒暄不算。
- 普通闲聊或玩梗也可能符合某条trigger，按当前提供的数据判断，不以消息形式一概排除。
- 没有匹配的数据trigger才返回null，不要为了选行为编造参与者或互动关系。
- 拿不准 → null。

★ 另外判一件事：**这是不是「连续第几轮在要同一件事」**（same_request）。
  · 看【最近对话】里用户那几句：如果**当前这句和前面几句是在要同一样东西**
    （换个说法也算，如「我要听你的声音」→「那我要听语音」→「发个语音条」），
    就说出是**同一件事**、以及**从几轮前开始连着**（含当前这轮，最小 1）。
  · **要的是「诉求」相同，不是「话题」相同**。用户连着问三个不同的生活问题 → 不算。
  · ⚠️ **注意方括号里的时间**（如 `[昨天 17:11｜28小时前]`）：如果用户那几句**中间隔了很久**
    （隔了几小时、或者跨了天），那**不算"连着"**——只从**同一场对话内**的连续几轮算起。
    上一句在几小时前 → 这轮就是 `count` 1。
  · `what` 写「在要什么」（如「要语音」「要被叫宝宝」「要一个表态」），不要知道就不写。
★ 再判一件事：**这是不是一个定时提醒请求**（reminder）。
  · 用户想在某个时间收到提醒，任务、时段和分钟可以在最近几轮分别给出。
    虚构示例：「记得提醒我收衣服」→「下午四点多」→「四点十分」合起来是当天下午 16:10 提醒收衣服。
  · 时间不够明确时 → ambiguous=true，clarify 写一句反问，不要自行猜测。
    ⚠️ **裸数字规则**：「十」「10」「就10」这种未带时间单位的数字一律 ambiguous=true。
    例如「下午四点多」之后只说「10」，仍需确认「你是指下午四点十分，还是十分钟后？」
    「十点」「16:10」「下午四点十分」是明确钟点；「十分钟后」是明确相对时间。
  · 没有提醒请求 → null。
  · 输出：`"reminder": {{"when_hint": "16:10", "what": "提醒收衣服",
    "due_at": "2026-01-02T16:10:00+08:00", "ambiguous": false, "clarify": ""}}`或 null。
    due_at 用当前时间的时区换算成绝对时间；换算不了就留空并设 ambiguous=true。
当前时间：{now}
最近对话：
{history}

用户消息：{user_msg}"""

_LEGACY_OUTPUT_CONTRACT = """只输出 JSON：{{"behavior": "<行为name>" 或 null, "same_request": {{"what": 「在要什么」, "count": 连续轮数}}, "reminder": <提醒对象或 null>}}"""
BEHAVIOR_CLASSIFY_PROMPT = _BEHAVIOR_CLASSIFY_BODY + "\n\n" + _LEGACY_OUTPUT_CONTRACT


_ANSWER_SCOPES = frozenset({"conversation", "general_task"})

RESPONSE_TASK_PROMPT = """【本轮任务与回答范围】
不依赖候选内容，只根据用户原话和最近对话，判断本轮实际要问、聊或处理什么，以及互动对象是谁。
先绑定人物，再写实际需求：understanding.focus只能是character/user/third_party/shared/general/unclear；
focus_text简述实际焦点和人物，request写本轮真正要回应什么，exclusions列仅被排除的用途，
premises列待核实前提；疑问、假设或用户转述的前提不能当作既成事实，task也不能把它们写成事实。
否定只约束被排除的用途，不能扩成对整个话题的禁止；排除亲历不等于排除概念或词义解释。
third_parties只列角色与当前用户之外实际提到的人；name使用原话称呼，evidence必须是原话或历史中
包含该称呼的一段连续原文，不能拼接、补出名字或把当前交谈双方当第三人；没有就[]。
focus_text至多160字，request至多320字；exclusions/premises各至多6条、每条160字；
third_parties至多6人、name至多64字、evidence至多240字。
task用一句话说明实际任务，scope只取conversation或general_task；输出结构见末尾唯一合同，不输出推理过程。
task保留主体、否定和引用边界，不回答问题、不猜库里有什么，也不判断资料存在哪个来源。
- conversation：默认范围。角色日常、个人事实、用户历史、自然联想或普通聊天均用它；
  未知对象或含义不清时也用conversation，不可只凭知识问句的形态断定是通用任务。
- general_task：只有明确是通用知识建议、操作求助或文本处理，且完成它不需要角色本人或当前用户的个人背景时才选。
  用户要求结合个人事实或历史时不能选；不要因候选经历同话题而把通用任务改成讨论她的过去。
范围不等于必须注入资料，具体材料会在下一步分别核对。行为判定与任务判定独立。"""


def _context_output_contract(song_text: str) -> str:
    schema = {"behavior": None, "same_request": {"what": "", "count": 1}, "reminder": None,
              "task": "一句话说明本轮实际任务", "scope": "conversation",
              "understanding": {"focus": "unclear", "focus_text": "本轮实际焦点和人物",
                                "request": "本轮实际需求", "exclusions": [],
                                "premises": [], "third_parties": []}}
    if song_text:
        schema["song_request"] = None
    return ("【唯一输出合同】\n" + json.dumps(schema, ensure_ascii=False)
            + "\n以上字段全部必须出现；behavior只能填提供的行为name或null，scope只能填conversation或general_task。"
              "按前述判据填写各字段，只输出一个完整JSON对象，不要代码围栏、解释或推理过程。")


def _parse_l3_fields(parsed: dict, names: set, now: datetime.datetime, *, contextual: bool,
                     song_text: str, raw_text: str = "", history: str = "") -> dict:
    raw_behavior = parsed.get("behavior")
    if isinstance(raw_behavior, str) and raw_behavior.strip() == "null":
        raw_behavior = None
    behavior = str(raw_behavior or "").strip()
    valid_behavior = "behavior" in parsed and (
        raw_behavior is None or (isinstance(raw_behavior, str) and behavior in names))
    result = {"behavior": behavior if behavior in names else "",
              "same_request": _parse_same_request(parsed.get("same_request")),
              "reminder": _parse_reminder(parsed.get("reminder"), now)}
    if contextual:
        raw_task, raw_scope = parsed.get("task"), parsed.get("scope")
        task = raw_task.strip() if isinstance(raw_task, str) else ""
        scope = raw_scope.strip() if isinstance(raw_scope, str) else ""
        valid_task = bool(task) and scope in _ANSWER_SCOPES
        understanding = parse_understanding(parsed.get("understanding"), raw_text, history)
        result.update(
            behavior=behavior if valid_behavior and behavior in names else "",
            classification_ok=valid_behavior,
            task=task if valid_task else "", scope=scope if valid_task else "",
            needs=sorted(MATERIAL_SOURCES) if valid_task else [], needs_ok=valid_task,
            understanding=understanding, understanding_ok=understanding is not None,
        )
    if song_text:
        result["song_request"] = parse_song_request(parsed.get("song_request"), song_text)
    return result


def _ground_behavior(result: dict, requires_third_party: set) -> dict:
    """A denied relation scene is an explicit null, so keyword fallback stays closed."""
    if result["behavior"] in requires_third_party:
        understanding = result.get("understanding")
        if understanding is None or not understanding.third_parties:
            result["behavior"] = ""
    return result


async def classify_l3(deepseek_client, user_msg: str, history_text: str,
                      behaviors: list, now: datetime.datetime | None = None, *,
                      context: RetrievalContext | None = None, song_text: str = "") -> dict:
    """一次调用判出：行为名（0/1 个）+ 连续索要轮数 + 可选定时提醒。

    返回 {"behavior": str, "same_request": {...}, "reminder": {...}|None}；
    **失败/拿不准返回空**（不触发任何东西）。返回的名字必须是数据里的真值（防模型编造）。
    传入共用语境时额外返回 task、scope 及展开后的 needs/needs_ok；行为与任务分别校验。

    ⚠️ 判据措辞是本模块最要紧的东西（改词=改行为，实测过判据稍动结果就从 0% 跳到 74%）。

    ⚠️ 原「措辞组」那一路已随 phrases.json 一起删除（2026-10-04）：
       输出的 `phrases` 字段、`phrase_groups` 参数、`PHRASE_SECTION` 全部拆掉。
       留着死字段只会浪费 token、并给 JSON 输出添乱（踩过：加字段后 40 token 截断过 JSON）。
       现在 `history_text` 由调用方传**带时间标签**的版本（`[昨天 17:11｜28小时前] 用户：…`），
       same_request 靠它判断"是不是同一场对话内的连续索要"；reminder 也用它合并不同轮次的提醒任务与钟点。
    """
    now = now or datetime.datetime.now().astimezone()
    empty = {"behavior": "", "same_request": {"what": "", "count": 1}, "reminder": None}
    if context is not None:
        empty["classification_ok"] = False
        empty.update(needs=[], needs_ok=False, task="", scope="",
                     understanding=None, understanding_ok=False)
    if not user_msg:
        return empty

    defs = []
    for b in behaviors or []:
        if not b.get("name"):
            continue
        line = f"- {b['name']}：{b.get('trigger', '')}"
        if context is not None and b.get("requires_third_party") is True:
            line += "（前提：understanding.third_parties中有原话或历史依据的第三人）"
        # 行为定义带真实粉丝话样例：领域黑话光靠 trigger 描述 LLM 认不出，
        # 给真实样例当参照（素材驱动），分类更准。
        for s in b.get("samples", [])[:2]:
            u = (s.get("user") or "").strip()
            if u:
                line += f"\n    例：{u}"
        defs.append(line)

    template = _BEHAVIOR_CLASSIFY_BODY if context is not None else BEHAVIOR_CLASSIFY_PROMPT
    prompt = template.format(
        behavior_defs="\n".join(defs) or "（无）",
        history=history_text or "（无）", user_msg=user_msg,
        role_name="灰泽满", now=now.isoformat(timespec="minutes"),
    )
    prompt += terms_hint_block(user_msg)
    if context is not None:
        prompt += "\n\n【本轮共用语境】\n" + context.note()
        prompt += "\n" + CONTEXT_GUIDANCE
        prompt += "\n\n" + RESPONSE_TASK_PROMPT
    if song_text:
        prompt += (
            '\n\n【点歌判定】在同一个 JSON 添加 "song_request"。'
            '只判断下面引号内的用户本人正文有没有当下请灰泽满唱或发歌。'
            '上文仅用于识别指代及回答“想听哪首/要发完整版吗”这种追问；引用、转述、称赞、'
            '问会不会唱、只说想听她唱歌、聊别人的歌、将来的邀约、取消或否定都返回 null。'
            '如果同批前面点歌、后面撤销，按撤销。\n'
            '有邀请时填 {"mode": "clip", "title": ""}，'
            'title 写用户点的歌名原文（可以据用户明确认领的近轮选项回填），没点名留空。'
            '说唱/哼、给两句、挑一首：clip；只有明确要求整首时填 full，是否点歌名都一样；'
            '“刚才那首发完整版”“完整版呢”指刚发的唱段：recent_full；'
            '“换一首”：another；要求从某时刻/某句开始或同首再来一段：other_part。'
            '紧接在她明确询问“要发某首完整版吗”之后答应：full，并填那首歌名。'
            '完整版默认也是语音条。只有用户本人本批明确提出要音频文件时，才在 full 或 recent_full 中加 "as_file": true。'
            '\n用户本人正文：' + json.dumps(song_text, ensure_ascii=False)
        )
    if context is not None:
        prompt += "\n\n" + _context_output_contract(song_text)
    original_prompt = prompt
    result = empty.copy()
    seen_fields = set()
    names = {b.get("name") for b in (behaviors or [])}
    required_participants = {b.get("name") for b in (behaviors or [])
                             if b.get("requires_third_party") is True}
    for attempt in range(2 if context is not None else 1):
        try:
            resp = await deepseek_client.chat.completions.create(
                model=_get_model_name(),
                messages=[{"role": "user", "content": prompt}],
                **({"response_format": {"type": "json_object"}} if context is not None else {}),
                temperature=0, max_tokens=900 if context is not None else 420, **THINKING_DISABLED,
            )
        except Exception as error:
            # HTTP/权限/传输错误由客户端原有降级链处理，不触发格式补救。
            print(f"⚠️ L3 意图分类失败（保留已确认结果）: {error}")
            return _ground_behavior(result, required_participants) if context is not None else result
        log_cache_usage(resp, "行为分类" if attempt == 0 else "行为分类格式补救")
        try:
            content = extract_chat_content(resp)
            parsed = parse_json_object(_repair_llm_json(content))
            incoming = _parse_l3_fields(parsed, names, now, contextual=context is not None,
                                       song_text=song_text, raw_text=user_msg, history=history_text)
        except (ValueError, TypeError, KeyError, RuntimeError, AttributeError, IndexError) as error:
            reason = f"不是完整JSON对象：{type(error).__name__}"
        else:
            if context is None:
                return incoming
            if incoming["classification_ok"] and not result["classification_ok"]:
                result.update(behavior=incoming["behavior"], classification_ok=True)
            if incoming["needs_ok"] and not result["needs_ok"]:
                for key in ("task", "scope", "needs", "needs_ok"):
                    result[key] = incoming[key]
            if incoming["understanding_ok"] and not result["understanding_ok"]:
                result.update(understanding=incoming["understanding"], understanding_ok=True)
            for key in ("same_request", "reminder", "song_request"):
                if key in parsed and key in incoming and key not in seen_fields:
                    result[key] = incoming[key]
                    seen_fields.add(key)
            invalid_understanding = "understanding" in parsed and not incoming["understanding_ok"]
            if incoming["needs_ok"] and not invalid_understanding:
                return _ground_behavior(result, required_participants)
            reason = ("understanding结构或第三人原文依据无效" if invalid_understanding else
                      "缺少非空字符串task或合法scope")
        if context is not None and attempt == 0:
            locked = {"behavior": result["behavior"] or None} if result["classification_ok"] else {}
            prompt = (original_prompt + "\n\n【一次格式补救】上一轮" + reason
                      + "。根据相同原话和上下文补齐唯一输出合同，只输出完整JSON，不要解释。"
                      + "已确认字段保持不变：" + json.dumps(locked, ensure_ascii=False))
    return _ground_behavior(result, required_participants) if context is not None else result


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


def _parse_reminder(raw, now: datetime.datetime) -> dict | None:
    """解析 L3 的 reminder。拿不准/缺内容返回 None；时间不明确则 ambiguous=True。

    due_at 必须是未来 30 天内的绝对时间；模型给的格式不对/超出范围一律降级成
    ambiguous（让回复先反问），**不拿一个可能错的绝对时间去排提醒**。
    """
    if not isinstance(raw, dict):
        return None
    what = str(raw.get("what") or "").strip()
    if not what:
        return None
    when_hint = str(raw.get("when_hint") or "").strip()
    due = None
    raw_due = str(raw.get("due_at") or "").strip()
    if raw_due:
        try:
            due = datetime.datetime.fromisoformat(raw_due.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            due = None
    if due is not None:
        if due.tzinfo is None:
            due = due.replace(tzinfo=now.tzinfo)
        if not (now < due <= now + datetime.timedelta(days=30)):
            due = None
    ambiguous = bool(raw.get("ambiguous")) or due is None
    clarify = str(raw.get("clarify") or "").strip()
    if ambiguous and not clarify:
        clarify = "是几点提醒你？"
    return {
        "when_hint": when_hint[:40],
        "what": what[:80],
        "due_at": "" if ambiguous or due is None else due.isoformat(),
        "ambiguous": ambiguous,
        "clarify": clarify[:80],
    }


async def classify_behavior(deepseek_client, user_msg: str, history_text: str, behaviors: list) -> str:
    """只取行为名（薄包装，评测/旧调用方用）。拿不准或失败返回空串。"""
    return (await classify_l3(deepseek_client, user_msg, history_text, behaviors))["behavior"]
