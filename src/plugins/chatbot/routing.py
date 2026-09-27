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
from .config import _get_clients, _get_model_name, extract_chat_content

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
#   · phrases  回答"该用哪些词"（11 组）
# 而措辞原本走向量检索，实测 100% 开火（trigger 是 6~13 字的**类别标签**，不是句子，
# 余弦根本分不开）——用检索工具干分类的活。并进来 = 零额外调用 + 判据统一。
BEHAVIOR_CLASSIFY_PROMPT = """你是{role_name}的意图分类器。判断用户刚发的这条消息落入哪些"已知情境"。只有明确匹配才选，拿不准一律不放（宁可不触发，不误触发）。

【行为】用户这条消息触发哪个行为场景？**最多选一个**（它决定{role_name}该怎么做）：
{behavior_defs}

行为判定要点：
- 只看用户这条消息本身的内容和语气，结合最近对话判断语境。
- "被夸"：消息确实在夸{role_name}（声音/外貌/才能/表现/生日祝福/唱歌好听等）。
- "被质疑/失约被催"：用户在质问、戳穿或催问{role_name}（骗人/敷衍/迟到/没播/鸽）。
- "被越界"：玩笑/幻想触及个人边界——含两类：低俗/黄段子/过度幻想，以及**关系向亲密越界**（求交往、要当女友/男朋友、喊老公/老婆、恋人昵称等）。这类也归"被越界"，别漏判成 null。
- "冷场"：提及或营造社交尴尬/冷场，要求{role_name}救场。
- "立Flag/感性流露/主动抛梗"：消息必须明显对应那个情境。
- 普通闲聊、提问、寒暄、表情、玩梗 → null。
- 拿不准 → null。
{phrase_section}
最近对话：
{history}

用户消息：{user_msg}

只输出 JSON：{{"behavior": "<行为name>" 或 null, "phrases": ["<措辞组id>", …]}}"""

# 措辞那一节（没有措辞组数据时整节不出现，免得给模型一个空列表）
PHRASE_SECTION = """
【措辞】用户这句话会让{role_name}用上哪些措辞组？**可以多选，也可以全不选**：
{phrase_defs}

措辞判定要点：
- 判的是"用户**冲着她**说了这类话"，不是"提到了同一个词"。
- 例：用户说"你唱歌真好听"→ 是夸她 → 选；用户说"今天股市怎么样"→ 跟她无关 → 不选。
- 例：用户说"你昨晚为什么没播"→ 在问责 → 选；用户说"感冒吃什么药"→ 在问药 → 不选。
- **多数消息一个都不选**（日常闲聊占大多数）。拿不准 → 不选。
"""


async def classify_l3(deepseek_client, user_msg: str, history_text: str,
                      behaviors: list, phrase_groups: list = None) -> dict:
    """一次调用判出：行为名（0/1 个）+ 措辞组 id（0~N 个）。

    返回 {"behavior": str, "phrases": [str]}；**失败/拿不准返回空**（不触发任何东西）。
    返回的名字/id 都必须是数据里的真值（防模型编造）。

    ⚠️ 判据措辞是本模块最要紧的东西（改词=改行为，实测过判据稍动结果就从 0% 跳到 74%）。
    """
    phrase_groups = phrase_groups or []
    if not user_msg:
        return {"behavior": "", "phrases": []}

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

    phrase_defs = "\n".join(
        f"- {g.get('id')}：{g.get('trigger', '')}（{g.get('meaning', '')}）"
        for g in phrase_groups if g.get("id")
    )
    prompt = BEHAVIOR_CLASSIFY_PROMPT.format(
        behavior_defs="\n".join(defs) or "（无）",
        phrase_section=PHRASE_SECTION.format(phrase_defs=phrase_defs, role_name="灰泽满") if phrase_defs else "",
        history=history_text or "（无）", user_msg=user_msg,
        role_name="灰泽满",
    )
    try:
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=40,
            **THINKING_DISABLED,
        )
        content = extract_chat_content(resp)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        parsed = json.loads(content)
        names = {b.get("name") for b in (behaviors or [])}
        behavior = str(parsed.get("behavior") or "").strip()
        ids = {g.get("id") for g in phrase_groups}
        raw = parsed.get("phrases")
        kept, seen = [], set()
        for gid in (raw if isinstance(raw, list) else []):
            gid = str(gid).strip()
            if gid in ids and gid not in seen:      # 拦住模型编的 id
                seen.add(gid)
                kept.append(gid)
        return {"behavior": behavior if behavior in names else "", "phrases": kept}
    except Exception as e:
        print(f"⚠️ L3 意图分类失败（降级不触发）: {e}")
        return {"behavior": "", "phrases": []}


async def classify_behavior(deepseek_client, user_msg: str, history_text: str, behaviors: list) -> str:
    """只取行为名（薄包装，评测/旧调用方用）。拿不准或失败返回空串。"""
    return (await classify_l3(deepseek_client, user_msg, history_text, behaviors))["behavior"]
