"""硬匹配路由：经典梗库 + LLM 语境确认 + 行为意图分类（L3）。

从 core.py 拆出。这些是"先判断这条消息是啥、要不要走固定回复"的入口逻辑，
与"组装消息 + 生成"分离——core 专注消息组装。

梗库数据化：LEGENDARY_REPLIES / LEGENDARY_CONFIRMS 从 persona/world/legendary.json 加载，
使用者填 JSON 即可，不用改代码。改后重启生效。
"""
import json
from pathlib import Path

from .constants import THINKING_DISABLED, PROJECT_ROOT
from .config import _get_clients, _get_model_name

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
        _legendary_cache = {
            "replies": data.get("replies", {}) or {},
            "confirms": data.get("confirms", {}) or {},
        }
    except (json.JSONDecodeError, OSError):
        _legendary_cache = {"replies": {}, "confirms": {}}
    return _legendary_cache


# ==================== 💬 经典梗硬匹配库（数据化） ====================
LEGENDARY_REPLIES = load_legendary()["replies"]
LEGENDARY_CONFIRMS = load_legendary()["confirms"]


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
        return "是" in (resp.choices[0].message.content or "")
    except Exception as e:
        print(f"⚠️ 梗确认失败（默认放行）: {e}")
        return True


# ==================== 🎭 行为意图分类（L3：LLM 判意图，不再用 embedding 猜） ====================
# 背景：embedding 聚的是"句式"不是"意图"——'灰泽满你唱歌好听'（夸）和
# '灰泽满你怎么又迟到了'（质问）句式相同，在向量空间挤成一团，余弦匹配会把
# 夸奖误判成质疑。治本：行为归属交给 LLM 理解，判别性词汇仍作关键词兜底。
BEHAVIOR_CLASSIFY_PROMPT = """你是{role_name}的行为意图分类器。判断用户刚发的这条消息是否明确落入某个"行为触发场景"。只有明确匹配才选，拿不准一律 null（宁可不触发，不误触发）。

可选行为（name：触发情境）：
{behavior_defs}

判定要点：
- 只看用户这条消息本身的内容和语气，结合最近对话判断语境。
- "被夸"：消息确实在夸{role_name}（声音/外貌/才能/表现/生日祝福/唱歌好听等）。
- "被质疑/失约被催"：用户在质问、戳穿或催问{role_name}（骗人/敷衍/迟到/没播/鸽）。
- "被越界"：玩笑/幻想触及个人边界（黄段子/低俗/过度幻想等）。
- "冷场"：提及或营造社交尴尬/冷场，要求{role_name}救场。
- "立Flag/感性流露/主动抛梗"：消息必须明显对应那个情境。
- 普通闲聊、提问、寒暄、表情、玩梗 → null。
- 拿不准 → null。

最近对话：
{history}

用户消息：{user_msg}

只输出 JSON：{{"behavior": "<可选行为name>" 或 null}}"""


async def classify_behavior(deepseek_client, user_msg: str, history_text: str, behaviors: list) -> str:
    """LLM 判定用户消息落入哪个行为场景；拿不准或失败返回空串（不触发任何行为）。

    返回的行为名必须是 behaviors 里的真实 name（防模型编造）。
    """
    if not behaviors or not user_msg:
        return ""
    # 行为定义带真实粉丝话样例：领域黑话光靠 trigger 描述 LLM 认不出，
    # 给真实样例当参照（素材驱动），分类更准。
    defs = []
    for b in behaviors:
        if not b.get("name"):
            continue
        line = f"- {b['name']}：{b.get('trigger', '')}"
        for s in b.get("samples", [])[:2]:
            u = (s.get("user") or "").strip()
            if u:
                line += f"\n    例：{u}"
        defs.append(line)
    prompt = BEHAVIOR_CLASSIFY_PROMPT.format(
        behavior_defs="\n".join(defs), history=history_text or "（无）", user_msg=user_msg,
        role_name="灰泽满",
    )
    try:
        resp = await deepseek_client.chat.completions.create(
            model=_get_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
            **THINKING_DISABLED,
        )
        content = (resp.choices[0].message.content or "").strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        parsed = json.loads(content)
        behavior = str(parsed.get("behavior") or "").strip()
        names = {b.get("name") for b in behaviors}
        return behavior if behavior in names else ""
    except Exception as e:
        print(f"⚠️ 行为意图分类失败（降级不触发）: {e}")
        return ""
