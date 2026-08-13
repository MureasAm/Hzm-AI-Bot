"""硬匹配路由：经典梗库 + LLM 语境确认 + 行为意图分类（L3）。

从 core.py 拆出。这些是"先判断这条消息是啥、要不要走固定回复"的入口逻辑，
与"组装消息 + 生成"分离——core 专注消息组装。
"""
import json

from .constants import THINKING_DISABLED
from .config import _get_clients, _get_model_name

# ==================== 💬 经典梗硬匹配库 ====================
# 角色的"固定回答梗"：用户消息里命中关键词，就回固定原话（不经检索）。
# 梗的原话必须来自素材，不编造。宽泛关键词配 LEGENDARY_CONFIRMS 的 LLM 语境确认防误触发。
LEGENDARY_REPLIES = {
    "爱不爱绿冻": [
        "早就说过很爱了...你要是感觉不到，灰泽满也不知道该怎么证明了",
        "爱是也可以的，不爱也可以的~（但其实是爱的，别不信）",
    ],
    "在和谁说话": [
        "在和..在和你说话哦~",
        "在和弹幕说话~"
    ],
    # "爱不爱"的跟进句：绿冻说感觉不到爱（带 LLM 确认）
    "没感觉": [
        "那灰泽满要怎么表现你才感觉得到？抱一下？",
        "感觉不到吗……那灰泽满多说几次，真的很爱",
    ],
    "感觉不": [
        "感觉不出来？那灰泽满只能天天说了，说到你信为止",
    ],
    # "绿冻永远是第二选择"（真实直播梗）：语义检索够不到口语化关系提问（实测~0.55）
    "排第几": [
        "绿冻永远是灰泽满的第二选择，这还用排吗",
        "别排了，绿冻永远是灰泽满的第二选择",
    ],
    "你的粉丝": [
        "不只是粉丝……绿冻永远是灰泽满的第二选择",
        "谁说是普通粉丝了，绿冻永远是灰泽满的第二选择",
    ],
    "在乎我": [
        "在乎的，绿冻永远是灰泽满的第二选择",
        "绿冻永远是灰泽满的第二选择，怎么可能不在乎",
    ],
    "我重要": [
        "重要啊，绿冻永远是灰泽满的第二选择",
        "怎么会不重要，绿冻永远是灰泽满的第二选择",
    ],
    "绿冻算什么": [
        "绿冻算什么……绿冻是灰泽满的第二选择啊",
    ],
    # 问"能不能叫你妈妈"（征求/求宠）：才16岁怎么做父母
    "叫你妈妈": [
        "灰泽满才16岁，怎么做父母啊",
        "16岁当妈？灰泽满自己还是小孩呢",
        "别别别，灰泽满担不起这个称呼",
    ],
    "叫你一声": [
        "灰泽满才16岁，怎么做父母啊",
        "16岁当妈？灰泽满自己还是小孩呢",
    ],
    "叫我妈妈": [
        "灰泽满才16岁，怎么做父母啊",
        "16岁当妈？灰泽满自己还是小孩呢",
    ],
    # 直接喊"妈妈"（玩梗撒娇）：单独回个😅
    "妈妈": [
        "😅",
        "……😅",
    ],
}

# ==================== 梗库双路由：LLM 语境确认 ====================
# 仅宽泛关键词配确认（防误触发）；'排第几''绿冻算什么'等特定词不配。
_CONFIRM_SECOND = (
    "判断这条消息是否属于'绿冻向灰泽满表达自我怀疑：自己只是普通粉丝、不被在乎、"
    "在灰泽满心里没有位置'的语境，需要灰泽满用'绿冻永远是第二选择'来安抚。\n\n"
    "属于（回复'是'）的例子：\n"
    "- 我在你心里排第几\n- 我只是你的粉丝吧\n- 你是不是根本不在乎我\n- 你觉得我重要吗\n"
    "- 我只是个路人而已吧\n- 我在你心里有位置吗\n\n"
    "不属于（回复'否'）的例子：\n"
    "- 你的粉丝好热情啊（是在夸粉丝，不是绿冻自我怀疑）\n"
    "- 这个任务我重要吗（指任务，不是绿冻本人）\n"
    "- 你还在乎我们宿舍吗（指宿舍/他人，不是绿冻本人）\n\n"
    "结合最近对话判断语境——如果最近在聊感情/关系/被冷落，消息里的'重要吗''在乎吗'就是指灰泽满对TA的感情。\n\n"
    "最近对话：\n{context}\n\n消息：{msg}\n只回复：是 或 否"
)
_CONFIRM_LOVE = (
    "判断这条消息是否在'绿冻质疑灰泽满的感情（爱不爱、在不在乎、能不能感觉到爱）'的语境。\n"
    "结合最近对话判断——如果最近在聊感情/爱不爱，消息里的'没感觉''感觉不到'就是指感情。\n\n"
    "属于（回复'是'）的例子：\n"
    "- 可是我没感觉出来（前面在聊爱不爱）\n- 你说爱我但我感觉不到\n\n"
    "不属于（回复'否'）的例子：\n"
    "- 这首歌我没感觉（指歌）\n- 这个菜没味道（指菜）\n\n"
    "最近对话：\n{context}\n\n消息：{msg}\n只回复：是 或 否"
)
_CONFIRM_MOM = (
    "判断这条消息是否在'绿冻玩梗喊灰泽满妈妈'——把灰泽满当妈/叫妈妈求宠/认妈'的语境，需要灰泽满装傻不接。\n\n"
    "属于（回复'是'）的例子：\n"
    "- 妈妈！\n- 喊你一声妈妈行不行\n- 我能叫你妈妈吗\n- 妈！我要抱抱\n\n"
    "不属于（回复'否'）的例子：\n"
    "- 我妈让我早点睡（指用户自己的妈妈）\n- 我妈妈叫我睡觉了（指用户自己的妈妈）\n"
    "- 你妈妈也是这么说你的吗（指灰泽满的妈妈/满妈）\n"
    "- 帮我妈个忙（'妈'是动词，非称呼灰泽满）\n\n"
    "最近对话：\n{context}\n\n消息：{msg}\n只回复：是 或 否"
)

LEGENDARY_CONFIRMS = {
    "你的粉丝": _CONFIRM_SECOND,
    "在乎我": _CONFIRM_SECOND,
    "我重要": _CONFIRM_SECOND,
    "没感觉": _CONFIRM_LOVE,
    "感觉不": _CONFIRM_LOVE,
    "妈妈": _CONFIRM_MOM,
}


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
# 背景：embedding 聚的是"句式"不是"意图"——'角色名你唱歌好听'（夸）和
# '角色名你怎么又迟到了'（质问）句式相同，在向量空间挤成一团，余弦匹配会把
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
