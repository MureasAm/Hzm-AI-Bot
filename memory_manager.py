import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

MEMORY_FILE = Path("long_term_memory.json")

# 关系等级阈值
RELATION_LEVELS = {
    "stranger": 0,
    "acquaintance": 3,
    "familiar": 8,
    "close": 30
}

def load_memory() -> dict:
    """加载整个记忆文件，空文件或格式错误时返回空字典"""
    if not MEMORY_FILE.exists():
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return {}

def save_memory(memory: dict) -> None:
    """保存整个记忆文件"""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def get_user_memory(user_id: str) -> dict:
    """获取指定用户的记忆卡，不存在则返回空卡片"""
    memory = load_memory()
    return memory.get(user_id, {})

def update_user_memory(user_id: str, updates: dict) -> None:
    """增量合并更新用户记忆"""
    memory = load_memory()
    card = memory.get(user_id, {})

    # 基础统计
    card["total_interactions"] = card.get("total_interactions", 0) + 1
    card["last_seen"] = datetime.now().isoformat()

    # 合并 impressions (标签)
    if updates.get("new_impression"):
        impressions = card.setdefault("impressions", [])
        new_imp = updates["new_impression"]
        found = False
        for imp in impressions:
            tag = imp["tag"] if isinstance(imp, dict) else imp
            if tag == new_imp:
                if isinstance(imp, dict):
                    imp["confidence"] = min(1.0, imp.get("confidence", 0.8) + 0.1)
                    imp["last_updated"] = datetime.now().isoformat()
                else:
                    impressions[impressions.index(imp)] = {
                        "tag": new_imp,
                        "confidence": 0.9,
                        "last_updated": datetime.now().isoformat()
                    }
                found = True
                break
        if not found:
            impressions.append({
                "tag": new_imp,
                "confidence": 0.8,
                "last_updated": datetime.now().isoformat()
            })

    # 合并用户事实
    if updates.get("new_user_fact"):
        user_facts = card.setdefault("user_facts", [])
        new_fact = updates["new_user_fact"]
        if isinstance(new_fact, str):
            new_fact_obj = {"fact": new_fact, "recorded": datetime.now().isoformat()}
        else:
            new_fact_obj = new_fact
        if not any(f.get("fact") == new_fact_obj.get("fact") for f in user_facts if isinstance(f, dict)):
            user_facts.append(new_fact_obj)

    # 合并自我披露
    if updates.get("new_self_fact"):
        self_facts = card.setdefault("self_facts", [])
        new_sf = updates["new_self_fact"]
        if isinstance(new_sf, str):
            new_sf_obj = {"fact": new_sf, "shared_on": datetime.now().isoformat()}
        else:
            new_sf_obj = new_sf
        if not any(s.get("fact") == new_sf_obj.get("fact") for s in self_facts if isinstance(s, dict)):
            self_facts.append(new_sf_obj)

    # 合并重要时刻
    if updates.get("new_moment"):
        moments = card.setdefault("significant_moments", [])
        moments.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": updates["new_moment"]
        })
        if len(moments) > 5:
            moments.pop(0)

    # 关系变化处理
    if updates.get("relationship_change"):
        change = updates["relationship_change"]
        current_level = card.get("relationship_level", "stranger")
        if change in ("warmed_up", "first_emotional") and current_level != "close":
            levels = ["stranger", "acquaintance", "familiar", "close"]
            idx = levels.index(current_level) if current_level in levels else 0
            if idx < len(levels) - 1:
                card["relationship_level"] = levels[idx + 1]

    # 根据互动次数自动调整等级
    interactions = card.get("total_interactions", 0)
    if interactions >= RELATION_LEVELS["close"]:
        card["relationship_level"] = "close"
    elif interactions >= RELATION_LEVELS["familiar"]:
        card["relationship_level"] = "familiar"
    elif interactions >= RELATION_LEVELS["acquaintance"]:
        card["relationship_level"] = "acquaintance"
    else:
        card.setdefault("relationship_level", "stranger")

    memory[user_id] = card
    save_memory(memory)

def build_memory_context(card: dict) -> str:
    """将记忆卡转化为提示文本"""
    if not card:
        return ""

    level = card.get("relationship_level", "stranger")
    parts = []

    # 关系等级 → 语气提示 + 动作限制 + 心虚限制
    if level == "close":
        parts.append('【关系】很熟悉的老朋友。可以放松防御，少用括号，偶尔说真话，允许使用"我"自称。')
        parts.append("【动作限制】禁止使用揉眼睛。禁止使用心虚。被催播时用自嘲代替心虚。")
    elif level == "familiar":
        parts.append("【关系】来过不少次的绿冻。比对新粉随意，可以少一点心虚。")
        parts.append("【动作限制】禁止使用揉眼睛。尽量少用心虚，每3轮最多1次。")
    elif level == "acquaintance":
        parts.append("【关系】见过几次的绿冻，保持标准防御模式，但不用太戒备。")
        parts.append("【动作限制】尽量不要揉眼睛，你不是刚睡醒。心虚每2轮最多1次。")
    else:
        parts.append("【关系】新面孔，用标准防御模式。")
        parts.append("【动作限制】允许首次回复揉眼睛一次。被催播时可以心虚。")

    # 印象标签
    impressions = card.get("impressions", [])
    if impressions:
        high_conf = []
        for imp in impressions:
            tag = imp["tag"] if isinstance(imp, dict) else imp
            conf = imp.get("confidence", 0.8) if isinstance(imp, dict) else 0.8
            if conf > 0.6:
                high_conf.append(tag)
        if high_conf:
            parts.append(f"这个绿冻给你的印象：{'、'.join(high_conf)}。")

    # 用户的事实
    user_facts = card.get("user_facts", [])
    if user_facts:
        fact_strs = []
        for f in user_facts:
            if isinstance(f, dict):
                fact_strs.append(f.get("fact", ""))
            else:
                fact_strs.append(str(f))
        if fact_strs:
            parts.append(f"这个绿冻曾提过：{'；'.join(fact_strs)}。可以自然提及。")

    # 已透露的事实（避免重复）
    self_facts = card.get("self_facts", [])
    if self_facts:
        fact_strs = []
        for s in self_facts:
            if isinstance(s, dict):
                fact_strs.append(s.get("fact", ""))
            else:
                fact_strs.append(str(s))
        if fact_strs:
            parts.append(f"你已跟TA说过：{'；'.join(fact_strs)}。不要再重复自曝这些事。")

    # 最近的亮点时刻
    moments = card.get("significant_moments", [])
    if moments:
        recent = moments[-1]["summary"]
        parts.append(f"你们之间最近的记忆：{recent}。聊到相关话题时可自然提起。")

    return "\n".join(parts)

MEMORY_EXTRACT_PROMPT = """
你是一个记忆提取助手。分析以下对话，只提取**值得长期记住的新信息**。忽略日常寒暄。

【当前记忆概要】
{current_summary}

【本轮对话】
用户：{user_msg}
灰泽满：{reply}

【提取要求】
只提取本轮新增的信息。如果本轮没有值得记录的新内容，返回 null。
- 判断标准：如果这条信息在明天、下周的对话中还能成立，才值得记录。
- **绝对不要记录为了附和用户而临时编造的状态**：如果用户说"我是上班族"，你跟着说"我也有作业压力"，这种附和性内容不要记录。
- **冲突检测**：如果提取的 self_fact 与当前记忆中的 impressions 或 user_facts 明显矛盾（如用户是上班族，你却记录自己也是上班族），不要提取。
- 如果不确定，宁可不提取。

**关于印象标签（new_impression）**：
- 从用户的话中抽象出长期身份或性格标签（如"上班族""学生党""夜猫子""喜欢催播"）
- 即使用户没有直接说"我是上班族"，如果说了"刚下班"，也应抽象为"上班族"
- 如果用户透露的信息只是一次性状态（如"今天很累"），不要提取

**关于用户事实（new_user_fact）**：
- 只记录用户的长期身份、职业、爱好等持续性信息（如"做设计的""在考研""养猫"）
- 不要记录瞬间状态（如"今天很累""刚下班"）

**关于自我披露（new_self_fact）的重要限制**：
- 只记录灰泽满透露的**长期个人特征或真实经历**（如"拖延症晚期""在国外留学""不会做饭"）。
- **绝对不要记录瞬间状态**：如"正在吃泡面""刚睡醒""今天嗓子哑"等一次性状态不要记录。
- 如果灰泽满的回复是为了附和用户而临时编造或类比的经验（如用户说考研，你跟着说"我也考过研"），不要提取。
- 判断标准：如果这条信息在明天、下周的对话中还能成立，才值得记录。如果不确定，宁可不提取。

返回 JSON（不要多余内容）：
{{
  "new_impression": "对用户的长期印象标签，如'上班族''学生''夜猫子'。一次性的状态不要提取，无则null",
  "new_user_fact": "用户透露的长期身份或爱好，如'做设计的''在考研'。瞬间状态不要记录，无则null",
  "new_self_fact": "你向用户新透露的关于自己的真实事实，无则null",
  "new_moment": "如果本轮对话有特殊意义，写简短摘要，无则null",
  "relationship_change": null 或 "warmed_up" 或 "first_emotional"
}}
"""