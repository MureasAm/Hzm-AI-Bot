import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

MEMORY_FILE = Path(__file__).resolve().parent / "data" / "long_term_memory.json"

# 长期记忆文件锁：保证「读-改-写」原子化，防止多消息并发互相覆盖
_memory_lock = threading.Lock()

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

def merge_memory_card(card: dict, updates: dict) -> dict:
    """纯函数：将 updates 增量合并到记忆卡，返回新的卡片。不做任何 IO。"""
    card = dict(card)  # 浅拷贝，避免污染外部引用

    # 基础统计
    card["total_interactions"] = card.get("total_interactions", 0) + 1
    card["last_seen"] = datetime.now().isoformat()

    # 合并 impressions (标签)
    if updates.get("new_impression"):
        impressions = card.setdefault("impressions", [])
        impressions = [dict(imp) if isinstance(imp, dict) else imp for imp in impressions]
        new_imp = updates["new_impression"]
        found = False
        for i, imp in enumerate(impressions):
            tag = imp["tag"] if isinstance(imp, dict) else imp
            if tag == new_imp:
                if isinstance(imp, dict):
                    imp = dict(imp)
                    imp["confidence"] = min(1.0, imp.get("confidence", 0.8) + 0.1)
                    imp["last_updated"] = datetime.now().isoformat()
                    impressions[i] = imp
                else:
                    impressions[i] = {
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
        card["impressions"] = impressions

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
    # 注意：V1 起不再从 AI 回复提取 self_fact（防止 AI 自嗨污染真实人格）。
    # 只保留显式传入的 self_fact（例如从真人素材蒸馏而来），提取流程在 core.py 停用。
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

    # 用户所在城市（天气感知用；保留最近一次，用户换城市则覆盖）
    if updates.get("new_city"):
        city = str(updates["new_city"]).strip()
        if city:
            card["weather_city"] = city

    # 合并承诺/约定（跨会话记住她答应过用户的事）
    if updates.get("new_promise"):
        promises = card.setdefault("promises", [])
        promises = [dict(p) if isinstance(p, dict) else p for p in promises]
        new_p = updates["new_promise"]
        new_p_obj = {"promise": new_p, "made_on": datetime.now().strftime("%Y-%m-%d")}
        found = False
        for i, p in enumerate(promises):
            text = p["promise"] if isinstance(p, dict) else p
            if text == new_p:
                if isinstance(promises[i], dict):
                    promises[i]["made_on"] = new_p_obj["made_on"]
                found = True
                break
        if not found:
            promises.append(new_p_obj)
        card["promises"] = promises[-5:]  # 保留最近 5 条

    return card


def update_user_memory(user_id: str, updates: dict) -> None:
    """增量合并更新用户记忆（读-改-写全程持锁）"""
    with _memory_lock:
        memory = load_memory()
        card = memory.get(user_id, {})
        card = merge_memory_card(card, updates)
        memory[user_id] = card
        save_memory(memory)

def build_memory_context(card: dict) -> str:
    """将记忆卡转化为提示文本"""
    if not card:
        return ""

    parts = []

    # 用户所在城市（天气感知用）
    city = card.get("weather_city")
    if city:
        parts.append(f"这个绿冻在{city}。聊天气时可以自然提及TA那边的天气。")

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

    # 承诺/约定（跨会话记住）
    promises = card.get("promises", [])
    if promises:
        promise_strs = []
        for p in promises:
            if isinstance(p, dict):
                promise_strs.append(p.get("promise", ""))
            else:
                promise_strs.append(str(p))
        promise_strs = [s for s in promise_strs if s]
        if promise_strs:
            parts.append(f"你答应过TA：{'；'.join(promise_strs)}。TA提到时要记得并回应，不要装作不知道。")

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

**关于承诺（new_promise）**：
- 记录灰泽满对用户明确做出的承诺/约定（如"明天一定直播""这周不鸽""下次补翻唱"）。
- 判断标准：对用户明确承诺的、值得跨会话记住的事才记录；随口客套（"以后再说吧""有机会一起"）不记。
- 无则 null

**关于用户城市（new_city）**：
- 从用户的话中提取用户**所在城市/地区**（如"我在广州""住深圳""人在悉尼"→"广州""深圳""悉尼"）。
- 只记城市名或区划名，不记街道/小区。未提到城市则 null。

返回 JSON（不要多余内容）：
{{
  "new_impression": "对用户的长期印象标签，如'上班族''学生''夜猫子'。一次性的状态不要提取，无则null",
  "new_user_fact": "用户透露的长期身份或爱好，如'做设计的''在考研'。瞬间状态不要记录，无则null",
  "new_self_fact": "你向用户新透露的关于自己的真实事实，无则null",
  "new_promise": "你本轮对用户做出的承诺/约定，无则null",
  "new_moment": "如果本轮对话有特殊意义，写简短摘要，无则null",
  "new_city": "用户所在城市/地区名（如'广州'），未提及则null"
}}
"""