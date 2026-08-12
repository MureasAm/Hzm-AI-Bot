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

_NULL_STRINGS = ("", "null", "none")


def _is_null_str(value) -> bool:
    """判断值是否为 LLM 输出的"无"占位（字符串 'null'/'None'/空串）。

    提取模型常把"没有"写成字符串 "null" 而非 JSON 的 null（无引号），
    字符串是 truthy，若直接存卡会把 '这个绿冻在null' 注进上下文（历史踩坑）。
    """
    return isinstance(value, str) and value.strip().lower() in _NULL_STRINGS


def _not_null_str(value) -> bool:
    """值是否非空非'null'占位（build_memory_context 注入前兜底过滤）。"""
    if value is None:
        return False
    if isinstance(value, str):
        return not _is_null_str(value)
    return True


def merge_memory_card(card: dict, updates: dict) -> dict:
    """纯函数：将 updates 增量合并到记忆卡，返回新的卡片。不做任何 IO。"""
    card = dict(card)  # 浅拷贝，避免污染外部引用

    # 清洗提取结果：值为字符串 'null'/'None' 的字段直接丢弃（模型常把"无"写成字符串而非 JSON null）
    updates = {
        k: v for k, v in updates.items()
        if not _is_null_str(v)
    }

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

    # supersede：用户明确说某已知信息已变，作废对应的旧印象/事实（防止旧标签永远残留）
    if updates.get("supersede"):
        gone = [str(x).strip() for x in updates["supersede"]
                if str(x).strip() and not _is_null_str(x)]
        if gone:
            if card.get("impressions"):
                card["impressions"] = [
                    imp for imp in card["impressions"]
                    if (imp.get("tag") if isinstance(imp, dict) else str(imp)) not in gone
                ]
            if card.get("user_facts"):
                card["user_facts"] = [
                    f for f in card["user_facts"]
                    if (f.get("fact") if isinstance(f, dict) else str(f)) not in gone
                ]

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

    # 用户名字/昵称（怎么称呼TA；用户改名则覆盖）
    if updates.get("new_name"):
        name = str(updates["new_name"]).strip()
        if name:
            card["user_name"] = name

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

    # 用户名字/昵称
    name = card.get("user_name")
    if _not_null_str(name):
        parts.append(f"这个绿冻叫{name}，聊天时用TA的名字称呼TA，别老叫TA'这个绿冻'。")

    # 用户所在城市（天气感知用）
    city = card.get("weather_city")
    if _not_null_str(city):
        parts.append(f"这个绿冻在{city}。聊天气时可以自然提及TA那边的天气。")

    # 印象标签
    impressions = card.get("impressions", [])
    if impressions:
        high_conf = []
        for imp in impressions:
            tag = imp["tag"] if isinstance(imp, dict) else imp
            if not _not_null_str(tag):
                continue  # 兜底：过滤历史遗留的 'null' 标签
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
        fact_strs = [s for s in fact_strs if _not_null_str(s)]
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
        fact_strs = [s for s in fact_strs if _not_null_str(s)]
        if fact_strs:
            parts.append(f"你已跟TA说过：{'；'.join(fact_strs)}。不要再重复自曝这些事。")

    # 最近的亮点时刻
    moments = card.get("significant_moments", [])
    if moments:
        recent = moments[-1]["summary"] if isinstance(moments[-1], dict) else str(moments[-1])
        if _not_null_str(recent):
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
        promise_strs = [s for s in promise_strs if _not_null_str(s)]
        if promise_strs:
            parts.append(f"你答应过TA：{'；'.join(promise_strs)}。TA提到时要记得并回应，不要装作不知道。")

    return "\n".join(parts)

def _format_profile_summary(card: dict) -> str:
    """把记忆卡格式化成给提取模型看的可读画像摘要（替代原生 JSON dump）。

    模型读可读画像才能可靠 dedup（判断"已知道没"）和冲突检测。
    """
    if not card:
        return "（无，首次对话）"
    parts = []
    imps = card.get("impressions", [])
    if imps:
        tags = [i["tag"] if isinstance(i, dict) else str(i) for i in imps]
        parts.append("印象：" + "、".join(str(t) for t in tags))
    facts = card.get("user_facts", [])
    if facts:
        fstrs = [f.get("fact") if isinstance(f, dict) else str(f) for f in facts]
        parts.append("事实：" + "；".join(str(f) for f in fstrs))
    promises = card.get("promises", [])
    if promises:
        pstrs = [p.get("promise") if isinstance(p, dict) else str(p) for p in promises]
        parts.append("承诺：" + "；".join(str(p) for p in pstrs))
    city = card.get("weather_city", "")
    if city:
        parts.append(f"城市：{city}")
    name = card.get("user_name", "")
    if name:
        parts.append(f"名字：{name}")
    moments = card.get("significant_moments", [])
    if moments:
        mstrs = [m["summary"] if isinstance(m, dict) else str(m) for m in moments[-2:]]
        parts.append("重要时刻：" + "；".join(str(m) for m in mstrs))
    if not parts:
        return "（无稳定画像，首次对话）"
    return "已知道这个绿冻：" + "；".join(parts)


MEMORY_EXTRACT_PROMPT = """
你是一个记忆提取助手。分析以下对话，只提取**值得长期记住的新信息**。忽略日常寒暄。

【当前已知画像】
{current_summary}

【本轮对话】
用户：{user_msg}
灰泽满：{reply}

【提取要求】
- **JSON 格式铁律**：某个字段"没有"时，输出真正的 JSON `null`（值直接写 null，**不带引号**）。禁止输出字符串 "null" 或 "None"——字符串 "null" 会被当有效内容存卡（曾导致上下文出现"这个绿冻在null"）。
- **不要重复提取**：如果某条信息已在【当前已知画像】里（已经知道了），不要重复提取，对应字段返回 null。
- 只提取本轮**新增**的信息。如果本轮没有值得记录的新内容，全部字段返回 null。
- **只记用户明确陈述的长期事实**：用户没说出口的、靠推断的、一次性的，都别记。"刚下班"不能推断成"上班族"——除非用户明确说"我是上班族"。
- **分层**：不可变身份（名字/生日/过敏）一次就可记；可变偏好/状态（职业/爱好/居住）要**明确陈述或重复提到**才记；一次性心情/状态（"今天好累""刚下班"）不记。
- **保留限定词**：提取时别丢细节——"在XX公司做设计"不是"做设计的"；"想养猫但还没养"不是"养了只猫"。
- **绝对不要记录为了附和用户而临时编造的状态**：用户说"我是上班族"，你跟着说"我也有作业压力"，这种附和性内容不要记录。
- **冲突检测**：新信息与已知画像矛盾时（如已知城市广州、用户又说在深圳），以新信息为准更新。
- **supersede（作废旧信息）**：如果用户明确说某个已知信息已经变了（如"我现在不上班了""我不在广州了"），在 supersede 字段列出要作废的旧印象/事实。
- 如果不确定，宁可不提取。

**印象标签 vs 用户事实的区别**：
- new_impression = 抽象的性格/行为/身份标签（简短）：如"夜猫子""上班族""喜欢催播""嘴硬心软"
- new_user_fact = 具体的长期信息（客观可描述）：如"在准备考研""养了只猫""做设计的"
- 只有用户明确说的才算数；一次"刚下班"不抽象成"上班族"。

**提取示例**：
该提取：
- "我平时挺喜欢熬夜的" → new_impression="夜猫子"
- "我在广州上班" → new_user_fact="在广州上班"，new_city="广州"
- "我打算周六直播" → new_promise="周六直播"
不该提取（返回 null）：
- "今天好累啊"（一次性状态）
- "哈哈哈""晚安""你吃饭了吗"（寒暄，无新信息）

**关于自我披露（new_self_fact）的重要限制**：
- 只记录灰泽满透露的**长期个人特征或真实经历**（如"拖延症晚期""在国外留学""不会做饭"）。
- **绝对不要记录瞬间状态**：如"正在吃泡面""刚睡醒""今天嗓子哑"等一次性状态不要记录。
- 如果灰泽满的回复是为了附和用户而临时编造或类比的经验（如用户说考研，你跟着说"我也考过研"），不要提取。

**关于承诺（new_promise）**：
- 记录灰泽满对用户明确做出的承诺/约定（如"明天一定直播""这周不鸽""下次补翻唱"）。
- 判断标准：对用户明确承诺的、值得跨会话记住的事才记录；随口客套（"以后再说吧""有机会一起"）不记。

**关于用户城市（new_city）**：
- 从用户的话中提取用户**常住**城市/地区（如"我在广州""住深圳""人在悉尼"→"广州""深圳""悉尼"）。
- **出差/旅游/暂住不算常住**："我去上海出差""周末去北京玩"不覆盖常住城市，new_city 返回 null。
- 只记城市名或区划名，不记街道/小区。未提到城市则 null。

**关于用户名字（new_name）**：
- 从用户的话中提取用户希望灰泽满怎么称呼 TA（如"我叫小明""你可以叫我阿伟"→"小明""阿伟"）。
- **只在用户明确告知名字/昵称时提取**；QQ 昵称不算（那不是聊天里说的）；随口称呼（"哥们""姐妹"）不算。
- 未告知则 null。

**提取示例补充**：
- "我叫小明，你以后叫我小明就行" → new_name="小明"

返回 JSON（不要多余内容）：
{{
  "new_name": "用户希望被称呼的名字/昵称，未告知则null",
  "new_impression": "对用户的长期印象标签，无则null",
  "new_user_fact": "用户透露的长期身份或爱好，无则null",
  "new_self_fact": "你向用户新透露的关于自己的真实事实，无则null",
  "new_promise": "你本轮对用户做出的承诺/约定，无则null",
  "new_moment": "如果本轮对话有特殊意义，写简短摘要，无则null",
  "new_city": "用户常住城市/地区名，出差旅游不算，未提及则null",
  "supersede": "要作废的旧印象/事实列表（用户明确说某已知信息已变时填），没有则[]"
}}
"""