"""Memory V2 shadow store.

This module deliberately does not affect live replies.  It validates typed LLM
extractions, consolidates evidence, persists a per-user shadow profile, and can
render the bounded context that *would* be injected after the shadow rollout is
approved.

Runtime data stays under ``user_memory/`` (gitignored).  Model output is treated
as untrusted input and must pass the allowlists below before it is stored.
"""
from __future__ import annotations

import hashlib
import json

from . import injection as inj
from . import daily_memory
import os
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .constants import PROJECT_ROOT, THINKING_DISABLED


DEFAULT_STATE_FILE = PROJECT_ROOT / "user_memory" / "memory_v2_shadow.json"
DEFAULT_AUDIT_FILE = PROJECT_ROOT / "user_memory" / "memory_v2_audit.jsonl"
# 每日记忆（第 20 块）跟长期记忆**同一次调用提取，但分开存**：它的寿命是"一天"，
# 塞进长期那份（268KB、无限增长）或会话那份（3 小时就清）都不合适。
# 存储实例在 `daily_memory` 里（读路径每轮也要用它，必须共用同一把锁）。

SCHEMA_VERSION = 2
MAX_SOURCE_CHARS = 240
MAX_VALUE_CHARS = 160
MAX_USER_ID_CHARS = 128
MAX_SESSION_ID_CHARS = 96

# 注入分段预算（字符）。偏好最大——它是针对这个人的适配，四层里优先级最高。
MAX_PREFERENCE_ITEMS = 6
PREFERENCE_BUDGET_CHARS = 360
COMMITMENT_BUDGET_CHARS = 220
# 时效性近况（fact）的期限档位 → 天数；"none" 不过期（长期事实/明确无期限）。
_VALID_FOR_DAYS = {"3d": 3, "14d": 14, "90d": 90}
# 承诺的默认寿命（2026-10-18 用户定：加 TTL）。
# 为什么必须过期：承诺没有"完成"信号，旧代码里 active 永久有效——实测 27 条里
# 10 条已超过 7 天、16 条是"明天/今晚/一会儿"这种一次性措辞，会在几周后被当成
# "还没兑现的约定"注入。分三档：一次性 / 长期态度 / 普通承诺。
COMMITMENT_TTL_DAYS = 14
COMMITMENT_ONE_TIME_TTL_HOURS = 48
COMMITMENT_STANDING_TTL_DAYS = 90
# 称呼提示的冷却：同一用户 10 分钟内最多提示一次，避免用户连叫外号时她每句都用。
NICKNAME_HINT_COOLDOWN_SECONDS = 600
_ONE_TIME_COMMITMENT_RE = re.compile(r"今天|今晚|一会儿|待会|等下|明天")
_STANDING_COMMITMENT_RE = re.compile(r"以后|天天|每天|每次|一直|长期")
# 这些 key 的事实**不管聊没聊到都带上**——名字/地点/时区/身份是"不矛盾的底线"。
# 换掉了原来那个对不上模型产出的旧名单（preferred_name/name/city）：
# 模型实际会产 location / timezone / identity 这些，旧名单等于没生效。
_ALWAYS_ON_FACT_KEYS = {
    "preferred_name", "name", "nickname", "location", "city",
    "timezone", "identity", "occupation", "background", "her_nickname",
}
# 上面那批里**天然只有一个值**的（名字/地点/时区/职业）——同一 key 的新值必须顶掉旧的。
# ⚠️ 不能直接用 `_ALWAYS_ON_FACT_KEYS` 当顶替名单：`identity` / `background` 会有多条
#    （"是学生" + "在澳洲留学"），顶替会真丢信息（踩过）。
# ⚠️ 也不能只靠 `temporal_scope == "current"` 判顶替：模型常把 `preferred_name` 标成
#    **timeless**，而 timeless 走"累积" → 同一个名字存 3 份、每轮注入 3 次（实测
#    1750961968 正是如此，还叠上"她每句都叫名字"的正反馈）。
_SINGLE_VALUED_FACT_KEYS = {
    "preferred_name", "name", "nickname", "location", "city", "timezone", "occupation",
    "her_nickname",
}
# 名字类事实的代码闸门（2026-10-19）：模型会把"云子"（网络用语，嫌她不懂/很云）、
# "我是绿冻"（粉丝身份）、"用户希望被称呼为X"（整句）都当成 preferred_name。
# 名字必须是**用户明确要求她这样称呼**，且 value 本身像名字。
_NAME_KEYS = {"preferred_name", "name", "nickname"}
_NAME_REQUEST_RE = re.compile(r"叫我|称呼我|喊我|管我叫|名字叫|我叫|以后叫|可以叫")
_NAME_VALUE_OK_RE = re.compile(r"^[一-鿿A-Za-z0-9_\-·]{1,12}$")
_NAME_VALUE_BAD_WORDS = ("用户", "希望", "称呼", "自称", "或者", "不叫")

_KINDS = {"fact", "interaction_preference", "commitment"}
_ACTIONS = {"upsert", "retract"}
_TEMPORAL_SCOPES = {"past", "current", "future", "timeless"}
# 每个维度都是"枚举值 → 行为句"，不存印象标签（黄金律第 2 条）。
# 8 个维度：低落时怎么接 / 玩笑尺度 / 拒绝方式 / 主动关心阈值（原有 4 个）
#         + 称呼偏好 / 回复长度 / 话题雷区 / 她怎么自称（2026-10-10 扩到 8 个）。
_PREFERENCE_VALUES = {
    "support_style": {"comfort", "problem_solving", "balanced"},
    "banter_tolerance": {"low", "light", "high"},
    "directness": {"gentle", "direct", "balanced"},
    "care_initiative": {"proactive", "reserved", "balanced"},
    "address_style": {"nickname", "plain", "casual"},
    "reply_length": {"short", "medium", "long"},
    "topic_preference": {"engage", "avoid"},
}
# 自由文本的相处方式键：值不是枚举，而是一句「场景：内容」（2026-10-23 加 ritual——
# 固定仪式没法塞进上面那张枚举表；用户 905860780 每晚的“仪式感晚安”就是这么漏掉的）。
# key 用 `ritual` 或 `ritual_<场景>`（如 ritual_goodnight）：每个场景一个槽，
# 同一场景的新值顶掉旧值，不同场景互不影响。
def _is_ritual_key(key: str) -> bool:
    return key == "ritual" or key.startswith("ritual_")
_SENSITIVE_KEYS = {
    "phone", "phone_number", "email", "exact_address", "home_address",
    "id_number", "identity_number", "bank_account", "finance_account",
    "medical_diagnosis", "health_diagnosis", "password", "credential",
}
_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(?:password|passwd|pwd|密码)\s*[:：=]?\s*\S+"
)
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")

# 关于机器人自己（不是用户）的 fact 一律拒收。这类永远不改变她的回复，却是实测最顽固的
# 误记（用户贴更新日志 → fact/bot_version=V8.0；引用她的动态 → fact/identity=灰泽满自己是…）。
# 提示词已写进反例清单，这里再加一道确定性闸门（模型输出本就当不可信输入校验）。
_BOT_SELF_KEY_PREFIXES = ("bot", "hzm")
_BOT_SELF_KEYS = {"version", "changelog", "release_notes", "bot_info"}
_BOT_SELF_VALUE_SUBJECTS = ("灰泽满", "hzm", "机器人", "assistant")

# 「什么都能装」的栏目名。提取器提示词里明令禁止（"别用 `current_situation`/`past_experience`/
# `info` 这类装什么都行的通用键——通用键会让不同的事互相顶掉"），但实测它照用不误：
# 某用户攒了 **5 条 `fact/moment`**（分手／表白／求婚／亲一下各一条），四件不同的事挤在一个名字下。
# 坏的后果是**两头都坏**——既认不了亲（"同一件事"合不到一起，只会越加越多），也查不出来
# （要问"他表白那次"，翻 `moment` 出来五条，分不清哪条是哪条），还**占位置**
# （「相关用户事实」一次只带 4 条，五条 `moment` 能把位置占满）。
#
# ⚠️ **现在只记不拦**（2026-10-16）：命中就写进审计的 `generic_key` 标记，**条目照收**。
#    理由：退掉 ≠ 清理干净，退掉 = **这件事压根没被记住**——那五条 `moment` 里有一条是
#    "用户向灰泽满求婚"，退了她以后就完全不知道有这回事。我们要的是**逼它改个具体名字**，
#    不是**丢事**。所以先跑几天看命中率：如果它撞了黑名单会自己换个具体名字再写一遍，那就该真拦；
#    如果它宁可空手也不换，拦了就是纯丢东西、白搭。
#    要改成真拦：在 `_normalize_item` 里把末尾那行 `return normalized, "accepted"` 之前加一句
#    `if kind == "fact" and key in _GENERIC_FACT_KEYS: return None, "generic_key_blocked"`。
_GENERIC_FACT_KEYS = {
    # 空壳名：什么都能装，既认不了亲也查不出来
    "moment", "info", "information", "situation", "event", "events", "experience",
    "thing", "things", "stuff", "detail", "details", "note", "notes",
    "misc", "other", "general", "current_situation", "past_experience",
    # 光秃秃的类别名（本身就不止一件事的类别）——等于空壳。**`location`/`timezone`/
    # `preferred_name`/`occupation` 不在里面**：那几个天然只有一件事，一个词就够。
    # （2026-10-16 探针上出现的 `health=感冒了` —— 光写个 `health`，回头"腰疼"也写 `health`，
    #   两条就顶掉了，又是同一个坑。）
    "health", "family", "work", "relationship", "interest", "habit", "past_event",
}

# 相处方式的「键 → 行为句」表**在 persona/injection/03_profile/block.json 里**
# （2026-10-16 从这儿搬走的：它是注入文案，不该住在代码里）。取用见 injection.behavior_sentence。


# ⚠️ **变量一律放最后**（2026-10-10 修）：DeepSeek 缓存按**前缀**匹配——把 {observed_at} /
#    {current_summary} / 对话放到前面，会让**它后面的整段静态指令永远 miss**。原来第一个变量
#    在第 130 字（共 2630 字）→ 只有 5% 可缓存；跑几千次重建就攒出 400 万+ 未命中 token。
#    改成"指令在前、数据在后"后有 ~95% 可缓存（v1 的提取提示词一直是这个排法）。
MEMORY_V2_EXTRACT_PROMPT = """你是灰泽满私聊机器人的记忆候选提取器。你的输出不会直接成为事实，
还会经过代码校验和影子审计。只根据下面这一次真实对话提取，不要脑补。

【唯一的准入标准】
把这条信息抽出来，**会不会改变灰泽满下一次回复**？不会的，一律不记。

只允许三种 kind（**情绪不在长期记忆里**——它住在会话记忆，跟话题一起 3 小时消散）：
1. fact：关于**用户本人**的、会影响回应的稳定事实——称呼/昵称、地域时区、身份（学生/社畜）、
   正在经历的大事（搬家/考试/生病）、明确喜好（"月饼一般，芒果味的可以"）。
   **必须是关于"用户本人"的**；讲灰泽满/机器人自己的内容一律不算 fact。
   · key 的写法：**先写类别，再写具体是哪件事**——`health_recent_cold`、`family_daughter_accident`、
     `relationship_confession`、`work_exam_failed`。本身就只有一件事的类别（名字/住地/时区/职业）
     用一个词就够：`preferred_name`、`location`、`timezone`、`occupation`。
     **类别只能从这张表里挑**：
        住地 `location` ／ 身份 `identity` ／ 健康 `health` ／ 家人 `family` ／ 工作学习 `work`
        关系 `relationship` ／ 喜好 `interest` ／ 习惯 `habit` ／ 过往事件 `past_event`
   · **习惯类（`habit_*`）要把「频率 + 原因」一起写进去**：不是"早上要跑步"，
     而是"每天早上要去跑早操（学校要求）"。（2026-10-17 实测：只记了前者的半截，
     她连着几天都在问"你起这么早干嘛去"——用户已经说过好几遍是学校要求的。）
   · ⚠️ **同一活动有多个频率时必须分开记**：不要把它们合成一条。
     实测踩坑（2026-10-21，用户 2151983431）：原话是"早上跑的是早操，这个是一周三次的"，
     模型写成"早上跑早操是一周三次的"——把**每天早上的跑操**和**每周三次的 3 公里跑**
     合并了，之后早起对话全接不上。正确写法是两条：
     `habit_morning_run=每天早上起床后都要跑操（学校要求）`、
     `habit_run_3km=每周还有三次 3 公里跑`。
   · **习惯类还要填一个 `habit` 对象**（只填在 habit_* 上，其它 fact 不要填）：
     `type`：`routine`＝作息/日程（起床、上课、跑步）；`interaction`＝他和你的互动习惯
       （表情包撒娇、情话、念诗、睡前摸摸）。**不要把所有习惯都叫 routine。**
     `cadence`：`daily`（每天）/`weekly`（每周）/`monthly`（每月）/`regular`（经常、平时）/
       `once`（今晚、明天、这次——**一次性计划不是固定习惯**）/`unknown`。
     `period`：`morning`/`noon`/`afternoon`/`evening`/`night`/`any`/`unknown`。
       只有**作息类且每天**的习惯才会按这个时段触发；不确定就写 `any`。
     `cues`：他实际会说的 2–8 个触发短语（如"跑操""早八""摸摸""表情包"）。
       **不要把"每天/早上/晚上"这类时间词写进 cues**——时段由 period 管。
     例子：
       `habit_morning_run`：`habit={{"type":"routine","cadence":"daily","period":"morning","cues":["跑操","跑步","早操"]}}`
       `habit_expression_sticker`：`habit={{"type":"interaction","cadence":"daily","period":"any","cues":["表情包","撒娇","语音"]}}`
       `habit_send_poetry`：`habit={{"type":"interaction","cadence":"regular","period":"any","cues":["念诗","诗歌","歌词"]}}`
     ⚠️ "今晚要熬夜看比赛"只是一次性计划：要么不记，要么 `cadence":"once"`；
       不要让它每天早上/晚上都被当成固定习惯翻出来。
   · 判据：同一个人、同一个类别下面，两件不同的事必须能用名字分开。
     后半截写不出具体的东西，就说明这条本来就不该记。
   · ⛔ **不许用 `moment` / `info` / `situation` / `experience` / `thing` 这类空壳名**：
     它们既认不了亲（同一件事合不到一起，只会越加越多），也查不出来
     （要问"他表白那次"，翻出来五条分不清哪条是哪条），还白占位置。
   · `preferred_name` / `nickname` 的 value **只写名字本身**（如"草草子哥"），
     **不要写成"用户希望被称呼为X"这种整句**——那个句子会被当成名字注入，又长又怪。
     ⚠️ 只有用户**明确要求灰泽满这样称呼他**（"叫我X""以后叫我X"）才记。
     "我是绿冻/暖冻"是粉丝身份，不是名字；"云子""云玩家"是网络用语（说他不懂/很云），不是名字；
     玩梗称呼（"爸爸"）和整句（"用户希望被称呼为X"）都不要记。
   · **用户给灰泽满起的称呼**：key=`her_nickname`，value 只写称呼本身（如"灰宝"）。
     只在用户明确说"叫你X""就喊你X""可以叫你X吗"时记；这是对她的昵称，不是她的身份设定。
     ⚠️ 只记这个称呼事实，**不要**借此改变她的自称——她默认仍自称"灰泽满"；
     用户后来说"还是叫灰泽满吧/别叫X了"，对 `her_nickname` 输出 action=retract。
   · temporal_scope：**经历**用 past（可叠加，不倒旧的）；**地域/时区/身份**这类长期成立用 timeless；
     **当前状态**（在准备考试 / 在搬家）用 current（**同一 key 的新值会顶掉旧的**）。
   · valid_for：**只有"时效性近况"**（在准备考试 / 这周军训 / 最近感冒）才填其大概期限，
     从用户话里的时间词读（"这几天"→3d、"最近/这一阵"→14d、"这学期"→90d）；
     没有明确期限、或不是近况，一律 "none"。
   · explicit：用户在话里**明说**了才填 true；只是从语气/表现里猜出来的填 false。
     **false 的 fact 会被代码拦掉**——所以"猜的"等于这条别写。
   一次性状态不提取。
2. interaction_preference：用户想被怎样对待（针对这个人的相处方式，不是性格标签）。key/value 只能是：
   support_style=comfort|problem_solving|balanced    低落时怎么接
   banter_tolerance=low|light|high                    玩笑尺度
   directness=gentle|direct|balanced                  拒绝/纠正的方式
   care_initiative=proactive|reserved|balanced        主动关心阈值
   address_style=nickname|plain|casual                **他希望灰泽满怎么称呼他自己**
   reply_length=short|medium|long                     回复长度偏好
   topic_preference=engage|avoid                      话题偏好与雷区
   ritual_<场景>=自由文本                              **固定场景的长期仪式**（见下）
   · **ritual（固定仪式）**：用户在某个固定场景规定灰泽满以后都这样说/做——晚安、早安、
     出门、上线。key 用 `ritual_<场景>`（如 `ritual_goodnight`，每个场景一个槽，
     不同场景互不顶替），value 写成「场景：要说的内容」，如
     `ritual_goodnight` = 「晚安/睡前/要睡：晚安+喜欢你+爱绿冻+mua」。
     场景词写用户实际会用到的说法（用 / 分隔），不然他说“要睡了”时不会触发。
     判据：用户是在**规定以后都这样**（"要有仪式感的晚安""睡前必做""以后都…"），
     不是这一晚的玩笑。哪怕单轮看起来像寒暄，只要是在立这个规矩就记。
     **更新时**：清单里已有同一场景的 ritual，这轮只提到一半内容 → 沿用旧值、
     不要缩水（新 value 必须包含旧值里的具体内容）；只是催一句、没提内容 → 不更新。
     实测漏记（2026-10-15，用户 905860780）：连着十几天每晚要"仪式感的晚安"，
     单轮提取把它当寒暄跳过，她从没主动想起过这件事。
   用户明确说喜欢/不喜欢时 explicit=true；仅从互动表现推断时 explicit=false。
   ⚠️ **用户嫌她"阴阳怪气／挤兑人／说话冲／不好好讲话"，要她改语气 → `banter_tolerance=low`。**
      （2026-10-17 实测漏了：用户 3498594329 明说"以后能保证好好讲话、不要冲人、不要阴阳怪气"，
       记忆里却只有 `address_style` 一条。这类是用户对相处方式的**明确要求**，必须记。）
3. commitment：灰泽满本轮明确答应用户、需要以后兑现的事。承诺只从灰泽满回复中提取，
   用户自己的计划不是灰泽满的承诺；actor 必须是 assistant，explicit 必须为 true。
4. daily：用户**当天很单纯的日常**——吃了什么、几点上课/下课、去哪、什么时候睡的这类
   **纯事件**。只记"有事发生"的句子（"中午吃了炸鸡""下午的课取消了"）。
   · **状态、情绪、喜好、关系一律不记**（"他最近很累""不爱吃辣""和室友关系一般"）——
     那些归 fact / interaction_preference。
   · 讲灰泽满/她自己/机器人的不记（跟 fact 同一条铁律）。
   · 时点**不用你写**，系统会带上这句对话发生的时间。
   · 写成**一句话、30 字以内**；这轮没有就留空数组。

【反例清单——抽到这些一律不记】
- 灰泽满自己/机器人自身的事：版本号、功能、更新日志、"你是不是 AI"、她自己的身份设定。
  **不管用户是告知、询问还是吐槽，只要讲的是灰泽满/机器人自己，都不是"关于用户的事实"**。
  ⚠️ **例外**：用户给**她**起的称呼（`her_nickname`，如"我就叫你灰宝"）可以记——
  那是用户对她的昵称，不是她的身份设定。
- **网上抄来的段子 / 定型文 / 角色扮演**：长段工整得像"一篇自我介绍"的（"我重新介绍一下自己"
  "丫头，我…"这类套路开局）、网上流传的梗文案、演一个人设给你看——**那是在玩，不是他的真实信息**，
  一条都别记。判据：**这段字读起来像"一篇文章/一个模板"，而不是"他随口在说自己的事"**。
  和**已知事实明显打架**的也先当段子（他记着是走读生，这轮又说自己四十岁跑大货）。
  踩坑（2026-10-12）：某用户发了条"丫头，我重新介绍一下自己。我今年四十，河南周口人，现跑豫粤线
  大货，A2证…"，被当真存成 occupation/location/age/income/savings 五条，之后她张口就是
  "明天还得握方向盘的人"。
- 单次闲聊、寒暄、玩梗（"在吗""哈哈哈""晚安"）。
  ⚠️ **例外（ritual）**：单次"晚安"不记；但"每天晚安都要有仪式感／睡前必做"是
  **固定仪式**，归 `interaction_preference/ritual`，不要因为出现"晚安"就跳过。
- **称呼需求的方向搞反**：`address_style` 填的是「**用户希望灰泽满怎么称呼他自己**」。
  他说"我想叫你宝宝""我能不能叫你满姐"——那是**他想叫她**，方向相反，**不要填 address_style**。
  （实测踩坑 2026-10-16：同一个句式"我想叫你宝宝"，两条记录分别被填成 `nickname` 和 `plain`，
  两个相反的值——至少有一个是错的。另有一条从"小满好好休息哦"里读出了"他想被称呼"，
  原文里根本没有这回事。）
- **单个称呼不是在报名字**：用户单独发一个称呼（"云子""宝宝""满神""小满"），
  通常是在叫灰泽满或粉丝群体，不是他的自称——**不要据此填 preferred_name**。
  只有他明确说"叫我X""我是X"，才记 preferred_name。
  （实测踩坑 2026-10-19：用户只发了一句"云子"，被存成 preferred_name=云子，
  之后她一直拿"云子"称呼这个用户。）
- 只出现一次、不会影响下次回应的状态（"他刚打了个哈欠""他刚发了个表情"）。
- 已能从别处拿到的：时间/星期/农历、天气、直播场次状态——那是感知模块的活。
- 用户自己的计划/打算——那是用户的事，不是她的承诺。
- 敏感信息：电话、邮箱、精确地址、证件、账户、密码、医疗诊断。

隐私铁律：不要提取电话、邮箱、精确地址、证件、账户、密码等**身份/联系方式类**敏感信息。
**精神/心理类诊断也不要记**（抑郁、躁郁、焦虑症的确诊与服药情况）——那是最该保护的东西，
记下来会在她往后的回复里反复带出来。**而且网上说这个的多半是玩梗、自嘲、借题发挥**
（"我确诊了""我玉玉了"），本来就不该当真。
（⚠️ 这条实测被违反过 3 次，2026-10-12：`depression_anxiety_diagnosis` /
`anxiety_condition` / `bipolar_disorder_medication` 都被存成了 fact。
普通小病近况是另一回事——"最近感冒了"可以记。）
时间铁律：保留“以前/现在/打算/已经结束”等状态，不把过去事实写成当前事实。
temporal_scope 必须是 past/current/future/timeless：用户过去经历用 past，当前仍成立用 current，
无时间属性的稳定偏好用 timeless，承诺或尚未发生的计划用 future。
纠正铁律：用户明确否定或纠正旧信息时 action=retract；其他新增/确认使用 action=upsert。
不确定就不提取。不要为了显得有记忆而凑内容。

只输出严格 JSON：
{{
  "items": [
    {{
      "kind": "fact|interaction_preference|commitment",
      "key": "小写英文键；fact 要具体到事，别用通用键",
      "value": "简洁、保留限定词的中文内容",
      "action": "upsert|retract",
      "temporal_scope": "past|current|future|timeless",
      "valid_for": "none|3d|14d|90d（仅时效性近况填，其余 none）",
      "explicit": true,
      "confidence": 0.0,
      "actor": "assistant",
      "habit": {{"type": "routine|interaction", "cadence": "daily|weekly|monthly|regular|once|unknown", "period": "morning|noon|afternoon|evening|night|any|unknown", "cues": ["2-8个短触发词"]}}
    }}
  ],
  "daily": [
    {{"text": "一句话、30字以内的纯日常事件（吃了什么/几点上课/去哪）"}}
  ]
}}
没有候选时返回 {{"items":[], "daily":[]}}。不要 markdown，不要额外文字。

【本轮对话】
用户：{user_text}
灰泽满：{assistant_text}

【观测时间与会话】
观测时间：{observed_at}
证据会话：{session_id}

【这个人已经记过的（kind/key=value）】
{current_summary}

⚠️ **写之前先把上面这张清单对一遍——这是这份提示词里最要紧的一步。**
  · 这轮要说的事，**清单里已经有了（哪怕措辞不一样）→ 那是同一件事，用清单里那个 key，
    不要另起一个新名字**。
    例：清单里已经有 `recent_cold=用户最近感冒了`，这轮他说"感冒还挺严重的"——
    **这是同一件事**：写 `key="recent_cold"`，**不要**新起 `recent_illness`。
    （踩坑 2026-10-16：就是没做这一步，同一个人的感冒被记成了 `recent_cold` / `recent_illness` /
    `recent_insomnia` 三条，一次注入全端出来。）
  · 这轮的事**是对清单里某一条的更新**（状态变了、程度变了、续上了）→ **也用那一条的 key**，
    让新值去顶掉旧值。
  · ⚠️ **但"同一类事情又发生了一次"是另一回事——必须起新 key，不能复用。**
    第二次表白、又搬一次家、再考一场试，都是**新的一件事**。复用旧 key 会把上一条**顶掉**，
    那就**丢了一条历史**。
    例：清单里有 `relationship_confession=用户首次表白`，这轮他说"我们交往吧，我是认真的"——
    起 `relationship_dating_request`；**不要**写回 `relationship_confession`。
    （踩坑 2026-10-16：探针上真的复用了，首次表白那条当场被顶没。）
    判据：**复用后上一条会被顶掉，而它并不该消失 → 那就是新的一件事。**
  · 清单里的 key **哪怕不符合上面的类别写法，也照它原样用**——复用比整齐重要。
  · 只有**确实是另一件事**时，才起一个新 key。
"""


def shadow_enabled() -> bool:
    """V2 提取**默认开启**（2026-10-10 转正：真实聊天已用它重建过一整库）。

    要临时关掉（回退排查）在 `.env.prod` 设 `MEMORY_V2_SHADOW=0`，不用改代码。
    """
    return os.environ.get("MEMORY_V2_SHADOW", "1") != "0"


def injection_enabled() -> bool:
    """V2 注入**默认开启**（同上转正）。临时关掉设 `MEMORY_V2_INJECT=0`。

    注入的都是长期适配（相处方式靠前 / 未完成约定中段）；情绪走会话记忆，不在这里。
    """
    return os.environ.get("MEMORY_V2_INJECT", "1") != "0"


def parse_extraction_payload(content: str) -> dict:
    """Parse model output without repairing or guessing malformed data.

    `daily` 是 2026-10-17 加的（每日记忆，跟 items 走**同一次调用**）：这里只做
    "是字符串、非空、一轮最多 3 条"的收口，真正的入库闸门在 `daily_memory` 里。
    """
    empty = {"items": [], "daily": []}
    raw = (content or "").strip()
    if raw.startswith("```json"):
        raw = raw[len("```json"):]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    if not raw or raw == "null":
        return empty
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return empty
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        return empty
    texts: list[str] = []
    daily = payload.get("daily")
    if isinstance(daily, list):
        for d in daily[:3]:
            text = str(d.get("text") or "").strip() if isinstance(d, dict) else ""
            if text:
                texts.append(text)
    return {"items": payload["items"][:20], "daily": texts}


def build_extraction_prompt(
    *,
    user_text: str,
    assistant_text: str,
    current_summary: str,
    session_id: str,
    observed_at: datetime,
) -> str:
    """Render the strict write-path prompt with bounded untrusted text."""
    return MEMORY_V2_EXTRACT_PROMPT.format(
        observed_at=observed_at.isoformat(),
        session_id=_bounded_text(session_id, MAX_SESSION_ID_CHARS),
        current_summary=_bounded_text(current_summary, 1200) or "（无）",
        user_text=_bounded_text(user_text, 1000),
        assistant_text=_bounded_text(assistant_text, 1000),
    )


def _response_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(getattr(part, "text", ""))
            for part in content
        )
    return ""


def _extraction_summary(store: "ShadowMemoryStore", user_id: str) -> str:
    memories = store.snapshot(user_id).get("memories", [])
    visible = [
        f"{m.get('kind')}/{m.get('key')}={m.get('value')}({m.get('status')})"
        for m in memories
        if m.get("status") not in {"superseded", "expired"}
    ]
    return "；".join(visible)[:1200] or "（无）"


async def extract_and_ingest(
    *,
    client: Any,
    model: str,
    store: "ShadowMemoryStore",
    user_id: str,
    user_text: str,
    assistant_text: str,
    session_id: str,
    observed_at: datetime | None = None,
) -> list[dict]:
    """Run one strict shadow extraction; failures never escape to chat flow."""
    observed_at = observed_at or datetime.now()
    prompt = build_extraction_prompt(
        user_text=user_text,
        assistant_text=assistant_text,
        current_summary=_extraction_summary(store, user_id),
        session_id=session_id,
        observed_at=observed_at,
    )
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
            **THINKING_DISABLED,
        )
    except Exception:
        return []
    # 提取是**第二大 token 消费者**（全量重建约 4000 次调用），也不该是隐形的。
    from .config import log_cache_usage

    log_cache_usage(response, "记忆提取")
    payload = parse_extraction_payload(_response_text(response))
    # 每日记忆搭同一次调用（2026-10-17）：判据同源、输入同一份，省一次 LLM。
    if daily_memory.enabled() and payload.get("daily"):
        daily_memory.add_events(user_id, payload["daily"], now=observed_at)
    return store.ingest(
        user_id,
        payload["items"],
        source={
            "session_id": session_id,
            "observed_at": observed_at.isoformat(),
            "user_text": user_text,
            "assistant_text": assistant_text,
        },
        now=observed_at,
    )


def _iso(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _contains_sensitive_value(value: str) -> bool:
    return (
        any(pattern.search(value) for pattern in _SENSITIVE_VALUE_PATTERNS)
        or bool(_SECRET_ASSIGNMENT_RE.search(value))
    )


def _redact_sensitive_excerpt(value: str) -> str:
    redacted = value
    for pattern in _SENSITIVE_VALUE_PATTERNS:
        redacted = pattern.sub("[redacted]", redacted)
    return _SECRET_ASSIGNMENT_RE.sub("[redacted]", redacted)


def _is_bot_self_fact(key: str, value: str) -> bool:
    """这条 fact 讲的是机器人/她自己，而不是用户本人？"""
    lowered = key.lower()
    if lowered in _BOT_SELF_KEYS or lowered.startswith(_BOT_SELF_KEY_PREFIXES):
        return True
    return value.strip().lower().startswith(_BOT_SELF_VALUE_SUBJECTS)


def _fact_expires_at(item: dict, now: datetime) -> str | None:
    """时效性近况（fact）的过期时刻；长期事实 / 偏好 / 承诺不过期 → None。"""
    if item.get("kind") != "fact":
        return None
    days = _VALID_FOR_DAYS.get(str(item.get("valid_for") or ""))
    if not days:
        return None
    return (now + timedelta(days=days)).isoformat()


def _commitment_ttl_hours(value: str) -> int:
    """承诺的 TTL：一次性措辞 48h；长期态度 90 天；其余 14 天。"""
    if _ONE_TIME_COMMITMENT_RE.search(value or ""):
        return COMMITMENT_ONE_TIME_TTL_HOURS
    if _STANDING_COMMITMENT_RE.search(value or ""):
        return COMMITMENT_STANDING_TTL_DAYS * 24
    return COMMITMENT_TTL_DAYS * 24


def _commitment_expires_at(item: dict, now: datetime) -> str | None:
    """承诺的过期时刻（2026-10-18 加 TTL）。

    显式 `valid_for` 优先；否则按价值措辞分档。旧数据没有这个字段时，
    `_active_memories` 会用 `created_at + TTL` 兜底（见下）。
    """
    if item.get("kind") != "commitment":
        return None
    days = _VALID_FOR_DAYS.get(str(item.get("valid_for") or ""))
    if days:
        return (now + timedelta(days=days)).isoformat()
    return (now + timedelta(hours=_commitment_ttl_hours(str(item.get("value") or "")))).isoformat()


def _ngrams(text: str) -> set[str]:
    normalized = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text or "").lower()
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[i:i + 2] for i in range(len(normalized) - 1)}


def _relevance(query: str, record: dict) -> float:
    query_terms = _ngrams(query)
    memory_terms = _ngrams(f"{record.get('key', '')}{record.get('value', '')}")
    if not query_terms or not memory_terms:
        return 0.0
    return len(query_terms & memory_terms) / max(1, min(len(query_terms), len(memory_terms)))


# 单字兜底时要先排掉的**虚字**：不排的话「今天午饭吃的啥」靠一个"天"（来自"每天"）
# 或一个"的"就能跟任意事实蹭上，长句会被塞进一堆无关事实。
_FUNC_CHARS = set("的了呢吧吗啊嗯哈是不没都我你他她它们这个有和就在吗呀哦噢")


# 扩写句里常见的**泛词**：它们跟任何一条“用户/自己/今天…”的事实都能蹭上一个二字组，
# 把真正相关的挤到第 5 名、被「每轮只取 4 条」截掉（2026-10-20 实测用户 2151983431：
# 「用户说自己去跑步了」让时差、高考各拿 0.125，跑步那条也是 0.125 → 排第 5 被丢）。
# 只在 fact 相关度里剔除，不动 commitment 的严格匹配。
_GENERIC_QUERY_TERMS = (
    "用户", "自己", "对方", "灰泽满", "hzm", "今天", "现在", "什么",
    "这个", "那个", "事情", "之前", "说过",
)


def _strip_generic_terms(text: str) -> str:
    s = text or ""
    for w in _GENERIC_QUERY_TERMS:
        s = s.replace(w, "|")
    return s


def _fact_relevance(query: str, record: dict) -> float:
    """fact 的相关度——在 `_relevance` 上补一档**单字兜底**。

    为什么必须补（2026-10-17，用户报的）：查 `_relevance` 只看二字组重合，而**招呼/表情
    这类短消息几乎没有二字组能对上任何事实**——实测用户 2151983431 的事实里明明有
    `morning_running：早上要跑步`，但他发「早安」时 query 的二字组是 {早安}，跟
    {早上, 上要, 要跑, 跑步} 交集为空 → **0 分 → 这条事实根本不会被注入**。

    2026-10-20 又发现了两种失效：
      · 扩写句里的“用户/自己”让无关事实各蹭到 0.125，和真正相关的并列 → 被 4 条上限截掉；
      · 单字兜底对**长扩写句**也生效，长句靠“用/户”蹭分。
    改法：算二字组前先剔掉泛词（`_strip_generic_terms`）；单字兜底**只对 ≤4 字短查询**生效。
    实测：「用户说自己去跑步了」→ 跑步 0.25、其余 0；「早安/早上好/跑操」仍命中；
    「在吗/今天午饭吃的啥」仍为 0。

    ⚠️ 只用于 fact。**commitment 仍走严格的 `_relevance`**：把"早上要跑步"捞出来是补上下文，
    把一条不相干的约定捞出来会让她凭空提一嘴没做完的事。
    """
    q = _strip_generic_terms(query)
    score = _relevance(q, record)
    if score > 0:
        return score
    # 单字兜底只为**短消息**设计（「早安」这类）；长扩写句不许靠单字蹭分。
    if len(re.sub(r"[^一-鿿A-Za-z0-9]", "", query or "")) > 4:
        return 0.0
    strip = lambda s: set(re.sub(r"[^一-鿿A-Za-z0-9]", "", s or "")) - _FUNC_CHARS  # noqa: E731
    q_chars = strip(q)
    m_chars = strip(f"{record.get('key', '')}{record.get('value', '')}")
    if not q_chars or not m_chars:
        return 0.0
    ratio = len(q_chars & m_chars) / max(1, len(q_chars))
    if ratio < 0.25:
        return 0.0
    return 0.35 * ratio


# 当前时段 → 事实施加词。为什么需要（2026-10-20）：「你忘了吗」这种提醒句跟任何事实都
# 对不上字面，但主模型只要手里有“这个时段他固定要做什么”，就能自然接住。
_TIME_PERIOD_WORDS = {
    "morning": ("早上", "早晨", "上午", "晨跑", "早操", "早起", "早饭", "morning"),
    "noon": ("中午", "午饭", "午休", "noon"),
    "afternoon": ("下午", "傍晚", "afternoon"),
    "evening": ("晚上", "晚饭", "晚自习", "傍晚", "evening"),
    "night": ("半夜", "凌晨", "睡前", "熬夜", "晚安", "night"),
}


def _period_key(now: datetime | None) -> str:
    """当前小时的粗时段名（和 `_TIME_PERIOD_WORDS` 的键一一对应）。"""
    h = (now or datetime.now()).hour
    if 5 <= h < 11:
        return "morning"
    if 11 <= h < 14:
        return "noon"
    if 14 <= h < 18:
        return "afternoon"
    if 18 <= h < 23:
        return "evening"
    return "night"


def _period_words(now: datetime | None) -> tuple:
    return _TIME_PERIOD_WORDS[_period_key(now)]


# ===========================================================================
# 结构化习惯（2026-10-26）
#
# 为什么不再靠 key/value 的字符串猜（2026-10-24 的临时版）：
# 旧版只要 `habit_*` 的 value 里出现“早上/凌晨”等时段词，就在对应时段把这条
# 记忆塞到用户消息前。它解决了“早”这种短问候捞不到事实的问题，但也把
# `habit_expression_sticker_usage`（表情包撒娇）、`habit_user_bedtime_comfort`
# （睡前摸摸）这类**互动习惯**当成作息，在晚安时也会提醒；还分不清
# “每天早上跑操”和“每周三次 3 公里”，更分不清“今晚要熬夜看比赛”只是一次性计划。
#
# 现在每条 habit_* 带一个可选的结构化对象：
#   {"type": "routine|interaction", "cadence": "daily|weekly|monthly|regular|once|unknown",
#    "period": "morning|noon|afternoon|evening|night|any|unknown",
#    "cues": ["跑操", "跑步"]}
# - type=routine（作息/日程）才可能按“当前时段 + 问候”触发；
# - type=interaction（表情包/情话/摸摸）只按他这句话里的线索触发，不按时段；
# - cadence=once（今晚/明天/这次）不当作固定习惯，直接跳过；
# - 旧数据没有这个对象时，按 value 里的词惰性推导，行为不中断。
#
# 这不是“多写几个关键词”，而是把原来藏在自然语言里的
# 「什么时候相关、什么算同一类」变成可检索的字段。
# ===========================================================================
_HABIT_TYPES = {"routine", "interaction"}
_HABIT_CADENCES = {"daily", "weekly", "monthly", "regular", "once", "unknown"}
_HABIT_PERIODS = {"morning", "noon", "afternoon", "evening", "night", "any", "unknown"}

_HABIT_TYPE_ALIASES = {
    "routine": "routine", "作息": "routine", "日常": "routine", "活动": "routine",
    "日程": "routine", "习惯": "routine", "作息习惯": "routine",
    "interaction": "interaction", "互动": "interaction", "相处": "interaction",
    "情感": "interaction", "撒娇": "interaction", "互动习惯": "interaction",
}
_HABIT_CADENCE_ALIASES = {
    "daily": "daily", "每天": "daily", "每日": "daily", "天天": "daily",
    "每晚": "daily", "每早": "daily", "几乎每天": "daily",
    "weekly": "weekly", "每周": "weekly", "一周": "weekly", "每周末": "weekly",
    "monthly": "monthly", "每月": "monthly",
    "regular": "regular", "经常": "regular", "平时": "regular", "频繁": "regular",
    "常常": "regular", "多次": "regular", "多天多次": "regular", "时常": "regular",
    "once": "once", "今晚": "once", "明天": "once", "这次": "once",
    "一次性": "once", "仅这次": "once", "仅此一次": "once",
    "unknown": "unknown", "未知": "unknown", "不确定": "unknown",
}
_HABIT_PERIOD_ALIASES = {
    "morning": "morning", "早上": "morning", "早晨": "morning", "上午": "morning",
    "清晨": "morning", "早": "morning",
    "noon": "noon", "中午": "noon", "午间": "noon",
    "afternoon": "afternoon", "下午": "afternoon", "傍晚": "afternoon",
    "evening": "evening", "晚上": "evening", "晚间": "evening", "晚饭": "evening",
    "night": "night", "半夜": "night", "凌晨": "night", "深夜": "night",
    "夜里": "night", "睡前": "night", "熬夜": "night",
    "any": "any", "任意": "any", "不限": "any", "随时": "any", "无": "any",
    "unknown": "unknown", "未知": "unknown",
}
_HABIT_PERIOD_TEXT = {
    "morning": ("早上", "早晨", "上午", "清晨", "早八", "早操", "晨跑", "早起", "早饭", "每早"),
    "noon": ("中午", "午饭", "午休", "午间"),
    "afternoon": ("下午", "傍晚"),
    "evening": ("晚上", "晚饭", "晚自习", "晚间"),
    "night": ("半夜", "凌晨", "深夜", "夜里", "睡前", "熬夜", "晚安", "每晚"),
}
_HABIT_INTERACTION_RE = re.compile(
    r"表情包|撒娇|情话|肉麻|念诗|诗歌|歌词|摸摸|抚慰|求语音|索要|称呼|夸|黏|表达心意"
)
_HABIT_SLEEP_CUES = (
    "晚安", "要睡", "睡了", "困了", "好困", "困死", "熬夜", "没睡", "还没睡", "失眠",
)
_HABIT_MORNING_PREFIXES = (
    "早安", "早上好", "上午好", "早啊", "早呀", "早哦", "早哇", "早早早",
)
_HABIT_MORNING_STATE_CUES = ("起床了", "刚起床", "刚起", "刚醒", "醒了", "睡醒")
_HABIT_TIME_ONLY = {
    "每天", "每日", "天天", "每晚", "每早", "每周", "每月", "一周", "平时",
    "经常", "频繁", "常常", "多次", "几乎", "日常", "多天多次",
}
_HABIT_GENERIC_TERMS = {
    "每天", "每日", "天天", "每周", "每月", "平时", "经常", "频繁", "常常",
    "多次", "几乎", "用户", "自己", "今天", "现在", "什么", "这个", "那个",
    "时候", "已经", "还有", "都要", "起床", "早上", "晚上", "睡前", "凌晨",
    "深夜", "半夜", "日常",
}
_HABIT_CUE_STRIP = (
    "对话里", "每天", "每日", "天天", "每晚", "每早", "每周", "每月", "一周",
    "平时", "经常", "频繁", "常常", "多次", "多天多次", "几乎", "通常", "一般",
    "习惯", "用户", "他", "她", "还要", "还有", "都要", "三次", "每次",
    "今晚", "明天", "这次", "仍然", "仍", "也", "常带", "常", "并", "在",
    "于", "会", "要", "去", "有", "的", "了", "和", "或", "与", "向", "用",
    "发", "找", "给", "早上", "早晨", "上午", "中午", "下午",
    "晚上", "深夜", "凌晨", "半夜", "睡前", "起床后", "起床", "除", "除了",
    "日常", "外",
)
_HABIT_CUE_BAD = (
    "灰泽满", "机器人", "原因", "频率", "因为", "要求", "时候",
    "每天", "每周", "每月", "三次", "多次", "几乎", "还有",
    "凌晨", "深夜", "半夜", "睡前", "早上", "晚上", "起床",
    "今晚", "明天", "增强", "语气", "称呼",
)
_HABIT_EXCEPTION_RE = re.compile(
    r"(?:取消|请假|改天|休息|算了|不去了|不跑了|不练了|不背了|不喝了|不做了|"
    r"不上了|没去|没跑|没练|没背|没喝|没做|没上|不想去|不想跑|不想做|不想练|"
    r"不熬夜|没熬夜|不想熬夜|"
    r"(?:今天|今晚|明天|这周|早上|这次)?(?:不|没)(?:去|跑|做|练|上|背|喝|吃|起))"
)


def _compact_text(text: Any) -> str:
    return re.sub(r"[^一-鿿A-Za-z0-9]+", "", str(text or "")).lower()


def _alias(value: Any, table: dict) -> str:
    text = str(value or "").strip().lower()
    return table.get(text, "") if text else ""


def _derive_habit_cadence(text: str) -> str:
    # 先认“每天”：同一句里出现“每天晚上…今晚也…”时，主频率仍是每天。
    if re.search(r"每天|每日|天天|每晚|每早|几乎每天", text):
        return "daily"
    if re.search(r"今晚|明天|这次|这一次|仅此一次|一次性", text):
        return "once"
    if re.search(r"每周|一周|每周末", text):
        return "weekly"
    if re.search(r"每月", text):
        return "monthly"
    if re.search(r"经常|平时|频繁|常常|多次|多天多次|时常", text):
        return "regular"
    return "unknown"


def _derive_habit_period(text: str) -> str:
    hits = {
        period for period, words in _HABIT_PERIOD_TEXT.items()
        if any(w in text for w in words)
    }
    if len(hits) == 1:
        return next(iter(hits))
    # 一句话里跨了两个时段（“早上和晚上”）→ 不按时段触发，只按话题线索。
    return "any" if hits else "unknown"


def _strip_cue_prefixes(token: str) -> str:
    changed = True
    while token and changed:
        changed = False
        for prefix in _HABIT_CUE_STRIP:
            if token.startswith(prefix):
                token = token[len(prefix):]
                changed = True
                break
    return token


def _derive_habit_cues(value: str) -> list[str]:
    """从自然语言里抠出“他说什么时这条习惯相关”的短语。

    只做保守的切分/去时间词；抠不出来就返回空，检索仍会退回二字组匹配。
    时间词（早上/每天）不进 cues——时段由 `period` 管，别让“早”既当时间又当线索。
    """
    cues: list[str] = []
    for chunk in re.split(r"[，。；、（）()\[\]【】{}<>《》“”\"'「」/／|]+", str(value or "")):
        chunk = chunk.strip()
        if not chunk:
            continue
        for part in re.split(r"[和及或]", chunk):
            token = _strip_cue_prefixes(part)
            token = _compact_text(token)
            if not (2 <= len(token) <= 8):
                continue
            if any(bad in token for bad in _HABIT_CUE_BAD):
                continue
            if token in _HABIT_TIME_ONLY:
                continue
            if re.fullmatch(r"[0-9一二三四五六七八九十两]+次", token):
                continue
            if token.endswith(("外", "后", "时", "中", "的", "了")):
                continue
            if token not in cues:
                cues.append(token)
            if len(cues) >= 8:
                return cues
    return cues


def _derive_habit_metadata(key: str, value: str) -> dict:
    text = f"{key} {value}".lower()
    habit_type = "interaction" if _HABIT_INTERACTION_RE.search(value or "") else "routine"
    return {
        "type": habit_type,
        "cadence": _derive_habit_cadence(text),
        "period": _derive_habit_period(text),
        "cues": _derive_habit_cues(value),
    }


def _clean_habit_cues(raw: Any) -> list[str]:
    if isinstance(raw, str):
        parts = re.split(r"[，,、;；|/]+", raw)
    elif isinstance(raw, list):
        parts = raw
    else:
        parts = []
    out: list[str] = []
    for part in parts:
        cue = _compact_text(part)[:12]
        if len(cue) < 2 or cue in out:
            continue
        out.append(cue)
        if len(out) >= 8:
            break
    return out


def normalize_habit_metadata(raw: Any, *, key: str = "", value: str = "") -> dict:
    """合并“模型给的结构化字段”和“从旧 value 惰性推导的字段”。

    模型字段优先；缺哪项就用推导值补哪项。返回值永远是一个完整 dict，
    所以新写入的 habit_* 都会带上 type/cadence/period/cues，旧数据则在读取时现推。
    """
    meta = _derive_habit_metadata(key, value)
    if not isinstance(raw, dict):
        return meta
    habit_type = _alias(raw.get("type"), _HABIT_TYPE_ALIASES)
    cadence = _alias(raw.get("cadence"), _HABIT_CADENCE_ALIASES)
    period = _alias(raw.get("period"), _HABIT_PERIOD_ALIASES)
    cues = _clean_habit_cues(raw.get("cues"))
    if habit_type in _HABIT_TYPES:
        meta["type"] = habit_type
    if cadence in _HABIT_CADENCES:
        meta["cadence"] = cadence
    if period in _HABIT_PERIODS:
        meta["period"] = period
    if cues:
        meta["cues"] = cues
    return meta


def _habit_meta(memory: dict) -> dict:
    return normalize_habit_metadata(
        memory.get("habit"),
        key=str(memory.get("key") or ""),
        value=str(memory.get("value") or ""),
    )


def _habit_display_value(value: Any) -> str:
    """把记忆里的“用户…”改成“他…”，方便直接念给模型听。"""
    text = str(value or "").strip().rstrip("。")
    text = re.sub(r"^(?:该)?用户", "他", text)
    if not text.startswith("他"):
        text = "他" + text
    return text


def _habit_effective_period(query: str, now: datetime | None) -> str:
    current = _period_key(now)
    # “晚安/睡了”常发生在 22 点（evening 档），但对夜间习惯应按 night 处理。
    if current in {"evening", "night"} and any(c in query for c in _HABIT_SLEEP_CUES):
        return "night"
    return current


def _habit_greeting(query: str, now: datetime | None) -> bool:
    """这句话是不是在打招呼/报状态——不用“字数少”猜，免得“在吗”也算。"""
    q = str(query or "").strip()
    q_compact = _compact_text(q)
    period = _habit_effective_period(q, now)
    if period == "morning":
        if re.match(r"^早(?:\s|[,，。.!！~～]|$)", q) or q.startswith(_HABIT_MORNING_PREFIXES):
            return True
        if q_compact == "起床" or any(c in q for c in _HABIT_MORNING_STATE_CUES):
            return True
    if period in {"evening", "night"}:
        if any(c in q for c in _HABIT_SLEEP_CUES):
            return True
        if "晚上好" in q:
            return True
    if period == "noon" and any(c in q for c in ("中午好", "午安")):
        return True
    if period == "afternoon" and "下午好" in q:
        return True
    if period == "evening" and "晚上好" in q:
        return True
    return False


def _habit_relevance(query: str, memory: dict) -> float:
    """习惯的话题相关度：先剔掉“每天/今天”这类泛词，避免靠一个泛词蹭上。"""
    q = str(query or "")
    for term in _HABIT_GENERIC_TERMS:
        q = q.replace(term, "|")
    return _relevance(q, memory)


def _habit_cue_hit(query: str, memory: dict, meta: dict) -> bool:
    q = _compact_text(query)
    if not q:
        return False
    for cue in meta.get("cues") or []:
        if len(cue) >= 2 and cue in q:
            return True
    return _habit_relevance(query, memory) > 0


def _habit_mentioned_recently(memory: dict, meta: dict, recent_text: str) -> bool:
    """她最近几轮是不是已经提过这条习惯了——提过就别再塞提示，免得每轮都问。"""
    recent = _compact_text(recent_text)
    if not recent:
        return False
    for cue in meta.get("cues") or []:
        if len(cue) >= 2 and cue in recent:
            return True
    for term in _ngrams(str(memory.get("value") or "")):
        if len(term) < 2 or term in _HABIT_GENERIC_TERMS:
            continue
        if term in recent:
            return True
    return False


def _ritual_matches(memory: dict, query: str) -> bool:
    """固定仪式只在**触发场景**出现时注入。

    value 约定写成「场景：内容」（如「晚安：晚安+喜欢你+爱绿冻+mua」），
    场景那半截按 +、/、逗号等切开；用户消息里出现任一个场景词才注入。
    这样“晚安仪式”不会在白天每条消息里都冒出来。
    """
    value = str(memory.get("value") or "")
    head = re.split(r"[：:]", value, maxsplit=1)[0]
    cues = [t for t in re.split(r"[+＋、,，/|\s]+", head) if t]
    q = query or ""
    return any(c in q for c in cues)


def _name_fact_rejection(key: str, value: str, source: dict) -> str:
    """名字类事实的代码闸门：必须由用户明确要求称呼，且 value 本身像名字。"""
    if key not in _NAME_KEYS:
        return ""
    excerpt = str((source or {}).get("excerpt") or "")
    if not _NAME_REQUEST_RE.search(excerpt):
        return "name_requires_explicit_request"
    if not _NAME_VALUE_OK_RE.fullmatch(value or ""):
        return "name_value_not_a_name"
    if any(w in value for w in _NAME_VALUE_BAD_WORDS):
        return "name_value_not_a_name"
    return ""


class ShadowMemoryStore:
    """Validate, consolidate, and inspect Memory V2 shadow state.

    The interface is intentionally small: callers ingest model candidates,
    request a hypothetical context, inspect a user partition, or delete it.
    """

    def __init__(
        self,
        state_file: Path = DEFAULT_STATE_FILE,
        audit_file: Path = DEFAULT_AUDIT_FILE,
        context_budget_chars: int = 700,
    ) -> None:
        self.state_file = Path(state_file)
        self.audit_file = Path(audit_file)
        self.context_budget_chars = max(100, int(context_budget_chars))
        self._nickname_hint_last: dict[str, float] = {}
        self._lock = threading.Lock()

    def _empty_state(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "users": {}}

    def _load(self) -> dict:
        if not self.state_file.exists():
            return self._empty_state()
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._empty_state()
        if not isinstance(data, dict) or not isinstance(data.get("users"), dict):
            return self._empty_state()
        return data

    def _save(self, state: dict) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.state_file)

    def _audit(self, entry: dict) -> None:
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _normalize_source(self, source: dict, role: str, now: datetime) -> dict | None:
        if not isinstance(source, dict):
            return None
        session_id = _bounded_text(source.get("session_id"), MAX_SESSION_ID_CHARS)
        if not session_id:
            return None
        observed = _iso(source.get("observed_at")) or now
        raw = source.get("assistant_text") if role == "assistant" else source.get("user_text")
        excerpt = _redact_sensitive_excerpt(_bounded_text(raw, MAX_SOURCE_CHARS))
        if not excerpt:
            return None
        return {
            "role": role,
            "excerpt": excerpt,
            "session_id": session_id,
            "observed_at": observed.isoformat(),
        }

    def _normalize_item(self, item: dict, source: dict, now: datetime) -> tuple[dict | None, str]:
        if not isinstance(item, dict):
            return None, "item_not_object"
        kind = _bounded_text(item.get("kind"), 40)
        key = _bounded_text(item.get("key"), 48)
        value = _bounded_text(item.get("value"), MAX_VALUE_CHARS)
        action = _bounded_text(item.get("action") or "upsert", 16)
        explicit = item.get("explicit")
        confidence = item.get("confidence")
        temporal_scope = _bounded_text(item.get("temporal_scope"), 16)

        if kind not in _KINDS or action not in _ACTIONS:
            return None, "unsupported_kind_or_action"
        if not _KEY_RE.fullmatch(key) or not value:
            return None, "invalid_key_or_value"
        if not isinstance(explicit, bool):
            return None, "explicit_must_be_boolean"
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            return None, "confidence_must_be_number"
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            return None, "confidence_out_of_range"
        if key in _SENSITIVE_KEYS or _contains_sensitive_value(value):
            return None, "sensitive_data_blocked"
        if kind == "fact" and _is_bot_self_fact(key, value):
            return None, "bot_self_fact_blocked"

        default_scope = {
            "fact": "current",
            "interaction_preference": "timeless",
            "commitment": "future",
        }[kind]
        temporal_scope = temporal_scope or default_scope
        if temporal_scope not in _TEMPORAL_SCOPES:
            return None, "invalid_temporal_scope"
        allowed_scopes = {
            "fact": {"past", "current", "timeless"},
            "interaction_preference": {"current", "timeless"},
            "commitment": {"future", "current"},
        }[kind]
        if temporal_scope not in allowed_scopes:
            return None, "temporal_scope_mismatch"

        # 时效性近况的期限档位（只有 fact 用；其余一律 none）。
        valid_for = _bounded_text(item.get("valid_for"), 8).lower()
        if valid_for not in _VALID_FOR_DAYS:
            valid_for = "none"

        role = "assistant" if kind == "commitment" else "user"
        normalized_source = self._normalize_source(source, role, now)
        if normalized_source is None:
            return None, "invalid_source"

        if kind == "fact" and not explicit:
            return None, "inferred_fact_blocked"
        if kind == "fact" and key in _NAME_KEYS and action != "retract":
            reason = _name_fact_rejection(key, value, normalized_source)
            if reason:
                return None, reason
        if kind == "interaction_preference":
            if key not in _PREFERENCE_VALUES and not _is_ritual_key(key):
                return None, "invalid_preference_dimension"
            if key in _PREFERENCE_VALUES and value not in _PREFERENCE_VALUES[key]:
                return None, "invalid_preference_dimension"
            if _is_ritual_key(key) and ("：" not in value and ":" not in value):
                # 固定仪式必须写成「场景：内容」，否则注入时无法按场景触发。
                return None, "ritual_needs_cue"
        if kind == "commitment":
            if item.get("actor") != "assistant" or not explicit:
                return None, "assistant_commitment_required"

        normalized = {
            "kind": kind,
            "key": key,
            "value": value,
            "action": action,
            "explicit": explicit,
            "confidence": confidence,
            "temporal_scope": temporal_scope,
            "valid_for": valid_for,
            "source": normalized_source,
        }
        # 习惯的结构化字段（2026-10-26）：模型没给就从 value 推导。
        # 不是 key/value 之外的“另存一份事实”，只是把“什么时候相关、怎么触发”变成可检索的字段。
        if kind == "fact" and key.startswith("habit_"):
            normalized["habit"] = normalize_habit_metadata(
                item.get("habit"), key=key, value=value)
        # 栏目名是"什么都能装"的那种 → 打标记（只记不拦，见 _GENERIC_FACT_KEYS 的说明）
        if kind == "fact" and key in _GENERIC_FACT_KEYS:
            normalized["generic_key"] = True
        return normalized, "accepted"

    @staticmethod
    def _memory_id(kind: str, key: str, value: str) -> str:
        raw = f"{kind}\0{key}\0{value}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:20]

    @staticmethod
    def _evidence_id(source: dict) -> str:
        raw = "\0".join(str(source.get(key) or "") for key in (
            "role", "excerpt", "session_id", "observed_at",
        )).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:20]

    def ingest(self, user_id: str, items: list, *, source: dict, now: datetime | None = None) -> list[dict]:
        now = now or datetime.now()
        user_id = _bounded_text(user_id, MAX_USER_ID_CHARS)
        if not user_id or not isinstance(items, list):
            return []

        accepted: list[dict] = []
        with self._lock:
            state = self._load()
            if not items and user_id in state["users"]:
                # 这轮什么都没提取到：往下走只会改一个没人读的 updated_at，
                # 却要把**全体用户**的状态文件整份重写一遍——跳过。新用户第一次照旧建档。
                return []
            user = state["users"].setdefault(user_id, {"memories": [], "updated_at": now.isoformat()})
            memories = user.setdefault("memories", [])

            for raw_item in items[:20]:
                item, reason = self._normalize_item(raw_item, source, now)
                if item is None:
                    self._audit({
                        "t": now.isoformat(), "user_id": user_id,
                        "decision": "rejected", "reason": reason,
                    })
                    continue

                if item["action"] == "retract":
                    changed = False
                    for memory in memories:
                        if (memory.get("kind") == item["kind"] and memory.get("key") == item["key"]
                                and memory.get("status") not in {"superseded", "expired"}
                                and (not item["value"] or memory.get("value") == item["value"])):
                            memory["status"] = "superseded"
                            memory["invalidated_at"] = now.isoformat()
                            changed = True
                    self._audit({
                        "t": now.isoformat(), "user_id": user_id,
                        "decision": "retracted" if changed else "ignored",
                        "kind": item["kind"], "key": item["key"],
                    })
                    continue

                memory_id = self._memory_id(item["kind"], item["key"], item["value"])
                evidence_id = self._evidence_id(item["source"])
                existing = next((m for m in memories if m.get("id") == memory_id), None)
                if existing is None:
                    if item["explicit"] and item["kind"] in {"fact", "interaction_preference"}:
                        status = "confirmed"
                    elif item["kind"] == "commitment":
                        status = "active"
                    else:
                        status = "candidate"
                    existing = {
                        "id": memory_id,
                        "kind": item["kind"],
                        "key": item["key"],
                        "value": item["value"],
                        "status": status,
                        "explicit": item["explicit"],
                        "confidence": item["confidence"],
                        "temporal_scope": item["temporal_scope"],
                        "valid_for": item["valid_for"],
                        "evidence_count": 1,
                        "evidence_ids": [evidence_id],
                        "session_ids": [item["source"]["session_id"]],
                        "session_count": 1,
                        "created_at": now.isoformat(),
                        "last_observed_at": now.isoformat(),
                        "source": item["source"],
                    }
                    if item.get("habit"):
                        existing["habit"] = item["habit"]
                    expires_at = _fact_expires_at(item, now) or _commitment_expires_at(item, now)
                    if expires_at:
                        existing["expires_at"] = expires_at
                    memories.append(existing)
                else:
                    evidence_ids = list(existing.get("evidence_ids") or [])
                    if not evidence_ids and isinstance(existing.get("source"), dict):
                        evidence_ids.append(self._evidence_id(existing["source"]))
                    if evidence_id in evidence_ids:
                        self._audit({
                            "t": now.isoformat(), "user_id": user_id,
                            "decision": "duplicate_evidence",
                            "memory_id": existing["id"],
                        })
                        continue
                    evidence_ids.append(evidence_id)
                    existing["evidence_ids"] = evidence_ids[-50:]
                    existing["evidence_count"] = int(existing.get("evidence_count", 1)) + 1
                    session_ids = list(existing.get("session_ids") or [])
                    if item["source"]["session_id"] not in session_ids:
                        session_ids.append(item["source"]["session_id"])
                    existing["session_ids"] = session_ids[-20:]
                    existing["session_count"] = len(existing["session_ids"])
                    existing["last_observed_at"] = now.isoformat()
                    existing["source"] = item["source"]
                    existing["confidence"] = max(float(existing.get("confidence", 0)), item["confidence"])
                    existing["temporal_scope"] = item["temporal_scope"]
                    existing["valid_for"] = item["valid_for"]
                    if item.get("habit"):
                        # 同一条习惯又被提到：用最新的结构化字段（模型修正/人工补的覆盖旧推导）。
                        existing["habit"] = item["habit"]
                    expires_at = _fact_expires_at(item, now) or _commitment_expires_at(item, now)
                    if expires_at:
                        existing["expires_at"] = expires_at   # 近况/承诺被再次提到 → 续期
                    if item["explicit"]:
                        existing["explicit"] = True
                        existing["status"] = "active" if item["kind"] == "commitment" else "confirmed"
                    elif (item["kind"] == "interaction_preference"
                          and existing["evidence_count"] >= 3
                          and existing["session_count"] >= 2):
                        existing["status"] = "confirmed"

                # 只有"单值槽"才顶替：偏好（每维度一个值）、temporal_scope=current 的 fact（近况），
                # 以及 `_SINGLE_VALUED_FACT_KEYS` 里那些**天然单值**的键（名字/地点/时区/职业）。
                # `habit_*` 也按单值槽处理（2026-10-26）：一个具体 key 就是一项习惯，
                # 新说法/新频率应顶掉旧的，不能同 key 并存两条。
                # **其余 past / timeless 的 fact 仍累积，不倒旧的**——否则"小时候被霸凌"会被
                # 后来的"高考车祸"顶掉（实测真丢过 3 条）。
                if item["explicit"] and (
                    item["kind"] == "interaction_preference"
                    or item["kind"] == "commitment"   # 同一承诺 key 只留最新一条（旧值顶替）
                    or (item["kind"] == "fact" and (
                        item["temporal_scope"] == "current"
                        or item["key"] in _SINGLE_VALUED_FACT_KEYS
                        or item["key"].startswith("habit_")
                    ))
                ):
                    for other in memories:
                        if (other is not existing and other.get("kind") == item["kind"]
                                and other.get("key") == item["key"]
                                and other.get("status") not in {"superseded", "expired"}):
                            other["status"] = "superseded"
                            other["invalidated_at"] = now.isoformat()

                accepted.append(json.loads(json.dumps(existing, ensure_ascii=False)))
                record = {
                    "t": now.isoformat(), "user_id": user_id, "decision": "accepted",
                    "memory_id": existing["id"], "kind": existing["kind"],
                    "key": existing["key"], "status": existing["status"],
                }
                if item.get("generic_key"):
                    # 命中"什么都能装"的栏目名。**只记不拦**（2026-10-16）——先攒几天命中率，
                    # 看撞上之后它会不会自己换个具体名字（见 _GENERIC_FACT_KEYS 的说明）。
                    record["generic_key"] = True
                self._audit(record)

            user["updated_at"] = now.isoformat()
            self._save(state)
        return accepted

    def snapshot(self, user_id: str) -> dict:
        with self._lock:
            user = self._load().get("users", {}).get(str(user_id), {})
        if not isinstance(user, dict):
            return {"memories": []}
        result = json.loads(json.dumps(user, ensure_ascii=False))
        result.setdefault("memories", [])
        return result

    def _active_memories(
        self,
        user_id: str,
        *,
        now: datetime | None = None,
        session_id: str | None = None,
    ) -> list[dict]:
        """存活的条目：confirmed/active，且没过 TTL（时效性近况到期就不算）。

        session_id 参数保留仅为兼容旧调用；情绪已搬去会话记忆，这里不再按会话过滤。
        """
        now = now or datetime.now()
        memories = self.snapshot(user_id).get("memories", [])
        active = []
        for memory in memories:
            if memory.get("status") not in {"confirmed", "active"}:
                continue
            expires = _iso(memory.get("expires_at"))
            if expires is None and memory.get("kind") == "commitment":
                # 旧数据没有 expires_at（2026-10-18 前写入）→ 按 created_at + TTL 兜底，
                # 不让历史里的"明天…"永久 active。
                created = _iso(memory.get("created_at"))
                if created is not None:
                    expires = created + timedelta(
                        hours=_commitment_ttl_hours(str(memory.get("value") or "")))
            if expires is not None and expires <= now:
                continue
            active.append(memory)
        return active

    def _select_facts(self, active: list[dict], query: str) -> list[dict]:
        # `habit_*` 单独走 `_preference_lines` 的“固定习惯”段——别混进通用事实里被
        # 名字/地点挤掉，也别让“早上跑操”这种**个人习惯**看起来像全局设定。
        facts = [m for m in active
                 if m.get("kind") == "fact" and not str(m.get("key", "")).startswith("habit_")]
        facts.sort(key=lambda m: (m.get("key") not in _ALWAYS_ON_FACT_KEYS,
                                  -_fact_relevance(query, m)))
        return [
            m for m in facts
            if (
                (m.get("key") in _ALWAYS_ON_FACT_KEYS and m.get("temporal_scope") != "past")
                or _fact_relevance(query, m) > 0
                or (m.get("temporal_scope") == "past" and any(k in query for k in ("以前", "过去", "曾经")))
            )
        ][:4]

    def _matching_habits(self, active: list[dict], query: str,
                         *, now: datetime | None = None, limit: int = 2) -> list[dict]:
        """这个用户此刻真正相关的固定习惯（habit_*），按相关度排序。

        2026-10-26 起按结构化字段判断，不再靠“消息短 + value 里有早上”猜：

        - `type=interaction`（表情包/情话/摸摸）**只按他这句话的线索**召回，
          不在晚上把所有互动习惯都翻出来；
        - `type=routine` 且 `cadence=daily`，才可能按“当前时段 + 打招呼/报状态”召回；
        - `cadence=once`（今晚/明天/这次）不是固定习惯，直接跳过；
        - 他刚说“不跑了/取消了/请假了”时，routine 习惯不再追着问；
        - 多条同时相关时按“线索命中 + 时段命中 + 文本重合”排序，不再取文件里第一条。
        """
        q = query or ""
        effective_period = _habit_effective_period(q, now)
        greeting = _habit_greeting(q, now)
        negative = bool(_HABIT_EXCEPTION_RE.search(q))
        scored: list[tuple[float, str, dict]] = []
        for memory in active:
            if memory.get("kind") != "fact" or not str(memory.get("key", "")).startswith("habit_"):
                continue
            meta = _habit_meta(memory)
            if meta.get("cadence") == "once":
                continue
            cue_hit = _habit_cue_hit(q, memory, meta)
            period_hit = (
                meta.get("type") == "routine"
                and meta.get("cadence") == "daily"
                and meta.get("period") not in {"any", "unknown"}
                and meta.get("period") == effective_period
            )
            if not (cue_hit or (period_hit and greeting)):
                continue
            # 否定/例外只压作息类习惯：他说“今天不跑操”，别再问“今天也去吗”。
            # 互动类（例如“今天不摸摸”）不在这里压，避免把真正的互动线索一起吞掉。
            if negative and meta.get("type") == "routine":
                continue
            score = 0.0
            if cue_hit:
                score += 3.0
            if period_hit and greeting:
                score += 2.0
            score += _habit_relevance(q, memory)
            scored.append((score, str(memory.get("last_observed_at") or ""), memory))
        scored.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
        out: list[dict] = []
        seen: set[str] = set()
        for _, _, memory in scored:
            mid = str(memory.get("id") or memory.get("key"))
            if mid in seen:
                continue
            seen.add(mid)
            out.append(memory)
            if len(out) >= limit:
                break
        return out

    def _preference_lines(self, active: list[dict], query: str,
                          used_out: list | None = None,
                          now: datetime | None = None,
                          include_facts: bool = True) -> list[str]:
        """靠前段：相处方式（行为句）+ 话题相关事实 + **当前时段的固定安排**。

        used_out：传入 list 时追加**实际选中注入**的记忆 id（条目级注入轨迹用）。
        预算截断的边界情况可能多记一条（它的前缀仍在提示词里），只做诊断、不做账。
        """
        preferences = [m for m in active if m.get("kind") == "interaction_preference"]
        lines: list[str] = []
        pref_lines: list[tuple[str, dict]] = []
        for memory in preferences[:MAX_PREFERENCE_ITEMS]:
            behavior = inj.behavior_sentence(memory.get("key"), memory.get("value"))
            if behavior:
                pref_lines.append((behavior, memory))
            elif _is_ritual_key(str(memory.get("key"))) and _ritual_matches(memory, query):
                pref_lines.append((inj.T("profile", "ritual", value=memory.get("value")), memory))
        if pref_lines:
            lines.append(inj.T("profile", "head"))
            for text, memory in pref_lines:
                lines.append(text)
                if used_out is not None and memory.get("id"):
                    used_out.append(str(memory["id"]))
        facts = self._select_facts(active, query) if include_facts else []
        selected_ids = set()
        if facts:
            rendered = "；".join(
                f"{'过去事实/' if m.get('temporal_scope') == 'past' else ''}{m.get('key')}：{m.get('value')}"
                for m in facts
            )
            lines.append(inj.T("profile", "fact", rendered=rendered))
            if used_out is not None:
                used_out.extend(str(m["id"]) for m in facts if m.get("id"))
            selected_ids = {str(m.get("id")) for m in facts if m.get("id")}
        # `habit_*` 不在这里注：改由 `build_habit_hint` 放在用户消息前（只对拥有该记忆的用户），
        # 避免和尾部的时段提示重复，也避免把个人习惯混进通用事实。
        # 当前时段的固定安排：和查询相关度无关。「你忘了吗」这类提醒句靠它接得住。
        # 已经按查询/习惯段注入过的不重复（同一件事一轮只说一次）。
        time_facts = [m for m in self._time_facts(active, now=now)
                      if str(m.get("id")) not in selected_ids
                      and not str(m.get("key", "")).startswith("habit_")]
        if time_facts:
            rendered = "；".join(f"{m.get('key')}：{m.get('value')}" for m in time_facts)
            lines.append(inj.T("profile", "time_fact", rendered=rendered))
            if used_out is not None:
                used_out.extend(str(m["id"]) for m in time_facts if m.get("id"))
        return lines

    def _time_facts(self, active: list[dict], *, now: datetime | None = None,
                    limit: int = 2) -> list[dict]:
        """当前时段对得上的固定事实（最多 2 条）。过去事实不算。"""
        words = _period_words(now)
        out = []
        for memory in active:
            if memory.get("kind") != "fact" or memory.get("temporal_scope") == "past":
                continue
            text = f"{memory.get('key', '')}{memory.get('value', '')}".lower()
            if any(w.lower() in text for w in words):
                out.append(memory)
        return out[:limit]

    def _commitment_lines(self, active: list[dict], query: str,
                          used_out: list | None = None) -> list[str]:
        """中段：未完成约定——别忘、要兑现。used_out 同 `_preference_lines`。"""
        commitments = [m for m in active if m.get("kind") == "commitment" and _relevance(query, m) > 0]
        if not commitments:
            return []
        # 命中多条时先注最相关的；旧实现按存储顺序取前 2 条，会把最旧的那条塞进去。
        commitments.sort(key=lambda m: -_relevance(query, m))
        chosen = commitments[:2]
        if used_out is not None:
            used_out.extend(str(m["id"]) for m in chosen if m.get("id"))
        rendered = "；".join(str(m.get("value")) for m in chosen)
        return [inj.T("commitment", rendered=rendered)]

    def _fit(self, lines: list[str], budget: int) -> str:
        if not lines:
            return ""
        output: list[str] = []
        used = 0
        for line in lines:
            separator = 1 if output else 0
            remaining = budget - used - separator
            if remaining <= 0:
                break
            piece = line[:remaining]
            if piece:
                output.append(piece)
                used += len(piece) + separator
            if len(piece) < len(line):
                break
        return "\n".join(output)

    def build_preference_context(
        self,
        user_id: str,
        query: str,
        *,
        now: datetime | None = None,
        session_id: str | None = None,
        used_ids: list | None = None,
        include_facts: bool = True,
    ) -> str:
        """靠前注入：相处方式行为句 + 话题相关事实（system，个体适配）。

        used_ids：传入 list 时追加实际注入的记忆 id（条目级注入轨迹用）。
        include_facts=False 用于注入 v3：fact 走 `recall`，这里只留相处方式行为句。
        """
        active = self._active_memories(user_id, now=now, session_id=session_id)
        return self._fit(self._preference_lines(active, query or "", used_out=used_ids,
                                                now=now, include_facts=include_facts),
                         PREFERENCE_BUDGET_CHARS)

    def build_habit_hint(self, user_id: str, query: str,
                         *, now: datetime | None = None,
                         recent_bot_text: str = "") -> str:
        """把“他此刻相关的固定习惯”渲染成**贴近用户消息**的一条提示；没有返回空串。

        为什么单独有它（2026-10-24）：习惯放在靠前的档案里，模型在短问候里经常不看；
        放到用户消息前、并且只对拥有这条习惯的用户出现，才能既生效又不串到别人。
        这不是全局 behaviors——内容完全来自这个用户的 `habit_*` 记忆。

        2026-10-26 起：
        - 只渲染 value（不再把英文 key 念给模型），并把“用户…”改成“他…”；
        - 作息/互动/一次性用不同文案；早间作息才用“今天还照常吗”的说法；
        - `recent_bot_text` 里已经提过这条习惯 → 这轮不再提示，避免连着几轮追问。
        """
        active = self._active_memories(user_id, now=now)
        habits = self._matching_habits(active, query, now=now, limit=1)
        if not habits:
            return ""
        memory = habits[0]
        meta = _habit_meta(memory)
        if _habit_mentioned_recently(memory, meta, recent_bot_text):
            return ""
        rendered = _habit_display_value(memory.get("value"))
        if meta.get("type") == "interaction":
            key = "habit_hint_interaction"
        elif (meta.get("type") == "routine"
              and meta.get("cadence") == "daily"
              and meta.get("period") == "morning"
              and _period_key(now) == "morning"
              and _habit_greeting(query, now)):
            key = "habit_hint_morning"
        else:
            key = "habit_hint"
        return inj.T("profile", key, rendered=rendered)

    def her_nickname(self, user_id: str, *, now: datetime | None = None) -> str:
        """用户给她起的外号（没有返回空串）。只用于识别/情境提示，不改变默认自称。"""
        for memory in self._active_memories(user_id, now=now):
            if memory.get("kind") == "fact" and memory.get("key") == "her_nickname":
                return str(memory.get("value") or "").strip()
        return ""

    def nickname_hint(self, user_id: str, user_msg: str, *,
                      now: datetime | None = None) -> str:
        """用户**这一轮**叫了她的外号时，返回一条一次性自称提示；否则空串。

        2026-10-19 用户定：默认永远自称"灰泽满"，只有用户这轮主动叫外号
        （"灰宝在吗"）时，才允许这一轮偶尔用外号自称一次。不做长期偏好、不猜情绪。
        """
        now = now or datetime.now()
        last = self._nickname_hint_last.get(user_id, 0.0)
        if now.timestamp() - last < NICKNAME_HINT_COOLDOWN_SECONDS:
            return ""
        nickname = self.her_nickname(user_id, now=now)
        if nickname and nickname in (user_msg or ""):
            self._nickname_hint_last[user_id] = now.timestamp()
            return inj.T("profile", "nickname_hint", nickname=nickname)
        return ""

    def build_commitment_context(
        self,
        user_id: str,
        query: str,
        *,
        now: datetime | None = None,
        session_id: str | None = None,
        used_ids: list | None = None,
    ) -> str:
        """中段注入：未完成约定（system，中段——别忘、要兑现）。used_ids 同 build_preference_context。"""
        active = self._active_memories(user_id, now=now, session_id=session_id)
        return self._fit(self._commitment_lines(active, query or "", used_out=used_ids),
                         COMMITMENT_BUDGET_CHARS)

    def build_context(
        self,
        user_id: str,
        query: str,
        *,
        now: datetime | None = None,
        session_id: str | None = None,
    ) -> str:
        """聚合视图（两段拼接、共享同一预算）——审计/测试用，注入走各自的方法。

        情绪不在这里：它住在会话记忆（`session_memory.mood_note`），跟话题一起 3h 消散。
        """
        active = self._active_memories(user_id, now=now, session_id=session_id)
        question = query or ""
        lines = (
            self._preference_lines(active, question, now=now)
            + self._commitment_lines(active, question)
        )
        return self._fit(lines, self.context_budget_chars)

    def delete_user(self, user_id: str, *, now: datetime | None = None) -> None:
        now = now or datetime.now()
        user_id = _bounded_text(user_id, MAX_USER_ID_CHARS)
        if not user_id:
            return
        with self._lock:
            state = self._load()
            existed = state.get("users", {}).pop(user_id, None) is not None
            self._save(state)
            self._audit({
                "t": now.isoformat(), "user_id": user_id,
                "decision": "deleted" if existed else "delete_noop",
            })


__all__ = [
    "ShadowMemoryStore",
    "DEFAULT_STATE_FILE",
    "DEFAULT_AUDIT_FILE",
    "MEMORY_V2_EXTRACT_PROMPT",
    "shadow_enabled",
    "injection_enabled",
    "parse_extraction_payload",
    "build_extraction_prompt",
    "extract_and_ingest",
]
