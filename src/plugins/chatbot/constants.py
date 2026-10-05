"""集中管理路径 / API / 阈值的常量。

项目根目录：src/plugins/chatbot/ 往上数四级。
"""
from pathlib import Path

# ==================== 路径 ====================
# constants.py -> chatbot -> plugins -> src -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SYSTEM_PROMPT_FILE = PROJECT_ROOT / "persona" / "core" / "system_prompt.txt"
TRAITS_FILE = PROJECT_ROOT / "persona" / "core" / "traits.json"
STYLES_FILE = PROJECT_ROOT / "persona" / "core" / "styles.json"
BEHAVIORS_FILE = PROJECT_ROOT / "persona" / "behavior" / "behaviors.json"
TERMS_FILE = PROJECT_ROOT / "persona" / "world" / "terms.json"                 # 灰泽满名词库(lorebook)：核心词always注入+命中注入

MEMORY_FILE = PROJECT_ROOT / "user_memory" / "short_term.json"                 # 用户短期记忆
GROUP_MEMORY_FILE = PROJECT_ROOT / "user_memory" / "groups.json"              # 群级记忆（成员身份 + 群近况/梗，按群号）

# ==================== 群聊：群近况记忆 ====================
# 群近况事件写入冷却（秒）+ 上限条数（防碎碎念刷爆记忆）
GROUP_EVENT_COOLDOWN = 300
GROUP_EVENT_MAX = 20

# ==================== 群聊：@ 她的信号 ====================
# 群里 @ 她是一个**独立消息段**（at），`extract_plain_text()` 里看不见——
# 所以"@她 + 正文"进来时正文里没有名字，接话门"点名必接"的确定性判据命中不了。
# 检测到就在消息文本前插这个标记，让 ① 接话门确定性放行、② LLM 判据也看见"这@的是她"。
# 标记本身带"灰泽满"三字，`_addressed_to_her` 的名字表是**兜底**，别只靠它。
AT_SELF_MARK = "[有人@了灰泽满]"

# 群里"纯事件"（陈述一件跟她无关的事，如"明天要下雨""服务器又炸了"）的接话概率。
# 原来的判据只有 点名/情绪/经历/事务 四档，纯事件落到"事务"→ 一律不接，太安静。
# 真人在群里是这样的：**大多数时候不接，但偶尔会插一句**，所以给一个概率而不是开关。
GROUP_EVENT_REPLY_PROB = 0.3

# ==================== 表情包 ====================
# 距上次发过表情包至少隔几轮才再考虑发一张。真人不会每轮都甩表情包
# （qq-bridge 那边的经验值也是"普通闲聊每 3~5 轮一张"）。
# 注意这只是**冷却**，不是频率——真正决定发不发的是"表情与这句话是否十分对应"。
STICKER_COOLDOWN_TURNS = 3
# 最近发过的**类别**记几个（这些类别里的图都会被排除）。
# 用户 2026-10-04：去重按**类别**而不是单张——同一类里可能有好几张
# （委屈 3 张、无语无奈 4 张），按单张挡不住"换一张还是同一个味道"。
# ⚠️ 别按"张"来理解这个数字：库里有 18 类，记 8 类等于把近一半锁死，
# 所以从 8 降到 3（≈ 最近三次表情包不重类）。
STICKER_RECENT_KEEP = 3
SCHEDULE_FILE = PROJECT_ROOT / "persona" / "world" / "schedule.json"          # 她的周表(手动维护,注入地面真值)

VECTOR_FILE = PROJECT_ROOT / "persona" / "world" / "corpus_vectors.json"         # 直播记忆向量库（灰泽满的人物记忆，归 world/）
# 偏好：**已不用向量**，按 keywords 子串命中（2026-09-26 改判据）。
PREFERENCES_FILE = PROJECT_ROOT / "persona" / "world" / "preferences.json"
CORPUS_KEYWORDS_FILE = PROJECT_ROOT / "persona" / "world" / "corpus_keywords.json"  # corpus 的「钩子」

# ==================== API ====================
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
# 与 .env.prod 的 OPENAI_MODEL 保持一致——运行时不读这里（config.openai_model 优先），
# 但两者名字不同时排查会误导（曾以为是 v4-flash 在跑，其实是 deepseek-flash）。
DEFAULT_MODEL = "deepseek-flash"
EMBEDDING_MODEL = "embedding-3"

# DeepSeek V4 默认开启思考模式；思考模式下 temperature 等参数不被支持，
# 故对聊天 / 记忆提取调用显式禁用思考。
THINKING_DISABLED = {"extra_body": {"thinking": {"type": "disabled"}}}

# ==================== 对话参数 ====================
CHAT_TEMPERATURE = 0.85       # 实验：回高温度（低温度让模型更走 RP 默认括号模板）
CHAT_FREQUENCY_PENALTY = 0.3  # 实验：penalty 降低（0.5 未压括号还引入整句重复）
CHAT_MAX_TOKENS = 150
MEMORY_EXTRACT_TEMPERATURE = 0.1
MEMORY_EXTRACT_MAX_TOKENS = 100

# ==================== 检索阈值 ====================
RAG_THRESHOLD = 0.48  # corpus 语义下限。原 0.55 对自然问句过高（问句 vs 陈述式嵌入鸿沟，实测相关 0.51 也被卡掉）→ 降后靠关键词门挡噪声

# ==================== V6 corpus 关键词门 ====================
# 纯 cosine 无法区分"真相关(0.51)"与"名词撞车但不相关(0.62)"，任何单一阈值都两难。
# 门规则：语义 ≥ RAG_THRESHOLD 且与 statement 有区分性词重叠 → 放行；
#        区分性词重叠占比 ≥ 强阈值（专名/罕见短语如"乌色月"）→ 直接放行（语义低也收）。
# 区分性词 = 去掉领域停用字（灰泽满/绿冻/直播/你我他的…）后的 bigram 重叠占比。
CORPUS_KEYWORD_FLOOR = 0.12        # 区分性词重叠占比下限（**现在只决定"值不值得让 LLM 看一眼"**）
# ⚠️ 别想靠调这个值挡误报 —— 2026-09-29 试过了，方向是错的：
#   误报 n7（用户在说**自己**的作息）ov=0.40 > 真阳性 co9（在问**她**的旧事）ov=0.143。
#   **误报的重叠度反而更高**，提到 0.5 能挡住 n7，却把 co9/co10 一起打回红。
#   根因：词重叠拿不到"**说的是谁**"这个信息（bigram 文档频率也分不开，方向同样相反）。
#   所以杠杆改成"**取消直通**"——门/钩子只负责把候选递给 LLM 判（见 retrieval._corpus_gate_pass）。
#   这里保持 0.12（宽一点）：它现在的作用是**多给候选、别漏**，精度交给判定。
CORPUS_STRONG_KEYWORD = 0.8        # 强关键词重叠（专名/罕见短语）—— 现在也只是"优先递过去看"
# ⚠️ 这条在 50 条评测上实测**一次都没触发**（最高的 co1 目标条才 0.833）。留着无害，但它没出过力。

# ==================== 六路检索（corpus/样本/行为/措辞 走 RRF；偏好/核心记忆命中才带）====================
CORPUS_TOP_N = 3              # 直播记忆每路取 top（315 条库后 2→3，相关背景更容易命中）

# ==================== corpus 语义判（LLM 判"这段经历和用户刚说的话是一回事吗"）====================
# 背景：corpus 的召回原本只靠「余弦阈值 + 关键词门」，而口语问句与第三人称陈述**不同构**
# （实测「你多高啊」对身高那条只有 0.443，低于噪声地板 0.474）→ 她明明有的经历想不起来、只能现编。
# 实测（2026-09-25）：
#   · 靠 asks 索引（"用户可能怎么问"）**不成立**：正例确实涨到 0.88，但噪声地板同步抬到 0.822——
#     「明天几点开会」vs「明天几点下课」在语义/共享 bigram/IDF 重叠上**全都分不开**
#     （差别不在文本里，在她"到底有没有这段经历"这个事实）→ 只有 LLM 判得开。
#   · 但**排序有用**：正例的 top-1 往往就是对的那条（身高→#147、室友→#134、唱歌→#173），只是分数被淹没。
#   · LLM 判实测（top-6 候选，deepseek-flash）：反例 10/10 全拒（含「明天几点开会」），正例 2/5 保留。
# 所以：**门/钩子只决定"把哪些候选递给 LLM 看"，判定全部交 LLM**
# （2026-09-29 取消"门放行直通"——实测词重叠分不开"说的是谁"，见 retrieval._corpus_gate_pass 的注释）。
CORPUS_CANDIDATE_N = 6        # 交给 LLM 判的候选条数（实测 6 条判得准，多了反而稀释注意力）
CORPUS_LEXICAL_EXTRA_N = 6    # 除语义 top-N 外，**按词面沾边**再补几条候选（钩子命中 或 ov≥FLOOR）
# ⚠️ 为什么要单独一条、且**不看语义分**：要救的恰恰是"语义分低但用户确实提到了"那批——
#   co9「你不是以前有喜欢的男学霸吗，你直播里说的」对 #42 只有 ov=0.143、sim 不到 0.48，
#   若补位沿用 _corpus_gate_pass（要求 sim≥0.48），**它正好被挡在外面**（实测：候选只有 7 条、#42 不在里面）。
# ⚠️ 为什么封顶：ov≥0.12 很松（实测有的 query 能过 29 条），不封会把判定池灌爆
#   ——§2.3 的判据对比实验量过"候选给多了反而更差"。按 ov 从高到低取。
CORPUS_JUDGE_MAX_KEEP = 2     # 判定最多保留几条（限制爆破半径：判错也只是多说一句，不是灌一堆）

# ==================== V3 RRF 融合 ====================
RRF_K = 60                    # RRF 平滑常数
SOURCE_WEIGHTS = {"behavior": 1.5, "corpus": 1.0}
RETRIEVAL_TOPK = 6            # 融合后条数硬上限

# ==================== V3 预算控制 ====================
RETRIEVAL_BUDGET_CHARS = 1200      # 融合检索注入字符预算（corpus+samples）
MAX_RETRIEVAL_ITEM_CHARS = 300     # 单条检索结果字符上限

# ==================== 短期记忆 ====================
SHORT_MEMORY_LINES = 10  # 最近 5 轮，每轮 2 条（用户 + AI）；>3 轮可缓解承诺/借口遗忘

# ==================== 读秒窗口（方案B：消息攒批） ====================
# 静默窗口随机区间（真人打字回复普遍 5-10s；随机避免固定等长=机械感）
READ_WINDOW_MIN_SECONDS = 5.0
READ_WINDOW_MAX_SECONDS = 10.0

# ==================== 分批发送（打字感） ====================
SPLIT_REPLY_ENABLED = True     # 长回复拆成几句分开发送
SPLIT_MIN_LEN = 10             # 回复短于该长度不拆（灰泽满短句多，阈值别太高）
SPLIT_MERGE_MIN_CHARS = 8      # 拆分后比这短的分段并入下一段，防"哦？"这类微碎片单独成条；完整句(≥8字)保留分割
SPLIT_MAX_PARTS = 4            # 最多拆成几条消息，超出并入最后一条（防刷屏）
# 句间延迟按内容长度模拟打字（基础 + 每字，夹 MIN~MAX，±15% 抖动）
# 用户反馈 0.6~2.6s 太快"像涌出"，调大到 ~1.8~5s 还原打字过程
SPLIT_DELAY_BASE_MS = 1500
SPLIT_DELAY_PER_CHAR_MS = 100
SPLIT_DELAY_MIN_MS = 1800
SPLIT_DELAY_MAX_MS = 5000
SPLIT_DELAY_JITTER = 0.15

# ==================== 感知增强（路线B） ====================
# 方向1 感知
WEATHER_BASE_URL = "https://devapi.qweather.com"  # 和风 API Host 根；每项目专属域名，需在控制台查并配 WEATHER_BASE_URL
WEATHER_CACHE_SECONDS = 3600      # 天气进程内缓存 1 小时，省免费额度
WEATHER_GEO_CACHE_SECONDS = 86400  # 城市名→LocationID 解析缓存 1 天
# 方向2 视觉
VISION_MODEL = "glm-4.6v"         # 视觉理解模型（智谱 OpenAI 兼容；用户有免费 token）
VISION_MAX_TOKENS = 512           # 视觉描述输出上限
# 视觉调用关掉思考模式：glm-4.6v 默认会先推理（可拖到 9s+），关掉后秒出，消除"图片迟到"感
VISION_THINKING_DISABLED = {"extra_body": {"thinking": {"type": "disabled"}}}
# 方向3 B站联动
PUSH_INTERVAL = 180               # B站轮询间隔（秒）。别低于 120：带 SESSDATA 的会话调太频繁会被 B站风控顶掉（曾 45s→2-3天失效）
BILI_STATE_FILE = PROJECT_ROOT / "data" / "bili_state.json"   # 开播/动态去重状态持久化
WEIBO_STATE_FILE = PROJECT_ROOT / "data" / "weibo_state.json" # 微博去重状态持久化

# ==================== 语音回复（GPT-SoVITS） ====================
SOVITS_URL = "http://127.0.0.1:9880/tts"         # GPT-SoVITS api_v2.py 起的服务
SOVITS_REF_DIR = PROJECT_ROOT / "assets" / "voice_refs"  # 参考音频（ref_voice.wav+同名.txt 即可起步；ref_happy/lazy/serious 情绪档可选）
VOICE_CACHE_DIR = PROJECT_ROOT / "data" / "voice_cache"   # TTS 合成缓存（text_hash→wav）
# 语音触发下限：成句(≥30字)才朗读成语音条，更短的话打字——语音要像"话多一点才录给你听"，
# 不是成句就出声。实测她回复常 20+ 字，下限 20 时语音过密，提到 30 更克制。
# 寒暄词(晚安/辛苦啦…)可突破下限，见 voice.py _GREETING_PHRASES。
VOICE_MIN_LEN = 30
VOICE_MAX_LEN = 120   # 纯保险上限（~30s 音频，仍在 QQ 语音 ~60s 内），几乎不会触发

# 好友申请自动通过（否则新加的绿冻是单向好友，推送发不到，见 NapCat issue #72）
AUTO_ACCEPT_FRIEND = True

# 偏好档案（第 5 路）：**按 keywords 子串命中**，不走向量（路径见文件头 PREFERENCES_FILE）
PREFERENCE_TOP_N = 2              # 最多注入几条偏好条目

# 核心记忆（印象最深的结晶，独立于 corpus 单独检索，低阈值高浮现）
CORE_STORY_VECTOR_FILE = PROJECT_ROOT / "persona" / "world" / "core_story_vectors.json"
CORE_STORY_THRESHOLD = 0.42       # 比偏好/行为低，核心故事更容易浮出（但低于此不注入，防乱触发）
CORE_STORY_TOP_N = 2              # 最多注入几条核心记忆
