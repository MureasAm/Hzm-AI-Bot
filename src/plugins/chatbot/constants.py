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

# ==================== 表情包 ====================
# 距上次发过表情包至少隔几轮才再考虑发一张。真人不会每轮都甩表情包
# （qq-bridge 那边的经验值也是"普通闲聊每 3~5 轮一张"）。
# 注意这只是**冷却**，不是频率——真正决定发不发的是"表情与这句话是否十分对应"。
STICKER_COOLDOWN_TURNS = 3
SCHEDULE_FILE = PROJECT_ROOT / "persona" / "world" / "schedule.json"          # 她的周表(手动维护,注入地面真值)

# ==================== 已删除：schedule 的「近况」====================
# 曾经的「近况」（一个自由文本 + 更新日期 + 7 天 TTL，如"这周在收拾搬家"）已**整条删除**。
# 脉络：① 它被无条件注入时把"搬家"当成了当前理由，且被【一致性规则】锁死用了一周；
#       ② 于是加了 TTL，超期就不注入；③ 但 2026-09-25 实测发现——近况当时早已过期（19 天），
#          漏的根本不是它，而是 voice_samples 里一条"她本人说过"的搬家（详见 待办清单.md「本土化」）。
# **结论：近况治不了「用旧记忆回答当下」**——那条通道是 assistant turn（"她本人的话"），
# 远强于任何 system 说明。且这个字段手动维护必烂（实测烂了 20 天）。
# **别再把它加回来**：真正该做的是保持样本库/corpus 与直播内容同步，而不是再设一个会烂的字段。
VECTOR_FILE = PROJECT_ROOT / "persona" / "world" / "corpus_vectors.json"         # 直播记忆向量库（灰泽满的人物记忆，归 world/）
VOICE_SAMPLE_VECTOR_FILE = PROJECT_ROOT / "persona" / "speech" / "voice_sample_vectors.json"  # 声音样本向量缓存（跟 voice_samples.json）
PHRASE_VECTOR_FILE = PROJECT_ROOT / "persona" / "speech" / "phrase_vectors.json"   # 措辞指纹向量缓存（跟 phrases.json）

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
CORPUS_KEYWORD_FLOOR = 0.12        # 区分性词重叠占比下限（与语义阈值联合放行）
CORPUS_STRONG_KEYWORD = 0.8        # 强关键词重叠直接放行（专名/罕见短语）

# ==================== 六路检索（corpus/样本/行为/措辞 走 RRF；偏好/核心记忆命中才带）====================
CORPUS_TOP_N = 3              # 直播记忆每路取 top（315 条库后 2→3，相关背景更容易命中）
VOICE_SAMPLE_TOP_N = 3        # 风格样本每路取 top
VOICE_SAMPLE_THRESHOLD = 0.60 # 风格样本阈值（0.60：真直播命中≈0.70，日常"聊时间"伪关联≈0.575，取中间值切断伪关联）
VOICE_SAMPLE_KEEPALIVE = True # 样本全低于阈值时保底注入 1 条（保住口癖）
VOICE_SAMPLE_KEEPALIVE_MIN_SIM = 0.60  # 保底注入的最低相关度：与主阈值一致，低于则不注入（宁断档不错话题，防"日常聊时间"被塞直播样本）
VOICE_SAMPLE_MIN_K = 1        # 保底注入条数
VOICE_SAMPLE_PREFER_SHORT = True  # 注入时优先 short 档样本（控制回复长度）

# ==================== V3 措辞指纹检索 ====================
PHRASE_TOP_N = 2              # 措辞组每路取 top
PHRASE_THRESHOLD = 0.40       # 措辞组阈值（略高于样本，避免误命中）
PHRASE_PHASES_MAX = 3         # 每个措辞组注入的短语条数上限

# ==================== V3 RRF 融合 ====================
RRF_K = 60                    # RRF 平滑常数
SOURCE_WEIGHTS = {"behavior": 1.5, "corpus": 1.0, "voice_sample": 1.0, "phrase": 1.2}
RETRIEVAL_TOPK = 6            # 融合后条数硬上限

# ==================== V3 预算控制 ====================
RETRIEVAL_BUDGET_CHARS = 1200      # 融合检索注入字符预算（corpus+samples）
MAX_RETRIEVAL_ITEM_CHARS = 300     # 单条检索结果字符上限
VOICE_SAMPLE_REPLY_TRIM_CHARS = 60   # 长样本回复裁剪到该字数（引导短句）

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

# 偏好档案（第 5 路语义检索）
PREFERENCE_VECTOR_FILE = PROJECT_ROOT / "persona" / "world" / "preference_vectors.json"
PREFERENCE_THRESHOLD = 0.55       # 偏好命中阈值（实测：真实命中 0.58+，短句偏好向量泛化过头会误命中，取 0.55 压误命中）
PREFERENCE_TOP_N = 2              # 最多注入几条偏好条目

# 核心记忆（印象最深的结晶，独立于 corpus 单独检索，低阈值高浮现）
CORE_STORY_VECTOR_FILE = PROJECT_ROOT / "persona" / "world" / "core_story_vectors.json"
CORE_STORY_THRESHOLD = 0.42       # 比偏好/行为低，核心故事更容易浮出（但低于此不注入，防乱触发）
CORE_STORY_TOP_N = 2              # 最多注入几条核心记忆
