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
# 最近发过的表情记多少个（这些都会被排除）。
# 光防"连着发"不够——用户报的是"**同类**太频繁"：换个话题回来还会撞上同一张。
# 记 8 个 ≈ 同一张至少隔 8 次表情包才会再出现。
STICKER_RECENT_KEEP = 8
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

# ==================== corpus 语义判（LLM 判"这段经历和用户刚说的话是一回事吗"）====================
# 背景：corpus 的召回原本只靠「余弦阈值 + 关键词门」，而口语问句与第三人称陈述**不同构**
# （实测「你多高啊」对身高那条只有 0.443，低于噪声地板 0.474）→ 她明明有的经历想不起来、只能现编。
# 实测（2026-09-25）：
#   · 靠 asks 索引（"用户可能怎么问"）**不成立**：正例确实涨到 0.88，但噪声地板同步抬到 0.822——
#     「明天几点开会」vs「明天几点下课」在语义/共享 bigram/IDF 重叠上**全都分不开**
#     （差别不在文本里，在她"到底有没有这段经历"这个事实）→ 只有 LLM 判得开。
#   · 但**排序有用**：正例的 top-1 往往就是对的那条（身高→#147、室友→#134、唱歌→#173），只是分数被淹没。
#   · LLM 判实测（top-6 候选，deepseek-flash）：反例 10/10 全拒（含「明天几点开会」），正例 2/5 保留。
# 所以：**门放行的照旧直通（不动，零成本）**；门没放行的取 top-N 交 LLM 判。
CORPUS_CANDIDATE_N = 6        # 交给 LLM 判的候选条数（实测 6 条判得准，多了反而稀释注意力）
CORPUS_JUDGE_MAX_KEEP = 2     # 判定最多保留几条（限制爆破半径：判错也只是多说一句，不是灌一堆）

VOICE_SAMPLE_TOP_N = 3        # 风格样本每路取 top
# 阈值不是拍的：跑 `python scripts/threshold_scan.py` 量「噪声地板」
# （15 条明确无关的输入打到每路上的最高分），阈值必须在地板之上，否则等于没有阈值。
# 2026-09-25 体检：voice_sample 地板 0.610 → 原 0.60 压在 Threshold 上，提到 0.66。
VOICE_SAMPLE_THRESHOLD = 0.66 # 风格样本阈值（噪声地板 0.610 + 余量）
VOICE_SAMPLE_KEEPALIVE = True # 样本全低于阈值时保底注入 1 条（保住口癖）
VOICE_SAMPLE_KEEPALIVE_MIN_SIM = 0.66  # 保底注入的最低相关度：与主阈值一致，低于则不注入（宁断档不错话题，防"日常聊时间"被塞直播样本）
VOICE_SAMPLE_MIN_K = 1        # 保底注入条数
VOICE_SAMPLE_PREFER_SHORT = True  # 注入时优先 short 档样本（控制回复长度）

# ==================== V3 措辞指纹检索 ====================
PHRASE_TOP_N = 2              # 措辞组每路取 top
# ⚠️ 这里是**已知失效但暂时回滚**的：措辞组的 trigger 只有六到十个字（"日常高频使用"），
# 短文本嵌入挤在向量空间中心 —— 15 条明确无关的输入能打出 0.575（中位 0.461），
# 而**正例**（"你唱得真好听！"→brag_deny）只有 **0.573**。**两者完全重叠，余弦分不开。**
# 试过提到 0.63：噪声挡住了，但正例一起被误杀（eval 27/29 → 25/29）。
# **结论：这一路该换判据（关键词/LLM 分类，像 behaviors 那样），不是调阈值。**
PHRASE_THRESHOLD = 0.40
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
# ⚠️ 同上：噪声地板 0.565 插在正例（0.501/0.552/0.590）**正中间** —— 余弦分不开。
# 试过 0.62：三个正例全被误杀。**该换判据，不是调阈值。**
PREFERENCE_THRESHOLD = 0.55
PREFERENCE_TOP_N = 2              # 最多注入几条偏好条目

# 核心记忆（印象最深的结晶，独立于 corpus 单独检索，低阈值高浮现）
CORE_STORY_VECTOR_FILE = PROJECT_ROOT / "persona" / "world" / "core_story_vectors.json"
CORE_STORY_THRESHOLD = 0.42       # 比偏好/行为低，核心故事更容易浮出（但低于此不注入，防乱触发）
CORE_STORY_TOP_N = 2              # 最多注入几条核心记忆
